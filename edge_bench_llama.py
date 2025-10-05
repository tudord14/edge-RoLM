#!/usr/bin/env python3
# edge_bench_llama.py
# Benchmarks GGUF models with llama-cpp-python on Jetson devices (Nano/Xavier/Orin).
# Records: prompt vs. decode timing, robust quantiles, integrated energy (raw & net of idle),
# TTFT (including prompt eval), memory, temps, throttle flags, and key runtime knobs.

import os, sys, time, csv, re, queue, threading, statistics as stats, socket, platform, subprocess
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import argparse

# --- Dependencies ---
# pip install llama-cpp-python psutil numpy
# Optional: pip install nvidia-ml-py3
# Jetson: tegrastats at /usr/bin/tegrastats

try:
    import psutil
except Exception as e:
    print("psutil is required: pip install psutil", file=sys.stderr); raise

try:
    from llama_cpp import Llama, __version__ as LLAMA_CPP_PY_VERSION
except Exception as e:
    print("llama-cpp-python is required: pip install llama-cpp-python", file=sys.stderr); raise

# numpy is optional; used for robust percentiles if present
try:
    import numpy as np
    NUMPY_OK = True
except Exception:
    NUMPY_OK = False

# NVML (rarely useful on Nano, but keep)
NVML_OK = False
try:
    import pynvml
    pynvml.nvmlInit()
    NVML_OK = True
except Exception:
    NVML_OK = False

TEGRASTATS_PATH = "/usr/bin/tegrastats"

# ------------------------- Tegrastats monitoring -------------------------

@dataclass
class TSamples:
    t_sec: List[float] = field(default_factory=list)      # wall-clock seconds from start()
    power_w: List[float] = field(default_factory=list)    # POM_5V_IN (W)
    gpu_temp_c: List[float] = field(default_factory=list)
    cpu_temp_c: List[float] = field(default_factory=list)
    throttling: List[str] = field(default_factory=list)
    vram_mb: List[float] = field(default_factory=list)    # rarely exposed
    ram_mb: List[float] = field(default_factory=list)

class TegrastatsMonitor:
    """
    Spawns 'tegrastats --interval <ms>' and parses:
    - POM_5V_IN power (mW) -> W
    - GPU@ temp C, CPU@ temp C
    - RAM used/total (MB)
    - Throttling hints
    Stores timestamps to enable energy integration.
    """
    def __init__(self, interval_ms: int = 200):
        self.interval_ms = interval_ms
        self.proc = None
        self.thread = None
        self.running = False
        self.samples = TSamples()
        self._t0 = None

    def start(self) -> bool:
        if not os.path.exists(TEGRASTATS_PATH):
            return False
        cmd = [TEGRASTATS_PATH, "--interval", str(self.interval_ms)]
        self.proc = psutil.Popen(
            cmd,
            stdout=psutil.subprocess.PIPE,
            stderr=psutil.subprocess.PIPE,
            text=True
        )
        self.running = True
        self._t0 = time.time()
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()
        return True

    def stop(self):
        self.running = False
        if self.proc:
            try:
                self.proc.terminate()
                self.proc.wait(timeout=1.0)
            except Exception:
                try: self.proc.kill()
                except Exception: pass
        if self.thread:
            self.thread.join(timeout=1.0)

    def _reader(self):
        while self.running and self.proc and self.proc.stdout:
            line = self.proc.stdout.readline()
            if not line:
                break
            self._parse_line(line)

    def _parse_line(self, line: str):
        now = time.time()
        if self._t0 is None:  # should not happen
            self._t0 = now
        t_rel = now - self._t0
        # Power: POM_5V_IN X/Y (mW), take the instantaneous (X)
        m = re.search(r"POM_5V_IN\s+(\d+)\s*/\s*(\d+)", line)
        if m:
            mw_now = float(m.group(1))
            self.samples.t_sec.append(t_rel)
            self.samples.power_w.append(mw_now / 1000.0)
        else:
            # still append time so arrays stay aligned if needed
            self.samples.t_sec.append(t_rel)

        # Temps
        mg = re.search(r"GPU@(\d+)", line)
        if mg:
            self.samples.gpu_temp_c.append(float(mg.group(1)))
        mc = re.search(r"CPU@(\d+)", line)
        if mc:
            self.samples.cpu_temp_c.append(float(mc.group(1)))

        # RAM used/total MB
        mr = re.search(r"RAM\s+(\d+)/(\d+)MB", line)
        if mr:
            used = float(mr.group(1))
            self.samples.ram_mb.append(used)

        # Throttling hints
        if "throt" in line.lower() or "OC" in line:
            self.samples.throttling.append(line.strip())

def integrate_energy(samples: TSamples) -> Optional[float]:
    """Integrate power over time via trapezoidal rule to get Joules."""
    if not samples.power_w or len(samples.power_w) < 2:
        return None
    power = samples.power_w
    if not samples.t_sec or len(samples.t_sec) != len(power):
        dt = 0.2  # default 200 ms
        return sum(p * dt for p in power)
    energy = 0.0
    for i in range(1, len(power)):
        p0, p1 = power[i-1], power[i]
        dt = max(0.0, samples.t_sec[i] - samples.t_sec[i-1])
        energy += 0.5 * (p0 + p1) * dt
    return energy

def idle_power_tegra(duration_s: float = 30.0, interval_ms: int = 200) -> Optional[float]:
    mon = TegrastatsMonitor(interval_ms=interval_ms)
    if not mon.start():
        return None
    time.sleep(duration_s)
    mon.stop()
    if mon.samples.power_w:
        return stats.fmean(mon.samples.power_w)
    return None

# ------------------------- NVML helpers -------------------------

def nvml_vram_mb() -> Optional[Tuple[float, float]]:
    if not NVML_OK:
        return None
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        used_mb = mem.used / (1024*1024)
        total_mb = mem.total / (1024*1024)
        return used_mb, total_mb
    except Exception:
        return None

def nvml_power_w() -> Optional[float]:
    if not NVML_OK:
        return None
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        mw = pynvml.nvmlDeviceGetPowerUsage(handle)
        return mw / 1000.0
    except Exception:
        return None

def nvml_temp_c() -> Optional[float]:
    if not NVML_OK:
        return None
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        return float(pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU))
    except Exception:
        return None

# ------------------------- Prompt builder (exact input_ids) -------------------------

def build_prompt_ids(llm: Llama, target_tokens: int) -> List[int]:
    # Use a short seed and tile its token ids exactly to target length.
    seed = "Acesta este un prompt de test pentru evaluare. "
    seed_ids = llm.tokenize(seed.encode("utf-8"))
    toks: List[int] = []
    while len(toks) < target_tokens:
        toks.extend(seed_ids)
    return toks[:target_tokens]

# ------------------------- Percentiles -------------------------

def percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    if NUMPY_OK:
        return float(np.percentile(np.array(values, dtype=float), pct))
    # nearest-rank fallback
    vals = sorted(values)
    k = max(1, int(round((pct / 100.0) * len(vals))))
    k = min(k, len(vals))
    return float(vals[k-1])

# ------------------------- Data structures -------------------------

@dataclass
class RunResult:
    hostname: str
    model: str
    quant: str
    n_ctx: int
    n_threads: int
    n_batch: int
    n_gpu_layers: int
    prompt_tokens: int
    new_tokens: int
    first_token_ms: float
    prompt_time_s: float
    prompt_tps: float
    decode_time_s: float
    decode_tps_overall: float
    decode_tps_mean: float
    decode_tps_p50: float
    decode_tps_p95: float
    peak_rss_mb: float
    avg_ram_mb: Optional[float]
    avg_vram_mb: Optional[float]
    max_vram_mb: Optional[float]
    tegra_energy_j: Optional[float]
    tegra_energy_per_tok_j: Optional[float]
    tegra_energy_per_tok_net_j: Optional[float]
    tegra_avg_power_w: Optional[float]
    tegra_max_power_w: Optional[float]
    tegra_avg_gpu_temp_c: Optional[float]
    tegra_max_gpu_temp_c: Optional[float]
    nvml_avg_power_w: Optional[float]
    nvml_max_power_w: Optional[float]
    nvml_avg_gpu_temp_c: Optional[float]
    nvml_max_gpu_temp_c: Optional[float]
    throttled: int
    notes: str = ""

# ------------------------- Benchmark core -------------------------

def run_benchmark(
    model_path: str,
    prompt_len: int,
    new_tokens: int,
    n_ctx: int,
    n_gpu_layers: int,
    use_gpu: bool,
    n_threads: int,
    n_batch: int,
    temperature: float = 0.2,
    top_p: float = 0.95,
    seed: int = 42,
    tegra_interval_ms: int = 200,
    idle_baseline_w: Optional[float] = None,
) -> RunResult:

    model_name = os.path.basename(model_path)
    quant = "Q5_K_M" if "Q5" in model_name.upper() else ("FP16" if "F16" in model_name.lower() or "FP16" in model_name.upper() else "UNKNOWN")
    hostname = socket.gethostname()

    # Load model
    t0_load = time.time()
    llm = Llama(
        model_path=model_path,
        n_ctx=n_ctx,
        n_gpu_layers=(n_gpu_layers if use_gpu else 0),
        logits_all=False,
        seed=seed,
        vocab_only=False,
        use_mmap=True,
        n_threads=n_threads,
        n_batch=n_batch,  # prompt KV batch size
    )
    t_load = time.time() - t0_load

    # Build prompt ids exactly
    input_ids = build_prompt_ids(llm, prompt_len)

    # Warn if context not really stressed (for "4096" claims)
    if prompt_len < int(0.85 * n_ctx):
        print(f"[WARN] prompt_len={prompt_len} < 0.85*n_ctx ({n_ctx}). "
              f"Consider adding a 4096-length case for long-context claims.", file=sys.stderr)

    # Warm-ups: cheap token & short prompt to page in
    llm("Ok.", max_tokens=1, temperature=0.0, top_p=1.0, echo=False)
    _ = llm(input_ids=input_ids[:128], max_tokens=0, temperature=0.0, top_p=1.0, echo=False, stream=False)

    # -------- Prompt-only timing (non-stream, no generation) --------
    t0_prompt = time.time()
    _ = llm(input_ids=input_ids, max_tokens=0, temperature=0.0, top_p=1.0, echo=False, stream=False)
    prompt_time_s = time.time() - t0_prompt
    prompt_tps = (len(input_ids) / prompt_time_s) if prompt_time_s > 0 else 0.0

    # -------- Decode (stream) with TTFT and token inter-arrival --------
    # Start tegrastats monitor
    tegra = TegrastatsMonitor(interval_ms=tegra_interval_ms)
    tegra_started = tegra.start()

    proc = psutil.Process(os.getpid())
    peak_rss_mb = proc.memory_info().rss / (1024*1024)
    ram_avg_accum = []

    # NVML sampling (secondary on Jetsons)
    nvml_power_samples, nvml_temp_samples = [], []

    timings = []            # inter-arrival seconds for tokens 2..N
    produced_tokens = 0
    first_token_ms = None

    gen_params = dict(
        input_ids=input_ids,
        max_tokens=new_tokens,
        temperature=temperature,
        top_p=top_p,
        stream=True,
        stop=[],
        echo=False,
    )

    start = time.time()
    last = start
    for _chunk in llm(**gen_params):
        now = time.time()
        if first_token_ms is None:
            first_token_ms = (now - start) * 1000.0
        dt = now - last
        last = now
        produced_tokens += 1
        if produced_tokens > 1:
            timings.append(dt)

        # memory (RSS), approximate average RAM via tegrastats (system-wide)
        try:
            rss = proc.memory_info().rss / (1024*1024)
            if rss > peak_rss_mb:
                peak_rss_mb = rss
        except Exception:
            pass

        if tegra_started and tegra.samples.ram_mb:
            ram_avg_accum.append(tegra.samples.ram_mb[-1])

        if NVML_OK:
            p = nvml_power_w()
            if p is not None:
                nvml_power_samples.append(p)
            t = nvml_temp_c()
            if t is not None:
                nvml_temp_samples.append(t)

    total_time = time.time() - start
    decode_time_s = total_time  # by construction

    # Stop monitor
    tegra.stop()

    # Throughputs
    inst_tps = [1.0/dt for dt in timings if dt > 1e-6]
    decode_tps_mean = stats.fmean(inst_tps) if inst_tps else 0.0
    decode_tps_p50  = percentile(inst_tps, 50.0) if inst_tps else 0.0
    decode_tps_p95  = percentile(inst_tps, 95.0) if inst_tps else 0.0
    decode_tps_overall = (produced_tokens / decode_time_s) if decode_time_s > 0 else 0.0

    # Tegrastats aggregates
    tegra_energy_j = integrate_energy(tegra.samples) if tegra_started else None
    tegra_avg_power_w = (stats.fmean(tegra.samples.power_w) if (tegra_started and tegra.samples.power_w) else None)
    tegra_max_power_w = (max(tegra.samples.power_w) if (tegra_started and tegra.samples.power_w) else None)
    tegra_avg_gpu_temp_c = (stats.fmean(tegra.samples.gpu_temp_c) if tegra.samples.gpu_temp_c else None)
    tegra_max_gpu_temp_c = (max(tegra.samples.gpu_temp_c) if tegra.samples.gpu_temp_c else None)
    avg_ram_mb = (stats.fmean(ram_avg_accum) if ram_avg_accum else None)

    tegra_energy_per_tok_j = None
    tegra_energy_per_tok_net_j = None
    if tegra_energy_j is not None and produced_tokens > 0:
        tegra_energy_per_tok_j = tegra_energy_j / produced_tokens
        if idle_baseline_w is not None and tegra.samples.t_sec:
            duration = max(0.0, tegra.samples.t_sec[-1] - tegra.samples.t_sec[0])
            net_energy = tegra_energy_j - (idle_baseline_w * duration)
            tegra_energy_per_tok_net_j = net_energy / produced_tokens

    # NVML aggregates (optional)
    nvml_avg_power_w = (stats.fmean(nvml_power_samples) if nvml_power_samples else None)
    nvml_max_power_w = (max(nvml_power_samples) if nvml_power_samples else None)
    nvml_avg_gpu_temp_c = (stats.fmean(nvml_temp_samples) if nvml_temp_samples else None)
    nvml_max_gpu_temp_c = (max(nvml_temp_samples) if nvml_temp_samples else None)

    # VRAM (often N/A on Nano)
    if NVML_OK:
        v = nvml_vram_mb()
        avg_vram_mb = v[0] if v else None
        max_vram_mb = v[0] if v else None
    else:
        avg_vram_mb = None
        max_vram_mb = None

    # Environment crumbs into notes (keeps header unchanged)
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    uname = " ".join(platform.uname())
    idle_str = (f"idle_w={idle_baseline_w:.2f}" if idle_baseline_w is not None else "idle_w=NA")
    notes = [
        f"load_s={t_load:.2f}",
        idle_str,
        f"llama_cpp_py={LLAMA_CPP_PY_VERSION}",
        f"python={py_ver}",
        f"uname={uname}"
    ]
    if tegra.samples.throttling:
        notes.append("THROTTLING_DETECTED")
    note_str = ";".join(notes)

    return RunResult(
        hostname=hostname,
        model=model_name,
        quant=quant,
        n_ctx=n_ctx,
        n_threads=n_threads,
        n_batch=n_batch,
        n_gpu_layers=(n_gpu_layers if use_gpu else 0),
        prompt_tokens=len(input_ids),
        new_tokens=produced_tokens,
        first_token_ms=first_token_ms or 0.0,
        prompt_time_s=prompt_time_s,
        prompt_tps=prompt_tps,
        decode_time_s=decode_time_s,
        decode_tps_overall=decode_tps_overall,
        decode_tps_mean=decode_tps_mean,
        decode_tps_p50=decode_tps_p50,
        decode_tps_p95=decode_tps_p95,
        peak_rss_mb=peak_rss_mb,
        avg_ram_mb=avg_ram_mb,
        avg_vram_mb=avg_vram_mb,
        max_vram_mb=max_vram_mb,
        tegra_energy_j=tegra_energy_j,
        tegra_energy_per_tok_j=tegra_energy_per_tok_j,
        tegra_energy_per_tok_net_j=tegra_energy_per_tok_net_j,
        tegra_avg_power_w=tegra_avg_power_w,
        tegra_max_power_w=tegra_max_power_w,
        tegra_avg_gpu_temp_c=tegra_avg_gpu_temp_c,
        tegra_max_gpu_temp_c=tegra_max_gpu_temp_c,
        nvml_avg_power_w=nvml_avg_power_w,
        nvml_max_power_w=nvml_max_power_w,
        nvml_avg_gpu_temp_c=nvml_avg_gpu_temp_c,
        nvml_max_gpu_temp_c=nvml_max_gpu_temp_c,
        throttled=int(bool(tegra.samples.throttling)),
        notes=note_str
    )

# ------------------------- Long-context stress -------------------------

def long_context_stress(model_path: str, n_ctx: int, seed: int = 123) -> str:
    try:
        llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            logits_all=False,
            seed=seed,
            vocab_only=False,
            n_threads=os.cpu_count() or 4,
        )
    except TypeError as e:
        return f"Error during Llama init: {str(e)}"
    input_ids = build_prompt_ids(llm, n_ctx - 16)
    try:
        _ = llm(input_ids=input_ids, max_tokens=8, temperature=0.0, top_p=1.0, stream=False)
        return "OK"
    except Exception as e:
        return f"FAIL: {type(e).__name__}: {e}"

# ------------------------- CLI -------------------------

def main():
    ap = argparse.ArgumentParser(description="Edge metrics for GGUF models on Jetson devices.")
    ap.add_argument("--models", nargs="+", default=[
        "New-Model-125M-Q5_K_M.gguf",
        "New-Model-125M-f16.gguf",
        "New-Model-260M-Q5_K_M.gguf",
        "New-Model-260M-f16.gguf",
    ], help="Paths to GGUF models")
    ap.add_argument("--n_ctx", type=int, default=4096, help="Context window (default 4096)")
    ap.add_argument("--new_tokens", type=int, default=128, help="Decode tokens to generate for timing")
    ap.add_argument("--prompt_lens", nargs="+", type=int, default=[64, 256, 1024, 4096], help="Prompt token lengths to test")
    ap.add_argument("--n_threads", type=int, default=max(1, (os.cpu_count() or 4) // 2), help="Threads for llama.cpp")
    ap.add_argument("--n_batch", type=int, default=256, help="Prompt KV batch size (n_batch)")
    ap.add_argument("--gpu_layers", type=int, default=0, help="n_gpu_layers (Jetson Nano typically CPU-only)")
    ap.add_argument("--cpu_only", action="store_true", help="Force CPU (n_gpu_layers=0)")
    ap.add_argument("--out_csv", default="edge_metrics.csv", help="Output CSV path")
    ap.add_argument("--no_idle_baseline", action="store_true", help="Skip idle power baseline capture")
    ap.add_argument("--idle_seconds", type=float, default=30.0, help="Idle baseline duration (s)")
    ap.add_argument("--ts_interval_ms", type=int, default=200, help="tegrastats sampling interval (ms)")
    ap.add_argument("--reps", type=int, default=3, help="Repetitions per (model, prompt_len) config")
    ap.add_argument("--cooldown", type=float, default=2.0, help="Seconds to sleep between repetitions")
    args = ap.parse_args()

    use_gpu = (not args.cpu_only) and (args.gpu_layers > 0)

    # Idle baseline (tegra). If unavailable, None.
    idle_w = None if args.no_idle_baseline else idle_power_tegra(args.idle_seconds, args.ts_interval_ms)
    if idle_w is not None:
        print(f"[IDLE] Baseline avg power: {idle_w:.2f} W")
    else:
        print("[IDLE] Baseline not available (no tegrastats or disabled).")

    # CSV header
    header = [
        "hostname","model","quant","n_ctx","n_threads","n_batch","n_gpu_layers",
        "prompt_tokens","new_tokens",
        "ttft_ms","prompt_time_s","prompt_tps",
        "decode_time_s","decode_tps_overall","decode_tps_mean","decode_tps_p50","decode_tps_p95",
        "peak_rss_mb","avg_ram_mb","avg_vram_mb","max_vram_mb",
        "tegra_energy_j","tegra_energy_per_tok_j","tegra_energy_per_tok_net_j",
        "tegra_avg_power_w","tegra_max_power_w",
        "tegra_avg_gpu_temp_c","tegra_max_gpu_temp_c",
        "nvml_avg_power_w","nvml_max_power_w",
        "nvml_avg_gpu_temp_c","nvml_max_gpu_temp_c",
        "throttled","notes"
    ]
    write_header = not os.path.exists(args.out_csv)

    with open(args.out_csv, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(header)

        for mp in args.models:
            if not os.path.exists(mp):
                print(f"[WARN] Model not found: {mp}")
                continue

            # Long-context smoke test
            stress = long_context_stress(mp, args.n_ctx)
            print(f"[STRESS] {os.path.basename(mp)} @ n_ctx={args.n_ctx}: {stress}")

            for pl in args.prompt_lens:
                for _rep in range(args.reps):
                    print(f"[RUN] {os.path.basename(mp)} | prompt={pl} | new={args.new_tokens} | "
                          f"threads={args.n_threads} | n_batch={args.n_batch} | gpu_layers={(args.gpu_layers if use_gpu else 0)}")
                    res = run_benchmark(
                        model_path=mp,
                        prompt_len=pl,
                        new_tokens=args.new_tokens,
                        n_ctx=args.n_ctx,
                        n_gpu_layers=args.gpu_layers,
                        use_gpu=use_gpu,
                        n_threads=args.n_threads,
                        n_batch=args.n_batch,
                        tegra_interval_ms=args.ts_interval_ms,
                        idle_baseline_w=idle_w,
                    )

                    w.writerow([
                        res.hostname, res.model, res.quant, res.n_ctx, res.n_threads, res.n_batch, res.n_gpu_layers,
                        res.prompt_tokens, res.new_tokens,
                        f"{res.first_token_ms:.2f}", f"{res.prompt_time_s:.3f}", f"{res.prompt_tps:.2f}",
                        f"{res.decode_time_s:.3f}", f"{res.decode_tps_overall:.3f}", f"{res.decode_tps_mean:.3f}",
                        f"{res.decode_tps_p50:.3f}", f"{res.decode_tps_p95:.3f}",
                        f"{res.peak_rss_mb:.1f}",
                        (f"{res.avg_ram_mb:.1f}" if res.avg_ram_mb is not None else ""),
                        (f"{res.avg_vram_mb:.1f}" if res.avg_vram_mb is not None else ""),
                        (f"{res.max_vram_mb:.1f}" if res.max_vram_mb is not None else ""),
                        (f"{res.tegra_energy_j:.4f}" if res.tegra_energy_j is not None else ""),
                        (f"{res.tegra_energy_per_tok_j:.6f}" if res.tegra_energy_per_tok_j is not None else ""),
                        (f"{res.tegra_energy_per_tok_net_j:.6f}" if res.tegra_energy_per_tok_net_j is not None else ""),
                        (f"{res.tegra_avg_power_w:.2f}" if res.tegra_avg_power_w is not None else ""),
                        (f"{res.tegra_max_power_w:.2f}" if res.tegra_max_power_w is not None else ""),
                        (f"{res.tegra_avg_gpu_temp_c:.1f}" if res.tegra_avg_gpu_temp_c is not None else ""),
                        (f"{res.tegra_max_gpu_temp_c:.1f}" if res.tegra_max_gpu_temp_c is not None else ""),
                        (f"{res.nvml_avg_power_w:.2f}" if res.nvml_avg_power_w is not None else ""),
                        (f"{res.nvml_max_power_w:.2f}" if res.nvml_max_power_w is not None else ""),
                        (f"{res.nvml_avg_gpu_temp_c:.1f}" if res.nvml_avg_gpu_temp_c is not None else ""),
                        (f"{res.nvml_max_gpu_temp_c:.1f}" if res.nvml_max_gpu_temp_c is not None else ""),
                        res.throttled,
                        res.notes
                    ])
                    f.flush()
                    time.sleep(args.cooldown)

    print(f"\nDone. Wrote results to {args.out_csv}")
    print("Tip: collect per-device CSVs (Nano/Pi/Phone) and aggregate by hostname for the paper.")

if __name__ == "__main__":
    main()

