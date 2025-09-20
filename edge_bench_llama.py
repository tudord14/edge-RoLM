#!/usr/bin/env python3
# edge_bench_llama.py
# Benchmarks GGUF models with llama-cpp-python on Jetson devices (Nano/Xavier/Orin).
# Metrics: first-token latency, tokens/sec (mean/p50/p95), RAM/VRAM, temperature,
# power (W) and energy/token (J/token), plus long-context (2k) stability.

import os, sys, time, csv, math, re, queue, threading, statistics as stats
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import argparse

# --- Dependencies you need on the device ---
# pip install llama-cpp-python psutil numpy
# Optional (prefer): pip install nvidia-ml-py3
# tegrastats must be present at /usr/bin/tegrastats (default on Jetsons)

try:
    import psutil
except Exception as e:
    print("psutil is required: pip install psutil", file=sys.stderr); raise

try:
    from llama_cpp import Llama
except Exception as e:
    print("llama-cpp-python is required: pip install llama-cpp-python", file=sys.stderr); raise

# Try NVML for VRAM/power/temps (not always available on Nano)
NVML_OK = False
try:
    import pynvml
    pynvml.nvmlInit()
    NVML_OK = True
except Exception:
    NVML_OK = False

# ------------------------- Tegrastats monitoring -------------------------

TEGRASTATS_PATH = "/usr/bin/tegrastats"

@dataclass
class TSamples:
    power_w: List[float] = field(default_factory=list)
    gpu_temp_c: List[float] = field(default_factory=list)
    cpu_temp_c: List[float] = field(default_factory=list)
    throttling: List[str] = field(default_factory=list)
    vram_mb: List[float] = field(default_factory=list)  # parsed VRAM if exposed
    ram_mb: List[float] = field(default_factory=list)

class TegrastatsMonitor:
    """
    Spawns 'tegrastats --interval <ms>' and parses:
    - POM_5V_IN power (mW) -> W
    - GPU@ temp C, CPU@ temp C
    - RAM used/total MB
    - (If present) GR3D or GPU memory (not always available)
    - Throttling notes if appear in line
    """
    def __init__(self, interval_ms: int = 200):
        self.interval_ms = interval_ms
        self.proc = None
        self.thread = None
        self.q = queue.Queue()
        self.running = False
        self.samples = TSamples()

    def start(self):
        if not os.path.exists(TEGRASTATS_PATH):
            return False
        cmd = [TEGRASTATS_PATH, f"--interval", str(self.interval_ms)]
        self.proc = psutil.Popen(cmd, stdout=psutil.subprocess.PIPE, stderr=psutil.subprocess.PIPE, text=True)
        self.running = True
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
        # Power: look for POM_5V_IN X/Y (mW)
        m = re.search(r"POM_5V_IN\s+(\d+)\s*/\s*(\d+)", line)
        if m:
            mw_now = float(m.group(1))
            self.samples.power_w.append(mw_now / 1000.0)

        # Temps: GPU@xxC CPU@xxC
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

        # VRAM (rarely exposed in tegrastats; keep placeholder if present)
        # Some tegrastats versions expose "GVDD" or EMC lines; we skip if not obvious.

        # Throttling flags
        if "throt" in line.lower() or "OC" in line:
            self.samples.throttling.append(line.strip())

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
        mw = pynvml.nvmlDeviceGetPowerUsage(handle)  # milliwatts
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

# ------------------------- Prompt builder -------------------------

def build_prompt_of_tokens(llm: Llama, target_tokens: int) -> str:
    seed = (
        "Acesta este un text de test pentru evaluarea performanței modelelor de limbă română. "
        "Vom genera un prompt de lungime controlată pentru măsurarea latenței și a vitezei de inferență. "
    )
    toks = []
    seed_toks = llm.tokenize(seed.encode("utf-8"), add_bos=False)
    while len(toks) < target_tokens:
        toks.extend(seed_toks)
    toks = toks[:target_tokens]
    prompt = llm.detokenize(toks).decode("utf-8", errors="ignore")
    return prompt

# ------------------------- Benchmark core -------------------------

@dataclass
class RunResult:
    model: str
    quant: str
    prompt_tokens: int
    new_tokens: int
    batch: int
    first_token_ms: float
    tps_mean: float
    tps_p50: float
    tps_p95: float
    peak_rss_mb: float
    avg_vram_mb: Optional[float]
    max_vram_mb: Optional[float]
    avg_power_w: Optional[float]
    max_power_w: Optional[float]
    energy_per_token_j: Optional[float]
    avg_gpu_temp_c: Optional[float]
    max_gpu_temp_c: Optional[float]
    notes: str = ""

def monitor_memory_peak_mb(proc: psutil.Process) -> float:
    peak = 0.0
    for _ in range(5):
        try:
            rss = proc.memory_info().rss / (1024*1024)
            peak = max(peak, rss)
        except Exception:
            pass
        time.sleep(0.05)
    return peak

def run_benchmark(
    model_path: str,
    prompt_len: int,
    new_tokens: int,
    batch: int,
    n_ctx: int,
    n_gpu_layers: int,
    use_gpu: bool,
    temperature: float = 0.2,
    top_p: float = 0.95,
    seed: int = 42,
) -> RunResult:
    model_name = os.path.basename(model_path)
    quant = "Q5_K_M" if "Q5" in model_name.upper() else ("FP16" if "F16" in model_name.lower() or "fp16" in model_name.lower() else "UNKNOWN")

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
        n_threads=os.cpu_count() or 4,
    )
    t_load = time.time() - t0_load

    # Build prompt
    prompt = build_prompt_of_tokens(llm, prompt_len)

    # Start tegrastats/NVML monitors
    ts = TegrastatsMonitor(interval_ms=200)
    ts_started = ts.start()

    # Record memory in background
    proc = psutil.Process(os.getpid())
    peak_rss_mb = proc.memory_info().rss / (1024*1024)

    vram_samples = []
    power_samples = []
    gpu_temp_samples = []

    # Generation with streaming to time tokens
    timings = []
    first_token_ms = None
    produced_tokens = 0

    gen_params = dict(
        prompt=prompt,
        max_tokens=new_tokens,
        temperature=temperature,
        top_p=top_p,
        stream=True,
        stop=[]
    )

    # NOTE: llama.cpp "batch" here refers to KV prompt batch size (n_batch) not multi-sequence batching.
    # We'll set n_batch via context size heuristics; true multi-seq batching would require server API.
    # For now, we just record requested 'batch' for the report.
    # Warmup small call to reduce first-token skew from lazy init:
    llm("Ok.", max_tokens=1, temperature=0.0, top_p=1.0, echo=False)

    start = time.time()
    last = start
    for chunk in llm(**gen_params):
        now = time.time()
        if first_token_ms is None:
            first_token_ms = (now - start) * 1000.0
        dt = now - last
        last = now
        # chunk may include "choices"[0]["text"]; we measure per chunk arrival (per token granularity)
        produced_tokens += 1
        if produced_tokens > 1:   # first dt is from first->second token; keep all for stats
            timings.append(dt)

        # sample RAM/VRAM/Power opportunistically
        try:
            rss = proc.memory_info().rss / (1024*1024)
            if rss > peak_rss_mb:
                peak_rss_mb = rss
        except Exception:
            pass

        if NVML_OK:
            v = nvml_vram_mb()
            if v: vram_samples.append(v[0])
            p = nvml_power_w()
            if p is not None: power_samples.append(p)
            t = nvml_temp_c()
            if t is not None: gpu_temp_samples.append(t)

    total_time = time.time() - start

    # Stop monitors
    ts.stop()

    # Compute TPS stats
    # Instantaneous TPS for each interval = 1/dt
    inst_tps = [1.0/dt for dt in timings if dt > 1e-6]
    tps_mean = stats.fmean(inst_tps) if inst_tps else 0.0
    tps_p50 = stats.median(inst_tps) if inst_tps else 0.0
    tps_p95 = (stats.quantiles(inst_tps, n=100)[94] if len(inst_tps) >= 100 else (max(inst_tps) if inst_tps else 0.0))

    # Aggregate power/temps
    if NVML_OK and not ts_started:
        avg_power_w = stats.fmean(power_samples) if power_samples else None
        max_power_w = max(power_samples) if power_samples else None
        avg_gpu_temp_c = stats.fmean(gpu_temp_samples) if gpu_temp_samples else None
        max_gpu_temp_c = max(gpu_temp_samples) if gpu_temp_samples else None
    else:
        avg_power_w = stats.fmean(ts.samples.power_w) if ts.samples.power_w else None
        max_power_w = max(ts.samples.power_w) if ts.samples.power_w else None
        avg_gpu_temp_c = stats.fmean(ts.samples.gpu_temp_c) if ts.samples.gpu_temp_c else None
        max_gpu_temp_c = max(ts.samples.gpu_temp_c) if ts.samples.gpu_temp_c else None

    # Energy per token: average power * total_time / produced_tokens
    energy_per_token_j = None
    if avg_power_w is not None and produced_tokens > 0:
        energy_per_token_j = (avg_power_w * total_time) / produced_tokens

    # VRAM
    if NVML_OK and vram_samples:
        avg_vram_mb = stats.fmean(vram_samples)
        max_vram_mb = max(vram_samples)
    else:
        avg_vram_mb = None
        max_vram_mb = None

    notes = []
    if t_load > 0.0:
        notes.append(f"load_s={t_load:.2f}")
    if ts.samples.throttling:
        notes.append("THROTTLING_DETECTED")
    note_str = ";".join(notes)

    return RunResult(
        model=model_name,
        quant=quant,
        prompt_tokens=prompt_len,
        new_tokens=produced_tokens,
        batch=batch,
        first_token_ms=first_token_ms or 0.0,
        tps_mean=tps_mean,
        tps_p50=tps_p50,
        tps_p95=tps_p95,
        peak_rss_mb=peak_rss_mb,
        avg_vram_mb=avg_vram_mb,
        max_vram_mb=max_vram_mb,
        avg_power_w=avg_power_w,
        max_power_w=max_power_w,
        energy_per_token_j=energy_per_token_j,
        avg_gpu_temp_c=avg_gpu_temp_c,
        max_gpu_temp_c=max_gpu_temp_c,
        notes=note_str
    )

# ------------------------- Long-context stress -------------------------

def long_context_stress(model_path: str, n_ctx: int, n_gpu_layers: int, use_gpu: bool, seed: int = 123) -> str:
    llm = Llama(
        model_path=model_path,
        n_ctx=n_ctx,
        n_gpu_layers=(n_gpu_layers if use_gpu else 0),
        logits_all=False,
        seed=seed,
        vocab_only=False,
        use_mmap=True,
        n_threads=os.cpu_count() or 4,
    )
    # Build ~n_ctx prompt; reserve room for a few output tokens
    prompt = build_prompt_of_tokens(llm, n_ctx - 16)
    try:
        _ = llm(prompt=prompt, max_tokens=8, temperature=0.0, top_p=1.0, stream=False)
        return "OK"
    except Exception as e:
        return f"FAIL: {type(e).__name__}: {e}"

# ------------------------- CLI -------------------------

def main():
    ap = argparse.ArgumentParser(description="Edge metrics for GGUF models on Jetson devices.")
    ap.add_argument("--models", nargs="+", default=[
        "Model-125M-Q5_K_M.gguf",
        "Model-125M-f16.gguf",
        "Model-350M-Q5_K_M.gguf",
        "Model-350M-f16.gguf",
    ], help="Paths to GGUF models")
    ap.add_argument("--n_ctx", type=int, default=2048, help="Context window")
    ap.add_argument("--new_tokens", type=int, default=128, help="New tokens to generate for timing")
    ap.add_argument("--prompt_lens", nargs="+", type=int, default=[64, 256, 1024], help="Prompt token lengths to test")
    ap.add_argument("--batches", nargs="+", type=int, default=[1, 2], help="Logical batch sizes to record (see note)")
    ap.add_argument("--gpu_layers", type=int, default=999, help="n_gpu_layers (use 999 to push as many layers to GPU as compiled)")
    ap.add_argument("--cpu_only", action="store_true", help="Force CPU (n_gpu_layers=0)")
    ap.add_argument("--out_csv", default="edge_metrics.csv", help="Output CSV path")
    args = ap.parse_args()

    use_gpu = (not args.cpu_only)
    results: List[RunResult] = []

    # Write CSV header
    header = [
        "model","quant","prompt_tokens","new_tokens","batch",
        "first_token_ms","tps_mean","tps_p50","tps_p95",
        "peak_rss_mb","avg_vram_mb","max_vram_mb",
        "avg_power_w","max_power_w","energy_per_token_j",
        "avg_gpu_temp_c","max_gpu_temp_c","notes"
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

            # Long-context stress test
            stress = long_context_stress(mp, args.n_ctx, args.gpu_layers, use_gpu)
            print(f"[STRESS] {os.path.basename(mp)} @ n_ctx={args.n_ctx}: {stress}")

            for pl in args.prompt_lens:
                for b in args.batches:
                    print(f"[RUN] {os.path.basename(mp)} | prompt={pl} | batch={b} | new={args.new_tokens} | gpu={use_gpu}")
                    res = run_benchmark(
                        model_path=mp,
                        prompt_len=pl,
                        new_tokens=args.new_tokens,
                        batch=b,
                        n_ctx=args.n_ctx,
                        n_gpu_layers=args.gpu_layers,
                        use_gpu=use_gpu,
                    )
                    results.append(res)
                    w.writerow([
                        res.model, res.quant, res.prompt_tokens, res.new_tokens, res.batch,
                        f"{res.first_token_ms:.2f}",
                        f"{res.tps_mean:.3f}", f"{res.tps_p50:.3f}", f"{res.tps_p95:.3f}",
                        f"{res.peak_rss_mb:.1f}",
                        (f"{res.avg_vram_mb:.1f}" if res.avg_vram_mb is not None else ""),
                        (f"{res.max_vram_mb:.1f}" if res.max_vram_mb is not None else ""),
                        (f"{res.avg_power_w:.2f}" if res.avg_power_w is not None else ""),
                        (f"{res.max_power_w:.2f}" if res.max_power_w is not None else ""),
                        (f"{res.energy_per_token_j:.4f}" if res.energy_per_token_j is not None else ""),
                        (f"{res.avg_gpu_temp_c:.1f}" if res.avg_gpu_temp_c is not None else ""),
                        (f"{res.max_gpu_temp_c:.1f}" if res.max_gpu_temp_c is not None else ""),
                        res.notes
                    ])
                    # Flush each row
                    f.flush()

    print(f"\nDone. Wrote results to {args.out_csv}")
    print("Tip: run the same script unchanged on Xavier/Orin and tag rows by hostname:")
    print("     e.g., append --out_csv edge_metrics_$(hostname).csv")

if __name__ == "__main__":
    main()
