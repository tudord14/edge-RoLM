import os, sys, time, csv, socket, platform, statistics as stats, subprocess
from dataclasses import dataclass
from typing import List, Optional
import argparse

try:
    import psutil
except Exception as e:
    print("psutil is required: pip install psutil", file=sys.stderr); raise

try:
    from llama_cpp import Llama, __version__ as LLAMA_CPP_PY_VERSION
except Exception as e:
    print("llama-cpp-python is required: pip install llama-cpp-python", file=sys.stderr); raise

try:
    import numpy as np
    NUMPY_OK = True
except Exception:
    NUMPY_OK = False

def percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    if NUMPY_OK:
        return float(np.percentile(np.asarray(values, dtype=float), pct))
    vals = sorted(values)
    k = max(1, int(round((pct / 100.0) * len(vals))))
    k = min(k, len(vals))
    return float(vals[k-1])

def read_cpu_temp() -> Optional[float]:
    try:
        out = subprocess.check_output(["vcgencmd", "measure_temp"], text=True).strip()
        if "temp=" in out:
            val = out.split("temp=")[1].split("'")[0]
            return float(val)
    except Exception:
        pass
    try:
        candidates = ["/sys/class/thermal/thermal_zone0/temp",
                      "/sys/class/thermal/thermal_zone1/temp"]
        for p in candidates:
            if os.path.exists(p):
                raw = open(p).read().strip()
                if raw.isdigit():
                    return float(raw) / 1000.0
                try:
                    return float(raw)
                except Exception:
                    pass
    except Exception:
        pass
    return None

def build_prompt_ids(llm: Llama, target_tokens: int) -> List[int]:
    seed = "Acesta este un prompt de test pentru evaluare. "
    seed_ids = llm.tokenize(seed.encode("utf-8"))
    toks: List[int] = []
    while len(toks) < target_tokens:
        toks.extend(seed_ids)
    return toks[:target_tokens]

@dataclass
class RunResult:
    hostname: str
    model: str
    quant: str
    n_ctx: int
    n_threads: int
    n_batch: int
    prompt_tokens: int
    new_tokens: int
    ttft_ms: float
    prompt_time_s: float
    prompt_tps: float
    decode_time_s: float
    decode_tps_overall: float
    decode_tps_mean: float
    decode_tps_p50: float
    decode_tps_p95: float
    peak_rss_mb: float
    avg_cpu_temp_c: Optional[float]
    notes: str

def run_benchmark(
    model_path: str,
    prompt_len: int,
    new_tokens: int,
    n_ctx: int,
    n_threads: int,
    n_batch: int,
    temperature: float = 0.2,
    top_p: float = 0.95,
    seed: int = 42,
) -> RunResult:

    model_name = os.path.basename(model_path)
    quant = "Q5_K_M" if "Q5" in model_name.upper() else ("FP16" if "F16" in model_name.upper() else "UNKNOWN")
    hostname = socket.gethostname()

    t0_load = time.time()
    llm = Llama(
        model_path=model_path,
        n_ctx=n_ctx,
        logits_all=False,
        seed=seed,
        vocab_only=False,
        use_mmap=True,
        n_threads=n_threads,
        n_batch=n_batch,
    )
    load_s = time.time() - t0_load

    input_ids = build_prompt_ids(llm, prompt_len)

    # Warm-up
    llm("Ok.", max_tokens=1, temperature=0.0, top_p=1.0, echo=False)
    _ = llm(input_ids=input_ids[:128], max_tokens=0, temperature=0.0, top_p=1.0, echo=False, stream=False)

    # Prompt-only timing
    t0_prompt = time.time()
    _ = llm(input_ids=input_ids, max_tokens=0, temperature=0.0, top_p=1.0, echo=False, stream=False)
    prompt_time_s = time.time() - t0_prompt
    prompt_tps = (len(input_ids) / prompt_time_s) if prompt_time_s > 0 else 0.0

    # Decode streaming
    proc = psutil.Process(os.getpid())
    peak_rss_mb = proc.memory_info().rss / (1024 * 1024)

    timings = []
    produced_tokens = 0
    ttft_ms = None

    temps = []
    last_temp_sample = 0.0

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
        if ttft_ms is None:
            ttft_ms = (now - start) * 1000.0
        dt = now - last
        last = now
        produced_tokens += 1
        if produced_tokens > 1:
            timings.append(dt)
        try:
            rss = proc.memory_info().rss / (1024 * 1024)
            if rss > peak_rss_mb:
                peak_rss_mb = rss
        except Exception:
            pass
        if now - last_temp_sample >= 0.1:
            t = read_cpu_temp()
            if t is not None:
                temps.append(t)
            last_temp_sample = now

    decode_time_s = time.time() - start
    inst_tps = [1.0 / dt for dt in timings if dt > 1e-6]
    decode_tps_mean = stats.fmean(inst_tps) if inst_tps else 0.0
    decode_tps_p50 = percentile(inst_tps, 50.0) if inst_tps else 0.0
    decode_tps_p95 = percentile(inst_tps, 95.0) if inst_tps else 0.0
    decode_tps_overall = (produced_tokens / decode_time_s) if decode_time_s > 0 else 0.0

    avg_cpu_temp_c = (stats.fmean(temps) if temps else None)

    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    uname = " ".join(platform.uname())
    notes = f"load_s={load_s:.2f};llama_cpp_py={LLAMA_CPP_PY_VERSION};python={py_ver};uname={uname}"

    return RunResult(
        hostname=hostname,
        model=model_name,
        quant=quant,
        n_ctx=n_ctx,
        n_threads=n_threads,
        n_batch=n_batch,
        prompt_tokens=len(input_ids),
        new_tokens=produced_tokens,
        ttft_ms=ttft_ms or 0.0,
        prompt_time_s=prompt_time_s,
        prompt_tps=prompt_tps,
        decode_time_s=decode_time_s,
        decode_tps_overall=decode_tps_overall,
        decode_tps_mean=decode_tps_mean,
        decode_tps_p50=decode_tps_p50,
        decode_tps_p95=decode_tps_p95,
        peak_rss_mb=peak_rss_mb,
        avg_cpu_temp_c=avg_cpu_temp_c,
        notes=notes
    )

def main():
    ap = argparse.ArgumentParser(description="Raspberry Pi edge metrics for GGUF models (llama.cpp).")
    ap.add_argument("--models", nargs="+", default=[
        "New-Model-125M-Q5_K_M.gguf",
        "New-Model-125M-f16.gguf",
        "New-Model-260M-Q5_K_M.gguf",
        "New-Model-260M-f16.gguf",
    ], help="Paths to GGUF models")
    ap.add_argument("--n_ctx", type=int, default=4096, help="Context window")
    ap.add_argument("--new_tokens", type=int, default=128, help="Decode tokens to generate")
    ap.add_argument("--prompt_lens", nargs="+", type=int, default=[64, 256, 1024, 4096], help="Prompt lengths to test")
    ap.add_argument("--n_threads", type=int, default=max(1, (os.cpu_count() or 4) // 2), help="Threads for llama.cpp")
    ap.add_argument("--n_batch", type=int, default=256, help="Prompt KV batch size")
    ap.add_argument("--reps", type=int, default=1, help="Repetitions per (model, prompt_len)")
    ap.add_argument("--cooldown", type=float, default=1.0, help="Seconds to sleep between repetitions")
    ap.add_argument("--out_csv", default="edge_metrics_pi.csv", help="Output CSV")
    args = ap.parse_args()

    print(f"[INFO] CSV output: {os.path.join(os.getcwd(), args.out_csv)}")

    header = [
        "hostname","model","quant","n_ctx","n_threads","n_batch",
        "prompt_tokens","new_tokens",
        "ttft_ms","prompt_time_s","prompt_tps",
        "decode_time_s","decode_tps_overall","decode_tps_mean","decode_tps_p50","decode_tps_p95",
        "peak_rss_mb","avg_cpu_temp_c","notes"
    ]
    write_header = not os.path.exists(args.out_csv)

    with open(args.out_csv, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(header)

        for mp in args.models:
            if not os.path.exists(mp):
                print(f"[WARN] Model file not found: {mp}")
                continue

            try:
                _ = Llama(model_path=mp, n_ctx=args.n_ctx, n_threads=max(1, (os.cpu_count() or 4)//2))
            except Exception as e:
                print(f"[STRESS] {os.path.basename(mp)} @ n_ctx={args.n_ctx}: FAIL {e}")
            else:
                print(f"[STRESS] {os.path.basename(mp)} @ n_ctx={args.n_ctx}: OK")
            finally:
                time.sleep(0.5)

            for pl in args.prompt_lens:
                for rep in range(1, args.reps + 1):
                    print(f"[RUN] {os.path.basename(mp)} | rep={rep} | prompt={pl} | new={args.new_tokens} | "
                          f"threads={args.n_threads} | n_batch={args.n_batch}")
                    try:
                        res = run_benchmark(
                            model_path=mp,
                            prompt_len=pl,
                            new_tokens=args.new_tokens,
                            n_ctx=args.n_ctx,
                            n_threads=args.n_threads,
                            n_batch=args.n_batch,
                        )
                    except Exception as e:
                        print(f"[WARN] benchmark failed: {e}")
                        time.sleep(args.cooldown)
                        continue

                    w.writerow([
                        res.hostname, res.model, res.quant, res.n_ctx, res.n_threads, res.n_batch,
                        res.prompt_tokens, res.new_tokens,
                        f"{res.ttft_ms:.2f}", f"{res.prompt_time_s:.3f}", f"{res.prompt_tps:.2f}",
                        f"{res.decode_time_s:.3f}", f"{res.decode_tps_overall:.3f}", f"{res.decode_tps_mean:.3f}",
                        f"{res.decode_tps_p50:.3f}", f"{res.decode_tps_p95:.3f}",
                        f"{res.peak_rss_mb:.1f}",
                        (f"{res.avg_cpu_temp_c:.1f}" if res.avg_cpu_temp_c is not None else ""),
                        res.notes
                    ])
                    f.flush()

                    print(f"   -> tokens: prompt={res.prompt_tokens}, gen={res.new_tokens} | "
                          f"TTFT={res.ttft_ms:.0f} ms | gen_tps_overall={res.decode_tps_overall:.2f} | "
                          f"RSS={res.peak_rss_mb:.0f} MB | CPU≈{(res.avg_cpu_temp_c or 0):.1f}°C")

                    time.sleep(args.cooldown)

    print(f"\ndone wrote results to {args.out_csv}")
    

if __name__ == "__main__":
    main()
