"""Benchmark torch matmul across sizes and dtypes, report TFLOPS.

This is the baseline for the PyTorch Lab. Run it as-is first, then change
sizes / dtypes / batched-vs-matmul to see where TFLOPS plateaus.
"""
import time
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SIZES = [1024, 2048, 4096, 8192]
DTYPES = [torch.float32, torch.float16, torch.bfloat16]


def bench(M: int, N: int, K: int, dtype: torch.dtype, iters: int = 20) -> float:
    a = torch.randn(M, K, device=DEVICE, dtype=dtype)
    b = torch.randn(K, N, device=DEVICE, dtype=dtype)
    # warmup
    for _ in range(5):
        torch.matmul(a, b)
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        c = torch.matmul(a, b)
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / iters
    flops = 2.0 * M * N * K
    return flops / dt / 1e12  # TFLOPS


def main():
    print(f"device={DEVICE} gpu={torch.cuda.get_device_name(0) if DEVICE=='cuda' else 'cpu'}")
    print(f"{'size':>6}  {'dtype':>10}  {'TFLOPS':>8}")
    for s in SIZES:
        for dt in DTYPES:
            try:
                t = bench(s, s, s, dt)
            except Exception as e:
                t = float("nan")
            print(f"{s:>6}  {str(dt).split('.')[-1]:>10}  {t:8.2f}")


if __name__ == "__main__":
    main()
