import argparse
import math
import statistics
import time

import torch

from miniattn.ref import ref
from miniattn.tiled import tiled


BATCH = 1
HEADS = 4
SEQ_LEN = 1024
DIM = 64
TILE_QS = (16, 32, 64, 128, 256)
WARMUPS = 5
REPEATS = 20


def measure(fn):
    with torch.inference_mode():
        for _ in range(WARMUPS):
            fn()
        samples = []
        for _ in range(REPEATS):
            begin = time.perf_counter()
            fn()
            samples.append((time.perf_counter() - begin) * 1000)
    return statistics.median(samples)


def score_mib(rows):
    return BATCH * HEADS * rows * SEQ_LEN * 4 / (1024 * 1024)


def main():
    parser = argparse.ArgumentParser(description="CPU Q-only tiled attention benchmark")
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    if args.threads <= 0:
        parser.error("--threads must be positive")

    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    q = torch.randn(BATCH, HEADS, SEQ_LEN, DIM)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    expected = ref(q, k, v)
    reference_ms = measure(lambda: ref(q, k, v))
    print("impl       tile_q  q_tiles  threads  median_ms  max_abs_error_vs_ref  max_score_mib")
    print(
        f"reference  {'-':>6}  {1:>7}  {args.threads:>7}  "
        f"{reference_ms:>9.3f}  {'-':>20}  {score_mib(SEQ_LEN):>13.2f}"
    )

    results = []
    for tile_q in TILE_QS:
        actual = tiled(q, k, v, tile_q)
        max_error = (actual - expected).abs().max().item()
        if not torch.allclose(actual, expected, rtol=1e-5, atol=1e-6):
            raise AssertionError(
                f"correctness failed for tile_q={tile_q}: max_abs_error={max_error}"
            )

        latency_ms = measure(lambda tile_q=tile_q: tiled(q, k, v, tile_q))
        q_tiles = math.ceil(SEQ_LEN / tile_q)
        results.append((tile_q, latency_ms))
        print(
            f"tiled      {tile_q:>6}  {q_tiles:>7}  {args.threads:>7}  "
            f"{latency_ms:>9.3f}  {max_error:>20.3e}  "
            f"{score_mib(min(tile_q, SEQ_LEN)):>13.2f}"
        )

    best_tile, best_latency = min(results, key=lambda item: item[1])
    print(f"best_tile_q={best_tile}")
    print(f"best_tiled_median_ms={best_latency:.3f}")


if __name__ == "__main__":
    main()
