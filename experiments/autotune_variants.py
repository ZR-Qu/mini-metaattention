import argparse
import json
import re
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
from matplotlib import font_manager

from examples.causal_relu import spec as causal_relu_spec
from examples.causal_softmax import spec as causal_softmax_spec
from examples.relu import spec as relu_spec
from miniattn.ref import ref, ref_causal_relu, ref_relu
from miniattn.tune import measure_latency, tune


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
PLOTS_DIR = ROOT / "plots"
BATCH = 1
HEADS = 4
SEQ_LEN = 512
DIM = 64
VALUE_DIM = 64
THREADS = 8
SEED = 0
TILE_CANDIDATES = (16, 32, 64, 128, 256, 512)
FIXED_TILE_Q = 64
FIXED_TILE_K = 64
VARIANT_LABELS = {
    "causal_softmax": "Causal Softmax",
    "relu": "ReLU",
    "causal_relu": "Causal ReLU",
}
IMPLEMENTATION_LABELS = {
    "reference": "Reference",
    "fixed": "Fixed 64×64",
    "tuned": "Tuned",
}
IMPLEMENTATION_COLORS = {
    "reference": "#D4D7DB",
    "fixed": "#A85C5C",
    "tuned": "#2F6FA5",
}
IMPLEMENTATION_HATCHES = {"reference": None, "fixed": "///", "tuned": None}
VARIANT_ORDER = ("causal_softmax", "relu", "causal_relu")
IMPLEMENTATION_ORDER = ("reference", "fixed", "tuned")

VARIANTS = (
    ("causal_softmax", causal_softmax_spec, ref),
    ("relu", relu_spec, ref_relu),
    ("causal_relu", causal_relu_spec, ref_causal_relu),
)


def _op_dict(op):
    return None if op is None else {"name": op.name, "value": op.value}


def _spec_fields(spec):
    return {
        "spec_name": spec.name,
        "pattern": spec.pattern,
        "score_mod": json.dumps(
            [_op_dict(op) for op in spec.score_mod], separators=(",", ":")
        ),
        "mask_mod": json.dumps(_op_dict(spec.mask_mod), separators=(",", ":")),
        "rownorm": json.dumps(_op_dict(spec.rownorm), separators=(",", ":")),
    }


def _context_fields(batch, heads, seq_len, dim, value_dim, threads):
    return {
        "batch": batch,
        "heads": heads,
        "sq": seq_len,
        "sk": seq_len,
        "dim": dim,
        "value_dim": value_dim,
        "dtype": "float32",
        "device": "cpu",
        "threads": threads,
    }


def _make_workload(batch, heads, seq_len, dim, value_dim):
    torch.manual_seed(SEED)
    q = torch.randn(batch, heads, seq_len, dim)
    k = torch.randn(batch, heads, seq_len, dim)
    v = torch.randn(batch, heads, seq_len, value_dim)
    return q, k, v


def _throughput(latency_ms, batch, seq_len):
    calls_per_second = 1000.0 / latency_ms
    tokens_per_second = batch * seq_len * calls_per_second
    return calls_per_second, tokens_per_second


def _reference_row(variant, spec, latency_ms, context):
    calls_per_second, tokens_per_second = _throughput(
        latency_ms, context["batch"], context["sq"]
    )
    return {
        "variant": variant,
        **_spec_fields(spec),
        **context,
        "implementation": "reference",
        "tile_q": None,
        "tile_k": None,
        "median_latency_ms": latency_ms,
        "calls_per_second": calls_per_second,
        "tokens_per_second": tokens_per_second,
    }


def _comparison_row(variant, spec, implementation, result, context):
    return {
        "variant": variant,
        **_spec_fields(spec),
        **context,
        "implementation": implementation,
        "tile_q": result.get("tile_q"),
        "tile_k": result.get("tile_k"),
        "median_latency_ms": result["median_latency_ms"],
        "calls_per_second": result["calls_per_second"],
        "tokens_per_second": result["tokens_per_second"],
    }


def _write_csv(path, rows):
    pd.DataFrame(rows).to_csv(path, index=False)


def _plot(compare_path, plot_path):
    font_path = Path("/mnt/c/Windows/Fonts/times.ttf")
    if font_path.exists():
        font_manager.fontManager.addfont(font_path)
    plt.rcParams["font.family"] = ["Times New Roman", "serif"]

    compare = pd.read_csv(compare_path)
    variant_order = list(VARIANT_ORDER)
    implementation_order = list(IMPLEMENTATION_ORDER)
    latency = compare.pivot(
        index="variant", columns="implementation", values="median_latency_ms"
    ).reindex(index=variant_order, columns=implementation_order)
    display_variants = [VARIANT_LABELS[name] for name in variant_order]
    tuned = latency["tuned"]
    fixed = latency["fixed"]
    speedup = fixed / tuned
    tuned_rows = (
        compare[compare["implementation"] == "tuned"]
        .set_index("variant")
        .reindex(variant_order)
    )

    fig, axis = plt.subplots(figsize=(5.2, 3.2), constrained_layout=True)
    fig.patch.set_facecolor("white")

    x_positions = list(range(len(variant_order)))
    bar_width = 0.20
    offsets = (-0.22, 0.0, 0.22)
    for implementation, offset in zip(implementation_order, offsets):
        bars = axis.bar(
            [x + offset for x in x_positions],
            latency[implementation],
            width=bar_width,
            label=IMPLEMENTATION_LABELS[implementation],
            color=IMPLEMENTATION_COLORS[implementation],
            edgecolor="black",
            linewidth=0.65,
            hatch=IMPLEMENTATION_HATCHES[implementation],
            zorder=3,
        )
        for bar in bars:
            axis.annotate(
                f"{bar.get_height():.2f}",
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=6.2,
                color="#222222",
            )

    for index in range(len(variant_order)):
        arrow_x = index + 0.34
        fixed_latency = float(fixed.iloc[index])
        tuned_latency = float(tuned.iloc[index])
        axis.annotate(
            "",
            xy=(arrow_x, fixed_latency),
            xytext=(arrow_x, tuned_latency),
            arrowprops={
                "arrowstyle": "<->",
                "color": "#333333",
                "linewidth": 0.65,
                "shrinkA": 0,
                "shrinkB": 0,
            },
        )
        axis.text(
            arrow_x + 0.035,
            (fixed_latency + tuned_latency) / 2,
            f"{speedup.iloc[index]:.2f}×",
            ha="left",
            va="center",
            fontsize=6.2,
            color="#9A3F3F",
        )

    for separator in (0.5, 1.5):
        axis.axvline(
            separator,
            color="#B8B8B8",
            linewidth=0.7,
            linestyle=(0, (3, 2)),
            zorder=1,
        )

    axis.set_ylabel("Median latency (ms)", fontsize=8)
    axis.set_xticks(x_positions, display_variants)
    for index, row in enumerate(tuned_rows.itertuples()):
        axis.text(
            index,
            -0.075,
            f"Best {int(row.tile_q)}×{int(row.tile_k)}",
            transform=axis.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=6.2,
            color=IMPLEMENTATION_COLORS["tuned"],
            clip_on=False,
        )
    axis.set_xlim(-0.55, 2.68)
    axis.set_ylim(0, float(latency.max().max()) * 1.18)
    axis.set_facecolor("white")
    axis.set_axisbelow(True)
    axis.grid(axis="y", color="#E4E7EA", linewidth=0.55)
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(0.7)
    axis.tick_params(axis="x", labelsize=7.2, width=0.6, length=2.5)
    axis.tick_params(axis="y", labelsize=7.2, width=0.6, length=2.5)
    axis.legend(
        frameon=False,
        fontsize=6.8,
        ncol=3,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        borderaxespad=0,
        handletextpad=0.4,
        columnspacing=0.9,
    )

    fig.savefig(
        plot_path,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


def run(
    *,
    batch=BATCH,
    heads=HEADS,
    seq_len=SEQ_LEN,
    dim=DIM,
    value_dim=VALUE_DIM,
    threads=THREADS,
    tile_candidates=TILE_CANDIDATES,
    warmups=5,
    repeats=20,
    label="variants",
):
    if not re.fullmatch(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*", label):
        raise ValueError("label must contain only letters, numbers, and hyphens")
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S") + f"-{label}"
    results_dir = RESULTS_DIR / run_id
    plots_dir = PLOTS_DIR / run_id
    results_dir.mkdir(parents=True)
    plots_dir.mkdir(parents=True)
    (results_dir / "run.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "label": label,
                "batch": batch,
                "heads": heads,
                "seq_len": seq_len,
                "dim": dim,
                "value_dim": value_dim,
                "dtype": "float32",
                "device": "cpu",
                "threads": threads,
                "tile_candidates": list(tile_candidates),
                "warmups": warmups,
                "repeats": repeats,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    torch.set_num_threads(threads)
    q, k, v = _make_workload(batch, heads, seq_len, dim, value_dim)
    context_fields = _context_fields(
        batch, heads, seq_len, dim, value_dim, threads
    )
    search_rows = []
    compare_rows = []

    for variant, spec, reference_fn in VARIANTS:
        reference_latency = measure_latency(
            lambda reference_fn=reference_fn: reference_fn(q, k, v),
            warmups=warmups,
            repeats=repeats,
        )
        compare_rows.append(
            _reference_row(variant, spec, reference_latency, context_fields)
        )

        results, best = tune(
            spec,
            q=q,
            k=k,
            v=v,
            reference_fn=reference_fn,
            tile_q_candidates=tile_candidates,
            tile_k_candidates=tile_candidates,
            warmups=warmups,
            repeats=repeats,
            atol=1e-4 if spec.rownorm.name == "identity" else 1e-6,
        )
        context = {"variant": variant, **_spec_fields(spec), **context_fields}
        search_rows.extend({**context, **result} for result in results)

        fixed = next(
            result
            for result in results
            if result["tile_q"] == FIXED_TILE_Q
            and result["tile_k"] == FIXED_TILE_K
        )
        compare_rows.append(
            _comparison_row(variant, spec, "fixed", fixed, context_fields)
        )
        compare_rows.append(
            _comparison_row(variant, spec, "tuned", best, context_fields)
        )

        speedup = fixed["median_latency_ms"] / best["median_latency_ms"]
        print(
            f"{variant:16s} "
            f"reference={reference_latency:.3f} ms "
            f"fixed={fixed['median_latency_ms']:.3f} ms "
            f"tuned={best['median_latency_ms']:.3f} ms "
            f"best={best['tile_q']}x{best['tile_k']} "
            f"speedup_vs_fixed={speedup:.3f}x"
        )

    search_path = results_dir / "variants_search.csv"
    compare_path = results_dir / "variants_compare.csv"
    _write_csv(search_path, search_rows)
    _write_csv(compare_path, compare_rows)
    _plot(compare_path, plots_dir / "variants.png")
    print(f"run_id={run_id}")
    print(f"results={results_dir.relative_to(ROOT)}")
    print(f"plots={plots_dir.relative_to(ROOT)}")
    return search_path, compare_path


def main():
    parser = argparse.ArgumentParser(
        description="Run the Mini MetaAttention cross-variant CPU autotuning experiment"
    )
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--label", default="variants")
    parser.add_argument("--batch", type=int, default=BATCH)
    parser.add_argument("--heads", type=int, default=HEADS)
    parser.add_argument("--seq-len", type=int, default=SEQ_LEN)
    parser.add_argument("--dim", type=int, default=DIM)
    parser.add_argument("--value-dim", type=int, default=VALUE_DIM)
    parser.add_argument("--threads", type=int, default=THREADS)
    parser.add_argument(
        "--tile-candidates", type=int, nargs="+", default=TILE_CANDIDATES
    )
    args = parser.parse_args()
    if args.warmups < 0:
        parser.error("--warmups must be non-negative")
    if args.repeats <= 0:
        parser.error("--repeats must be positive")
    if any(
        value <= 0
        for value in (
            args.batch,
            args.heads,
            args.seq_len,
            args.dim,
            args.value_dim,
            args.threads,
        )
    ):
        parser.error("workload dimensions and --threads must be positive")
    if any(candidate <= 0 for candidate in args.tile_candidates):
        parser.error("--tile-candidates must contain only positive integers")
    if args.seq_len < FIXED_TILE_Q:
        parser.error("--seq-len must be at least 64 for the fixed 64x64 baseline")
    if FIXED_TILE_Q not in args.tile_candidates:
        parser.error("--tile-candidates must include 64 for the fixed 64x64 baseline")
    if not re.fullmatch(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*", args.label):
        parser.error("--label must contain only letters, numbers, and hyphens")
    run(
        batch=args.batch,
        heads=args.heads,
        seq_len=args.seq_len,
        dim=args.dim,
        value_dim=args.value_dim,
        threads=args.threads,
        tile_candidates=args.tile_candidates,
        warmups=args.warmups,
        repeats=args.repeats,
        label=args.label,
    )


if __name__ == "__main__":
    main()
