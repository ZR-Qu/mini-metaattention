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
DEFAULT_TILE_CANDIDATES = (16, 32, 64, 128, 256, 512, 1024, 2048)
FIXED_TILE_Q = 64
FIXED_TILE_K = 64
VARIANTS = {
    "causal_softmax": (causal_softmax_spec, ref),
    "relu": (relu_spec, ref_relu),
    "causal_relu": (causal_relu_spec, ref_causal_relu),
}
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


def _spec_json(spec):
    return {
        "name": spec.name,
        "pattern": spec.pattern,
        "score_mod": [_op_dict(op) for op in spec.score_mod],
        "mask_mod": _op_dict(spec.mask_mod),
        "rownorm": _op_dict(spec.rownorm),
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
    torch.manual_seed(0)
    q = torch.randn(batch, heads, seq_len, dim)
    k = torch.randn(batch, heads, seq_len, dim)
    v = torch.randn(batch, heads, seq_len, value_dim)
    return q, k, v


def _throughput(latency_ms, batch, seq_len):
    calls_per_second = 1000.0 / latency_ms
    return calls_per_second, batch * seq_len * calls_per_second


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


def _plot(
    compare_path,
    plot_path,
    label,
    variant,
    heads,
    seq_len,
    dim,
    value_dim,
    threads,
):
    font_path = Path("/mnt/c/Windows/Fonts/times.ttf")
    if font_path.exists():
        font_manager.fontManager.addfont(font_path)
    plt.rcParams["font.family"] = (
        "Times New Roman" if font_path.exists() else "Liberation Serif"
    )

    compare = pd.read_csv(compare_path).set_index("implementation")
    implementations = ("reference", "fixed", "tuned")
    values = [float(compare.loc[name, "median_latency_ms"]) for name in implementations]
    fixed = values[1]
    tuned = values[2]
    speedup = fixed / tuned
    best_tile = f"{int(compare.loc['tuned', 'tile_q'])}×{int(compare.loc['tuned', 'tile_k'])}"

    fig, axis = plt.subplots(figsize=(4.1, 3.1), constrained_layout=True)
    fig.patch.set_facecolor("white")
    x_positions = (0.0, 0.42, 0.84)
    for x, implementation, value in zip(x_positions, implementations, values):
        bar = axis.bar(
            x,
            value,
            width=0.16,
            color=IMPLEMENTATION_COLORS[implementation],
            edgecolor="black",
            linewidth=0.7,
            hatch=IMPLEMENTATION_HATCHES[implementation],
            zorder=3,
        )[0]
        axis.annotate(
            f"{value:.2f}",
            (bar.get_x() + bar.get_width() / 2, value),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7,
            color="#222222",
        )

    axis.annotate(
        "",
        xy=(0.63, fixed),
        xytext=(0.63, tuned),
        arrowprops={
            "arrowstyle": "<->",
            "color": "#333333",
            "linewidth": 0.7,
            "shrinkA": 0,
            "shrinkB": 0,
        },
    )
    axis.text(
        0.68,
        (fixed + tuned) / 2,
        f"{speedup:.2f}×",
        ha="left",
        va="center",
        fontsize=7,
        color="#9A3F3F",
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.5},
    )
    axis.set_title(label, loc="left", fontsize=9.5, pad=25)
    axis.text(
        0,
        1.015,
        f"{VARIANT_LABELS[variant]} · H={heads} · S={seq_len} · Dqk={dim} · Dv={value_dim} · threads={threads}",
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=6.8,
        color="#444444",
    )
    axis.set_ylabel("Median latency (ms)", fontsize=8)
    implementation_labels = [IMPLEMENTATION_LABELS[name] for name in implementations]
    implementation_labels[2] = f"Tuned\nbest {best_tile}"
    axis.set_xticks(list(x_positions), implementation_labels)
    axis.set_xlim(-0.5, 1.34)
    axis.set_ylim(0, max(values) * 1.18)
    axis.set_facecolor("white")
    axis.set_axisbelow(True)
    axis.grid(axis="y", color="#E4E7EA", linewidth=0.55)
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(0.7)
    axis.tick_params(axis="both", labelsize=7.2, width=0.6, length=2.5)
    fig.savefig(plot_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def run(
    *,
    label,
    variant,
    heads,
    seq_len,
    dim,
    value_dim,
    batch=1,
    threads=8,
    tile_candidates=DEFAULT_TILE_CANDIDATES,
    warmups=5,
    repeats=20,
):
    if variant not in VARIANTS:
        raise ValueError(f"unsupported variant: {variant}")
    if not re.fullmatch(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*", label):
        raise ValueError("label must contain only letters, numbers, and hyphens")
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S") + f"-{label}"
    results_dir = RESULTS_DIR / run_id
    plots_dir = PLOTS_DIR / run_id
    results_dir.mkdir(parents=True)
    plots_dir.mkdir(parents=True)
    spec, reference_fn = VARIANTS[variant]
    run_json = {
        "run_id": run_id,
        "label": label,
        "variant": variant,
        "spec": _spec_json(spec),
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
    }
    (results_dir / "run.json").write_text(
        json.dumps(run_json, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    torch.set_num_threads(threads)
    q, k, v = _make_workload(batch, heads, seq_len, dim, value_dim)
    context = _context_fields(batch, heads, seq_len, dim, value_dim, threads)
    reference_latency = measure_latency(
        lambda: reference_fn(q, k, v), warmups=warmups, repeats=repeats
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
    search_rows = [
        {"variant": variant, **_spec_fields(spec), **context, **result}
        for result in results
    ]
    fixed = next(
        result
        for result in results
        if result["tile_q"] == FIXED_TILE_Q and result["tile_k"] == FIXED_TILE_K
    )
    compare_rows = [_reference_row(variant, spec, reference_latency, context)]
    compare_rows.append(_comparison_row(variant, spec, "fixed", fixed, context))
    compare_rows.append(_comparison_row(variant, spec, "tuned", best, context))

    search_path = results_dir / "search.csv"
    compare_path = results_dir / "compare.csv"
    _write_csv(search_path, search_rows)
    _write_csv(compare_path, compare_rows)
    _plot(
        compare_path,
        plots_dir / "comparison.png",
        label,
        variant,
        heads,
        seq_len,
        dim,
        value_dim,
        threads,
    )
    speedup = fixed["median_latency_ms"] / best["median_latency_ms"]
    print(
        f"{variant:16s} reference={reference_latency:.3f} ms "
        f"fixed={fixed['median_latency_ms']:.3f} ms "
        f"tuned={best['median_latency_ms']:.3f} ms "
        f"best={best['tile_q']}x{best['tile_k']} "
        f"speedup_vs_fixed={speedup:.3f}x"
    )
    print(f"run_id={run_id}")
    print(f"results={results_dir.relative_to(ROOT)}")
    print(f"plots={plots_dir.relative_to(ROOT)}")
    return search_path, compare_path


def main():
    parser = argparse.ArgumentParser(
        description="Run one Mini MetaAttention single-workload autotuning experiment"
    )
    parser.add_argument("--label", required=True)
    parser.add_argument("--variant", choices=tuple(VARIANTS), required=True)
    parser.add_argument("--heads", type=int, required=True)
    parser.add_argument("--seq-len", type=int, required=True)
    parser.add_argument("--dim", type=int, required=True)
    parser.add_argument("--value-dim", type=int, required=True)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument(
        "--tile-candidates", type=int, nargs="+", default=DEFAULT_TILE_CANDIDATES
    )
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
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
    if args.seq_len < FIXED_TILE_Q:
        parser.error("--seq-len must be at least 64 for the fixed 64x64 baseline")
    if any(candidate <= 0 for candidate in args.tile_candidates):
        parser.error("--tile-candidates must contain only positive integers")
    if FIXED_TILE_Q not in args.tile_candidates:
        parser.error("--tile-candidates must include 64 for the fixed 64x64 baseline")
    if not re.fullmatch(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*", args.label):
        parser.error("--label must contain only letters, numbers, and hyphens")
    run(
        label=args.label,
        variant=args.variant,
        heads=args.heads,
        seq_len=args.seq_len,
        dim=args.dim,
        value_dim=args.value_dim,
        batch=args.batch,
        threads=args.threads,
        tile_candidates=args.tile_candidates,
        warmups=args.warmups,
        repeats=args.repeats,
    )


if __name__ == "__main__":
    main()
