import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from matplotlib import font_manager

from examples.causal_softmax import spec
from miniattn.generate import generate
from miniattn.ref import ref
from miniattn.tune import DEFAULT_TILE_CANDIDATES, measure_latency, tune


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
PLOTS_DIR = ROOT / "plots"
BATCH = 1
HEADS = 4
SEQ_LENS = (128, 256, 512, 1024)
THREADS = (1, 8)
DIM = 64
VALUE_DIM = 64
SEED = 0
FIXED_TILE_Q = 64
FIXED_TILE_K = 64


def _op_dict(op):
    return None if op is None else {"name": op.name, "value": op.value}


def _spec_fields():
    return {
        "spec_name": spec.name,
        "pattern": spec.pattern,
        "score_mod": json.dumps(
            [_op_dict(op) for op in spec.score_mod], separators=(",", ":")
        ),
        "mask_mod": json.dumps(_op_dict(spec.mask_mod), separators=(",", ":")),
        "rownorm": json.dumps(_op_dict(spec.rownorm), separators=(",", ":")),
    }


def _spec_json():
    return {
        "name": spec.name,
        "pattern": spec.pattern,
        "score_mod": [_op_dict(op) for op in spec.score_mod],
        "mask_mod": _op_dict(spec.mask_mod),
        "rownorm": _op_dict(spec.rownorm),
    }


def _context_fields(sq, sk, threads):
    return {
        "batch": BATCH,
        "heads": HEADS,
        "sq": sq,
        "sk": sk,
        "dim": DIM,
        "value_dim": VALUE_DIM,
        "dtype": "float32",
        "device": "cpu",
        "threads": threads,
    }


def _make_workload(seq_len):
    torch.manual_seed(SEED)
    q = torch.randn(BATCH, HEADS, seq_len, DIM)
    k = torch.randn(BATCH, HEADS, seq_len, DIM)
    v = torch.randn(BATCH, HEADS, seq_len, VALUE_DIM)
    return q, k, v


def _throughput(latency_ms, sq):
    calls_per_second = 1000.0 / latency_ms
    return calls_per_second, BATCH * sq * calls_per_second


def _reference_row(sq, latency_ms):
    calls_per_second, tokens_per_second = _throughput(latency_ms, sq)
    elements = BATCH * HEADS * sq * sq
    bytes_ = elements * 4
    return {
        **_spec_fields(),
        **_context_fields(sq, sq, torch.get_num_threads()),
        "implementation": "reference",
        "tile_q": None,
        "tile_k": None,
        "median_latency_ms": latency_ms,
        "calls_per_second": calls_per_second,
        "tokens_per_second": tokens_per_second,
        "score_tile_elements": elements,
        "score_tile_bytes": bytes_,
        "score_tile_kib": bytes_ / 1024.0,
        "q_tiles": None,
        "kv_tile_interactions": None,
    }


def _comparison_row(implementation, result, sq, threads):
    return {
        **_spec_fields(),
        **_context_fields(sq, sq, threads),
        "implementation": implementation,
        "tile_q": result.get("tile_q"),
        "tile_k": result.get("tile_k"),
        "median_latency_ms": result["median_latency_ms"],
        "calls_per_second": result["calls_per_second"],
        "tokens_per_second": result["tokens_per_second"],
        "score_tile_elements": result["score_tile_elements"],
        "score_tile_bytes": result["score_tile_bytes"],
        "score_tile_kib": result["score_tile_kib"],
        "q_tiles": result.get("q_tiles"),
        "kv_tile_interactions": result.get("kv_tile_interactions"),
    }


def _write_csv(path, rows):
    pd.DataFrame(rows).to_csv(path, index=False)


def _plot_search(path):
    search = pd.read_csv(path)
    search = search[search["sq"] == 1024]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    for axis, threads in zip(axes, THREADS):
        subset = search[search["threads"] == threads]
        table = subset.pivot(
            index="tile_q", columns="tile_k", values="median_latency_ms"
        ).sort_index(ascending=True)
        table = table.reindex(sorted(table.columns), axis=1)
        sns.heatmap(table, annot=True, fmt=".2f", cmap="mako", ax=axis, cbar=axis is axes[-1])
        best = subset[subset["is_best"].astype(str).str.lower() == "true"].iloc[0]
        row = list(table.index).index(best["tile_q"])
        column = list(table.columns).index(best["tile_k"])
        axis.add_patch(plt.Rectangle((column, row), 1, 1, fill=False, edgecolor="red", linewidth=3))
        axis.set_title(f"S=1024, threads={threads}; best={int(best['tile_q'])}x{int(best['tile_k'])}")
        axis.set_xlabel("tile_k")
        axis.set_ylabel("tile_q")
    fig.savefig(PLOTS_DIR / "search.png", dpi=180)
    plt.close(fig)


def _plot_throughput(path):
    font_path = Path("/mnt/c/Windows/Fonts/times.ttf")
    if font_path.exists():
        font_manager.fontManager.addfont(font_path)
    plt.rcParams["font.family"] = ["Times New Roman", "serif"]

    compare = pd.read_csv(path)
    fig, axes = plt.subplots(
        1, 2, figsize=(7.2, 3.0), sharex=True, sharey=True, constrained_layout=True
    )
    fig.patch.set_facecolor("white")
    styles = {
        "reference": {
            "label": "Reference",
            "color": "#9FA6AE",
            "marker": "o",
            "linestyle": "-",
        },
        "fixed": {
            "label": "Fixed 64×64",
            "color": "#A85C5C",
            "marker": "s",
            "linestyle": "--",
        },
        "tuned": {
            "label": "Tuned",
            "color": "#2F6FA5",
            "marker": "^",
            "linestyle": "-",
        },
    }
    panel_titles = {1: "(a) 1 thread", 8: "(b) 8 threads"}

    for axis, threads in zip(axes, THREADS):
        subset = compare[compare["threads"] == threads]
        for implementation, style in styles.items():
            values = subset[subset["implementation"] == implementation].sort_values(
                "sq"
            )
            axis.plot(
                values["sq"],
                values["tokens_per_second"] / 1_000_000,
                label=style["label"],
                color=style["color"],
                marker=style["marker"],
                linestyle=style["linestyle"],
                linewidth=1.25,
                markersize=4.2,
                markeredgecolor="black",
                markeredgewidth=0.4,
            )
        axis.set_title(panel_titles[threads], loc="left", fontsize=9, pad=7)
        axis.set_xlabel("Sequence length S", fontsize=8)
        axis.set_xticks(SEQ_LENS)
        axis.set_facecolor("white")
        axis.set_axisbelow(True)
        axis.grid(axis="y", color="#E4E7EA", linewidth=0.55)
        for spine in axis.spines.values():
            spine.set_visible(True)
            spine.set_color("black")
            spine.set_linewidth(0.7)
        axis.tick_params(axis="both", labelsize=7.2, width=0.6, length=2.5)

    axes[0].set_ylabel("Input tokens/s (×10⁶)", fontsize=8)
    axes[0].legend(
        frameon=False,
        fontsize=6.8,
        ncol=3,
        loc="upper right",
        handletextpad=0.4,
        columnspacing=0.9,
    )
    fig.savefig(
        PLOTS_DIR / "throughput.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


def run(*, warmups=5, repeats=20):
    RESULTS_DIR.mkdir(exist_ok=True)
    PLOTS_DIR.mkdir(exist_ok=True)
    search_rows = []
    compare_rows = []
    best_results = []

    for threads in THREADS:
        torch.set_num_threads(threads)
        for seq_len in SEQ_LENS:
            q, k, v = _make_workload(seq_len)
            reference_latency = measure_latency(
                lambda: ref(q, k, v), warmups=warmups, repeats=repeats
            )
            compare_rows.append(_reference_row(seq_len, reference_latency))

            results, best = tune(
                spec,
                q=q,
                k=k,
                v=v,
                reference_fn=ref,
                tile_q_candidates=DEFAULT_TILE_CANDIDATES,
                tile_k_candidates=DEFAULT_TILE_CANDIDATES,
                warmups=warmups,
                repeats=repeats,
            )
            context = {**_spec_fields(), **_context_fields(seq_len, seq_len, threads)}
            for result in results:
                search_rows.append({**context, **result})

            fixed = next(
                result
                for result in results
                if result["tile_q"] == FIXED_TILE_Q
                and result["tile_k"] == FIXED_TILE_K
            )
            compare_rows.append(_comparison_row("fixed", fixed, seq_len, threads))
            compare_rows.append(_comparison_row("tuned", best, seq_len, threads))
            best_results.append(
                {
                    **_context_fields(seq_len, seq_len, threads),
                    "best": {
                        "tile_q": best["tile_q"],
                        "tile_k": best["tile_k"],
                        "median_latency_ms": best["median_latency_ms"],
                    },
                }
            )
            print(
                f"threads={threads} S={seq_len} "
                f"best={best['tile_q']}x{best['tile_k']} "
                f"latency={best['median_latency_ms']:.3f} ms"
            )

    search_path = RESULTS_DIR / "search.csv"
    compare_path = RESULTS_DIR / "compare.csv"
    best_path = RESULTS_DIR / "best_causal_softmax.json"
    _write_csv(search_path, search_rows)
    _write_csv(compare_path, compare_rows)
    best_path.write_text(
        json.dumps(
            {"spec": _spec_json(), "best_results": best_results},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    representative = next(
        result
        for result in best_results
        if result["sq"] == 1024 and result["threads"] == 8
    )
    best = representative["best"]
    header = (
        "# Generated by miniattn.generate.\n"
        "# Tuning context: B=1, H=4, Sq=Sk=1024, D=Dv=64, "
        "dtype=float32, device=cpu, threads=8.\n"
        f"# Measured best: tile_q={best['tile_q']}, tile_k={best['tile_k']}, "
        f"median_latency_ms={best['median_latency_ms']:.6f}.\n"
        "# Input shapes remain dynamic; this schedule was measured best only "
        "for the context above.\n"
        "# Do not edit manually.\n\n"
    )
    (ROOT / "generated" / "tuned_causal_softmax.py").write_text(
        header + generate(spec, best["tile_q"], best["tile_k"]),
        encoding="utf-8",
    )
    _plot_search(search_path)
    _plot_throughput(compare_path)
    return search_path, compare_path, best_path


def main():
    parser = argparse.ArgumentParser(description="Run the Mini MetaAttention CPU autotuning experiment")
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()
    if args.warmups < 0:
        parser.error("--warmups must be non-negative")
    if args.repeats <= 0:
        parser.error("--repeats must be positive")
    run(warmups=args.warmups, repeats=args.repeats)


if __name__ == "__main__":
    main()
