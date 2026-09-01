"""Linear CKA analysis for the zero-overlap 32-property pilot."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
from typing import Any, Mapping, Sequence

import torch

from .cka import (
    ActivationSet,
    _atomic_csv,
    _atomic_json,
    _atomic_text,
    _git_commit,
    _load_random_activations,
    _load_trained_activations,
    _sha256,
    linear_cka,
    probe_identity,
    select_probe_examples,
)
from .property_experiments import (
    DEFAULT_CONFIG,
    PropertyExperimentRun,
    build_property_matrix,
    matrix_summary,
)


DEFAULT_OUTPUT_DIR = Path("results/property32-zero-overlap/cka")
DEFAULT_PROBE_COUNT = 4_096
DEFAULT_PROBE_SEED = 20_260_902

PAIR_FIELDS = (
    "comparison",
    "architecture",
    "pool_a",
    "task_count_a",
    "run_id_a",
    "pool_b",
    "task_count_b",
    "run_id_b",
    "layer",
    "probe_examples",
    "linear_cka",
)


def _pair(
    comparison: str,
    a: ActivationSet,
    b: ActivationSet,
    *,
    pool_a: str,
    pool_b: str,
    layer: str,
    device: torch.device,
) -> dict[str, Any]:
    x = a.layers[layer].to(device)
    y = b.layers[layer].to(device)
    value = float(linear_cka(x, y).cpu())
    if not math.isfinite(value):
        raise ValueError("CKA result is not finite")
    return {
        "comparison": comparison,
        "architecture": a.architecture,
        "pool_a": pool_a,
        "task_count_a": a.task_count,
        "run_id_a": a.run_id,
        "pool_b": pool_b,
        "task_count_b": b.task_count,
        "run_id_b": b.run_id,
        "layer": layer,
        "probe_examples": x.shape[0],
        "linear_cka": value,
    }


def build_property_cka_rows(
    activations: Sequence[ActivationSet],
    pools_by_run: Mapping[str, str],
    random_baselines: Sequence[ActivationSet],
    *,
    device: torch.device,
) -> list[dict[str, Any]]:
    """Build matched A/B comparisons and explicitly labeled controls."""

    by_key = {
        (pools_by_run[item.run_id], item.task_count): item for item in activations
    }
    task_counts = sorted({item.task_count for item in activations})
    if set(pools_by_run.values()) != {"a", "b"}:
        raise ValueError("property CKA requires pools a and b")
    if len(by_key) != 2 * len(task_counts):
        raise ValueError("property activation matrix is incomplete or duplicated")
    layers = tuple(by_key["a", task_counts[0]].layers)
    rows: list[dict[str, Any]] = []

    # Primary comparison: equal k, identical seed/architecture, zero task
    # overlap.  Thus changing k changes diversity without direct overlap.
    for task_count in task_counts:
        a = by_key["a", task_count]
        b = by_key["b", task_count]
        if tuple(a.layers) != layers or tuple(b.layers) != layers:
            raise ValueError("property model layer grids differ")
        for layer in layers:
            rows.append(
                _pair(
                    "disjoint_pools_equal_k",
                    a,
                    b,
                    pool_a="a",
                    pool_b="b",
                    layer=layer,
                    device=device,
                )
            )

    # Secondary nested control: alignment with k=16 inside each pool.  This is
    # intentionally labeled as overlapping and is not the primary evidence.
    reference_k = max(task_counts)
    for pool in ("a", "b"):
        reference = by_key[pool, reference_k]
        for task_count in task_counts:
            if task_count == reference_k:
                continue
            current = by_key[pool, task_count]
            for layer in layers:
                rows.append(
                    _pair(
                        "within_pool_k16_alignment",
                        current,
                        reference,
                        pool_a=pool,
                        pool_b=pool,
                        layer=layer,
                        device=device,
                    )
                )

    if len(random_baselines) != 2:
        raise ValueError("exactly two random-seed baselines are required")
    random_a, random_b = sorted(random_baselines, key=lambda item: item.seed)
    for layer in layers:
        rows.append(
            _pair(
                "random_cross_seed",
                random_a,
                random_b,
                pool_a="random",
                pool_b="random",
                layer=layer,
                device=device,
            )
        )
    return rows


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        raise ValueError("correlation requires matched sequences of length >= 2")
    x_mean = statistics.fmean(xs)
    y_mean = statistics.fmean(ys)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    denominator = math.sqrt(
        sum((x - x_mean) ** 2 for x in xs)
        * sum((y - y_mean) ** 2 for y in ys)
    )
    return numerator / denominator if denominator else 0.0


def summarize_primary_trend(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize the preregistered final-layer A/B trend across k."""

    selected = sorted(
        (
            row
            for row in rows
            if row["comparison"] == "disjoint_pools_equal_k"
            and row["layer"] == "final_norm"
        ),
        key=lambda row: int(row["task_count_a"]),
    )
    task_counts = [int(row["task_count_a"]) for row in selected]
    values = [float(row["linear_cka"]) for row in selected]
    if task_counts != [1, 2, 4, 8, 16]:
        raise ValueError("primary trend requires k = 1, 2, 4, 8, 16")
    log2_k = [math.log2(value) for value in task_counts]
    monotonic_non_decreasing = all(
        left <= right for left, right in zip(values, values[1:])
    )
    return {
        "task_counts": task_counts,
        "final_layer_linear_cka": values,
        "pearson_r_log2_k": _pearson(log2_k, values),
        "spearman_rho": _pearson(list(range(len(values))), _rank(values)),
        "monotonic_non_decreasing": monotonic_non_decreasing,
        "delta_k16_minus_k1": values[-1] - values[0],
    }


def summarize_final_layer_controls(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Return explicitly labeled final-layer random and overlap controls."""

    random_rows = [
        row
        for row in rows
        if row["comparison"] == "random_cross_seed" and row["layer"] == "final_norm"
    ]
    if len(random_rows) != 1:
        raise ValueError("final-layer controls require one random cross-seed row")
    within: dict[str, dict[str, float]] = {"a": {}, "b": {}}
    for row in rows:
        if (
            row["comparison"] == "within_pool_k16_alignment"
            and row["layer"] == "final_norm"
        ):
            within[str(row["pool_a"])][str(int(row["task_count_a"]))] = float(
                row["linear_cka"]
            )
    if any(set(values) != {"1", "2", "4", "8"} for values in within.values()):
        raise ValueError("within-pool final-layer control grid is incomplete")
    return {
        "random_cross_seed": float(random_rows[0]["linear_cka"]),
        "within_pool_k16_alignment": within,
    }


def _rank(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = (start + end - 1) / 2
        for index in order[start:end]:
            ranks[index] = rank
        start = end
    return ranks


def _run_mapping(run: PropertyExperimentRun) -> dict[str, Any]:
    marker = json.loads(
        (Path(run.output_dir) / "completed.json").read_text(encoding="utf-8")
    )
    return {
        "run_id": run.run_id,
        "architecture": run.architecture,
        "task_count": run.task_count,
        "seed": run.seed,
        "checkpoint_sha256": marker["checkpoint_sha256"],
        "checkpoint_path": marker["checkpoint"],
    }


def _render_readme(
    trend: Mapping[str, Any],
    controls: Mapping[str, Any],
    *,
    probe_count: int,
) -> str:
    rows = [
        "# Zero-overlap 32-property CKA pilot",
        "",
        "The primary comparison uses independently trained Pool A and Pool B ",
        "Transformers at equal k. The pools have zero task overlap at every k, ",
        "all targets are one scalar token, and activations are extracted at ",
        "`<ONE_END>` before any task token is supplied.",
        "",
        f"Probe examples: {probe_count:,} deterministic validation prefixes.",
        "",
        "| k | Final-layer linear CKA (A vs B) |",
        "|---:|---:|",
    ]
    rows.extend(
        f"| {task_count} | {value:.6f} |"
        for task_count, value in zip(
            trend["task_counts"], trend["final_layer_linear_cka"]
        )
    )
    rows.extend(
        [
            "",
            f"Spearman rho across k: {trend['spearman_rho']:.6f}.",
            f"Pearson r against log2(k): {trend['pearson_r_log2_k']:.6f}.",
            f"k=16 minus k=1: {trend['delta_k16_minus_k1']:+.6f}.",
            "",
            "The sequence is not monotonic: the largest value occurs at `k=8`, ",
            "followed by a substantial decline at `k=16`. Thus the pilot shows a ",
            "positive descriptive association, not stable convergence as tasks grow.",
            "",
            "## Controls",
            "",
            f"Random-initialization cross-seed final-layer CKA: "
            f"{float(controls['random_cross_seed']):.6f}.",
            "",
            "| Pool | k vs 16 final-layer CKA |",
            "|---|---:|",
        ]
    )
    for pool in ("a", "b"):
        for task_count in (1, 2, 4, 8):
            value = controls["within_pool_k16_alignment"][pool][str(task_count)]
            rows.append(f"| {pool.upper()} {task_count} vs {pool.upper()} 16 | {value:.6f} |")
    rows.extend(
        [
            "",
            "The within-pool rows are overlapping-task controls and are not primary ",
            "zero-overlap evidence. The random baseline is also not a performance ",
            "baseline: high CKA can arise from common architecture and input geometry.",
            "",
            "Pool A and Pool B share no task names, but several properties are natural ",
            "duals (for example descents/recoils and LIS/LDS). The `k=8` spike may ",
            "therefore reflect conceptual symmetry rather than generic task diversity.",
            "",
            "This is a one-seed pilot. It establishes a descriptive trend, not ",
            "an error-bar-supported population claim. Test data were not used.",
            "",
        ]
    )
    return "\n".join(row.rstrip() for row in rows)


def run_property_cka(
    config_path: Path,
    output_dir: Path,
    *,
    probe_count: int,
    probe_seed: int,
    batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    summary = matrix_summary(config_path)
    if summary["complete_count"] != summary["run_count"]:
        raise ValueError("all property pilot models must complete before CKA")
    runs = build_property_matrix(config_path)
    repository = Path.cwd()
    validation_manifest = Path(_read_toml(config_path)["validation_manifest"])
    manifest_sha256 = _sha256(validation_manifest)
    examples = select_probe_examples(
        validation_manifest,
        count=probe_count,
        seed=probe_seed,
        shard_index=0,
    )
    probe = probe_identity(
        examples,
        dataset_manifest_sha256=manifest_sha256,
        shard_index=0,
        seed=probe_seed,
    )
    _atomic_json(probe, output_dir / "probe_manifest.json")

    cache_dir = output_dir / "cache"
    run_mappings = [_run_mapping(run) for run in runs]
    activations: list[ActivationSet] = []
    for index, run in enumerate(run_mappings, start=1):
        print(f"extracting property activation {index}/{len(runs)}: {run['run_id']}")
        activations.append(
            _load_trained_activations(
                run,
                repository=repository,
                examples=examples,
                probe_sha256=probe["probe_sha256"],
                cache_dir=cache_dir,
                device=device,
                batch_size=batch_size,
            )
        )

    reference = run_mappings[0]
    random_baselines = [
        _load_random_activations(
            reference,
            repository=repository,
            seed=seed,
            examples=examples,
            probe_sha256=probe["probe_sha256"],
            cache_dir=cache_dir,
            device=device,
            batch_size=batch_size,
        )
        for seed in (17, 42)
    ]
    pools_by_run = {run.run_id: run.pool for run in runs}
    rows = build_property_cka_rows(
        activations, pools_by_run, random_baselines, device=device
    )
    trend = summarize_primary_trend(rows)
    controls = summarize_final_layer_controls(rows)
    pair_path = output_dir / "pairwise_layer_cka.csv"
    readme_path = output_dir / "README.md"
    _atomic_csv(rows, PAIR_FIELDS, pair_path)
    _atomic_text(
        _render_readme(trend, controls, probe_count=probe_count), readme_path
    )
    result = {
        "status": "completed",
        "protocol_version": "property32-zero-overlap-cka/v1",
        "analysis_commit": _git_commit(repository),
        "config_path": str(config_path),
        "config_sha256": summary["config_sha256"],
        "probe": probe,
        "primary_trend": trend,
        "final_layer_controls": controls,
        "test_split_used": False,
        "artifacts": {
            pair_path.name: _sha256(pair_path),
            readme_path.name: _sha256(readme_path),
            "probe_manifest.json": _sha256(output_dir / "probe_manifest.json"),
        },
    }
    _atomic_json(result, output_dir / "manifest.json")
    return result


def _read_toml(path: Path) -> dict[str, Any]:
    import tomllib

    return tomllib.loads(path.read_text(encoding="utf-8"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--probe-count", type=int, default=DEFAULT_PROBE_COUNT)
    parser.add_argument("--probe-seed", type=int, default=DEFAULT_PROBE_SEED)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="auto")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    result = run_property_cka(
        args.config,
        args.output_dir,
        probe_count=args.probe_count,
        probe_seed=args.probe_seed,
        batch_size=args.batch_size,
        device=device,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "DEFAULT_CONFIG",
    "DEFAULT_OUTPUT_DIR",
    "build_property_cka_rows",
    "main",
    "run_property_cka",
    "summarize_final_layer_controls",
    "summarize_primary_trend",
]
