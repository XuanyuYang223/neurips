"""Reproducible CKA analysis for the completed nested V3 model matrix.

The primary analysis asks whether representations become more reproducible
across random seeds as the nested training-task count increases.  Every model
receives the same deterministic sample of one-line permutation prefixes from
validation shard 098.  Activations are read at ``<ONE_END>``, before any task
token is present, so the comparison concerns the learned representation of the
permutation rather than a particular requested operation.

Linear CKA is computed without materializing an examples-by-examples Gram
matrix.  The formula is equivalent to biased linear-kernel CKA in Kornblith et
al. (2019) and to ``ckatorch.core.cka_base(..., kernel="linear")`` in the
MIT-licensed implementation by Alessandro Ristori:
https://github.com/RistoAle97/centered-kernel-alignment
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import gzip
import hashlib
import heapq
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import tempfile
from typing import Any, Iterable, Mapping, Sequence

import torch
from torch import Tensor, nn

from .audit import audit_experiment
from .passage import TOKEN_TO_ID
from .training import TrainConfig, _default_model_factory, resolve_shards


FORMAT_VERSION = 1
DEFAULT_CONFIG = Path("configs/henry_permutation_revised.toml")
DEFAULT_OUTPUT_DIR = Path("results/v3/cka")
DEFAULT_PROBE_COUNT = 4_096
DEFAULT_PROBE_SEED = 20_260_831
DEFAULT_VALIDATION_SHARD = 98
REFERENCE_IMPLEMENTATION_COMMIT = "f7e2aefee17b6440088d62830881ba30b797fe92"

PAIR_FIELDS = (
    "comparison",
    "architecture_a",
    "trained_task_count_a",
    "seed_a",
    "run_id_a",
    "layer_a",
    "architecture_b",
    "trained_task_count_b",
    "seed_b",
    "run_id_b",
    "layer_b",
    "probe_examples",
    "linear_cka",
)

SUMMARY_FIELDS = (
    "comparison",
    "architecture",
    "trained_task_count",
    "reference_task_count",
    "layer",
    "pair_count",
    "cka_mean",
    "cka_sample_sd",
    "cka_min",
    "cka_max",
)


@dataclass(frozen=True)
class ProbeExample:
    """One deterministic task-free permutation prefix."""

    record_id: int
    n: int
    token_ids: tuple[int, ...]


@dataclass(frozen=True)
class ActivationSet:
    """Authenticated activations for one checkpoint or random baseline."""

    run_id: str
    architecture: str
    task_count: int
    seed: int
    checkpoint_sha256: str
    probe_sha256: str
    layers: Mapping[str, Tensor]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _atomic_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_text(value: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_csv(rows: Sequence[Mapping[str, Any]], fields: Sequence[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_torch_save(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    try:
        torch.save(value, temporary)
        with open(temporary, "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def linear_cka(x: Tensor, y: Tensor) -> Tensor:
    """Return biased linear CKA for two ``[examples, features]`` matrices.

    Centering is performed over examples.  The feature dimensions may differ,
    but the models must be evaluated on the same examples in the same order.
    Float64 accumulation is used for stable, reproducible corpus analysis.
    """

    if not isinstance(x, Tensor) or not isinstance(y, Tensor):
        raise TypeError("x and y must be torch tensors")
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError("x and y must have shape [examples, features]")
    if x.shape[0] != y.shape[0]:
        raise ValueError("x and y must contain the same number of examples")
    if x.shape[0] < 2 or x.shape[1] < 1 or y.shape[1] < 1:
        raise ValueError("CKA requires at least two examples and one feature")
    if not bool(torch.isfinite(x).all()) or not bool(torch.isfinite(y).all()):
        raise ValueError("CKA inputs must be finite")
    x64 = x.to(dtype=torch.float64)
    y64 = y.to(device=x64.device, dtype=torch.float64)
    x64 = x64 - x64.mean(dim=0, keepdim=True)
    y64 = y64 - y64.mean(dim=0, keepdim=True)
    cross = x64.T @ y64
    self_x = x64.T @ x64
    self_y = y64.T @ y64
    numerator = cross.square().sum()
    denominator = torch.sqrt(self_x.square().sum() * self_y.square().sum())
    if not bool(torch.isfinite(denominator)) or float(denominator) <= 0.0:
        raise ValueError("CKA is undefined for a constant representation")
    return numerator / denominator


def _record_to_probe(record: Mapping[str, Any]) -> ProbeExample:
    record_id = record.get("id")
    n = record.get("n")
    tokens = record.get("tokens")
    if type(record_id) is not int or record_id < 0:
        raise ValueError("probe record has an invalid id")
    if type(n) is not int or not 2 <= n <= 30:
        raise ValueError(f"probe record {record_id} has invalid n")
    if not isinstance(tokens, list) or not all(isinstance(token, str) for token in tokens):
        raise ValueError(f"probe record {record_id} has invalid tokens")
    if "<ONE_END>" not in tokens:
        raise ValueError(f"probe record {record_id} must contain <ONE_END>")
    end = tokens.index("<ONE_END>") + 1
    prefix = tokens[:end]
    if not prefix or prefix[0] != "<BOS>" or prefix[-1] != "<ONE_END>":
        raise ValueError(f"probe record {record_id} has malformed one-line prefix")
    try:
        token_ids = tuple(TOKEN_TO_ID[token] for token in prefix)
    except KeyError as error:
        raise ValueError(
            f"probe record {record_id} has unknown token {error.args[0]!r}"
        ) from None
    return ProbeExample(record_id=record_id, n=n, token_ids=token_ids)


def select_probe_examples(
    manifest: Path,
    *,
    count: int = DEFAULT_PROBE_COUNT,
    seed: int = DEFAULT_PROBE_SEED,
    shard_index: int = DEFAULT_VALIDATION_SHARD,
) -> tuple[ProbeExample, ...]:
    """Select a deterministic hash-priority sample from one physical shard."""

    if count < 2:
        raise ValueError("probe count must be at least two")
    shard = resolve_shards(manifest, (shard_index,))[0]

    def ranked_records() -> Iterable[tuple[bytes, int, ProbeExample]]:
        opener = gzip.open if shard.name.endswith(".gz") else open
        with opener(shard, "rt", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"invalid JSON in {shard}:{line_number}: {error}"
                    ) from error
                if not isinstance(record, Mapping):
                    raise ValueError(f"record {shard}:{line_number} is not an object")
                example = _record_to_probe(record)
                rank = hashlib.sha256(
                    f"{seed}:{example.record_id}".encode("ascii")
                ).digest()
                yield rank, example.record_id, example

    selected = heapq.nsmallest(count, ranked_records())
    if len(selected) != count:
        raise ValueError(f"probe shard contains only {len(selected)} records")
    examples = tuple(item[2] for item in sorted(selected, key=lambda item: item[1]))
    if len({example.record_id for example in examples}) != count:
        raise ValueError("probe record ids are not unique")
    return examples


def probe_identity(
    examples: Sequence[ProbeExample],
    *,
    dataset_manifest_sha256: str,
    shard_index: int,
    seed: int,
) -> dict[str, Any]:
    histogram: dict[str, int] = {}
    for example in examples:
        histogram[str(example.n)] = histogram.get(str(example.n), 0) + 1
    content = [asdict(example) for example in examples]
    return {
        "format_version": FORMAT_VERSION,
        "split": "validation",
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "shard_index": shard_index,
        "selection": "smallest_sha256(seed:record_id)",
        "seed": seed,
        "example_count": len(examples),
        "n_histogram": dict(sorted(histogram.items(), key=lambda item: int(item[0]))),
        "record_ids": [example.record_id for example in examples],
        "probe_sha256": _canonical_sha256(content),
    }


def _layer_modules(model: nn.Module) -> tuple[tuple[str, nn.Module], ...]:
    embedding = getattr(model, "embedding_dropout", None)
    blocks = getattr(model, "blocks", None)
    final_norm = getattr(model, "final_norm", None)
    if not isinstance(embedding, nn.Module) or not isinstance(blocks, nn.ModuleList):
        raise TypeError("model does not expose the expected embedding/blocks interface")
    if not isinstance(final_norm, nn.Module):
        raise TypeError("model does not expose final_norm")
    layers: list[tuple[str, nn.Module]] = [("embedding", embedding)]
    layers.extend((f"block_{index + 1:02d}", block) for index, block in enumerate(blocks))
    layers.append(("final_norm", final_norm))
    return tuple(layers)


def extract_landmark_activations(
    model: nn.Module,
    examples: Sequence[ProbeExample],
    *,
    device: torch.device,
    batch_size: int,
) -> dict[str, Tensor]:
    """Extract every layer's final-prefix vector at the ``<ONE_END>`` token."""

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if not examples:
        raise ValueError("examples cannot be empty")
    modules = _layer_modules(model)
    captured: dict[str, Tensor] = {}
    outputs: dict[str, list[Tensor]] = {name: [] for name, _ in modules}

    def make_hook(name: str):
        def hook(_module: nn.Module, _inputs: tuple[Any, ...], value: Any) -> None:
            if not isinstance(value, Tensor) or value.ndim != 3:
                raise TypeError(f"layer {name} did not return [batch, sequence, features]")
            captured[name] = value

        return hook

    handles = [module.register_forward_hook(make_hook(name)) for name, module in modules]
    model = model.to(device).eval()
    pad_id = TOKEN_TO_ID["<PAD>"]
    try:
        with torch.inference_mode():
            for start in range(0, len(examples), batch_size):
                batch = examples[start : start + batch_size]
                width = max(len(example.token_ids) for example in batch)
                input_ids = torch.full(
                    (len(batch), width), pad_id, dtype=torch.long, device=device
                )
                attention_mask = torch.zeros(
                    (len(batch), width), dtype=torch.bool, device=device
                )
                positions = torch.empty(len(batch), dtype=torch.long, device=device)
                for row, example in enumerate(batch):
                    length = len(example.token_ids)
                    input_ids[row, :length] = torch.tensor(
                        example.token_ids, dtype=torch.long, device=device
                    )
                    attention_mask[row, :length] = True
                    positions[row] = length - 1
                captured.clear()
                _ = model(input_ids=input_ids, attention_mask=attention_mask)
                if set(captured) != set(outputs):
                    raise RuntimeError("not every requested hidden layer was captured")
                rows = torch.arange(len(batch), device=device)
                for name in outputs:
                    landmark = captured[name][rows, positions]
                    outputs[name].append(landmark.detach().float().cpu())
    finally:
        for handle in handles:
            handle.remove()
        model.to("cpu")
    result = {name: torch.cat(parts, dim=0) for name, parts in outputs.items()}
    expected_shape = (len(examples), getattr(model, "config").d_model)
    for name, value in result.items():
        if tuple(value.shape) != expected_shape or not bool(torch.isfinite(value).all()):
            raise ValueError(f"invalid extracted activation matrix for {name}")
    return result


def _cache_payload(activation: ActivationSet) -> dict[str, Any]:
    return {
        "format_version": FORMAT_VERSION,
        "run_id": activation.run_id,
        "architecture": activation.architecture,
        "task_count": activation.task_count,
        "seed": activation.seed,
        "checkpoint_sha256": activation.checkpoint_sha256,
        "probe_sha256": activation.probe_sha256,
        "layers": dict(activation.layers),
    }


def _load_cache(path: Path, identity: Mapping[str, Any]) -> ActivationSet | None:
    if not path.is_file():
        return None
    value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, Mapping):
        raise ValueError(f"invalid activation cache: {path}")
    for key, expected in identity.items():
        if value.get(key) != expected:
            raise ValueError(f"activation cache identity mismatch for {key}: {path}")
    layers = value.get("layers")
    if not isinstance(layers, Mapping) or not layers:
        raise ValueError(f"activation cache has no layers: {path}")
    if not all(
        isinstance(name, str)
        and isinstance(tensor, Tensor)
        and tensor.ndim == 2
        and bool(torch.isfinite(tensor).all())
        for name, tensor in layers.items()
    ):
        raise ValueError(f"activation cache contains invalid tensors: {path}")
    return ActivationSet(
        run_id=str(value["run_id"]),
        architecture=str(value["architecture"]),
        task_count=int(value["task_count"]),
        seed=int(value["seed"]),
        checkpoint_sha256=str(value["checkpoint_sha256"]),
        probe_sha256=str(value["probe_sha256"]),
        layers=dict(layers),
    )


def _load_trained_activations(
    run: Mapping[str, Any],
    *,
    repository: Path,
    examples: Sequence[ProbeExample],
    probe_sha256: str,
    cache_dir: Path,
    device: torch.device,
    batch_size: int,
) -> ActivationSet:
    identity = {
        "format_version": FORMAT_VERSION,
        "run_id": run["run_id"],
        "architecture": run["architecture"],
        "task_count": run["task_count"],
        "seed": run["seed"],
        "checkpoint_sha256": run["checkpoint_sha256"],
        "probe_sha256": probe_sha256,
    }
    cache_path = cache_dir / f"{run['run_id']}.pt"
    cached = _load_cache(cache_path, identity)
    if cached is not None:
        return cached
    checkpoint_path = repository / str(run["checkpoint_path"])
    if _sha256(checkpoint_path) != run["checkpoint_sha256"]:
        raise ValueError(f"checkpoint changed after audit: {run['run_id']}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config = TrainConfig.from_value(checkpoint["config"])
    model = _default_model_factory(config)
    model.load_state_dict(checkpoint["model"], strict=True)
    del checkpoint
    layers = extract_landmark_activations(
        model, examples, device=device, batch_size=batch_size
    )
    activation = ActivationSet(layers=layers, **{key: value for key, value in identity.items() if key != "format_version"})
    _atomic_torch_save(_cache_payload(activation), cache_path)
    return activation


def _load_random_activations(
    reference_run: Mapping[str, Any],
    *,
    repository: Path,
    seed: int,
    examples: Sequence[ProbeExample],
    probe_sha256: str,
    cache_dir: Path,
    device: torch.device,
    batch_size: int,
) -> ActivationSet:
    architecture = str(reference_run["architecture"])
    run_id = f"random-{architecture}-seed{seed}"
    identity = {
        "format_version": FORMAT_VERSION,
        "run_id": run_id,
        "architecture": architecture,
        "task_count": 0,
        "seed": seed,
        "checkpoint_sha256": "random_init",
        "probe_sha256": probe_sha256,
    }
    cache_path = cache_dir / f"{run_id}.pt"
    cached = _load_cache(cache_path, identity)
    if cached is not None:
        return cached
    checkpoint_path = repository / str(reference_run["checkpoint_path"])
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config = TrainConfig.from_value(checkpoint["config"])
    del checkpoint
    torch.manual_seed(seed)
    model = _default_model_factory(config)
    layers = extract_landmark_activations(
        model, examples, device=device, batch_size=batch_size
    )
    activation = ActivationSet(layers=layers, **{key: value for key, value in identity.items() if key != "format_version"})
    _atomic_torch_save(_cache_payload(activation), cache_path)
    return activation


def _corresponding_layers(a: ActivationSet, b: ActivationSet) -> tuple[str, ...]:
    if tuple(a.layers) != tuple(b.layers):
        raise ValueError(
            f"same-architecture layer grids differ: {a.run_id}, {b.run_id}"
        )
    return tuple(a.layers)


def _pair_row(
    comparison: str,
    a: ActivationSet,
    b: ActivationSet,
    layer_a: str,
    layer_b: str,
    *,
    compute_device: torch.device,
) -> dict[str, Any]:
    x = a.layers[layer_a].to(compute_device)
    y = b.layers[layer_b].to(compute_device)
    value = float(linear_cka(x, y).cpu())
    if not math.isfinite(value):
        raise ValueError("CKA result is not finite")
    return {
        "comparison": comparison,
        "architecture_a": a.architecture,
        "trained_task_count_a": a.task_count,
        "seed_a": a.seed,
        "run_id_a": a.run_id,
        "layer_a": layer_a,
        "architecture_b": b.architecture,
        "trained_task_count_b": b.task_count,
        "seed_b": b.seed,
        "run_id_b": b.run_id,
        "layer_b": layer_b,
        "probe_examples": x.shape[0],
        "linear_cka": value,
    }


def build_pairwise_rows(
    trained: Sequence[ActivationSet],
    random_baselines: Sequence[ActivationSet],
    *,
    compute_device: torch.device,
) -> list[dict[str, Any]]:
    """Compute preregistered primary and secondary CKA comparisons."""

    by_key = {
        (item.architecture, item.task_count, item.seed): item for item in trained
    }
    architectures = sorted({item.architecture for item in trained})
    task_counts = sorted({item.task_count for item in trained})
    seeds = sorted({item.seed for item in trained})
    if len(by_key) != len(architectures) * len(task_counts) * len(seeds):
        raise ValueError("trained activation matrix is incomplete or duplicated")
    rows: list[dict[str, Any]] = []

    # Primary: representation reproducibility across random seeds at fixed k.
    for architecture in architectures:
        for task_count in task_counts:
            for left_index, seed_a in enumerate(seeds):
                for seed_b in seeds[left_index + 1 :]:
                    a = by_key[architecture, task_count, seed_a]
                    b = by_key[architecture, task_count, seed_b]
                    for layer in _corresponding_layers(a, b):
                        rows.append(
                            _pair_row(
                                "seed_stability",
                                a,
                                b,
                                layer,
                                layer,
                                compute_device=compute_device,
                            )
                        )

    # Secondary: alignment of each smaller nested subset with the k=16 model.
    reference_k = max(task_counts)
    for architecture in architectures:
        for task_count in task_counts:
            if task_count == reference_k:
                continue
            for seed in seeds:
                a = by_key[architecture, task_count, seed]
                b = by_key[architecture, reference_k, seed]
                for layer in _corresponding_layers(a, b):
                    rows.append(
                        _pair_row(
                            "k16_alignment",
                            a,
                            b,
                            layer,
                            layer,
                            compute_device=compute_device,
                        )
                    )

    # Exploratory: final representations across architectures at matched k/seed.
    if {"transformer", "mlp"} <= set(architectures):
        for task_count in task_counts:
            for seed in seeds:
                a = by_key["transformer", task_count, seed]
                b = by_key["mlp", task_count, seed]
                rows.append(
                    _pair_row(
                        "cross_architecture",
                        a,
                        b,
                        "final_norm",
                        "final_norm",
                        compute_device=compute_device,
                    )
                )

    random_by_key = {
        (item.architecture, item.seed): item for item in random_baselines
    }
    for architecture in architectures:
        for left_index, seed_a in enumerate(seeds):
            for seed_b in seeds[left_index + 1 :]:
                a = random_by_key[architecture, seed_a]
                b = random_by_key[architecture, seed_b]
                for layer in _corresponding_layers(a, b):
                    rows.append(
                        _pair_row(
                            "random_baseline",
                            a,
                            b,
                            layer,
                            layer,
                            compute_device=compute_device,
                        )
                    )
    return rows


def summarize_pairwise_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, int, int, str], list[float]] = {}
    for row in rows:
        comparison = str(row["comparison"])
        if comparison == "cross_architecture":
            architecture = "transformer_vs_mlp"
            task_count = int(row["trained_task_count_a"])
            reference = task_count
        elif comparison == "random_baseline":
            architecture = str(row["architecture_a"])
            task_count = 0
            reference = 0
        else:
            architecture = str(row["architecture_a"])
            task_count = int(row["trained_task_count_a"])
            reference = int(row["trained_task_count_b"])
        key = (comparison, architecture, task_count, reference, str(row["layer_a"]))
        groups.setdefault(key, []).append(float(row["linear_cka"]))
    summaries: list[dict[str, Any]] = []
    for key in sorted(groups):
        comparison, architecture, task_count, reference, layer = key
        values = groups[key]
        summaries.append(
            {
                "comparison": comparison,
                "architecture": architecture,
                "trained_task_count": task_count,
                "reference_task_count": reference,
                "layer": layer,
                "pair_count": len(values),
                "cka_mean": statistics.fmean(values),
                "cka_sample_sd": statistics.stdev(values) if len(values) > 1 else 0.0,
                "cka_min": min(values),
                "cka_max": max(values),
            }
        )
    return summaries


def _average_ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = (start + 1 + end) / 2.0
        for position in range(start, end):
            ranks[order[position]] = rank
        start = end
    return ranks


def _spearman(x: Sequence[float], y: Sequence[float]) -> float:
    if len(x) != len(y) or len(x) < 2:
        raise ValueError("Spearman correlation requires matched nontrivial inputs")
    ranks_x = _average_ranks(x)
    ranks_y = _average_ranks(y)
    centered_x = [value - statistics.fmean(ranks_x) for value in ranks_x]
    centered_y = [value - statistics.fmean(ranks_y) for value in ranks_y]
    denominator = math.sqrt(
        sum(value * value for value in centered_x)
        * sum(value * value for value in centered_y)
    )
    if denominator == 0.0:
        raise ValueError("Spearman correlation is undefined for a constant input")
    return sum(a * b for a, b in zip(centered_x, centered_y)) / denominator


def _git_commit(repository: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _render_readme(
    summaries: Sequence[Mapping[str, Any]],
    *,
    probe: Mapping[str, Any],
    commit: str,
) -> str:
    lookup = {
        (
            row["comparison"],
            row["architecture"],
            int(row["trained_task_count"]),
            row["layer"],
        ): row
        for row in summaries
    }
    task_counts = (1, 2, 4, 8, 16)
    trend_lines: list[str] = []
    for architecture in ("transformer", "mlp"):
        values = [
            float(
                lookup[
                    ("seed_stability", architecture, task_count, "final_norm")
                ]["cka_mean"]
            )
            for task_count in task_counts
        ]
        delta = values[-1] - values[0]
        increases = sum(right > left for left, right in zip(values, values[1:]))
        rho = _spearman([math.log2(value) for value in task_counts], values)
        label = "Transformer" if architecture == "transformer" else "MLP"
        trend_lines.append(
            f"- {label}: k=1 to k=16 change {delta:+.4f}; Spearman rho "
            f"{rho:+.3f}; {increases}/4 adjacent steps increased."
        )
    lines = [
        "# V3 representation similarity with linear CKA",
        "",
        "This analysis tests whether increasing the number of nested training tasks",
        "makes the learned permutation representation more reproducible across random",
        "seeds. It does not compare raw parameters. Every model is frozen, receives",
        "the same task-free one-line prefix, and is measured at `<ONE_END>` before a",
        "task token appears.",
        "",
        "## Primary result: cross-seed stability",
        "",
        "Values are mean +/- sample SD over the three seed pairs (17-42, 17-314159,",
        "42-314159). Higher linear CKA means more similar representation geometry.",
        "",
        "| Architecture | Trained tasks (k) | Final-layer CKA |",
        "|---|---:|---:|",
    ]
    for architecture in ("transformer", "mlp"):
        label = "Transformer" if architecture == "transformer" else "MLP"
        for task_count in task_counts:
            row = lookup[("seed_stability", architecture, task_count, "final_norm")]
            lines.append(
                f"| {label} | {task_count} | "
                f"{float(row['cka_mean']):.4f} +/- {float(row['cka_sample_sd']):.4f} |"
            )
    lines.extend(
        [
            "",
            "Trend diagnostics across the five k values:",
            "",
            *trend_lines,
            "",
            "A positive endpoint change alone is not evidence of a monotonic trend; the",
            "Spearman and adjacent-step diagnostics show whether the intermediate k",
            "values support the same direction.",
        ]
    )
    lines.extend(
        [
            "",
            "## Secondary result: alignment with k=16",
            "",
            "This compares each smaller model with the same-seed k=16 model. Because",
            "the task sets are nested, increasing k also increases task overlap with the",
            "reference. These values are descriptive and are not an isolated causal",
            "effect of task count.",
            "",
            "| Architecture | Smaller k | Final-layer CKA to k=16 |",
            "|---|---:|---:|",
        ]
    )
    for architecture in ("transformer", "mlp"):
        label = "Transformer" if architecture == "transformer" else "MLP"
        for task_count in (1, 2, 4, 8):
            row = lookup[("k16_alignment", architecture, task_count, "final_norm")]
            lines.append(
                f"| {label} | {task_count} | "
                f"{float(row['cka_mean']):.4f} +/- {float(row['cka_sample_sd']):.4f} |"
            )
    lines.extend(
        [
            "",
            "## Random-initialization controls",
            "",
            "| Architecture | Final-layer CKA |",
            "|---|---:|",
        ]
    )
    for architecture in ("transformer", "mlp"):
        label = "Transformer" if architecture == "transformer" else "MLP"
        row = lookup[("random_baseline", architecture, 0, "final_norm")]
        lines.append(
            f"| {label} | {float(row['cka_mean']):.4f} +/- "
            f"{float(row['cka_sample_sd']):.4f} |"
        )
    lines.extend(
        [
            "",
            "## Protocol",
            "",
            f"- Probe split: validation shard 098; test shard 099 was not read.",
            f"- Probe examples: {probe['example_count']:,}, selected deterministically",
            f"  with seed {probe['seed']} and probe SHA-256 `{probe['probe_sha256']}`.",
            "- Landmark: the hidden vector at `<ONE_END>` from the embedding output,",
            "  every model block, and final layer normalization.",
            "- Metric: biased linear CKA over examples, accumulated in float64.",
            "- Reference implementation: Ristori's `ckatorch` at commit",
            f"  `{REFERENCE_IMPLEMENTATION_COMMIT}`; the local Gram-free formula is",
            "  regression-tested against direct centered-Gram CKA.",
            "- Primary units: three pairwise comparisons among three independently",
            "  trained seeds for each architecture and k.",
            "- Random-init controls use the same architectures, tokenizer, positions,",
            "  inputs, and seeds without training.",
            f"- Analysis implementation commit: `{commit}`.",
            "",
            "## Interpretation limits",
            "",
            "Only one frozen nested task order was trained. Thus k is confounded with",
            "which tasks were added and with reduced per-task exposure at larger k. The",
            "three seed pairs are also dependent because each of three models appears in",
            "two pairs. The results can establish an observed representation-stability",
            "trend, but not a general causal law that task diversity creates a universal",
            "permutation representation. Cross-architecture CKA is exploratory because",
            "the Transformer and causal MLP have different computational structures.",
            "",
            "## Machine-readable files",
            "",
            "- `pairwise_layer_cka.csv`: every preregistered model/layer comparison.",
            "- `summary.csv`: pair means, sample SDs, minima, and maxima.",
            "- `probe_manifest.json`: exact probe IDs, length histogram, and data hash.",
            "- `manifest.json`: model checkpoint hashes and output checksums.",
            "",
        ]
    )
    return "\n".join(lines)


def run_analysis(
    config_path: Path,
    output_dir: Path,
    *,
    probe_count: int,
    probe_seed: int,
    batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    audit = audit_experiment(config_path, matrix="nested")
    if not audit["ok"] or audit["passed_count"] != 30:
        raise ValueError("nested experiment must pass strict 30/30 audit before CKA")
    repository = Path(audit["repository"])
    manifest = repository / "data/permutation-10m-v3/manifest.json"
    manifest_sha256 = _sha256(manifest)
    examples = select_probe_examples(
        manifest, count=probe_count, seed=probe_seed, shard_index=DEFAULT_VALIDATION_SHARD
    )
    probe = probe_identity(
        examples,
        dataset_manifest_sha256=manifest_sha256,
        shard_index=DEFAULT_VALIDATION_SHARD,
        seed=probe_seed,
    )
    _atomic_json(probe, output_dir / "probe_manifest.json")

    cache_dir = output_dir / "cache"
    trained: list[ActivationSet] = []
    for index, run in enumerate(audit["runs"], start=1):
        print(f"extracting nested activation {index}/30: {run['run_id']}", flush=True)
        trained.append(
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

    random_baselines: list[ActivationSet] = []
    seeds = sorted({item.seed for item in trained})
    for architecture in sorted({item.architecture for item in trained}):
        reference = next(
            run for run in audit["runs"] if run["architecture"] == architecture
        )
        for seed in seeds:
            print(f"extracting random baseline: {architecture}, seed {seed}", flush=True)
            random_baselines.append(
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
            )

    rows = build_pairwise_rows(trained, random_baselines, compute_device=device)
    summaries = summarize_pairwise_rows(rows)
    pair_path = output_dir / "pairwise_layer_cka.csv"
    summary_path = output_dir / "summary.csv"
    _atomic_csv(rows, PAIR_FIELDS, pair_path)
    _atomic_csv(summaries, SUMMARY_FIELDS, summary_path)
    commit = _git_commit(repository)
    readme_path = output_dir / "README.md"
    _atomic_text(
        _render_readme(summaries, probe=probe, commit=commit), readme_path
    )
    checkpoint_rows = [
        {
            "run_id": item.run_id,
            "architecture": item.architecture,
            "trained_task_count": item.task_count,
            "seed": item.seed,
            "checkpoint_sha256": item.checkpoint_sha256,
        }
        for item in trained
    ]
    result = {
        "format_version": FORMAT_VERSION,
        "status": "completed",
        "method": "biased_linear_cka",
        "reference_implementation": {
            "repository": "https://github.com/RistoAle97/centered-kernel-alignment",
            "commit": REFERENCE_IMPLEMENTATION_COMMIT,
        },
        "landmark": "<ONE_END>",
        "activation_dtype": "float32",
        "cka_accumulation_dtype": "float64",
        "analysis_commit": commit,
        "config_path": str(config_path),
        "config_sha256": audit["config_sha256"],
        "probe": probe,
        "models": checkpoint_rows,
        "artifacts": {
            pair_path.name: _sha256(pair_path),
            summary_path.name: _sha256(summary_path),
            "probe_manifest.json": _sha256(output_dir / "probe_manifest.json"),
            readme_path.name: _sha256(readme_path),
        },
    }
    _atomic_json(result, output_dir / "manifest.json")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--probe-count", type=int, default=DEFAULT_PROBE_COUNT)
    parser.add_argument("--probe-seed", type=int, default=DEFAULT_PROBE_SEED)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--device",
        default="auto",
        help="auto, cpu, cuda, or another torch device",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    result = run_analysis(
        args.config,
        args.output_dir,
        probe_count=args.probe_count,
        probe_seed=args.probe_seed,
        batch_size=args.batch_size,
        device=device,
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
