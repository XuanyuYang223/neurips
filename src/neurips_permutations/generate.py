"""Deterministic, streaming generation of permutation-task corpora."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import gzip
import hashlib
from itertools import islice, permutations
import json
import math
import os
from pathlib import Path
from random import Random
from typing import Any, Iterable, Mapping, Sequence

from . import math_ops as ops
from .math_ops import PROPERTY32_TASK_NAMES, V2_TASK_NAMES, V3_TASK_NAMES
from .passage import TASK_SPECS, passage_tokens


V2_SCHEMA_VERSION = "permutation-20/v2"
V3_SCHEMA_VERSION = "permutation-20/v3"
PROPERTY32_SCHEMA_VERSION = "permutation-properties-32/v1"
# The unqualified API and CLI now create Henry's revised task suite.  The
# explicit v2 protocol remains available for reproducing and validating the
# already-generated corpus.
SCHEMA_VERSION = V3_SCHEMA_VERSION
TASKS_BY_SCHEMA: Mapping[str, tuple[str, ...]] = {
    V2_SCHEMA_VERSION: V2_TASK_NAMES,
    V3_SCHEMA_VERSION: V3_TASK_NAMES,
    PROPERTY32_SCHEMA_VERSION: PROPERTY32_TASK_NAMES,
}
DEFAULT_COUNT = 10_000_000
DEFAULT_MIN_ENTRIES = 2
DEFAULT_MAX_ENTRIES = 30
DEFAULT_BASE = 100
DEFAULT_SEED = 20_260_830
DEFAULT_SHARD_SIZE = 100_000
DEFAULT_OUTPUT_DIR = Path("data/permutation-10m-v3")
DEFAULT_WORKERS = max(1, min(8, os.cpu_count() or 1))
GZIP_COMPRESSLEVEL = 6

_MASK_64 = (1 << 64) - 1


@dataclass(frozen=True, slots=True)
class _ShardJob:
    index: int
    start_id: int
    record_count: int
    filename: str
    output_dir: str
    min_entries: int
    max_entries: int
    base: int
    seed: int
    schema_version: str


def _checked_int(name: str, value: int, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def task_names_for_schema(schema_version: str) -> tuple[str, ...]:
    """Return the frozen ordered task registry for a supported protocol."""

    try:
        return TASKS_BY_SCHEMA[schema_version]
    except (KeyError, TypeError):
        supported = ", ".join(TASKS_BY_SCHEMA)
        raise ValueError(
            f"unsupported schema_version {schema_version!r}; choose {supported}"
        ) from None


def _validate_config(
    *,
    count: int,
    min_entries: int,
    max_entries: int,
    base: int,
    shard_size: int,
    seed: int,
    workers: int,
    schema_version: str,
) -> None:
    _checked_int("count", count, minimum=1)
    # V2/V3 include Bruhat examples whose balanced negative construction needs
    # S_4.  The scalar-only property registry is well-defined from S_2.
    _checked_int("min_entries", min_entries, minimum=2)
    minimum_maximum = 2 if schema_version == PROPERTY32_SCHEMA_VERSION else 4
    _checked_int("max_entries", max_entries, minimum=minimum_maximum)
    if min_entries > max_entries:
        raise ValueError("min_entries must not exceed max_entries")
    _checked_int("base", base, minimum=2)
    _checked_int("shard_size", shard_size, minimum=1)
    _checked_int("seed", seed, minimum=0)
    _checked_int("workers", workers, minimum=1)
    task_names = task_names_for_schema(schema_version)
    if not task_names or len(set(task_names)) != len(task_names):
        raise RuntimeError("each schema must contain a nonempty unique task grid")
    if any(task not in TASK_SPECS for task in task_names):
        raise RuntimeError("math and Passage task registries disagree")
    if count % len(task_names):
        raise ValueError(
            f"count must be divisible by {len(task_names)} for exact task balance"
        )
    if schema_version in {V2_SCHEMA_VERSION, V3_SCHEMA_VERSION} and (
        count // len(task_names)
    ) % 2:
        raise ValueError(
            "count must be divisible by 40 for exact binary-label balance"
        )
    # passage.py implements the supplied, fixed base-100 grammar.
    if base != 100:
        raise ValueError("the Passage Math grammar currently requires base=100")


def _splitmix64(value: int) -> int:
    value = (value + 0x9E3779B97F4A7C15) & _MASK_64
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _MASK_64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _MASK_64
    return value ^ (value >> 31)


def _record_rng(seed: int, record_id: int) -> Random:
    """Return an ID-addressable RNG independent of shard and worker order."""

    return Random(_splitmix64(seed ^ _splitmix64(record_id)))


def _random_permutation(size: int, rng: Random) -> tuple[int, ...]:
    values = list(range(1, size + 1))
    # Spell out Fisher--Yates so the sampling contract is explicit.
    for right in range(size - 1, 0, -1):
        left = rng.randrange(right + 1)
        values[left], values[right] = values[right], values[left]
    return tuple(values)


def _json_value(value: object) -> object:
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    return value


def _balanced_pattern(
    primary: tuple[int, ...], *, should_avoid: bool, rng: Random
) -> tuple[int, ...]:
    """Choose a size-(n-1) pattern with an exact requested containment label."""

    def standardize(values: Iterable[int]) -> tuple[int, ...]:
        ranks = {value: rank for rank, value in enumerate(sorted(values), start=1)}
        return tuple(ranks[value] for value in values)

    contained = {
        standardize((*primary[:deleted], *primary[deleted + 1 :]))
        for deleted in range(len(primary))
    }
    if not should_avoid:
        ordered = sorted(contained)
        return ordered[rng.randrange(len(ordered))]

    pattern_size = len(primary) - 1
    for _ in range(16):
        candidate = _random_permutation(pattern_size, rng)
        if candidate not in contained:
            return candidate
    # There are at most n deletion patterns.  Checking n+1 distinct candidates
    # therefore guarantees a missing one without enumerating (n-1)! patterns.
    candidates = islice(
        permutations(range(1, pattern_size + 1)), len(primary) + 1
    )
    return next(candidate for candidate in candidates if candidate not in contained)


def _permutation_with_inversion_count(
    size: int, inversion_count: int, rng: Random
) -> tuple[int, ...]:
    """Sample a varied Lehmer code with an exact requested digit sum."""

    maximum = size * (size - 1) // 2
    if not 0 <= inversion_count <= maximum:
        raise ValueError("requested inversion count is outside S_n")
    remaining = inversion_count
    code: list[int] = []
    for position in range(size):
        capacity = size - position - 1
        future_capacity = capacity * (capacity - 1) // 2
        lower = max(0, remaining - future_capacity)
        upper = min(capacity, remaining)
        digit = rng.randint(lower, upper)
        code.append(digit)
        remaining -= digit
    if remaining:
        raise AssertionError("failed to construct an exact Lehmer digit sum")
    available = list(range(1, size + 1))
    permutation: list[int] = []
    for digit in code:
        permutation.append(available.pop(digit))
    return tuple(permutation)


def _negative_bruhat_fallback(
    size: int, gap: int, rng: Random
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return a certified incomparable pair, embedded in larger S_n."""

    if size == 4:
        lower = (1, 2, 4, 3)
        uppers = {
            1: (2, 3, 1, 4),
            2: (3, 2, 1, 4),
        }
    else:
        lower = (1, 2, 3, 5, 4)
        uppers = {
            1: (1, 3, 4, 2, 5),
            2: (1, 4, 3, 2, 5),
            3: (2, 4, 3, 1, 5),
            4: (3, 4, 2, 1, 5),
        }
    try:
        upper = uppers[gap]
    except KeyError:
        raise ValueError(f"no negative Bruhat fallback for S_{size} gap {gap}") from None
    if size > len(lower):
        suffix = tuple(range(len(lower) + 1, size + 1))
        lower = (*lower, *suffix)
        upper = (*upper, *suffix)
    # Inversion is a length-preserving Bruhat automorphism and cheaply gives
    # the deterministic fallback more than one surface form.
    if rng.randrange(2):
        lower, upper = ops.inverse(lower), ops.inverse(upper)
    return lower, upper


def _balanced_bruhat_pair(
    size: int, *, gap: int, should_be_leq: bool, rng: Random
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Construct a strict pair with fixed positive length gap and label.

    Positive examples follow a saturated chain of right-ascent covers.
    Negative examples independently sample two exact inversion ranks and
    reject comparable pairs.  Both paths have bounded work; certified small
    parabolic examples provide a deterministic fallback.
    """

    if size < 4:
        raise ValueError("positive-gap incomparable Bruhat pairs require n >= 4")
    gap_limit = 2 if size == 4 else min(4, size - 1)
    if not 1 <= gap <= gap_limit:
        raise ValueError(f"gap must satisfy 1 <= gap <= {gap_limit} for S_{size}")

    if should_be_leq:
        for _ in range(64):
            lower = _random_permutation(size, rng)
            upper = lower
            for _ in range(gap):
                ascents = [
                    index
                    for index in range(1, size)
                    if upper[index - 1] < upper[index]
                ]
                if not ascents:
                    break
                upper = ops.right_multiply_simple(
                    upper, ascents[rng.randrange(len(ascents))]
                )
            else:
                break
        else:  # pragma: no cover - overwhelmingly unlikely bounded fallback
            lower = tuple(range(1, size + 1))
            upper = _permutation_with_inversion_count(size, gap, rng)
    else:
        maximum = size * (size - 1) // 2
        for _ in range(128):
            lower_rank = rng.randint(1, maximum - gap - 1)
            lower = _permutation_with_inversion_count(size, lower_rank, rng)
            upper = _permutation_with_inversion_count(
                size, lower_rank + gap, rng
            )
            if not ops.bruhat_leq(lower, upper):
                break
        else:
            lower, upper = _negative_bruhat_fallback(size, gap, rng)

    actual_gap = ops.inversion_count(upper) - ops.inversion_count(lower)
    actual_label = ops.bruhat_leq(lower, upper)
    if actual_gap != gap or actual_label is not should_be_leq:
        raise AssertionError("Bruhat construction violated its gap/label contract")
    return lower, upper


def build_record(
    record_id: int,
    *,
    min_entries: int = DEFAULT_MIN_ENTRIES,
    max_entries: int = DEFAULT_MAX_ENTRIES,
    base: int = DEFAULT_BASE,
    seed: int = DEFAULT_SEED,
    schema_version: str = SCHEMA_VERSION,
) -> dict[str, Any]:
    """Build one compact record, deterministically addressed by ``record_id``."""

    _checked_int("record_id", record_id, minimum=0)
    _checked_int("min_entries", min_entries, minimum=2)
    minimum_maximum = 2 if schema_version == PROPERTY32_SCHEMA_VERSION else 3
    _checked_int("max_entries", max_entries, minimum=minimum_maximum)
    if min_entries > max_entries:
        raise ValueError("min_entries must not exceed max_entries")
    if base != 100:
        raise ValueError("the Passage Math grammar currently requires base=100")

    task_names = task_names_for_schema(schema_version)
    task = task_names[record_id % len(task_names)]
    task_occurrence = record_id // len(task_names)
    rng = _record_rng(seed, record_id)
    if task == "bruhat_leq":
        if max_entries < 4:
            raise ValueError(
                "bruhat_leq requires max_entries >= 4 for nontrivial negatives"
            )
        # Consecutive positive/negative occurrences share size and gap exactly,
        # preventing either quantity from leaking the binary label.
        pair_index = task_occurrence // 2
        pair_rng = _record_rng(seed ^ 0xB8A7_4A7, pair_index)
        size = pair_rng.randint(max(4, min_entries), max_entries)
        gap_limit = 2 if size == 4 else min(4, size - 1)
        bruhat_gap = 1 + pair_index % gap_limit
    else:
        task_minimum = 3 if task == "pattern_avoidance" else 2
        size = rng.randint(max(task_minimum, min_entries), max_entries)
    primary = _random_permutation(size, rng)
    answer: object
    render_kwargs: dict[str, object] = {}
    inputs: dict[str, object] = {"primary": primary}

    if schema_version == PROPERTY32_SCHEMA_VERSION:
        answer = ops.PROPERTY_FUNCTIONS[task](primary)
    elif task == "to_cycle":
        answer = ops.canonical_cycles(primary)
    elif task == "to_lehmer":
        answer = ops.lehmer_code(primary)
    elif task == "to_inversion_vector":
        answer = ops.inversion_vector(primary)
    elif task == "to_reduced_word":
        answer = ops.reduced_coxeter_word(primary)
    elif task == "length":
        answer = ops.inversion_count(primary)
    elif task == "descents":
        answer = ops.descent_count(primary)
    elif task == "fixed_points":
        answer = ops.fixed_point_count(primary)
    elif task == "peaks":
        answer = ops.peak_count(primary)
    elif task == "exceedances":
        answer = ops.exceedance_count(primary)
    elif task == "recoils":
        answer = ops.recoil_count(primary)
    elif task == "parity":
        answer = ops.parity(primary)
    elif task == "cycle_type":
        answer = ops.cycle_type(primary)
    elif task == "rsk_shape":
        answer = ops.rsk_shape(primary)
    elif task == "lis_length":
        answer = ops.lis_length(primary)
    elif task == "lds_length":
        answer = ops.lds_length(primary)
    elif task == "pattern_avoidance":
        should_avoid = task_occurrence % 2 == 0
        if should_avoid and size == 3:
            # Some S_3 permutations contain both S_2 patterns after a deletion.
            # A monotone primary has exactly one, leaving the other to avoid.
            primary = (
                tuple(range(1, size + 1))
                if rng.randrange(2) == 0
                else tuple(range(size, 0, -1))
            )
            inputs["primary"] = primary
        pattern = _balanced_pattern(
            primary,
            should_avoid=should_avoid,
            rng=rng,
        )
        inputs["pattern"] = pattern
        render_kwargs["pattern"] = pattern
        answer = int(ops.avoids_pattern(primary, pattern))
    elif task == "inverse":
        answer = ops.inverse(primary)
    elif task in {"compose", "conjugate", "commutator"}:
        operand = _random_permutation(size, rng)
        inputs["operand"] = operand
        render_kwargs["operand"] = operand
        if task == "compose":
            answer = ops.compose(primary, operand)
        elif task == "conjugate":
            answer = ops.conjugate(operand, primary)
        else:
            answer = ops.commutator(primary, operand)
    elif task == "power":
        exponent = rng.randint(0, 100)
        inputs["exponent"] = exponent
        render_kwargs["exponent"] = exponent
        answer = ops.power(primary, exponent)
    elif task == "right_multiply_simple":
        simple_index = rng.randint(1, size - 1)
        inputs["simple_index"] = simple_index
        render_kwargs["simple_index"] = simple_index
        answer = ops.right_multiply_simple(primary, simple_index)
    elif task == "bruhat_leq":
        primary, operand = _balanced_bruhat_pair(
            size,
            gap=bruhat_gap,
            should_be_leq=(task_occurrence % 2 == 0),
            rng=rng,
        )
        inputs["primary"] = primary
        inputs["operand"] = operand
        render_kwargs["operand"] = operand
        answer = int(ops.bruhat_leq(primary, operand))
        expected_answer = int(task_occurrence % 2 == 0)
        if answer != expected_answer:
            raise AssertionError("Bruhat label schedule is not exactly balanced")
    else:  # pragma: no cover - guarded by the frozen task registry
        raise AssertionError(f"unhandled task {task!r}")

    tokens = passage_tokens(task, primary, answer, **render_kwargs)  # type: ignore[arg-type]
    return {
        "schema_version": schema_version,
        "id": record_id,
        "task": task,
        "n": size,
        "inputs": _json_value(inputs),
        "answer": _json_value(answer),
        "answer_kind": TASK_SPECS[task].answer_kind,
        "tokens": list(tokens),
        "canonical_text": " ".join(tokens),
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fsync_directory(directory: Path) -> None:
    try:
        descriptor = os.open(directory, os.O_RDONLY)
    except OSError:  # pragma: no cover - platform/filesystem dependent
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _metadata(
    job: _ShardJob, *, sha256: str, byte_size: int, task_counts: Mapping[str, int]
) -> dict[str, Any]:
    task_names = task_names_for_schema(job.schema_version)
    return {
        "index": job.index,
        "filename": job.filename,
        "first_id": job.start_id,
        "last_id": job.start_id + job.record_count - 1,
        "record_count": job.record_count,
        "byte_size": byte_size,
        "sha256": sha256,
        "task_counts": {task: int(task_counts.get(task, 0)) for task in task_names},
    }


def _write_shard(job: _ShardJob) -> dict[str, Any]:
    output_dir = Path(job.output_dir)
    final_path = output_dir / job.filename
    partial_path = output_dir / f"{job.filename}.partial"
    task_counts: Counter[str] = Counter()

    with partial_path.open("wb") as raw:
        with gzip.GzipFile(
            filename="",
            mode="wb",
            compresslevel=GZIP_COMPRESSLEVEL,
            fileobj=raw,
            mtime=0,
        ) as compressed:
            for record_id in range(job.start_id, job.start_id + job.record_count):
                record = build_record(
                    record_id,
                    min_entries=job.min_entries,
                    max_entries=job.max_entries,
                    base=job.base,
                    seed=job.seed,
                    schema_version=job.schema_version,
                )
                task_counts[record["task"]] += 1
                line = json.dumps(
                    record, ensure_ascii=True, sort_keys=True, separators=(",", ":")
                )
                compressed.write(line.encode("utf-8") + b"\n")
        raw.flush()
        os.fsync(raw.fileno())

    os.replace(partial_path, final_path)
    _fsync_directory(output_dir)
    return _metadata(
        job,
        sha256=_sha256_file(final_path),
        byte_size=final_path.stat().st_size,
        task_counts=task_counts,
    )


def _expected_jobs(
    *,
    count: int,
    shard_size: int,
    output_dir: Path,
    min_entries: int,
    max_entries: int,
    base: int,
    seed: int,
    schema_version: str,
) -> tuple[_ShardJob, ...]:
    shard_count = math.ceil(count / shard_size)
    width = max(5, len(str(max(0, shard_count - 1))))
    jobs: list[_ShardJob] = []
    for index, start_id in enumerate(range(0, count, shard_size)):
        jobs.append(
            _ShardJob(
                index=index,
                start_id=start_id,
                record_count=min(shard_size, count - start_id),
                filename=f"part-{index:0{width}d}.jsonl.gz",
                output_dir=str(output_dir),
                min_entries=min_entries,
                max_entries=max_entries,
                base=base,
                seed=seed,
                schema_version=schema_version,
            )
        )
    return tuple(jobs)


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read existing manifest {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("existing manifest must contain a JSON object")
    return value


def _assert_compatible_manifest(
    manifest: Mapping[str, Any],
    *,
    count: int,
    min_entries: int,
    max_entries: int,
    base: int,
    seed: int,
    shard_size: int,
    schema_version: str,
) -> None:
    task_names = task_names_for_schema(schema_version)
    expected = {
        "schema_version": schema_version,
        "count": count,
        "max_entries": max_entries,
        "base": base,
        "seed": seed,
        "shard_size": shard_size,
        "tasks": list(task_names),
    }
    if min_entries != DEFAULT_MIN_ENTRIES:
        expected["min_entries"] = min_entries
    elif manifest.get("min_entries", DEFAULT_MIN_ENTRIES) != DEFAULT_MIN_ENTRIES:
        expected["min_entries"] = min_entries
    mismatches = [key for key, value in expected.items() if manifest.get(key) != value]
    if mismatches:
        raise ValueError(
            "existing manifest uses a different configuration: "
            + ", ".join(mismatches)
        )


def _reuse_from_manifest(
    job: _ShardJob, entry: Mapping[str, Any] | None
) -> dict[str, Any] | None:
    if entry is None or entry.get("filename") != job.filename:
        return None
    path = Path(job.output_dir) / job.filename
    if not path.is_file():
        return None
    try:
        if path.stat().st_size != entry.get("byte_size"):
            return None
        if _sha256_file(path) != entry.get("sha256"):
            return None
        if entry.get("record_count") != job.record_count:
            return None
        if entry.get("first_id") != job.start_id:
            return None
        if entry.get("last_id") != job.start_id + job.record_count - 1:
            return None
        counts = entry.get("task_counts")
        if not isinstance(counts, dict) or sum(counts.values()) != job.record_count:
            return None
    except (OSError, TypeError):
        return None
    return dict(entry)


def _inspect_orphan_shard(job: _ShardJob) -> dict[str, Any] | None:
    """Fully inspect a shard left by a crash before manifest publication."""

    path = Path(job.output_dir) / job.filename
    if not path.is_file():
        return None
    counts: Counter[str] = Counter()
    expected_id = job.start_id
    task_names = task_names_for_schema(job.schema_version)
    try:
        with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
            for line in handle:
                record = json.loads(line)
                if not isinstance(record, dict):
                    return None
                if record.get("schema_version") != job.schema_version:
                    return None
                if record.get("id") != expected_id:
                    return None
                expected_task = task_names[expected_id % len(task_names)]
                if record.get("task") != expected_task:
                    return None
                counts[expected_task] += 1
                expected_id += 1
    except (OSError, EOFError, UnicodeError, json.JSONDecodeError):
        return None
    if expected_id != job.start_id + job.record_count:
        return None
    return _metadata(
        job,
        sha256=_sha256_file(path),
        byte_size=path.stat().st_size,
        task_counts=counts,
    )


def _write_manifest(path: Path, manifest: Mapping[str, Any]) -> None:
    partial = path.with_name(f"{path.name}.partial")
    payload = json.dumps(
        manifest, ensure_ascii=True, indent=2, sort_keys=True
    ).encode("utf-8") + b"\n"
    with partial.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(partial, path)
    _fsync_directory(path.parent)


def generate_dataset(
    *,
    count: int = DEFAULT_COUNT,
    min_entries: int = DEFAULT_MIN_ENTRIES,
    max_entries: int = DEFAULT_MAX_ENTRIES,
    base: int = DEFAULT_BASE,
    seed: int = DEFAULT_SEED,
    shard_size: int = DEFAULT_SHARD_SIZE,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    workers: int = DEFAULT_WORKERS,
    resume: bool = True,
    schema_version: str = SCHEMA_VERSION,
) -> Path:
    """Generate or safely resume a corpus and return its manifest path."""

    _validate_config(
        count=count,
        min_entries=min_entries,
        max_entries=max_entries,
        base=base,
        shard_size=shard_size,
        seed=seed,
        workers=workers,
        schema_version=schema_version,
    )
    task_names = task_names_for_schema(schema_version)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    jobs = _expected_jobs(
        count=count,
        shard_size=shard_size,
        output_dir=destination,
        min_entries=min_entries,
        max_entries=max_entries,
        base=base,
        seed=seed,
        schema_version=schema_version,
    )
    manifest_path = destination / "manifest.json"
    old_manifest: dict[str, Any] | None = None
    old_by_index: dict[int, Mapping[str, Any]] = {}
    if resume and manifest_path.exists():
        old_manifest = _read_json_object(manifest_path)
        _assert_compatible_manifest(
            old_manifest,
            count=count,
            min_entries=min_entries,
            max_entries=max_entries,
            base=base,
            seed=seed,
            shard_size=shard_size,
            schema_version=schema_version,
        )
        old_shards = old_manifest.get("shards", [])
        if not isinstance(old_shards, list):
            raise ValueError("existing manifest shards must be a list")
        old_by_index = {
            entry["index"]: entry
            for entry in old_shards
            if isinstance(entry, dict) and isinstance(entry.get("index"), int)
        }

    completed: dict[int, dict[str, Any]] = {}
    pending: list[_ShardJob] = []
    for job in jobs:
        reused = None
        if resume:
            reused = _reuse_from_manifest(job, old_by_index.get(job.index))
            if reused is None and old_manifest is None:
                reused = _inspect_orphan_shard(job)
        if reused is None:
            pending.append(job)
        else:
            completed[job.index] = reused

    if workers == 1:
        for job in pending:
            completed[job.index] = _write_shard(job)
    elif pending:
        with ProcessPoolExecutor(max_workers=min(workers, len(pending))) as executor:
            futures = {executor.submit(_write_shard, job): job for job in pending}
            for future in as_completed(futures):
                metadata = future.result()  # Worker failures intentionally propagate.
                completed[metadata["index"]] = metadata

    shards = [completed[index] for index in range(len(jobs))]
    task_counts: Counter[str] = Counter()
    for shard in shards:
        task_counts.update(shard["task_counts"])
    expected_per_task = count // len(task_names)
    if any(task_counts[task] != expected_per_task for task in task_names):
        raise RuntimeError("generated corpus is not exactly task-balanced")

    manifest: dict[str, Any] = {
        "schema_version": schema_version,
        "format": "jsonl.gz",
        "gzip_mtime": 0,
        "gzip_compresslevel": GZIP_COMPRESSLEVEL,
        "count": count,
        "max_entries": max_entries,
        "base": base,
        "seed": seed,
        "shard_size": shard_size,
        "shard_count": len(shards),
        "tasks": list(task_names),
        "task_counts": {task: task_counts[task] for task in task_names},
        "total_bytes": sum(shard["byte_size"] for shard in shards),
        "shards": shards,
    }
    if min_entries != DEFAULT_MIN_ENTRIES:
        manifest["min_entries"] = min_entries
    _write_manifest(manifest_path, manifest)
    return manifest_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate deterministic Passage Math permutation shards."
    )
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT)
    parser.add_argument("--min-entries", type=int, default=DEFAULT_MIN_ENTRIES)
    parser.add_argument("--max-entries", type=int, default=DEFAULT_MAX_ENTRIES)
    parser.add_argument("--base", type=int, default=DEFAULT_BASE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--shard-size", type=int, default=DEFAULT_SHARD_SIZE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument(
        "--schema-version",
        choices=tuple(TASKS_BY_SCHEMA),
        default=SCHEMA_VERSION,
        help="dataset protocol to generate (default: permutation-20/v3)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_false",
        dest="resume",
        help="regenerate every expected shard instead of validating and reusing it",
    )
    parser.set_defaults(resume=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = generate_dataset(
        count=args.count,
        min_entries=args.min_entries,
        max_entries=args.max_entries,
        base=args.base,
        seed=args.seed,
        shard_size=args.shard_size,
        output_dir=args.output_dir,
        workers=args.workers,
        resume=args.resume,
        schema_version=args.schema_version,
    )
    print(manifest)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "DEFAULT_BASE",
    "DEFAULT_COUNT",
    "DEFAULT_MIN_ENTRIES",
    "DEFAULT_MAX_ENTRIES",
    "DEFAULT_OUTPUT_DIR",
    "DEFAULT_SEED",
    "DEFAULT_SHARD_SIZE",
    "DEFAULT_WORKERS",
    "SCHEMA_VERSION",
    "PROPERTY32_SCHEMA_VERSION",
    "TASKS_BY_SCHEMA",
    "V2_SCHEMA_VERSION",
    "V3_SCHEMA_VERSION",
    "build_parser",
    "build_record",
    "generate_dataset",
    "main",
    "task_names_for_schema",
]
