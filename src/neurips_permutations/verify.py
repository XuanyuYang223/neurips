"""Integrity and schema verification for generated permutation shards."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import gzip
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .generate import TASKS_BY_SCHEMA, task_names_for_schema
from . import math_ops as ops
from .math_ops import validate_permutation
from .passage import TASK_SPECS, VOCABULARY, passage_tokens


DEFAULT_WORKERS = 1


@dataclass(frozen=True, slots=True)
class _ShardVerificationJob:
    path: str
    entry: dict[str, Any]
    max_entries: int
    full: bool
    schema_version: str


class VerificationError(ValueError):
    """Raised when a manifest, shard, or record violates the frozen protocol."""


def _fail(message: str) -> None:
    raise VerificationError(message)


def _load_manifest(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise VerificationError(f"cannot read manifest {path}: {exc}") from exc
    if not isinstance(value, dict):
        _fail("manifest must be a JSON object")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise VerificationError(f"cannot read shard {path}: {exc}") from exc
    return digest.hexdigest()


def _require_plain_filename(value: object, *, index: int) -> str:
    if not isinstance(value, str) or not value:
        _fail(f"shard {index} has an invalid filename")
    candidate = Path(value)
    if candidate.is_absolute() or candidate.name != value or value in {".", ".."}:
        _fail(f"shard {index} filename must be one safe relative basename")
    return value


def _require_int(
    value: object, *, name: str, minimum: int | None = None
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _fail(f"{name} must be an integer")
    if minimum is not None and value < minimum:
        _fail(f"{name} must be >= {minimum}")
    return value


def _require_task_counts(
    value: object,
    *,
    name: str,
    expected_total: int,
    task_names: Sequence[str],
) -> dict[str, int]:
    if not isinstance(value, dict) or set(value) != set(task_names):
        _fail(f"{name} must contain exactly the {len(task_names)} task keys")
    result: dict[str, int] = {}
    for task in task_names:
        result[task] = _require_int(
            value[task], name=f"{name}.{task}", minimum=0
        )
    if sum(result.values()) != expected_total:
        _fail(f"{name} does not sum to {expected_total}")
    return result


def _record_render_kwargs(inputs: Mapping[str, object]) -> dict[str, object]:
    return {
        key: inputs[key]
        for key in ("operand", "pattern", "exponent", "simple_index")
        if key in inputs
    }


def _json_value(value: object) -> object:
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    return value


def _mathematical_answer(
    task: str, primary: tuple[int, ...], inputs: Mapping[str, object]
) -> object:
    """Recompute one task answer from its mathematical inputs."""

    if task in ops.PROPERTY_FUNCTIONS:
        return ops.PROPERTY_FUNCTIONS[task](primary)

    operand = inputs.get("operand")
    pattern = inputs.get("pattern")
    exponent = inputs.get("exponent")
    simple_index = inputs.get("simple_index")
    if task == "to_cycle":
        return ops.canonical_cycles(primary)
    if task == "to_lehmer":
        return ops.lehmer_code(primary)
    if task == "to_inversion_vector":
        return ops.inversion_vector(primary)
    if task == "to_reduced_word":
        return ops.reduced_coxeter_word(primary)
    if task == "length":
        return ops.inversion_count(primary)
    if task == "descents":
        return ops.descent_count(primary)
    if task == "fixed_points":
        return ops.fixed_point_count(primary)
    if task == "peaks":
        return ops.peak_count(primary)
    if task == "exceedances":
        return ops.exceedance_count(primary)
    if task == "recoils":
        return ops.recoil_count(primary)
    if task == "parity":
        return ops.parity(primary)
    if task == "cycle_type":
        return ops.cycle_type(primary)
    if task == "rsk_shape":
        return ops.rsk_shape(primary)
    if task == "lis_length":
        return ops.lis_length(primary)
    if task == "lds_length":
        return ops.lds_length(primary)
    if task == "pattern_avoidance":
        return int(ops.avoids_pattern(primary, pattern))  # type: ignore[arg-type]
    if task == "inverse":
        return ops.inverse(primary)
    if task == "compose":
        return ops.compose(primary, operand)  # type: ignore[arg-type]
    if task == "power":
        if isinstance(exponent, bool) or not isinstance(exponent, int):
            raise TypeError("exponent must be an integer")
        if not 0 <= exponent <= 100:
            raise ValueError("exponent must satisfy 0 <= exponent <= 100")
        return ops.power(primary, exponent)
    if task == "conjugate":
        return ops.conjugate(operand, primary)  # type: ignore[arg-type]
    if task == "commutator":
        return ops.commutator(primary, operand)  # type: ignore[arg-type]
    if task == "right_multiply_simple":
        return ops.right_multiply_simple(primary, simple_index)  # type: ignore[arg-type]
    if task == "bruhat_leq":
        return int(ops.bruhat_leq(primary, operand))  # type: ignore[arg-type]
    raise AssertionError(f"unhandled task {task!r}")


def _verify_record(
    record: object,
    *,
    expected_id: int,
    max_entries: int,
    shard_name: str,
    line_number: int,
    schema_version: str,
    task_names: Sequence[str],
) -> str:
    location = f"{shard_name}:{line_number}"
    if not isinstance(record, dict):
        _fail(f"{location}: record must be a JSON object")
    if record.get("schema_version") != schema_version:
        _fail(f"{location}: wrong schema_version")
    if record.get("id") != expected_id:
        _fail(f"{location}: expected id {expected_id}, got {record.get('id')!r}")
    expected_task = task_names[expected_id % len(task_names)]
    if record.get("task") != expected_task:
        _fail(f"{location}: expected task {expected_task!r}")

    size = _require_int(record.get("n"), name=f"{location}.n", minimum=2)
    if size > max_entries:
        _fail(f"{location}: n exceeds manifest max_entries")
    inputs = record.get("inputs")
    if not isinstance(inputs, dict):
        _fail(f"{location}: inputs must be an object")
    primary = inputs.get("primary")
    try:
        checked_primary = validate_permutation(primary)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise VerificationError(f"{location}: invalid primary permutation: {exc}") from exc
    if len(checked_primary) != size:
        _fail(f"{location}: n does not match primary length")

    spec = TASK_SPECS[expected_task]
    expected_input_keys = {"primary"}
    if spec.operand_kind is not None:
        expected_input_keys.add(spec.operand_kind)
    if set(inputs) != expected_input_keys:
        _fail(
            f"{location}: inputs must contain exactly "
            f"{', '.join(sorted(expected_input_keys))}"
        )
    if record.get("answer_kind") != spec.answer_kind:
        _fail(f"{location}: answer_kind does not match task")
    tokens = record.get("tokens")
    if not isinstance(tokens, list) or not all(isinstance(token, str) for token in tokens):
        _fail(f"{location}: tokens must be a list of strings")
    if not tokens or tokens[0] != "<BOS>" or tokens[-1] != "<EOS>":
        _fail(f"{location}: sequence boundaries are invalid")
    if any(token not in VOCABULARY for token in tokens):
        _fail(f"{location}: sequence contains a token outside the vocabulary")
    if tokens.count(spec.token) != 1 or tokens.count("=") != 1:
        _fail(f"{location}: sequence must contain one task token and one equals token")
    canonical_text = record.get("canonical_text")
    if canonical_text != " ".join(tokens):
        _fail(f"{location}: canonical_text does not match tokens")

    try:
        truth = _mathematical_answer(expected_task, checked_primary, inputs)
    except (TypeError, ValueError) as exc:
        raise VerificationError(
            f"{location}: cannot recompute mathematical answer: {exc}"
        ) from exc
    json_truth = _json_value(truth)
    if record.get("answer") != json_truth:
        _fail(f"{location}: stored answer disagrees with mathematical answer")

    try:
        expected_tokens = passage_tokens(
            expected_task,
            checked_primary,
            truth,
            **_record_render_kwargs(inputs),  # type: ignore[arg-type]
        )
    except (TypeError, ValueError) as exc:
        raise VerificationError(f"{location}: invalid typed record: {exc}") from exc
    if tuple(tokens) != expected_tokens:
        _fail(f"{location}: tokens are not the canonical rendering of metadata")
    return expected_task


def _verify_full_shard(
    path: Path,
    *,
    entry: Mapping[str, Any],
    max_entries: int,
    schema_version: str,
) -> Counter[str]:
    task_names = task_names_for_schema(schema_version)
    expected_id = int(entry["first_id"])
    final_id = expected_id + int(entry["record_count"])
    counts: Counter[str] = Counter()
    line_number = 0
    try:
        with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
            for line_number, line in enumerate(handle, start=1):
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise VerificationError(
                        f"{path.name}:{line_number}: invalid JSON: {exc}"
                    ) from exc
                task = _verify_record(
                    record,
                    expected_id=expected_id,
                    max_entries=max_entries,
                    shard_name=path.name,
                    line_number=line_number,
                    schema_version=schema_version,
                    task_names=task_names,
                )
                counts[task] += 1
                expected_id += 1
    except VerificationError:
        raise
    except (OSError, EOFError, UnicodeError) as exc:
        raise VerificationError(f"cannot parse gzip shard {path}: {exc}") from exc
    if expected_id != final_id:
        _fail(
            f"{path.name}: parsed {line_number} records, expected {entry['record_count']}"
        )
    return counts


def _verify_shard(job: _ShardVerificationJob) -> tuple[int, int, dict[str, int] | None]:
    """Verify one physical shard; suitable for a process-pool worker."""

    path = Path(job.path)
    entry = job.entry
    shard_index = int(entry["index"])
    if not path.is_file():
        _fail(f"missing shard {path.name}")
    try:
        actual_size = path.stat().st_size
    except OSError as exc:
        raise VerificationError(f"cannot stat shard {path}: {exc}") from exc
    expected_size = int(entry["byte_size"])
    if actual_size != expected_size:
        _fail(
            f"shard {path.name} byte size mismatch: {actual_size} != {expected_size}"
        )
    if _sha256_file(path) != entry["sha256"]:
        _fail(f"shard {path.name} SHA-256 mismatch")
    parsed = None
    if job.full:
        parsed = dict(
            _verify_full_shard(
                path,
                entry=entry,
                max_entries=job.max_entries,
                schema_version=job.schema_version,
            )
        )
    return shard_index, actual_size, parsed


def _load_parent_manifest(
    manifest_path: Path, manifest: Mapping[str, Any]
) -> dict[str, Any] | None:
    parent_name = manifest.get("parent_manifest")
    parent_hash = manifest.get("parent_manifest_sha256")
    if parent_name is None and parent_hash is None:
        return None
    if parent_name is None or parent_hash is None:
        _fail("split view must provide both parent manifest name and SHA-256")
    safe_name = _require_plain_filename(parent_name, index=-1)
    if (
        not isinstance(parent_hash, str)
        or len(parent_hash) != 64
        or any(character not in "0123456789abcdef" for character in parent_hash)
    ):
        _fail("parent_manifest_sha256 is invalid")
    parent_path = manifest_path.parent / safe_name
    if parent_path == manifest_path:
        _fail("split view cannot name itself as its parent manifest")
    try:
        raw = parent_path.read_bytes()
    except OSError as exc:
        raise VerificationError(f"cannot read parent manifest {parent_path}: {exc}") from exc
    if hashlib.sha256(raw).hexdigest() != parent_hash:
        _fail("parent manifest SHA-256 mismatch")
    try:
        parent = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise VerificationError(f"parent manifest is invalid JSON: {exc}") from exc
    if not isinstance(parent, dict):
        _fail("parent manifest must be a JSON object")
    for key in (
        "schema_version",
        "format",
        "gzip_mtime",
        "gzip_compresslevel",
        "max_entries",
        "base",
        "seed",
        "shard_size",
        "tasks",
    ):
        if manifest.get(key) != parent.get(key):
            _fail(f"split view disagrees with parent manifest field {key}")
    return parent


def verify_manifest(
    manifest_path: str | Path,
    *,
    full: bool = False,
    workers: int = DEFAULT_WORKERS,
) -> dict[str, Any]:
    """Verify every compressed hash and optionally every mathematical answer."""

    workers = _require_int(workers, name="workers", minimum=1)
    path = Path(manifest_path)
    manifest = _load_manifest(path)
    schema_version = manifest.get("schema_version")
    if not isinstance(schema_version, str) or schema_version not in TASKS_BY_SCHEMA:
        _fail("manifest schema_version is not supported")
    task_names = task_names_for_schema(schema_version)
    if manifest.get("format") != "jsonl.gz":
        _fail("manifest format must be jsonl.gz")
    if manifest.get("gzip_mtime") != 0:
        _fail("manifest must declare deterministic gzip mtime=0")
    if manifest.get("tasks") != list(task_names):
        _fail("manifest task registry or order is invalid")

    split_view = "split" in manifest or "parent_manifest" in manifest
    if split_view and (
        not isinstance(manifest.get("split"), str) or not manifest.get("split")
    ):
        _fail("split view must have a nonempty split name")
    parent = _load_parent_manifest(path, manifest)

    count = _require_int(
        manifest.get("count"),
        name="count",
        minimum=1 if split_view else len(task_names),
    )
    if not split_view and count % len(task_names):
        _fail(f"manifest count is not divisible by {len(task_names)}")
    max_entries = _require_int(
        manifest.get("max_entries"), name="max_entries", minimum=2
    )
    if manifest.get("base") != 100:
        _fail("manifest base must be 100")
    _require_int(manifest.get("seed"), name="seed", minimum=0)
    shard_size = _require_int(
        manifest.get("shard_size"), name="shard_size", minimum=1
    )
    shards = manifest.get("shards")
    if not isinstance(shards, list):
        _fail("manifest shards must be a list")
    shard_count = _require_int(
        manifest.get("shard_count"), name="shard_count", minimum=1
    )
    if shard_count != len(shards):
        _fail("manifest shard_count does not match its listed shards")
    if not split_view and shard_count != math.ceil(count / shard_size):
        _fail("manifest shard_count is invalid")

    declared_global = _require_task_counts(
        manifest.get("task_counts"),
        name="task_counts",
        expected_total=count,
        task_names=task_names,
    )
    if not split_view:
        expected_per_task = count // len(task_names)
        if any(declared_global[task] != expected_per_task for task in task_names):
            _fail(
                f"manifest is not exactly balanced across the {len(task_names)} tasks"
            )

    parent_by_index: dict[int, Mapping[str, Any]] = {}
    if parent is not None:
        parent_shards = parent.get("shards")
        if not isinstance(parent_shards, list):
            _fail("parent manifest shards must be a list")
        for parent_entry in parent_shards:
            if not isinstance(parent_entry, dict):
                _fail("parent manifest contains a malformed shard entry")
            parent_index = parent_entry.get("index")
            if isinstance(parent_index, bool) or not isinstance(parent_index, int):
                _fail("parent manifest contains an invalid shard index")
            if parent_index in parent_by_index:
                _fail("parent manifest contains duplicate shard indices")
            parent_by_index[parent_index] = parent_entry

    aggregate_declared: Counter[str] = Counter()
    declared_by_index: dict[int, dict[str, int]] = {}
    seen_filenames: set[str] = set()
    seen_indices: set[int] = set()
    jobs: list[_ShardVerificationJob] = []
    listed_records = 0
    previous_index: int | None = None
    for list_position, raw_entry in enumerate(shards):
        if not isinstance(raw_entry, dict):
            _fail(f"shard entry {list_position} must be an object")
        entry: dict[str, Any] = raw_entry
        shard_index = _require_int(
            entry.get("index"), name=f"shard[{list_position}].index", minimum=0
        )
        if shard_index in seen_indices:
            _fail(f"duplicate shard index {shard_index}")
        if previous_index is not None and shard_index <= previous_index:
            _fail("shard entries must be ordered by increasing source index")
        previous_index = shard_index
        seen_indices.add(shard_index)
        if not split_view and shard_index != list_position:
            _fail(f"shard entry {list_position} has the wrong source index")

        filename = _require_plain_filename(entry.get("filename"), index=shard_index)
        if filename in seen_filenames:
            _fail(f"duplicate shard filename {filename!r}")
        seen_filenames.add(filename)
        first_id = _require_int(
            entry.get("first_id"), name=f"shard[{shard_index}].first_id", minimum=0
        )
        if first_id != shard_index * shard_size:
            _fail(f"shard {shard_index} first_id disagrees with its source index")
        record_count = _require_int(
            entry.get("record_count"),
            name=f"shard[{shard_index}].record_count",
            minimum=1,
        )
        if record_count > shard_size:
            _fail(f"shard {shard_index} record_count exceeds shard_size")
        if entry.get("last_id") != first_id + record_count - 1:
            _fail(f"shard {shard_index} has the wrong last_id")
        if not split_view:
            expected_records = min(shard_size, count - first_id)
            if record_count != expected_records:
                _fail(f"shard {shard_index} has the wrong record_count")

        if parent is not None:
            parent_entry = parent_by_index.get(shard_index)
            if parent_entry is None:
                _fail(f"split shard {shard_index} is absent from parent manifest")
            if entry != parent_entry:
                _fail(f"split shard {shard_index} metadata differs from parent manifest")

        _require_int(
            entry.get("byte_size"),
            name=f"shard[{shard_index}].byte_size",
            minimum=1,
        )
        expected_hash = entry.get("sha256")
        if (
            not isinstance(expected_hash, str)
            or len(expected_hash) != 64
            or any(character not in "0123456789abcdef" for character in expected_hash)
        ):
            _fail(f"shard {shard_index} has an invalid SHA-256")
        declared_counts = _require_task_counts(
            entry.get("task_counts"),
            name=f"shard[{shard_index}].task_counts",
            expected_total=record_count,
            task_names=task_names,
        )
        aggregate_declared.update(declared_counts)
        declared_by_index[shard_index] = declared_counts
        listed_records += record_count
        jobs.append(
            _ShardVerificationJob(
                path=str(path.parent / filename),
                entry=entry,
                max_entries=max_entries,
                full=full,
                schema_version=schema_version,
            )
        )

    if listed_records != count:
        _fail("listed shard record counts do not reproduce manifest count")
    if any(aggregate_declared[task] != declared_global[task] for task in task_names):
        _fail("per-shard task counts do not reproduce global counts")

    if workers == 1 or len(jobs) == 1:
        results = [_verify_shard(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=min(workers, len(jobs))) as executor:
            # executor.map preserves manifest order and therefore deterministic
            # error propagation while still doing shard work concurrently.
            results = list(executor.map(_verify_shard, jobs))

    aggregate_parsed: Counter[str] = Counter()
    total_bytes = 0
    for shard_index, actual_size, parsed in results:
        total_bytes += actual_size
        if full:
            if parsed is None:  # pragma: no cover - worker contract guard
                _fail(f"shard {shard_index} did not return full-verification counts")
            declared = declared_by_index[shard_index]
            if any(parsed.get(task, 0) != declared[task] for task in task_names):
                _fail(
                    f"shard {shard_index} parsed task counts disagree with manifest"
                )
            aggregate_parsed.update(parsed)
    if full and any(
        aggregate_parsed[task] != declared_global[task] for task in task_names
    ):
        _fail("parsed records do not reproduce global task counts")
    if manifest.get("total_bytes") != total_bytes:
        _fail("manifest total_bytes is invalid")

    return {
        "ok": True,
        "full": full,
        "workers": workers,
        "split": manifest.get("split"),
        "record_count": count,
        "shard_count": len(shards),
        "total_bytes": total_bytes,
        "task_counts": declared_global,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify hashes and schema for permutation dataset shards."
    )
    parser.add_argument("manifest", type=Path)
    parser.add_argument(
        "--full",
        action="store_true",
        help="decompress and validate every JSON record after hashing all shards",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help="number of shards to hash/verify concurrently",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = verify_manifest(args.manifest, full=args.full, workers=args.workers)
    print(json.dumps(summary, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "DEFAULT_WORKERS",
    "VerificationError",
    "build_parser",
    "main",
    "verify_manifest",
]
