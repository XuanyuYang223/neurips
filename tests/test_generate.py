"""Production generator, resume, and verifier tests on tiny corpora."""

from __future__ import annotations

from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
from typing import Callable
from unittest.mock import patch

import pytest

from neurips_permutations import math_ops as ops
from neurips_permutations.generate import (
    DEFAULT_BASE,
    DEFAULT_COUNT,
    DEFAULT_MAX_ENTRIES,
    DEFAULT_SHARD_SIZE,
    SCHEMA_VERSION,
    V2_SCHEMA_VERSION,
    V3_SCHEMA_VERSION,
    build_record,
    build_parser,
    generate_dataset,
    main as generate_main,
)
from neurips_permutations.math_ops import V2_TASK_NAMES, V3_TASK_NAMES
from neurips_permutations.passage import passage_tokens
from neurips_permutations.splits import create_split_manifests
from neurips_permutations.verify import (
    VerificationError,
    main as verify_main,
    verify_manifest,
)


def _manifest(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _records(manifest_path: Path) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    manifest = _manifest(manifest_path)
    for shard in manifest["shards"]:  # type: ignore[index]
        shard_path = manifest_path.parent / shard["filename"]
        with gzip.open(shard_path, "rt", encoding="utf-8") as handle:
            result.extend(json.loads(line) for line in handle)
    return result


def _rewrite_one_shard(
    manifest_path: Path,
    record_id: int,
    mutate: Callable[[dict[str, object]], None],
) -> None:
    """Mutate a test record and consistently refresh its physical metadata."""

    manifest = _manifest(manifest_path)
    shard = next(
        entry
        for entry in manifest["shards"]
        if entry["first_id"] <= record_id <= entry["last_id"]
    )
    shard_path = manifest_path.parent / shard["filename"]
    with gzip.open(shard_path, "rt", encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle]
    target = next(record for record in records if record["id"] == record_id)
    mutate(target)
    partial = shard_path.with_suffix(shard_path.suffix + ".partial")
    with partial.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as stream:
            for record in records:
                payload = json.dumps(
                    record, ensure_ascii=True, sort_keys=True, separators=(",", ":")
                )
                stream.write(payload.encode("utf-8") + b"\n")
    partial.replace(shard_path)
    shard["byte_size"] = shard_path.stat().st_size
    shard["sha256"] = hashlib.sha256(shard_path.read_bytes()).hexdigest()
    manifest["total_bytes"] = sum(entry["byte_size"] for entry in manifest["shards"])
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def test_cli_defaults_freeze_the_production_request() -> None:
    args = build_parser().parse_args([])
    assert args.count == DEFAULT_COUNT == 10_000_000
    assert args.max_entries == DEFAULT_MAX_ENTRIES == 30
    assert args.base == DEFAULT_BASE == 100
    assert args.shard_size == DEFAULT_SHARD_SIZE == 100_000
    assert args.output_dir == Path("data/permutation-10m-v3")
    assert args.resume is True
    assert args.schema_version == SCHEMA_VERSION == V3_SCHEMA_VERSION
    assert V2_SCHEMA_VERSION == "permutation-20/v2"
    assert V3_SCHEMA_VERSION == "permutation-20/v3"
    assert DEFAULT_COUNT // len(V3_TASK_NAMES) == 500_000


def test_multiple_shards_have_exact_global_task_and_label_balance(
    tmp_path: Path,
) -> None:
    manifest_path = generate_dataset(
        count=80,
        max_entries=8,
        seed=1234,
        shard_size=13,
        output_dir=tmp_path / "dataset",
        workers=1,
    )
    summary = verify_manifest(manifest_path, full=True)
    manifest = _manifest(manifest_path)
    records = _records(manifest_path)

    assert summary["record_count"] == 80
    assert manifest["shard_count"] == 7
    assert manifest["schema_version"] == V3_SCHEMA_VERSION
    assert manifest["tasks"] == list(V3_TASK_NAMES)
    assert manifest["task_counts"] == {task: 4 for task in V3_TASK_NAMES}
    assert [record["id"] for record in records] == list(range(80))
    assert Counter(record["task"] for record in records) == Counter(
        {task: 4 for task in V3_TASK_NAMES}
    )

    patterns = [record for record in records if record["task"] == "pattern_avoidance"]
    assert [record["answer"] for record in patterns] == [1, 0, 1, 0]
    for record in patterns:
        inputs = record["inputs"]
        primary = tuple(inputs["primary"])
        pattern = tuple(inputs["pattern"])
        assert len(pattern) == len(primary) - 1
        assert int(ops.avoids_pattern(primary, pattern)) == record["answer"]

    bruhat = [record for record in records if record["task"] == "bruhat_leq"]
    assert [record["answer"] for record in bruhat] == [1, 0, 1, 0]
    gaps_by_label: dict[int, list[int]] = {0: [], 1: []}
    for record in bruhat:
        inputs = record["inputs"]
        primary = tuple(inputs["primary"])
        operand = tuple(inputs["operand"])
        actual = ops.bruhat_leq(primary, operand)
        assert int(actual) == record["answer"]
        gap = ops.inversion_count(operand) - ops.inversion_count(primary)
        assert gap > 0
        gaps_by_label[record["answer"]].append(gap)
    assert gaps_by_label[0] == gaps_by_label[1]


def test_v3_replaces_expensive_operations_with_three_scalar_statistics() -> None:
    assert V3_TASK_NAMES == (
        *V2_TASK_NAMES[:13],
        "peaks",
        "exceedances",
        "recoils",
        "inverse",
        "compose",
        "right_multiply_simple",
        "bruhat_leq",
    )
    assert {"power", "conjugate", "commutator"}.isdisjoint(V3_TASK_NAMES)
    assert {"peaks", "exceedances", "recoils"}.isdisjoint(V2_TASK_NAMES)
    assert set(V3_TASK_NAMES) - set(V2_TASK_NAMES) == {
        "peaks",
        "exceedances",
        "recoils",
    }
    assert set(V2_TASK_NAMES) - set(V3_TASK_NAMES) == {
        "power",
        "conjugate",
        "commutator",
    }

    implementations = {
        "peaks": ops.peak_count,
        "exceedances": ops.exceedance_count,
        "recoils": ops.recoil_count,
    }
    for task, implementation in implementations.items():
        task_index = V3_TASK_NAMES.index(task)
        record = build_record(
            task_index,
            max_entries=12,
            seed=20260830,
            schema_version=V3_SCHEMA_VERSION,
        )
        primary = tuple(record["inputs"]["primary"])
        assert record["task"] == task
        assert set(record["inputs"]) == {"primary"}
        assert record["answer_kind"] == "scalar"
        assert record["answer"] == implementation(primary)
        assert 0 <= record["answer"] < record["n"]


def test_v2_generation_and_full_verification_remain_supported(
    tmp_path: Path,
) -> None:
    manifest_path = generate_dataset(
        count=40,
        max_entries=7,
        seed=2718,
        shard_size=13,
        output_dir=tmp_path / "v2",
        workers=1,
        schema_version=V2_SCHEMA_VERSION,
    )
    manifest = _manifest(manifest_path)
    records = _records(manifest_path)

    assert manifest["schema_version"] == V2_SCHEMA_VERSION
    assert manifest["tasks"] == list(V2_TASK_NAMES)
    assert manifest["task_counts"] == {task: 2 for task in V2_TASK_NAMES}
    assert {record["schema_version"] for record in records} == {V2_SCHEMA_VERSION}
    assert {"power", "conjugate", "commutator"} <= {
        record["task"] for record in records
    }
    assert {"peaks", "exceedances", "recoils"}.isdisjoint(
        record["task"] for record in records
    )
    assert verify_manifest(manifest_path, full=True, workers=2)["ok"] is True
    views = create_split_manifests(
        manifest_path, train_shards=2, validation_shards=1, test_shards=1
    )
    assert _manifest(views["validation"])["schema_version"] == V2_SCHEMA_VERSION
    assert verify_manifest(views["validation"], full=True)["ok"] is True


def test_resume_rejects_cross_protocol_output_directory(tmp_path: Path) -> None:
    output_dir = tmp_path / "mixed-protocol"
    generate_dataset(
        count=40,
        max_entries=6,
        output_dir=output_dir,
        workers=1,
        schema_version=V2_SCHEMA_VERSION,
    )
    with pytest.raises(ValueError, match="schema_version|tasks"):
        generate_dataset(
            count=40,
            max_entries=6,
            output_dir=output_dir,
            workers=1,
            schema_version=V3_SCHEMA_VERSION,
        )


def test_bruhat_labels_and_positive_gap_histograms_are_exactly_balanced() -> None:
    task_index = V3_TASK_NAMES.index("bruhat_leq")
    gap_histograms: dict[int, Counter[int]] = {0: Counter(), 1: Counter()}
    label_counts: Counter[int] = Counter()

    for occurrence in range(400):
        record = build_record(
            task_index + len(V3_TASK_NAMES) * occurrence,
            max_entries=30,
            seed=20260830,
        )
        inputs = record["inputs"]
        primary = tuple(inputs["primary"])
        operand = tuple(inputs["operand"])
        label = record["answer"]
        gap = ops.inversion_count(operand) - ops.inversion_count(primary)

        assert record["n"] >= 4
        assert primary != operand
        assert 1 <= gap <= min(4, record["n"] - 1)
        assert ops.bruhat_leq(primary, operand) is bool(label)
        label_counts[label] += 1
        gap_histograms[label][gap] += 1

    assert label_counts == Counter({0: 200, 1: 200})
    assert gap_histograms[0] == gap_histograms[1]
    assert any(gap > 1 for gap in gap_histograms[0])
    assert set(gap_histograms[0]) == {1, 2, 3, 4}


def test_pattern_balance_remains_nontrivial_at_minimum_supported_size() -> None:
    positive = build_record(12, max_entries=3, seed=44)
    negative = build_record(32, max_entries=3, seed=44)
    assert [positive["answer"], negative["answer"]] == [1, 0]
    for record in (positive, negative):
        inputs = record["inputs"]
        assert len(inputs["pattern"]) == record["n"] - 1


@pytest.mark.parametrize("record_id", [4, 16])
def test_full_verifier_rejects_consistently_rendered_wrong_answers(
    tmp_path: Path, record_id: int
) -> None:
    """Hash/schema consistency must not let a mathematically false label pass."""

    manifest_path = generate_dataset(
        count=40,
        max_entries=7,
        seed=815,
        shard_size=40,
        output_dir=tmp_path / f"wrong-{record_id}",
        workers=1,
    )

    def mutate(record: dict[str, object]) -> None:
        if record_id == 4:  # scalar length
            wrong = int(record["answer"]) + 1
        else:  # permutation-valued inverse
            wrong = list(ops.right_multiply_simple(tuple(record["answer"]), 1))
        inputs = record["inputs"]
        kwargs = {
            key: inputs[key]
            for key in ("operand", "pattern", "exponent", "simple_index")
            if key in inputs
        }
        tokens = passage_tokens(
            str(record["task"]), inputs["primary"], wrong, **kwargs
        )
        record["answer"] = wrong
        record["tokens"] = list(tokens)
        record["canonical_text"] = " ".join(tokens)

    _rewrite_one_shard(manifest_path, record_id, mutate)
    # The refreshed hash and typed Passage rendering are internally consistent.
    assert verify_manifest(manifest_path, full=False)["ok"] is True
    with pytest.raises(VerificationError, match="mathematical answer"):
        verify_manifest(manifest_path, full=True)


def test_parallel_full_verification_matches_serial(tmp_path: Path) -> None:
    manifest_path = generate_dataset(
        count=80,
        max_entries=8,
        seed=601,
        shard_size=9,
        output_dir=tmp_path / "parallel-verify",
        workers=2,
    )
    serial = verify_manifest(manifest_path, full=True, workers=1)
    parallel = verify_manifest(manifest_path, full=True, workers=3)
    assert serial["task_counts"] == parallel["task_counts"]
    assert serial["record_count"] == parallel["record_count"] == 80
    assert serial["total_bytes"] == parallel["total_bytes"]
    assert parallel["workers"] == 3


def test_split_view_with_nonzero_source_indices_and_parent_is_verified(
    tmp_path: Path,
) -> None:
    source = generate_dataset(
        count=80,
        max_entries=7,
        seed=902,
        shard_size=13,
        output_dir=tmp_path / "split-view",
        workers=1,
    )
    views = create_split_manifests(
        source, train_shards=3, validation_shards=2, test_shards=2
    )
    validation_manifest = _manifest(views["validation"])
    assert validation_manifest["shards"][0]["index"] == 3
    assert validation_manifest["shards"][0]["first_id"] == 39

    summary = verify_manifest(views["validation"], full=True, workers=2)
    assert summary["ok"] is True
    assert summary["split"] == "validation"
    assert summary["record_count"] == validation_manifest["count"] == 26

    # A split view without optional parent provenance is still self-verifiable.
    standalone = dict(validation_manifest)
    standalone.pop("parent_manifest")
    standalone.pop("parent_manifest_sha256")
    standalone_path = source.parent / "standalone_split_manifest.json"
    standalone_path.write_text(json.dumps(standalone), encoding="utf-8")
    assert verify_manifest(standalone_path, full=True, workers=2)["ok"] is True


def test_split_view_parent_hash_and_metadata_are_enforced(tmp_path: Path) -> None:
    source = generate_dataset(
        count=80,
        max_entries=6,
        seed=903,
        shard_size=11,
        output_dir=tmp_path / "parent-check",
        workers=1,
    )
    views = create_split_manifests(
        source, train_shards=2, validation_shards=3, test_shards=3
    )
    view = _manifest(views["validation"])
    view["parent_manifest_sha256"] = "0" * 64
    views["validation"].write_text(json.dumps(view), encoding="utf-8")
    with pytest.raises(VerificationError, match="parent manifest SHA-256"):
        verify_manifest(views["validation"])

    view["parent_manifest_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    view["shards"][0]["byte_size"] += 1
    view["total_bytes"] += 1
    views["validation"].write_text(json.dumps(view), encoding="utf-8")
    with pytest.raises(VerificationError, match="metadata differs from parent"):
        verify_manifest(views["validation"])


def test_output_is_byte_deterministic_across_worker_counts(tmp_path: Path) -> None:
    first = generate_dataset(
        count=80,
        max_entries=7,
        seed=77,
        shard_size=11,
        output_dir=tmp_path / "one",
        workers=1,
    )
    second = generate_dataset(
        count=80,
        max_entries=7,
        seed=77,
        shard_size=11,
        output_dir=tmp_path / "two",
        workers=2,
    )
    first_manifest = _manifest(first)
    second_manifest = _manifest(second)
    assert first.read_bytes() == second.read_bytes()
    for left, right in zip(
        first_manifest["shards"], second_manifest["shards"], strict=True
    ):
        assert (first.parent / left["filename"]).read_bytes() == (
            second.parent / right["filename"]
        ).read_bytes()


def test_resume_reuses_valid_shards_without_rewriting(tmp_path: Path) -> None:
    arguments = dict(
        count=40,
        max_entries=6,
        seed=9,
        shard_size=9,
        output_dir=tmp_path / "resume",
        workers=1,
    )
    first = generate_dataset(**arguments)
    before = {
        path.name: path.stat().st_mtime_ns
        for path in first.parent.glob("part-*.jsonl.gz")
    }
    with patch(
        "neurips_permutations.generate._write_shard",
        side_effect=AssertionError("resume attempted to rewrite a valid shard"),
    ):
        second = generate_dataset(**arguments)
    after = {
        path.name: path.stat().st_mtime_ns
        for path in second.parent.glob("part-*.jsonl.gz")
    }
    assert before == after
    verify_manifest(second, full=True)


def test_corruption_is_detected_and_safe_resume_repairs_it(tmp_path: Path) -> None:
    arguments = dict(
        count=40,
        max_entries=6,
        seed=31,
        shard_size=17,
        output_dir=tmp_path / "corrupt",
        workers=1,
    )
    manifest_path = generate_dataset(**arguments)
    manifest = _manifest(manifest_path)
    damaged = manifest_path.parent / manifest["shards"][1]["filename"]
    original = damaged.read_bytes()
    damaged.write_bytes(original + b"damage")

    with pytest.raises(VerificationError, match="byte size mismatch|SHA-256"):
        verify_manifest(manifest_path, workers=2)

    repaired_manifest = generate_dataset(**arguments)
    assert damaged.read_bytes() == original
    assert verify_manifest(repaired_manifest, full=True)["ok"] is True


@pytest.mark.parametrize("count", [19, 21, 39, 41])
def test_count_must_be_divisible_by_twenty(tmp_path: Path, count: int) -> None:
    with pytest.raises(ValueError, match="divisible by 20"):
        generate_dataset(
            count=count,
            max_entries=6,
            shard_size=10,
            output_dir=tmp_path / str(count),
            workers=1,
        )


@pytest.mark.parametrize("count", [20, 60, 100])
def test_count_requires_even_examples_per_task(tmp_path: Path, count: int) -> None:
    with pytest.raises(ValueError, match="divisible by 40"):
        generate_dataset(
            count=count,
            max_entries=6,
            shard_size=10,
            output_dir=tmp_path / str(count),
            workers=1,
        )


def test_invalid_base_and_too_small_maximum_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="base=100"):
        generate_dataset(
            count=40,
            max_entries=6,
            base=10,
            output_dir=tmp_path / "base",
            workers=1,
        )
    with pytest.raises(ValueError, match="max_entries"):
        generate_dataset(
            count=40,
            max_entries=2,
            output_dir=tmp_path / "small",
            workers=1,
        )


def test_verifier_rejects_invalid_worker_count(tmp_path: Path) -> None:
    manifest = generate_dataset(
        count=40,
        max_entries=5,
        output_dir=tmp_path / "verify-workers",
        workers=1,
    )
    with pytest.raises(VerificationError, match="workers"):
        verify_manifest(manifest, workers=0)


def test_generate_and_verify_cli_smoke(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    output = tmp_path / "cli"
    assert (
        generate_main(
            [
                "--count",
                "40",
                "--max-entries",
                "6",
                "--seed",
                "5",
                "--shard-size",
                "7",
                "--workers",
                "1",
                "--output-dir",
                str(output),
            ]
        )
        == 0
    )
    capsys.readouterr()
    assert (
        verify_main(
            [str(output / "manifest.json"), "--full", "--workers", "2"]
        )
        == 0
    )
    printed = json.loads(capsys.readouterr().out)
    assert printed["ok"] is True
    assert printed["record_count"] == 40
    assert printed["workers"] == 2
