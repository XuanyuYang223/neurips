"""Streaming, resumable training utilities for Passage Math permutation data.

The data path is intentionally streaming: gzip shards are opened one at a
time, records are shuffled through a bounded buffer, and worker/rank
partitioning is performed before records enter that buffer.  A ten-million
example corpus therefore never needs to be materialized in memory.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import random
import tempfile
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

import torch
from torch import Tensor, nn
import torch.distributed as dist
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from .generate import V2_SCHEMA_VERSION, task_names_for_schema
from .passage import PERMUTATION20_VOCABULARY, TOKEN_TO_ID, tokenize


Record = dict[str, Any]
ModelFactory = Callable[["TrainConfig"], nn.Module]


def _task_tuple(tasks: str | Iterable[str] | None) -> tuple[str, ...] | None:
    if tasks is None:
        return None
    parts = tasks.split(",") if isinstance(tasks, str) else tasks
    normalized: list[str] = []
    for part in parts:
        for value in str(part).split(","):
            value = value.strip()
            if value and value not in normalized:
                normalized.append(value)
    return tuple(normalized) or None


def parse_shard_indices(selection: str | Iterable[int] | None) -> tuple[int, ...] | None:
    """Parse positional shard selections such as ``000-097,099``."""

    if selection is None:
        return None
    if not isinstance(selection, str):
        return tuple(int(index) for index in selection)
    indices: list[int] = []
    for part in selection.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            left_text, right_text = part.split("-", 1)
            left, right = int(left_text), int(right_text)
            if left > right:
                raise ValueError(f"descending shard range {part!r} is invalid")
            candidates: Iterable[int] = range(left, right + 1)
        else:
            candidates = (int(part),)
        for index in candidates:
            if index < 0:
                raise ValueError("shard indices cannot be negative")
            if index not in indices:
                indices.append(index)
    return tuple(indices) or None


def resolve_shards(
    source: str | os.PathLike[str] | Sequence[str | os.PathLike[str]],
    shard_indices: Sequence[int] | None = None,
) -> tuple[Path, ...]:
    """Resolve a manifest, directory, shard, or explicit shard list.

    Manifest shard entries may be strings or dictionaries containing one of
    ``path``, ``filename``, ``file``, or ``name``.  Relative names are resolved
    against the manifest directory.
    """

    if isinstance(source, (str, os.PathLike)):
        path = Path(source)
        if path.is_dir():
            shards = sorted(path.glob("*.jsonl.gz"))
        elif path.suffix == ".json" and not path.name.endswith(".jsonl.gz"):
            with path.open("r", encoding="utf-8") as handle:
                manifest = json.load(handle)
            entries = manifest.get("shards") or manifest.get("files")
            if not isinstance(entries, list):
                raise ValueError(f"manifest {path} does not contain a shard list")
            shards = []
            for index, entry in enumerate(entries):
                if isinstance(entry, str):
                    name = entry
                elif isinstance(entry, Mapping):
                    name = next(
                        (
                            entry[key]
                            for key in ("path", "filename", "file", "name")
                            if key in entry
                        ),
                        None,
                    )
                else:
                    name = None
                if not isinstance(name, str):
                    raise ValueError(f"manifest shard {index} has no usable path")
                shard = Path(name)
                shards.append(shard if shard.is_absolute() else path.parent / shard)
        else:
            shards = [path]
    else:
        shards = [Path(path) for path in source]

    if not shards:
        raise ValueError("no JSONL shards were selected")
    if shard_indices is not None:
        selected: list[Path] = []
        for raw_index in shard_indices:
            if isinstance(raw_index, bool) or not isinstance(raw_index, int):
                raise TypeError("shard indices must be integers")
            if raw_index < 0 or raw_index >= len(shards):
                raise IndexError(f"shard index {raw_index} is out of range")
            selected.append(shards[raw_index])
        shards = selected
    for shard in shards:
        if not shard.is_file():
            raise FileNotFoundError(shard)
        if not (shard.name.endswith(".jsonl.gz") or shard.name.endswith(".jsonl")):
            raise ValueError(f"unsupported shard extension: {shard}")
    return tuple(shards)


def task_names_for_data_source(
    source: str | os.PathLike[str] | None,
) -> tuple[str, ...]:
    """Infer the protocol task grid from a manifest-backed data source.

    Explicit shard lists and legacy ad-hoc manifests retain the historical v2
    default.  Formal v2/v3 manifests must declare the matching frozen task
    order, preventing a revised run from validating against the wrong grid.
    """

    if source is None:
        return task_names_for_schema(V2_SCHEMA_VERSION)
    path = Path(source)
    if not path.is_file() or path.suffix != ".json":
        return task_names_for_schema(V2_SCHEMA_VERSION)
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read data manifest {path}: {exc}") from exc
    if not isinstance(manifest, Mapping):
        raise ValueError(f"data manifest {path} must contain a JSON object")
    schema_version = manifest.get("schema_version")
    if schema_version is None:
        # Preserve generic manifest support used by callers that only provide a
        # shard list and predate the frozen permutation protocols.
        return task_names_for_schema(V2_SCHEMA_VERSION)
    if not isinstance(schema_version, str):
        raise ValueError(f"data manifest {path} has an invalid schema_version")
    task_names = task_names_for_schema(schema_version)
    declared = manifest.get("tasks")
    if declared != list(task_names):
        raise ValueError(f"data manifest {path} task registry does not match its schema")
    return task_names


def _bounded_shuffle(
    records: Iterable[Record], *, buffer_size: int, rng: random.Random
) -> Iterator[Record]:
    if buffer_size <= 1:
        yield from records
        return
    buffer: list[Record] = []
    for record in records:
        if len(buffer) < buffer_size:
            buffer.append(record)
            continue
        index = rng.randrange(len(buffer))
        yield buffer[index]
        buffer[index] = record
    rng.shuffle(buffer)
    yield from buffer


class StreamingPermutationDataset(IterableDataset[Record]):
    """Stream generator JSONL shards with filtering and deterministic shuffle."""

    def __init__(
        self,
        shards: str | os.PathLike[str] | Sequence[str | os.PathLike[str]],
        *,
        tasks: str | Iterable[str] | None = None,
        shard_indices: Sequence[int] | None = None,
        shuffle_buffer_size: int = 1,
        seed: int = 0,
        epoch: int = 0,
        rank: int | None = None,
        world_size: int | None = None,
    ) -> None:
        super().__init__()
        self.shards = resolve_shards(shards, shard_indices)
        self.tasks = frozenset(_task_tuple(tasks) or ())
        self.shuffle_buffer_size = int(shuffle_buffer_size)
        self.seed = int(seed)
        self.epoch = int(epoch)
        if self.shuffle_buffer_size < 1:
            raise ValueError("shuffle_buffer_size must be at least one")
        if (rank is None) != (world_size is None):
            raise ValueError("rank and world_size must be supplied together")
        if rank is not None and (world_size is None or not 0 <= rank < world_size):
            raise ValueError("rank must satisfy 0 <= rank < world_size")
        self.rank = rank
        self.world_size = world_size

    def set_epoch(self, epoch: int) -> None:
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise ValueError("epoch must be a non-negative integer")
        self.epoch = epoch

    def _rank_info(self) -> tuple[int, int]:
        if self.rank is not None:
            return self.rank, int(self.world_size)
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank(), dist.get_world_size()
        return 0, 1

    def _records(self, consumer_id: int, consumers: int) -> Iterator[Record]:
        eligible_index = 0
        for shard in self.shards:
            opener = gzip.open if shard.name.endswith(".gz") else open
            with opener(shard, "rt", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError as error:
                        raise ValueError(
                            f"invalid JSON in {shard}:{line_number}: {error.msg}"
                        ) from error
                    if not isinstance(record, dict):
                        raise ValueError(f"record in {shard}:{line_number} is not an object")
                    task = record.get("task")
                    if not isinstance(task, str):
                        raise ValueError(f"record in {shard}:{line_number} has no task")
                    if self.tasks and task not in self.tasks:
                        continue
                    if "tokens" not in record and isinstance(record.get("canonical_text"), str):
                        record["tokens"] = list(tokenize(record["canonical_text"]))
                    tokens = record.get("tokens")
                    if not isinstance(tokens, (list, tuple)) or not all(
                        isinstance(token, str) for token in tokens
                    ):
                        raise ValueError(f"record in {shard}:{line_number} has invalid tokens")
                    if eligible_index % consumers == consumer_id:
                        yield record
                    eligible_index += 1

    def __iter__(self) -> Iterator[Record]:
        rank, world_size = self._rank_info()
        worker = get_worker_info()
        worker_id = worker.id if worker is not None else 0
        workers = worker.num_workers if worker is not None else 1
        consumer_id = rank * workers + worker_id
        consumers = world_size * workers
        # Avoid Python's randomized hash so this seed is stable across processes.
        stream_seed = self.seed + 1_000_003 * self.epoch + 97_003 * consumer_id
        rng = random.Random(stream_seed)
        yield from _bounded_shuffle(
            self._records(consumer_id, consumers),
            buffer_size=self.shuffle_buffer_size,
            rng=rng,
        )


# Backward-friendly short alias for experiment code.
PermutationIterableDataset = StreamingPermutationDataset


class TokenBudgetBatcher(IterableDataset[list[Record]]):
    """Group a stream by both example count and padded-token budget.

    The budget is ``batch_size * longest_sequence``.  A 938-token reduced-word
    example therefore automatically receives a much smaller microbatch than a
    short scalar example, while no sequence is truncated.
    """

    def __init__(
        self,
        dataset: IterableDataset[Record],
        *,
        max_examples: int,
        max_padded_tokens: int | None,
    ) -> None:
        super().__init__()
        if max_examples < 1:
            raise ValueError("max_examples must be at least one")
        if max_padded_tokens is not None and max_padded_tokens < 1:
            raise ValueError("max_padded_tokens must be at least one")
        self.dataset = dataset
        self.max_examples = max_examples
        self.max_padded_tokens = max_padded_tokens

    def __iter__(self) -> Iterator[list[Record]]:
        batch: list[Record] = []
        longest = 0
        for record in self.dataset:
            length = len(record["tokens"])
            if self.max_padded_tokens is not None and length > self.max_padded_tokens:
                raise ValueError(
                    f"sequence of length {length} exceeds max_padded_tokens="
                    f"{self.max_padded_tokens}"
                )
            proposed_longest = max(longest, length)
            exceeds_examples = len(batch) >= self.max_examples
            exceeds_tokens = (
                self.max_padded_tokens is not None
                and proposed_longest * (len(batch) + 1) > self.max_padded_tokens
            )
            if batch and (exceeds_examples or exceeds_tokens):
                yield batch
                batch = []
                longest = 0
            batch.append(record)
            longest = max(longest, length)
        if batch:
            yield batch


class AnswerOnlyCollator:
    """Pad token IDs and supervise only answer tokens plus ``<EOS>``."""

    def __init__(self, *, max_seq_len: int | None = None) -> None:
        self.pad_id = TOKEN_TO_ID["<PAD>"]
        self.max_seq_len = max_seq_len
        if max_seq_len is not None and max_seq_len < 2:
            raise ValueError("max_seq_len must be at least two")

    def __call__(self, records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        if not records:
            raise ValueError("cannot collate an empty batch")
        encoded: list[list[int]] = []
        label_rows: list[list[int]] = []
        tasks: list[str] = []
        ids: list[str] = []
        for record_index, record in enumerate(records):
            raw_tokens = record.get("tokens")
            if not isinstance(raw_tokens, (list, tuple)) or not all(
                isinstance(token, str) for token in raw_tokens
            ):
                raise ValueError(f"record {record_index} has invalid tokens")
            tokens = list(raw_tokens)
            if self.max_seq_len is not None and len(tokens) > self.max_seq_len:
                raise ValueError(
                    f"record {record_index} length {len(tokens)} exceeds max_seq_len "
                    f"{self.max_seq_len}; answer-bearing sequences are never truncated"
                )
            if tokens.count("=") != 1:
                raise ValueError(f"record {record_index} must contain exactly one '=' token")
            if not tokens or tokens[-1] != "<EOS>":
                raise ValueError(f"record {record_index} must end in <EOS>")
            equals = tokens.index("=")
            if equals + 1 >= len(tokens) - 1:
                raise ValueError(f"record {record_index} has no answer after '='")
            try:
                token_row = [TOKEN_TO_ID[token] for token in tokens]
            except KeyError as error:
                raise ValueError(f"record {record_index} has unknown token {error.args[0]!r}") from None
            labels = [-100] * len(tokens)
            labels[equals + 1 :] = token_row[equals + 1 :]
            encoded.append(token_row)
            label_rows.append(labels)
            tasks.append(str(record.get("task", "")))
            ids.append(str(record.get("id", record_index)))

        width = max(len(row) for row in encoded)
        input_ids = torch.full((len(encoded), width), self.pad_id, dtype=torch.long)
        labels = torch.full((len(encoded), width), -100, dtype=torch.long)
        attention_mask = torch.zeros((len(encoded), width), dtype=torch.bool)
        for row_index, (token_row, label_row) in enumerate(zip(encoded, label_rows)):
            length = len(token_row)
            input_ids[row_index, :length] = torch.tensor(token_row, dtype=torch.long)
            labels[row_index, :length] = torch.tensor(label_row, dtype=torch.long)
            attention_mask[row_index, :length] = True
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "tasks": tasks,
            "ids": ids,
        }


@dataclass
class TrainConfig:
    """Serializable configuration for one resumable training run."""

    output_dir: str
    manifest: str | None = None
    train_shards: tuple[str, ...] = ()
    validation_manifest: str | None = None
    validation_shards: tuple[str, ...] = ()
    architecture: str = "transformer"
    d_model: int = 256
    num_layers: int | None = None
    num_heads: int = 8
    dropout: float = 0.0
    mlp_ratio: float = 4.0
    tie_embeddings: bool = True
    model_config: dict[str, Any] = field(default_factory=dict)
    tasks: tuple[str, ...] | None = None
    validation_tasks: tuple[str, ...] | None = None
    seed: int = 0
    max_steps: int = 1_000
    batch_size: int = 32
    validation_batch_size: int = 32
    gradient_accumulation_steps: int = 1
    max_seq_len: int = 1_024
    max_tokens_per_batch: int | None = 4_096
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 100
    min_lr_ratio: float = 0.1
    max_grad_norm: float = 1.0
    shuffle_buffer_size: int = 10_000
    num_workers: int = 0
    checkpoint_every: int = 1_000
    validate_every: int = 1_000
    validation_batches_per_task: int = 1
    device: str | None = None
    amp: bool = True
    bf16: bool = True
    resume: str | None = None
    shard_indices: tuple[int, ...] | None = None
    validation_shard_indices: tuple[int, ...] | None = None
    experiment_config: str | None = None
    experiment_config_sha256: str | None = None

    @classmethod
    def from_value(cls, value: "TrainConfig | Mapping[str, Any]") -> "TrainConfig":
        if isinstance(value, cls):
            return value
        data = dict(value)
        aliases = {
            "grad_accum": "gradient_accumulation_steps",
            "resume_from": "resume",
            "max_sequence_length": "max_seq_len",
        }
        for old, new in aliases.items():
            if old in data and new not in data:
                data[new] = data.pop(old)
        for key in ("train_shards", "validation_shards"):
            if key in data and data[key] is not None:
                data[key] = tuple(str(path) for path in data[key])
        for key in ("tasks", "validation_tasks"):
            if key in data:
                data[key] = _task_tuple(data[key])
        for key in ("shard_indices", "validation_shard_indices"):
            if key in data:
                data[key] = parse_shard_indices(data[key])
        return cls(**data)

    def validate(self) -> None:
        if not self.manifest and not self.train_shards:
            raise ValueError("manifest or train_shards is required")
        for name in (
            "max_steps",
            "batch_size",
            "validation_batch_size",
            "gradient_accumulation_steps",
            "max_seq_len",
            "shuffle_buffer_size",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be at least one")
        if self.num_workers < 0:
            raise ValueError("num_workers cannot be negative")
        if self.max_tokens_per_batch is not None:
            if self.max_tokens_per_batch < self.max_seq_len:
                raise ValueError(
                    "max_tokens_per_batch must be at least max_seq_len so a "
                    "maximum-length example fits"
                )
        if self.d_model < 1 or self.num_heads < 1:
            raise ValueError("d_model and num_heads must be positive")
        if self.num_layers is not None and self.num_layers < 1:
            raise ValueError("num_layers must be positive")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("invalid optimizer hyperparameters")
        if self.max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive")
        if self.warmup_steps < 0 or not 0 <= self.min_lr_ratio <= 1:
            raise ValueError("invalid scheduler hyperparameters")


@dataclass
class TrainState:
    epoch: int = 0
    batches_in_epoch: int = 0
    global_step: int = 0
    task_examples: dict[str, int] = field(default_factory=dict)
    task_supervised_tokens: dict[str, int] = field(default_factory=dict)
    task_loss_sum: dict[str, float] = field(default_factory=dict)


def _training_shards(config: TrainConfig) -> tuple[Path, ...]:
    source: str | Sequence[str]
    source = config.manifest if config.manifest else config.train_shards
    return resolve_shards(source, config.shard_indices)


def _validation_shards(config: TrainConfig, training: tuple[Path, ...]) -> tuple[Path, ...]:
    if config.validation_manifest:
        return resolve_shards(
            config.validation_manifest, config.validation_shard_indices
        )
    if config.validation_shards:
        return resolve_shards(
            config.validation_shards, config.validation_shard_indices
        )
    if config.validation_shard_indices is not None and config.manifest:
        return resolve_shards(config.manifest, config.validation_shard_indices)
    return training


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _manifest_schema_version(path: str | None) -> str | None:
    if not path:
        return None
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    value = payload.get("schema_version") if isinstance(payload, Mapping) else None
    return value if isinstance(value, str) else None


def _model_vocab_size(config: TrainConfig) -> int:
    """Recover the vocabulary frozen by the dataset protocol.

    Property32 tokens were appended after the original 163-token v2/v3
    experiments.  Reading the manifest keeps old checkpoints and new scaling
    replications architecture-identical even when both protocols share one
    installed tokenizer module.
    """

    explicit = config.model_config.get("vocab_size")
    if explicit is not None:
        return int(explicit)
    schema = _manifest_schema_version(config.manifest)
    if schema in {"permutation-20/v2", "permutation-20/v3"}:
        return len(PERMUTATION20_VOCABULARY)
    return len(TOKEN_TO_ID)


def _default_model_factory(config: TrainConfig) -> nn.Module:
    from .models import build_model

    kwargs = dict(config.model_config)
    kwargs.setdefault("model_type", config.architecture)
    kwargs.setdefault("vocab_size", _model_vocab_size(config))
    kwargs.setdefault("max_seq_len", config.max_seq_len)
    kwargs.setdefault("d_model", config.d_model)
    kwargs.setdefault(
        "layers",
        config.num_layers
        if config.num_layers is not None
        else (4 if config.architecture == "transformer" else 1),
    )
    kwargs.setdefault("dropout", config.dropout)
    kwargs.setdefault("mlp_ratio", config.mlp_ratio)
    kwargs.setdefault("tie_embeddings", config.tie_embeddings)
    if config.architecture == "transformer":
        kwargs.setdefault("n_heads", config.num_heads)
    return build_model(**kwargs)


def _model_logits(model: nn.Module, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
    output = model(input_ids=input_ids, attention_mask=attention_mask)
    if isinstance(output, Tensor):
        logits = output
    elif isinstance(output, Mapping):
        logits = output.get("logits")
    elif hasattr(output, "logits"):
        logits = output.logits
    elif isinstance(output, (tuple, list)) and output:
        logits = output[0]
    else:
        logits = None
    if not isinstance(logits, Tensor) or logits.ndim != 3:
        raise TypeError("model must return logits shaped [batch, sequence, vocabulary]")
    return logits


def _causal_loss(logits: Tensor, labels: Tensor, *, reduction: str = "mean") -> Tensor:
    token_losses = F.cross_entropy(
        logits[:, :-1, :].contiguous().view(-1, logits.shape[-1]),
        labels[:, 1:].contiguous().view(-1),
        ignore_index=-100,
        reduction="none",
    ).view(labels.shape[0], -1)
    supervised = labels[:, 1:].ne(-100)
    if reduction == "sum":
        return (token_losses * supervised).sum()
    if reduction == "none":
        counts = supervised.sum(dim=1).clamp_min(1)
        return (token_losses * supervised).sum(dim=1) / counts
    if reduction != "mean":
        raise ValueError(f"unsupported reduction {reduction!r}")

    # Average within each example first, then across examples.  Otherwise a
    # 400-token reduced word can outweigh a two-token scalar target by two
    # orders of magnitude even though corpus records are task-balanced.
    counts = supervised.sum(dim=1)
    valid = counts.gt(0)
    if not bool(valid.any()):
        return token_losses.sum() * 0.0
    per_example = (token_losses * supervised).sum(dim=1) / counts.clamp_min(1)
    return per_example[valid].mean()


def _config_identity(config: TrainConfig | Mapping[str, Any]) -> dict[str, Any]:
    value = asdict(config) if isinstance(config, TrainConfig) else dict(config)
    # The resume locator changes between invocations but not the experiment.
    value.pop("resume", None)
    return value


def _config_sha256(config: TrainConfig | Mapping[str, Any]) -> str:
    payload = json.dumps(
        _config_identity(config), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _data_fingerprints(config: TrainConfig) -> dict[str, str | None]:
    """Fingerprint manifest contents so same-path data changes break resume."""

    return {
        "training_manifest_sha256": (
            _sha256_file(Path(config.manifest)) if config.manifest else None
        ),
        "validation_manifest_sha256": (
            _sha256_file(Path(config.validation_manifest))
            if config.validation_manifest
            else None
        ),
    }


def _autocast_context(device: torch.device, enabled: bool, dtype: torch.dtype):
    if device.type != "cuda" or not enabled:
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=dtype)


def _capture_rng() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"].cpu())
    if "cuda" in state and torch.cuda.is_available():
        # A CUDA checkpoint is loaded with ``map_location=device`` above, so
        # its saved generator states are CUDA ByteTensors at this point.
        # ``set_rng_state_all`` deliberately accepts CPU ByteTensors only.
        # Normalising also keeps checkpoints portable if a state was decoded
        # as a plain byte sequence by another serializer.
        cuda_states = [
            value.detach().to(device="cpu", dtype=torch.uint8)
            if isinstance(value, Tensor)
            else torch.as_tensor(value, dtype=torch.uint8, device="cpu")
            for value in state["cuda"]
        ]
        torch.cuda.set_rng_state_all(cuda_states)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_torch_save(payload: Mapping[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            torch.save(dict(payload), handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _atomic_json(payload: Mapping[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(dict(payload), handle, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _scheduler(optimizer: AdamW, config: TrainConfig) -> LambdaLR:
    def multiplier(step: int) -> float:
        if config.warmup_steps and step < config.warmup_steps:
            return max(1e-12, (step + 1) / config.warmup_steps)
        span = max(1, config.max_steps - config.warmup_steps)
        progress = min(1.0, max(0.0, (step - config.warmup_steps) / span))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return config.min_lr_ratio + (1.0 - config.min_lr_ratio) * cosine

    return LambdaLR(optimizer, multiplier)


def _checkpoint_payload(
    *,
    model: nn.Module,
    optimizer: AdamW,
    scheduler: LambdaLR,
    scaler: torch.amp.GradScaler,
    state: TrainState,
    config: TrainConfig,
    data_fingerprints: Mapping[str, str | None],
    validation: Mapping[str, Any] | None,
) -> dict[str, Any]:
    return {
        "format_version": 1,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "rng": _capture_rng(),
        "state": asdict(state),
        "config": asdict(config),
        "data_fingerprints": dict(data_fingerprints),
        "validation": dict(validation or {}),
    }


@torch.no_grad()
def _evaluate_loader(
    model: nn.Module,
    loader: DataLoader[Record],
    *,
    device: torch.device,
    max_batches: int,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
) -> dict[str, float | int]:
    was_training = model.training
    model.eval()
    loss_sum = 0.0
    correct_tokens = 0
    token_count = 0
    correct_sequences = 0
    example_count = 0
    for batch_index, batch in enumerate(loader):
        if batch_index >= max_batches:
            break
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        with _autocast_context(device, amp_enabled, amp_dtype):
            logits = _model_logits(model, input_ids, attention_mask)
            batch_loss = _causal_loss(logits, labels, reduction="sum")
        targets = labels[:, 1:]
        predictions = logits[:, :-1, :].argmax(dim=-1)
        supervised = targets.ne(-100)
        matches = predictions.eq(targets) | ~supervised
        counts = supervised.sum(dim=1)
        loss_sum += float(batch_loss)
        correct_tokens += int((predictions.eq(targets) & supervised).sum())
        token_count += int(supervised.sum())
        correct_sequences += int((matches.all(dim=1) & counts.gt(0)).sum())
        example_count += input_ids.shape[0]
    if was_training:
        model.train()
    return {
        "loss": loss_sum / max(1, token_count),
        "token_accuracy": correct_tokens / max(1, token_count),
        "sequence_accuracy": correct_sequences / max(1, example_count),
        "tokens": token_count,
        "examples": example_count,
    }


def validate_per_task(
    model: nn.Module,
    shards: Sequence[str | os.PathLike[str]],
    *,
    tasks: Sequence[str],
    config: TrainConfig,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
) -> dict[str, dict[str, float | int]]:
    """Return independent validation metrics for every requested task.

    The validation records are collected for all tasks in one physical shard
    scan.  Each task still receives the same prefix of its own record stream,
    the same token-budget batching, and the same independent metric pass as in
    the historical one-scan-per-task implementation.  Collecting
    ``batch_size * max_batches`` records is an upper bound on the number any
    task can consume, including when the padded-token budget shortens a batch.
    """

    metrics: dict[str, dict[str, float | int]] = {}
    collator = AnswerOnlyCollator(max_seq_len=config.max_seq_len)
    requested_tasks = tuple(dict.fromkeys(tasks))
    if not requested_tasks:
        return metrics
    record_limit = config.validation_batch_size * config.validation_batches_per_task
    records_by_task: dict[str, list[Record]] = {
        task: [] for task in requested_tasks
    }
    dataset = StreamingPermutationDataset(
        shards,
        tasks=requested_tasks,
        shuffle_buffer_size=1,
        seed=config.seed,
        rank=0,
        world_size=1,
    )
    for record in dataset:
        task = record["task"]
        bucket = records_by_task[task]
        if len(bucket) < record_limit:
            bucket.append(record)
        if all(len(values) >= record_limit for values in records_by_task.values()):
            break

    for task in requested_tasks:
        batches = TokenBudgetBatcher(
            records_by_task[task],
            max_examples=config.validation_batch_size,
            max_padded_tokens=config.max_tokens_per_batch,
        )
        loader = DataLoader(
            batches,
            batch_size=None,
            collate_fn=collator,
            num_workers=0,
        )
        metrics[task] = _evaluate_loader(
            model,
            loader,
            device=device,
            max_batches=config.validation_batches_per_task,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
        )
    return metrics


def train_run(
    run_config: TrainConfig | Mapping[str, Any],
    *,
    model_factory: ModelFactory | None = None,
    stop_after_steps: int | None = None,
) -> dict[str, Any]:
    """Train or resume a run and return a compact, JSON-compatible summary.

    ``stop_after_steps`` is primarily useful for preemption tests: it limits
    optimizer steps in this invocation without changing the planned cosine
    schedule in ``config.max_steps``.
    """

    config = TrainConfig.from_value(run_config)
    config.validate()
    validation_tasks = (
        config.validation_tasks
        if config.validation_tasks is not None
        else task_names_for_data_source(config.validation_manifest or config.manifest)
    )
    if stop_after_steps is not None and stop_after_steps < 1:
        raise ValueError("stop_after_steps must be at least one")
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "checkpoint.pt"
    completed_path = output_dir / "completed.json"
    training_shards = _training_shards(config)
    validation_shards = _validation_shards(config, training_shards)
    data_fingerprints = _data_fingerprints(config)

    _seed_everything(config.seed)
    factory = model_factory or _default_model_factory
    model = factory(config)
    if not isinstance(model, nn.Module):
        raise TypeError("model_factory must return torch.nn.Module")
    device = torch.device(
        config.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model.to(device)
    optimizer = AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = _scheduler(optimizer, config)
    use_amp = config.amp and device.type == "cuda"
    use_bf16 = use_amp and config.bf16 and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and not use_bf16)
    state = TrainState()
    last_validation: dict[str, Any] = {}

    resume_path: Path | None = None
    if config.resume:
        if config.resume == "auto":
            resume_path = checkpoint_path if checkpoint_path.is_file() else None
        else:
            resume_path = Path(config.resume)
    if resume_path is not None:
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        checkpoint_config = checkpoint.get("config")
        if not isinstance(checkpoint_config, Mapping):
            raise ValueError("checkpoint does not contain a valid training config")
        if _config_identity(checkpoint_config) != _config_identity(config):
            raise ValueError(
                "checkpoint training config differs from the requested run; "
                "refusing an unsafe resume"
            )
        if checkpoint.get("data_fingerprints") != data_fingerprints:
            raise ValueError(
                "checkpoint dataset fingerprints differ from current manifests; "
                "refusing an unsafe resume"
            )
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        if checkpoint.get("scaler"):
            scaler.load_state_dict(checkpoint["scaler"])
        state = TrainState(**checkpoint["state"])
        last_validation = dict(checkpoint.get("validation") or {})
        _restore_rng(checkpoint["rng"])
        if state.global_step < config.max_steps:
            completed_path.unlink(missing_ok=True)

    collator = AnswerOnlyCollator(max_seq_len=config.max_seq_len)
    optimizer.zero_grad(set_to_none=True)
    invocation_steps = 0
    recent_loss = 0.0
    stopped_early = False

    def save_checkpoint() -> None:
        _atomic_torch_save(
            _checkpoint_payload(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                state=state,
                config=config,
                data_fingerprints=data_fingerprints,
                validation=last_validation,
            ),
            checkpoint_path,
        )

    def optimizer_step() -> None:
        nonlocal invocation_steps
        if scaler.is_enabled():
            scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()
        state.global_step += 1
        invocation_steps += 1

    while state.global_step < config.max_steps and not stopped_early:
        dataset = StreamingPermutationDataset(
            training_shards,
            tasks=config.tasks,
            shuffle_buffer_size=config.shuffle_buffer_size,
            seed=config.seed,
            epoch=state.epoch,
        )
        batches = TokenBudgetBatcher(
            dataset,
            max_examples=config.batch_size,
            max_padded_tokens=config.max_tokens_per_batch,
        )
        loader_generator = torch.Generator().manual_seed(
            config.seed + 65_537 * state.epoch
        )
        loader = DataLoader(
            batches,
            batch_size=None,
            collate_fn=collator,
            num_workers=config.num_workers,
            generator=loader_generator,
            persistent_workers=config.num_workers > 0,
        )
        iterator = iter(loader)
        skipped = 0
        while skipped < state.batches_in_epoch:
            try:
                next(iterator)
            except StopIteration as error:
                raise RuntimeError("checkpoint cursor exceeds the epoch data") from error
            skipped += 1

        accumulated = 0
        saw_batch = False
        for batch in iterator:
            saw_batch = True
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            with _autocast_context(device, use_amp, amp_dtype):
                logits = _model_logits(model, input_ids, attention_mask)
                per_example_loss = _causal_loss(logits, labels, reduction="none")
                loss = per_example_loss.mean()
                scaled_loss = loss / config.gradient_accumulation_steps
            scaler.scale(scaled_loss).backward()
            recent_loss = float(loss.detach())
            supervised_counts = labels[:, 1:].ne(-100).sum(dim=1).tolist()
            for task, token_count, example_loss in zip(
                batch["tasks"],
                supervised_counts,
                per_example_loss.detach().float().cpu().tolist(),
                strict=True,
            ):
                state.task_examples[task] = state.task_examples.get(task, 0) + 1
                state.task_supervised_tokens[task] = (
                    state.task_supervised_tokens.get(task, 0) + int(token_count)
                )
                state.task_loss_sum[task] = (
                    state.task_loss_sum.get(task, 0.0) + float(example_loss)
                )
            accumulated += 1
            state.batches_in_epoch += 1
            if accumulated < config.gradient_accumulation_steps:
                continue
            optimizer_step()
            accumulated = 0

            if (
                config.validate_every > 0
                and state.global_step % config.validate_every == 0
            ):
                last_validation = validate_per_task(
                    model,
                    validation_shards,
                    tasks=validation_tasks,
                    config=config,
                    device=device,
                    amp_enabled=use_amp,
                    amp_dtype=amp_dtype,
                )
            if (
                config.checkpoint_every > 0
                and state.global_step % config.checkpoint_every == 0
            ):
                save_checkpoint()
            if state.global_step >= config.max_steps:
                break
            if stop_after_steps is not None and invocation_steps >= stop_after_steps:
                stopped_early = True
                break

        if stopped_early or state.global_step >= config.max_steps:
            break
        if accumulated:
            optimizer_step()
            if stop_after_steps is not None and invocation_steps >= stop_after_steps:
                stopped_early = True
        if not saw_batch and state.batches_in_epoch == 0:
            raise RuntimeError("training dataset is empty after task filtering")
        state.epoch += 1
        state.batches_in_epoch = 0

    save_checkpoint()
    status = "stopped" if stopped_early else "completed"
    checkpoint_sha256 = _sha256_file(checkpoint_path)
    task_accounting = {
        task: {
            "examples": examples,
            "supervised_tokens": state.task_supervised_tokens.get(task, 0),
            "mean_example_loss": state.task_loss_sum.get(task, 0.0) / max(1, examples),
        }
        for task, examples in sorted(state.task_examples.items())
    }
    summary = {
        "status": status,
        "run_id": output_dir.name,
        "architecture": config.architecture,
        "tasks": list(config.tasks or ()),
        "seed": config.seed,
        "global_step": state.global_step,
        "epoch": state.epoch,
        "batches_in_epoch": state.batches_in_epoch,
        "last_loss": recent_loss,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "config_sha256": _config_sha256(config),
        "experiment_config_sha256": config.experiment_config_sha256,
        **data_fingerprints,
        "task_accounting": task_accounting,
        "validation": last_validation,
    }
    if status == "completed":
        _atomic_json(summary, completed_path)
    return summary


# Familiar alias for callers that do not use the experiment runner naming.
train = train_run


def _parse_cli_tasks(values: Sequence[str] | None) -> tuple[str, ...] | None:
    return _task_tuple(values)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--architecture", default="transformer")
    parser.add_argument("--tasks", action="append", help="repeat or use commas")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--grad-accum", type=int)
    parser.add_argument("--max-seq-len", type=int)
    parser.add_argument("--max-tokens-per-batch", type=int, default=4_096)
    parser.add_argument("--d-model", type=int)
    parser.add_argument("--num-layers", type=int)
    parser.add_argument("--num-heads", type=int)
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--mlp-ratio", type=float)
    parser.add_argument("--resume", nargs="?", const="auto")
    parser.add_argument("--validation-manifest")
    parser.add_argument("--train-shard-indices")
    parser.add_argument("--validation-shard-indices")
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--warmup-steps", type=int)
    parser.add_argument("--min-lr-ratio", type=float)
    parser.add_argument("--max-grad-norm", type=float)
    parser.add_argument("--shuffle-buffer-size", type=int)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--checkpoint-every", type=int)
    parser.add_argument("--validate-every", type=int)
    parser.add_argument("--validation-batches-per-task", type=int)
    parser.add_argument("--device")
    parser.add_argument(
        "--experiment-config",
        help="frozen experiment config path recorded in the checkpoint",
    )
    parser.add_argument(
        "--model-config-json", default="{}", help="JSON object passed to build_model"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    model_config = json.loads(args.model_config_json)
    if not isinstance(model_config, dict):
        raise SystemExit("--model-config-json must decode to an object")

    experiment: dict[str, Any] = {}
    experiment_sha256: str | None = None
    if args.experiment_config:
        experiment_path = Path(args.experiment_config)
        payload = experiment_path.read_bytes()
        experiment_sha256 = hashlib.sha256(payload).hexdigest()
        try:
            import tomllib
        except ModuleNotFoundError:  # pragma: no cover - Python 3.10 only
            try:
                import tomli as tomllib  # type: ignore[no-redef,import-not-found]
            except ModuleNotFoundError as error:
                raise SystemExit(
                    "reading --experiment-config on Python 3.10 requires tomli"
                ) from error
        experiment = tomllib.loads(payload.decode("utf-8"))

    data_section = experiment.get("data", {})
    model_section = experiment.get("model", {})
    training_section = experiment.get("training", {})

    def chosen(value: Any, section: Mapping[str, Any], key: str, default: Any) -> Any:
        return value if value is not None else section.get(key, default)

    layers_default = (
        model_section.get("transformer_layers", 4)
        if args.architecture == "transformer"
        else model_section.get("mlp_layers", 1)
    )
    precision = str(training_section.get("precision", "bf16")).lower()
    total_validation_batches = int(training_section.get("validation_batches", 20))
    validation_batches_per_task = chosen(
        args.validation_batches_per_task,
        training_section,
        "validation_batches_per_task",
        max(1, math.ceil(total_validation_batches / 20)),
    )
    config = TrainConfig(
        manifest=args.manifest,
        validation_manifest=args.validation_manifest,
        output_dir=args.output_dir,
        architecture=args.architecture,
        d_model=chosen(args.d_model, model_section, "d_model", 256),
        num_layers=args.num_layers if args.num_layers is not None else layers_default,
        num_heads=chosen(args.num_heads, model_section, "num_heads", 8),
        dropout=chosen(args.dropout, model_section, "dropout", 0.0),
        mlp_ratio=chosen(args.mlp_ratio, model_section, "ff_multiplier", 4.0),
        tie_embeddings=bool(model_section.get("tie_embeddings", True)),
        model_config=model_config,
        tasks=_parse_cli_tasks(args.tasks),
        seed=args.seed,
        max_steps=chosen(args.max_steps, training_section, "max_steps", 1_000),
        batch_size=chosen(args.batch_size, training_section, "micro_batch_size", 32),
        gradient_accumulation_steps=chosen(
            args.grad_accum, training_section, "gradient_accumulation_steps", 1
        ),
        max_seq_len=chosen(
            args.max_seq_len, data_section, "max_sequence_length", 1_024
        ),
        max_tokens_per_batch=args.max_tokens_per_batch,
        resume=args.resume,
        learning_rate=chosen(
            args.learning_rate, training_section, "learning_rate", 3e-4
        ),
        weight_decay=chosen(args.weight_decay, training_section, "weight_decay", 0.01),
        warmup_steps=chosen(args.warmup_steps, training_section, "warmup_steps", 100),
        min_lr_ratio=chosen(
            args.min_lr_ratio,
            training_section,
            "min_learning_rate_ratio",
            0.1,
        ),
        max_grad_norm=chosen(
            args.max_grad_norm, training_section, "gradient_clip_norm", 1.0
        ),
        shuffle_buffer_size=chosen(
            args.shuffle_buffer_size, data_section, "shuffle_buffer", 10_000
        ),
        num_workers=args.num_workers,
        checkpoint_every=chosen(
            args.checkpoint_every,
            training_section,
            "checkpoint_every_steps",
            1_000,
        ),
        validate_every=chosen(
            args.validate_every, training_section, "validate_every_steps", 1_000
        ),
        validation_batches_per_task=validation_batches_per_task,
        device=args.device,
        amp=precision in {"bf16", "fp16", "float16", "bfloat16"},
        bf16=precision in {"bf16", "bfloat16"},
        shard_indices=parse_shard_indices(
            args.train_shard_indices or data_section.get("train_shards")
        ),
        validation_shard_indices=parse_shard_indices(
            args.validation_shard_indices or data_section.get("validation_shards")
        ),
        experiment_config=args.experiment_config,
        experiment_config_sha256=experiment_sha256,
    )
    print(json.dumps(train_run(config), sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through the CLI
    raise SystemExit(main())


__all__ = [
    "AnswerOnlyCollator",
    "PermutationIterableDataset",
    "StreamingPermutationDataset",
    "TokenBudgetBatcher",
    "TrainConfig",
    "TrainState",
    "build_arg_parser",
    "main",
    "parse_shard_indices",
    "resolve_shards",
    "task_names_for_data_source",
    "train",
    "train_run",
    "validate_per_task",
]
