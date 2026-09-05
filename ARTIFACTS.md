# Datasets and model checkpoints

The synthetic datasets and formal model checkpoints are distributed as GitHub
Release assets rather than ordinary Git objects. This keeps normal clones small
while making the artifacts public, versioned, and independently verifiable.

The initial artifact snapshot is tied to source commit
`efe755b4251826c263379ebbace0cb667ddb10ae`.

## Releases

- [Permutation datasets (2026-09-04)](https://github.com/XuanyuYang223/neurips/releases/tag/permutation-data-2026-09-04)
- [Permutation model checkpoints (2026-09-04)](https://github.com/XuanyuYang223/neurips/releases/tag/permutation-checkpoints-2026-09-04)

Each release contains a `SHA256SUMS.txt` file covering every archive or archive
part. The data release is approximately 15 GiB; the checkpoint release is
approximately 17 GiB.

## Data included

- the fully verified 10-million-example v2 corpus;
- the fully verified 10-million-example v3 corpus;
- the 16-million-example Property32 corpus;
- the 100-million-example scaling corpus;
- the four-representation transfer corpus; and
- the size-extrapolation evaluation corpus.

## Checkpoints included

- v2 and v3 nested/category experiments;
- v3 5-, 20-, and 100-shot adaptations;
- Property32 zero-overlap, replicate, probing-support, and few-shot studies;
- Property32 task-geometry and relation-controlled studies;
- the four-representation transfer experiment; and
- the completed data-by-depth scaling study.

Temporary controller state, aborted duplicate runs, and smoke-only runs are not
included.

## Download and verify

With the GitHub CLI:

```bash
gh release download permutation-data-2026-09-04 \
  --repo XuanyuYang223/neurips \
  --dir downloads/permutation-data-2026-09-04

gh release download permutation-checkpoints-2026-09-04 \
  --repo XuanyuYang223/neurips \
  --dir downloads/permutation-checkpoints-2026-09-04
```

Verify every downloaded asset before extraction:

```bash
cd downloads/permutation-data-2026-09-04
sha256sum --check SHA256SUMS.txt
```

Repeat the checksum command in the checkpoint download directory.

## Extract

Each archive stores its original `data/...` or `runs/...` path. Extract from the
repository root. For a single-file archive:

```bash
tar --use-compress-program=unzstd -xf DOWNLOAD_DIR/ARCHIVE.tar.zst
```

For a multipart archive, concatenate the numerically ordered parts before
decompression:

```bash
cat DOWNLOAD_DIR/ARCHIVE.tar.zst.part-* | zstd -d | tar -xf -
```

The 100-million-example corpus and the task-geometry checkpoints are multipart
archives. All other initial artifacts are single-file archives.

PyTorch checkpoint files should be loaded only after checksum verification. When
supported by the installed PyTorch version, use `torch.load(...,
weights_only=True)` for inspection.
