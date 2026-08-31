# Permutation 多任务实验：完整数据与训练过程

本文档记录 permutation 部分从需求解释、数据生成、编码、模型设计、正式训练、断点恢复、验证、审计到结果发布的完整过程。第 1–14 节描述已经实际执行完成的 v2 baseline；第 0 节记录 Henry 反馈后的 v3：数据已经生成并 full-verified，但 revised 模型仍是尚未训练的冻结方案。

最后更新：2026-08-30

- GitHub：<https://github.com/XuanyuYang223/neurips>
- 已完成 baseline：数据协议 `permutation-20/v2`，实验协议 `henry-permutation/v1`
- Baseline 状态：30/30 个 base models 完成，严格审计 30 passed / 0 failed
- Revised main study：数据协议 `permutation-20/v3`，实验协议 `henry-permutation/v2-revised`
- Revised data 状态：10,000,000 records，100 shards，full verification passed
- Revised model 状态：方案已冻结，**尚未训练，0 个 completed models**
- V2 正式训练使用的 implementation commit：`6a40235`
- 完训后 audit/results 基线 commit：`32ff22a`

## 0. Henry 反馈后的 revised main study（v3）

Henry 的意见是保持 architecture 尽量标准，不必复现原始 PermuFormer；同时把学习成本较高的 `power`、`conjugate`、`commutator` 排除出论文主实验，并按任务类别比较 learned representations。Henry 没有指定替代任务；为继续保持 20 个平衡 tasks，本项目选择了 `peaks`、`exceedances`、`recoils`。

因此 v3 保留当前标准 small pre-LN decoder-only Transformer（4 layers、8 heads、`d_model=256`）及 MLP 对照，不引入新的 Transformer modification。任务替换为：

| v2 移除 | v3 新增 | 定义 | 新数据量 |
|---|---|---|---:|
| `power` | `peaks` | 内部局部峰值数量 | 500,000 |
| `conjugate` | `exceedances` | `pi(i) > i` 的位置数量 | 500,000 |
| `commutator` | `recoils` | `pi^-1` 的 descent 数量 | 500,000 |

三个新任务都是 `n <= 30` 下的 scalar property；答案用一个 `00`–`99` token，不需要 `<NUM_START>`。正式 v3 数据已经完成：20 tasks × 500,000 records = 10,000,000 条，写入独立目录 `data/permutation-10m-v3`，没有覆盖 v2 数据。

| V3 data fact | Value |
|---|---:|
| Records | 10,000,000 |
| Shards | 100 |
| Compressed bytes | 1,139,175,228 |
| Generation wall time | 43.76 seconds |
| Full verification wall time | 34.59 seconds |
| Parent manifest SHA-256 | `b20a16cee7710cee4a21cc4575c8651ade1bcfca18219d2e6c230d4a3ab0cf6f` |

V3 split：

| Split | Shards | Records | Manifest SHA-256 |
|---|---|---:|---|
| Train | `000-097` | 9,800,000 | `7ad40c63a7559c52640d233a5398125d14160d83acadfa30637de291292893fa` |
| Validation | `098` | 100,000 | `90e88845f3f58947f317c67144c83bc5e38c27b248227e311632af834d2fd068` |
| Test | `099` | 100,000 | `3ca12e6b6eeb29fc0ddd441b9c44c80d7a160faaf7e832eb55007f4c6a3ab52b` |

Revised nested matrix 继续使用 `1/2/4/8/16 tasks × 2 architectures × 3 seeds = 30` 个计划模型，并保留与 v2 相同的四个 holdouts：`to_reduced_word`、`compose`、`parity`、`to_lehmer`。相同 holdout identities 便于直接比较 v2 与 v3，且保留一个 algebra holdout。新版 output 独立写到 `runs/henry-permutation-v3`。

为回答 Henry 关于类别表示差异的问题，本项目将其 operationalize 为 task-count-matched 三组：

| 条件 | 4 个训练任务 |
|---|---|
| Encoding E4 | `to_cycle`, `to_lehmer`, `to_inversion_vector`, `to_reduced_word` |
| Statistics S4 | `length`, `cycle_type`, `rsk_shape`, `pattern_avoidance` |
| Algebra A4 | `inverse`, `compose`, `right_multiply_simple`, `bruhat_leq` |

这避免直接比较完整 `4 vs 12 vs 4` 类别造成 task-count confound。三组使用相同 per-task data、optimizer-update budget、architecture 和三个 seeds，共 `3 categories × 2 architectures × 3 seeds = 18` 个计划模型；在相同 held-out one-line permutation 的 `<ONE_END>` 位置比较 layerwise CKA、frozen linear probes 和 few-shot transfer。

权威设计见 [configs/henry_permutation_revised.toml](configs/henry_permutation_revised.toml) 与 [EXPERIMENTS.md](EXPERIMENTS.md)。TOML 中的 category table 当前是 declarative design；现有 nested runner 尚不能调度 E4/S4/A4。V3 数据完成不等于模型完成：这 30+18 个计划模型都没有 `completed.json`，所以本文和 README 不报告任何 v3 accuracy。

## 1. V2 baseline 的研究目标与完成范围

Henry 提出的核心问题是：随着一个模型训练过的任务种类增加，它的内部表示和 generalization 能力是否系统性增强。V2 permutation baseline 采用嵌套任务集合，并在相同优化步数预算下比较 Transformer 和 MLP。

V2 baseline 已经完成：

1. 定义并实现 20 个 permutation tasks；
2. 生成 10,000,000 条 Passage Math 训练序列；
3. 对全部数据逐条重算数学答案并验证编码；
4. 训练 `1/2/4/8/16 tasks × 2 architectures × 3 seeds = 30` 个 base models；
5. 保存可恢复 checkpoint、最终 marker 和 validation 指标；
6. 对全部 30 个 checkpoint 做严格结构、哈希和数值审计；
7. 汇总初步 zero-shot generalization 结果。

V2 baseline 尚未完成：

- 在从未用于模型评估的 test shard 099 上做冻结测试；
- holdout tasks 的 few-shot fine-tuning；
- 与随机初始化模型的 few-shot baseline 比较；
- linear probing 和 representation geometry；
- 第二份方案中的完整 `4 representations × 8 tasks` 输入组合实验。

因此，v2 baseline 完成的是 Henry 方案中的“任务选择、数据生成和 base-model training”阶段，以及一份 validation-set zero-shot 初步结果；还不能称为完整的 generalization study。这句话不描述 v3 模型状态；v3 目前只完成了正式数据和 full verification。

## 2. 需求解释与冻结决策

### 2.1 `maximum entries = 30`

所有对象都是标准对称群 `S_n` 中的 permutation：

```text
pi is a permutation of {1, 2, ..., n}, with 2 <= n <= 30.
```

因此 permutation 的最大长度为 30，entry 的最大值也自然是 30。没有把它解释为“从 1 到 100 中选 30 个不同数字”的 partial permutation，因为 cycle、Bruhat、Coxeter generators 等操作需要标准 `S_n` 语义。

### 2.2 `maximum number = 100`

在 v2 baseline 中，100 被实现为用户提供的 base-100 number tokenizer 约定和 power task 的指数上界，而不是 permutation entry 上界：

- `00` 到 `99` 是 atomic number tokens；
- 100 以上使用 `<NUM_START> ... <NUM_END>`；
- v2 的 power exponent 取 `0 <= k <= 100`；v3 已移除 power，但 number encoding 不变。

### 2.3 “10M data”的单位

10M 指 10,000,000 条最终 causal-LM sequences。每条 sequence 只包含一个 task 和一个 answer。

它不是：

```text
10M base permutations × 20 labels = 200M model sequences
```

V2 最终 20 个 tasks 完全平衡，每个 task 500,000 条。V3 也已按同样总量和平衡约束完成，其中三个新增 properties 各 500,000 条。这一选择避免了约 200M 条富 JSON 记录带来的数百 GB 存储和极长训练时间，同时保留了任务平衡。

### 2.4 输入 representation

当前 30 个 Henry base models 的 primary input 始终使用 one-line notation。Cycle、Lehmer、inversion vector 和 reduced Coxeter word 是 translation targets。

第二份附件提出的 `4 input representations × 8 tasks = 32 combinations` 是另一套 representation-transfer 实验；本轮没有把它与 20-task Henry nested matrix 混在一起。

## 3. V2 baseline 的二十个任务

以下列表属于已经训练完成的 v2。V3 用 `peaks`、`exceedances`、`recoils` 分别替换其中的 `power`、`conjugate`、`commutator`；完整 v3 registry 见第 0 节和 [PROTOCOL.md](PROTOCOL.md)。

### 3.1 Encoding / translation（4）

1. `to_cycle`：canonical disjoint cycles；包含 singleton cycle，每个 cycle 从最小元素开始，cycles 按最小元素排序。
2. `to_lehmer`：`L_i = #{j > i : pi_j < pi_i}`。
3. `to_inversion_vector`：value-indexed，`I_v = #{u > v : position(u) < position(v)}`。
4. `to_reduced_word`：稳定 bubble-sort 产生的 deterministic reduced adjacent-generator word。

### 3.2 Statistics / properties（9）

5. `length`：Coxeter length / inversion count。
6. `descents`：descent 数量。
7. `fixed_points`：fixed point 数量。
8. `parity`：inversion count modulo 2，`00=even`、`01=odd`。
9. `cycle_type`：包含 1-cycles 的 cycle lengths，降序排列。
10. `rsk_shape`：RSK insertion tableau 的 row lengths。
11. `lis_length`：longest strictly increasing subsequence length。
12. `lds_length`：longest strictly decreasing subsequence length。
13. `pattern_avoidance`：是否避免给定 classical pattern，`01=avoids`、`00=contains`。

### 3.3 Algebraic operations / comparisons（7）

采用 composition 约定：

```text
(a o b)(i) = a(b(i))
```

14. `inverse`：`pi^-1`。
15. `compose`：`pi o sigma`。
16. `power`：`pi^k`，`0 <= k <= 100`。
17. `conjugate`：`g o pi o g^-1`。
18. `commutator`：`pi o sigma o pi^-1 o sigma^-1`。
19. `right_multiply_simple`：`pi s_i`，交换 one-line positions `i` 与 `i+1`。
20. `bruhat_leq`：strong Bruhat comparison。

数学定义和 convention 的权威版本见 [PROTOCOL.md](PROTOCOL.md) 与 [math_ops.py](src/neurips_permutations/math_ops.py)。

## 4. Passage Math encoding

### 4.1 Vocabulary

V2 正式 vocabulary size 为 163：

- 100 个 number tokens：`00`–`99`；
- 36 个原始 fixed tokens；
- 为 20-task 数据新增 27 个 task、operand 和 structured-answer tokens。

输入 embedding 和输出 LM head 使用同一套 vocabulary，并进行 weight tying。

### 4.2 Number encoding

```text
0     -> 00
7     -> 07
28    -> 28
99    -> 99
100   -> <NUM_START> 01 00 <NUM_END>
137   -> <NUM_START> 01 37 <NUM_END>
9999  -> <NUM_START> 99 99 <NUM_END>
10000 -> <NUM_START> 01 00 00 <NUM_END>
```

这给每个非负整数一个唯一 canonical encoding。

### 4.3 通用 sequence grammar

```text
<BOS> <SIZE> ENCODE(n)
<ONE_START> PRIMARY <ONE_END>
[TYPED OPERANDS]
<TASK> = TYPED_ANSWER
<EOS>
```

每条 sequence 恰好有一个 task token 和一个 answer。训练 loss 只监督 answer tokens 与 `<EOS>`；prompt、primary permutation、operands、task token 和 `=` 的 labels 都设为 ignore index。

### 4.4 Representation 示例

令 `pi = [3,1,4,2]`。

One-line：

```text
<ONE_START> 03 , 01 , 04 , 02 <ONE_END>
```

Canonical cycle：

```text
<CYCLE_START> 01 , 03 , 04 , 02 <CYCLE_END>
```

Lehmer code `[2,0,1,0]`：

```text
<LEHMER_START> 02 , 00 , 01 , 00 <LEHMER_END>
```

Inversion vector `[1,2,0,0]`：

```text
<INVEC_START> 01 , 02 , 00 , 00 <INVEC_END>
```

Reduced word `[2,3,1]`：

```text
<REDUCED_WORD_START> 02 , 03 , 01 <REDUCED_WORD_END>
```

完整 translation sequence 示例：

```text
<BOS> <SIZE> 04 <ONE_START> 03 , 01 , 04 , 02 <ONE_END>
<TO_LEHMER> = <LEHMER_START> 02 , 00 , 01 , 00 <LEHMER_END> <EOS>
```

### 4.5 Typed operands

```text
Second permutation: <ARG_START> ... <ARG_END>
Pattern:            <PATTERN_START> ... <PATTERN_END>
Exponent:           <EXPONENT> ENCODE(k)
Simple index:       <SIMPLE_INDEX> ENCODE(i)
```

编码实现见 [passage.py](src/neurips_permutations/passage.py)。

### 4.6 JSONL record schema

每一行同时保存结构化字段与最终 token sequence：

```json
{
  "schema_version": "permutation-20/v2",
  "id": 0,
  "task": "...",
  "n": 12,
  "inputs": {"primary": [1, 2, 3]},
  "answer": 0,
  "answer_kind": "scalar",
  "tokens": ["<BOS>", "...", "<EOS>"],
  "canonical_text": "<BOS> ... <EOS>"
}
```

`answer` 与 `inputs` 会依任务包含 scalar、Boolean、permutation、pattern、exponent 或 nested lists。

## 5. 数据生成

### 5.1 确定性采样

- 全局 seed：`20260830`；
- 每条 record 的 RNG 由 global seed 与 record ID 确定；
- task 按 `record_id mod 20` 精确轮转；
- 大多数 tasks 的 `n` 从 2–30 采样；
- pattern avoidance 使用 `n >= 3`；
- Bruhat 使用 `n >= 4`；
- duplicates 允许存在，这是带 replacement 的 synthetic sampling。

Pattern avoidance 正负样本严格平衡。Pattern 长度为 `n-1`：对 label `00`（contains），pattern 从删除 primary 的一个 entry 后得到的 standardized deletion patterns 中选取；对 label `01`（avoids），保证 pattern 不在该集合中。

Bruhat 正负样本也严格平衡，并匹配相同的 permutation size 和正 Coxeter-length gap。Gap 为 1–4（`S_4` 中为 1–2）；正样本沿 strong Bruhat covers 构造，负样本是具有相同 gap 的 incomparable pairs。因此不能只看 inversion-length gap 猜 label。

### 5.2 Streaming 与 sharding

生成器不会把 10M records 放进内存：

- 100 个 shards；
- 每 shard 100,000 records；
- deterministic `jsonl.gz`；
- gzip level 6，header timestamp 固定为 0；
- 每个 shard 先写 temporary file、flush、`fsync`，再 atomic rename；
- manifest 保存 record range、byte size、SHA-256 和 task counts；
- resume 时只有 checksum、count 和 config 全匹配的 shard 才会复用。

### 5.3 v1 审查与 v2 重生成

初版数据审查确认 Bruhat label 存在直接 leakage：v1 正例的 inversion-length gap 全为 `+1`，负例全为 `-1`，因此只看方向即可读出 label。早期 verifier 也只检查“存储答案能否重新渲染”，没有从 inputs 重算数学真值。

正式训练前做了两项修复：

1. 将 Bruhat 改成上述 matched-gap comparable/incomparable 构造；
2. full verifier 从 inputs 重新调用权威 `math_ops` 计算 20 个 tasks 的 answer，再从该真值重建 tokens；`math_ops` 本身另有 exhaustive small-`n` 和 unit tests 交叉检查。

修复后重新生成 `permutation-20/v2`。旧目录 `data/permutation-10m` 没有用于任何正式训练；所有 formal runs 只读取 `data/permutation-10m-v2`。

### 5.4 最终数据量

| Split | Shards | Records | 每 task | Compressed bytes |
|---|---:|---:|---:|---:|
| Train | 98（000–097） | 9,800,000 | 490,000 | 1,263,940,793 |
| Validation | 1（098） | 100,000 | 5,000 | 12,866,288 |
| Test | 1（099） | 100,000 | 5,000 | 12,911,897 |
| Total | 100 | **10,000,000** | **500,000** | **1,289,718,978** |

总压缩大小为 1.290 GB；gzip 解压后的 JSONL 大小为 8,734,219,058 bytes，即 8.734 GB，平均约 873 bytes/record。

Parent manifest SHA-256：

```text
a9cc873bc82777c50fc2cfced96f54d727e3c3964eff457bd1a03ffabb179e87
```

### 5.5 数据验证

最终 full verification 使用 20 workers，完成以下检查：

- 100 个 shard SHA-256 与 byte size；
- ID、shard index 和 record count 连续性；
- schema、task balance 和 permutation validity；
- 10,000,000 条记录的所有 20-task answers 从 inputs 重新计算；
- stored answer、typed answer kind、tokens 与 canonical text 一致；
- parent full verification 检查全部 physical shards；split views 的 parent metadata、ranges 和 counts 由 split verifier/tests 与 formal audit 另行检查。

结果：10,000,000 / 10,000,000 records passed。实际生成约 41.4 秒，完整数学复核约 32.0 秒。

## 6. 模型 architecture

两种模型都实现相同接口：

```text
forward(input_ids, attention_mask) -> logits[B, L, 163]
```

共同组件：

- learned token embedding；
- learned absolute position embedding；
- maximum context length 1024；
- `d_model = 256`；
- pre-LayerNorm、GELU、residual connections；
- channel MLP hidden dimension 1024；
- dropout 0.1；
- final LayerNorm；
- bias-free tied LM head；
- strict prefix causality 和 padding masking。

| Architecture | Blocks | Attention | Token mixing | Parameters |
|---|---:|---|---|---:|
| Causal Transformer | 4 | 8 heads，head dim 32 | causal self-attention | 3,463,424 |
| Causal MLP | 1 | none | 两个 masked `1024 × 1024` linear maps | 2,930,176 |

### 6.1 Transformer

每个 block：

```text
x = x + CausalSelfAttention(LayerNorm(x))
x = x + ChannelMLP(LayerNorm(x))
```

Attention 使用显式 lower-triangular causal mask 和 padding-key mask。Channel MLP 为 `256 -> 1024 -> 256`。

### 6.2 Causal MLP

MLP 没有 attention，也没有 recurrence。每个 block：

```text
x = x + CausalTokenMixingMLP(LayerNorm(x))
x = x + ChannelMLP(LayerNorm(x))
```

Token-mixing MLP 包含两个 learned `1024 × 1024` matrices；forward 时只使用 lower-triangular entries，并在两层之间使用 GELU。矩阵跨 channel 共享，因此结构类似 causal MLP-Mixer。Prefix-invariance tests 验证了后缀 token 不会改变任何 earlier representation。

只使用 1 个 MLP block 是为了在 1024 context 下让 registered parameter count 与 4-layer Transformer 大致匹配，而不是声称“1-layer MLP 与 4-layer Transformer 深度等价”。这里的 2,930,176 是 nominal/registered count：两个 `1024 × 1024` matrices 的 1,047,552 个 strict upper-triangular parameters 会被 causal mask 屏蔽，forward 中不使用，因此 registered count 不等于 active degrees of freedom。

实现见 [models.py](src/neurips_permutations/models.py)。

## 7. 已完成的 Henry v2 nested-task 实验矩阵

### 7.1 冻结 task order

任务使用 seed `20260830` 冻结为以下顺序，而不是采用提出任务时的顺序：

```text
 1  power
 2  lis_length
 3  fixed_points
 4  to_inversion_vector
 5  pattern_avoidance
 6  lds_length
 7  right_multiply_simple
 8  conjugate
 9  to_cycle
10  descents
11  commutator
12  length
13  rsk_shape
14  inverse
15  cycle_type
16  bruhat_leq
17  to_reduced_word       fixed holdout
18  compose               fixed holdout
19  parity                fixed holdout
20  to_lehmer             fixed holdout
```

### 7.2 Nested subsets

| Subset | Training tasks |
|---:|---|
| 1 | tasks 1 |
| 2 | tasks 1–2 |
| 4 | tasks 1–4 |
| 8 | tasks 1–8 |
| 16 | tasks 1–16 |

最后 4 个 tasks 对所有 base models 都从 training sequences 和 gradient updates 中 hold out；但每 1,000 steps 的 validation diagnostics 会评估它们，所以它们不是未查看的 test set。

### 7.3 Run count

```text
5 subset sizes × 2 architectures × 3 seeds = 30 formal runs
```

Model seeds：`17`、`42`、`314159`。三个 seeds 改变 initialization 和 streaming shuffle，但 task order 本身只冻结了一次。

完整配置见 [configs/henry_permutation.toml](configs/henry_permutation.toml)，其 SHA-256 为：

```text
c5d9a0ea7a601588d1e07a520721dfeb3b8f96830d03c8c9f8632c6d37f70dfa
```

## 8. Training configuration

| Item | Value |
|---|---:|
| Optimizer | AdamW |
| Optimizer steps / run | 20,000 |
| Micro-batch max examples | 16 |
| Gradient accumulation | 4 |
| Nominal examples / optimizer step | 64 |
| Max padded tokens / micro-batch | 4,096 |
| Initial learning rate | 0.0003 |
| Warmup | 1,000 steps |
| Schedule | cosine decay |
| Minimum LR ratio | 0.1 |
| Weight decay | 0.01 |
| Gradient clipping | 1.0 |
| Precision | bf16 AMP |
| Shuffle buffer | 10,000 records |
| Checkpoint interval | 1,000 steps |
| Validation interval | 1,000 steps |
| Validation batches | 5 dynamic batches / task |
| DataLoader workers | 0 |

### 8.1 Streaming loader

训练 loader 直接流式读取 gzip shards，按当前 run 的 tasks 过滤记录，使用 bounded deterministic shuffle，不把整个 dataset 放入内存。`TokenBudgetBatcher` 同时约束 example count 和 padded-token count，因此 reduced word 等长序列会自动使用更小 micro-batch，而不会截断到错误答案。

### 8.2 Answer-only objective

模型进行 causal next-token prediction，但 loss 只覆盖 answer 和 `<EOS>`：

```text
prompt labels -> -100 / ignored
answer labels -> supervised
EOS label      -> supervised
```

Loss 先在每个 example 的 supervised tokens 内求平均，再跨 examples 平均。这样一个可能有数百 tokens 的 reduced word 不会仅因答案更长而比 scalar task 获得几十倍权重。

### 8.3 固定 update budget

所有 models 都获得相同的 20,000 optimizer updates，而不是每个 task 获得相同步数。这能控制总计算量，但 task 数增加时每个 task 分到的 examples 会减少。

| Tasks | Examples / run | Eligible train pool | Approx. pool passes |
|---:|---:|---:|---:|
| 1 | 1,279,904 | 490,000 | 2.612 |
| 2 | 1,279,968 | 980,000 | 1.306 |
| 4 | 1,280,000 | 1,960,000 | 0.653 |
| 8 | 1,280,000 | 3,920,000 | 0.327 |
| 16 | 1,280,000 | 7,840,000 | 0.163 |

30 runs 合计：

```text
38,399,232 example exposures
807,897,938 supervised target tokens（包含每条样本的 EOS）
600,000 optimizer steps
```

每个 task 的平均 exposure 因而约从 k=1 时的 1.28M 降为 k=16 时的 80k。监督 token budget 也不完全相同：只训练 `power` 的 k=1 runs 约有 43.5M supervised tokens，而其余 task mixtures 每 run 约为 22–23M。这是解释 task-count 曲线时必须保留的混杂因素。

### 8.4 Validation protocol

每 1,000 steps 在 shard 098 上评估全部 20 tasks，不仅是当前训练 tasks。每个 task 最多 5 个 dynamic batches；每个 run 的最终 validation 共 2,924 examples、57,953 supervised tokens。

保存指标：

- token-weighted negative log likelihood；
- supervised token accuracy；
- exact sequence accuracy；
- examples 和 supervised-token counts。

`token_accuracy` 是 teacher-forced：每个 token 都看到 gold prefix，所以 copy、punctuation 和 boundary tokens 会抬高数值。

`sequence_accuracy` 要求 answer 和 EOS 的每一个 next-token argmax 都正确。由于模型经过 strict causal tests，这个 all-token event 与在相同 prompt 上进行 greedy decoding 得到完整 canonical target 等价；但是本轮没有单独运行 parser-aware decoding harness。

Shard 099 在本轮中没有用于模型选择或任何模型评估。

## 9. Checkpoint、resume 与完成条件

每个 run 的 `checkpoint.pt` 包含：

- model state；
- optimizer state；
- scheduler state；
- AMP scaler state；
- Python、CPU Torch 和 CUDA RNG state；
- epoch、batch offset、global step；
- per-task examples、tokens 和 accumulated loss；
- complete `TrainConfig`；
- training/validation manifest fingerprints；
- last validation metrics。

Checkpoint 通过 temporary file + atomic replace 写入。Resume 前会严格比较 training config 和 data fingerprints；不允许用不同 task set、shards 或 hyperparameters 静默续训。

Run 只有在 step 20,000 完成、最终 checkpoint 写入并哈希后，才会生成 `completed.json`。Marker 保存 checkpoint SHA-256、config hashes、task accounting 和 validation metrics。任意垃圾 marker 不能被视为完成。

训练过程中发现并修复了 CUDA resume 问题：checkpoint 以 `map_location=device` 加载时，保存的 CUDA RNG states 会被映射成 CUDA tensors；恢复前需统一转回 CPU `uint8` tensors，再传给 `torch.cuda.set_rng_state_all`。修复后，从正式 run 的 step 7,000 checkpoint 成功恢复并继续完成。

## 10. 实际 GPU 执行过程

### 10.1 Hardware/software

```text
GPU: NVIDIA GeForce RTX 5070, 12,227 MiB
Driver: 610.88
Python: 3.13.13
PyTorch: 2.11.0+cu128
CUDA runtime: 12.8
cuDNN: 9.19
bf16 support: true
```

### 10.2 Pilots

正式矩阵前分别运行了 100-step、16-task Transformer 和 MLP pilots，验证：

- CUDA forward/backward；
- bf16 AMP；
- dynamic batching；
- validation；
- checkpoint/marker；
- resume；
- 显存和吞吐。

Pilot artifacts 位于：

```text
runs/pilots/transformer-16task-100step-v2/
runs/pilots/mlp-16task-100step-v2/
```

### 10.3 Formal orchestration

初始顺序 controller 在第一个正式 run 运行到 step 7,000 后，为提高单卡利用率被干净停止。完成 CUDA RNG resume 修复并验证后，矩阵被拆成两个互不重叠的 run-ID queues；每个 controller 一次只管理自己队列中的一个 run，因此不会同时写同一目录。

双队列共同使用一张 RTX 5070：

- observed combined VRAM 通常约 3.5–4.2 GB；
- observed GPU utilization 通常约 60–80%；
- 没有 OOM；
- 没有 duplicate run controller；
- 已完成 marker 的 run 会被跳过；
- 中断 run 从精确 checkpoint 自动恢复。

第一个 formal completion marker 写于 01:31:32，最后一个写于 05:12:24；两者间隔约 3 小时 41 分钟。两个 controllers 最终均正常退出。

### 10.4 Final artifacts

```text
runs/henry-permutation/<run-id>/checkpoint.pt
runs/henry-permutation/<run-id>/completed.json
```

- Transformer checkpoints：约 41.64 MB × 15；
- MLP checkpoints：约 35.20 MB × 15；
- 30 个正式 checkpoints 总计 1,152,516,936 bytes。

`runs/` 和 production `data/` 被 `.gitignore` 排除，不会错误推送到 GitHub。

## 11. 初步 generalization 结果

最干净的跨 task-count 比较是四个对所有模型都未见的 fixed holdouts：

```text
to_reduced_word, compose, parity, to_lehmer
```

下表为三个 seeds 的 task-macro mean ± sample standard deviation：

| Architecture | Trained tasks | Holdout token accuracy | Holdout exact-sequence accuracy |
|---|---:|---:|---:|
| Transformer | 1 | 19.18 ± 1.02% | 0.23 ± 0.00% |
| Transformer | 2 | 25.71 ± 1.36% | 0.88 ± 0.58% |
| Transformer | 4 | 30.89 ± 0.29% | 0.00 ± 0.00% |
| Transformer | 8 | **43.75 ± 0.22%** | 0.39 ± 0.13% |
| Transformer | 16 | 41.03 ± 1.94% | 0.31 ± 0.13% |
| MLP | 1 | 23.08 ± 0.38% | 0.00 ± 0.00% |
| MLP | 2 | 35.14 ± 2.71% | 1.17 ± 0.13% |
| MLP | 4 | 40.99 ± 2.75% | **3.12 ± 1.80%** |
| MLP | 8 | **49.05 ± 0.78%** | 1.06 ± 0.53% |
| MLP | 16 | 43.80 ± 0.56% | 0.83 ± 1.04% |

观察：

1. 从 1 task 增至 8 tasks，holdout token accuracy 显著上升；16 tasks 时小幅下降。
2. MLP 在所有 subset sizes 上的 holdout token accuracy 都高于 Transformer。
3. 这种 token-level transfer 没有转化成可靠的完整答案：macro exact accuracy 最高仅 3.12%。
4. `to_reduced_word` 与 `to_lehmer` 在所有条件下 exact accuracy 都为 0%。
5. `compose` 的最佳三-seed exact mean 为 Transformer 1.54%、MLP 2.16%；`parity` 分别为 2.92% 和 12.50%。

当前最稳妥的结论是：

> 增加训练任务多样性改善了 prefix-conditioned token transfer，峰值出现在 8-task 条件；但没有形成可靠的 hard zero-shot complete-answer generalization。

一个关键设计限制是四个 holdout 使用 opaque task tokens；这些 tokens 从未作为 base-training sequence 的输入 token 或正确 target 出现，因此没有获得 operation semantics 的 grounding，尽管它们仍属于 163-way vocabulary 并会收到非目标类别梯度。模型没有被教过它们的任务含义，所以 hard zero-shot task identification 本身是 underdetermined。Henry 建议的 few-shot adaptation 和 linear probing 比这个 hard zero-shot 指标更有解释力。

完整 seen、pool-unseen 和 holdout 表见 [TRAINING_RESULTS.md](TRAINING_RESULTS.md)。

k=16 的逐 holdout 结果进一步说明 token accuracy 与数学求解成功不能混为一谈：

| Holdout task | Transformer token / exact | MLP token / exact |
|---|---:|---:|
| Reduced word | 16.03 ± 1.48% / 0.00 ± 0.00% | 18.54 ± 4.99% / 0.00 ± 0.00% |
| Composition | 61.30 ± 0.63% / 1.23 ± 0.53% | 60.06 ± 0.54% / 0.62 ± 0.53% |
| Parity | 40.00 ± 9.85% / 0.00 ± 0.00% | 51.35 ± 2.35% / 2.71 ± 4.69% |
| Lehmer code | 46.80 ± 3.48% / 0.00 ± 0.00% | 45.27 ± 8.17% / 0.00 ± 0.00% |

例如 composition 有约 60% token accuracy，却只有约 1% exact accuracy，表明局部格式或 copy 规律正确不代表整个运算正确。MLP k=16 parity 的非零均值主要来自一个 seed，另外两个 seeds 为 0，也不能称为稳定 generalization。

## 12. 最终验证与审计

### 12.1 Dataset verifier

```text
full=true
ok=true
record_count=10,000,000
shard_count=100
20 tasks × 500,000 records
```

### 12.2 Formal checkpoint audit

最终只读 audit 结果：

```text
run_count=30
passed_count=30
incomplete_count=0
failed_count=0
global issues=[]
partial artifacts=[]
```

审计器不是只检查“文件存在”，而是：

- 从 frozen TOML 和 launch command 重建完整 expected `TrainConfig`；
- 校验 experiment/manifest/checkpoint SHA、manifest schema、固定 shard ranges、shard 文件存在与 byte size；数据 shard 内容 SHA 与逐条数学复核由前述 full dataset verifier 完成；
- 使用 `weights_only=True` 安全读取 checkpoint；
- 按 expected architecture 实例化模型并 strict-check keys、shapes 和 dtypes；
- 检查 optimizer、scheduler、scaler、RNG、state 和 validation schema；
- 递归检查 NaN/Inf 和不可能的负统计量；
- 对比 marker 与 checkpoint accounting/validation；
- 拒绝 symlink/path escape 和 `.tmp/.partial/.part` 残留；
- 校验 train/validation/test manifests 和固定 shard ranges。

### 12.3 Tests

全仓库共有 127 tests，最终全部通过。覆盖范围包括：

- 20 个数学 operations；
- Passage Math grammar 和 canonical encodings；
- deterministic generation、balance、resume 和 corruption；
- full truth recomputation；
- split views；
- Transformer/MLP causality、padding、gradients、serialization 和 CUDA；
- streaming loader、answer-only collator、training、resume 和 markers；
- experiment matrix；
- adversarial completion audit。

唯一 warnings 是 Python 3.13 multi-threaded process 使用 `fork()` 的 deprecation warnings，不是测试失败或训练数值问题。

## 13. 从零复现

以下命令均在 repository root 执行。

### 13.1 Environment

```bash
git clone https://github.com/XuanyuYang223/neurips.git
cd neurips

# Optional: freeze the audited post-training code/results baseline exactly.
git checkout 32ff22a2e77acdf1d18b634ed431e54d3c1341f0

# Python >=3.11 is recommended because the orchestration code uses tomllib.
python3.13 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip setuptools
python -m pip install -e '.[test,train]'
python -m pytest -q
```

`pyproject.toml` 目前声明 Python ≥3.10，但 `experiments.py` 和 `audit.py` 直接使用标准库 `tomllib`；不安装 backport 时应实际使用 Python ≥3.11。所有冻结路径相对于 repository root，以下命令都应从该目录运行。

### 13.2 生成与验证 10M 数据

```bash
python -m neurips_permutations.generate \
  --count 10000000 \
  --max-entries 30 \
  --base 100 \
  --seed 20260830 \
  --shard-size 100000 \
  --workers 20 \
  --schema-version permutation-20/v2 \
  --output-dir data/permutation-10m-v2

python -m neurips_permutations.verify \
  data/permutation-10m-v2/manifest.json \
  --full \
  --workers 20

python -m neurips_permutations.splits \
  data/permutation-10m-v2/manifest.json \
  --train-shards 98 \
  --validation-shards 1 \
  --test-shards 1
```

正式 TOML 有意让 training 和 validation 都引用 parent manifest，再用 `000-097` 与 `098` 的 shard indices 选择数据；不要将这两个路径随意换成 split-manifest 路径，否则 config hash 和严格审计会改变。

当前 split-manifest SHA-256：

```text
train       76e682a8afb217350fbe4454eb473593f2cf53850254f826697faf6fa0349de3
validation  6bdc14e4363c2b8a0d74d389543d5260ef28597537cb578e8c37b4a0284693ef
test        9f9822b0dbac51af8c40d57fa5df12237ba1893c788582f5c1898f5ca33ed2da
```

### 13.3 检查实验矩阵

```bash
python -m neurips_permutations.experiments \
  --config configs/henry_permutation.toml \
  --plan

python -m neurips_permutations.experiments \
  --config configs/henry_permutation.toml \
  --run \
  --dry-run
```

### 13.4 训练全部 30 runs

最安全的复现方式是使用一个 controller 顺序运行：

```bash
python -m neurips_permutations.experiments \
  --config configs/henry_permutation.toml \
  --run
```

重新执行同一命令时：

- valid `completed.json` runs 会跳过；
- incomplete runs 会从 `checkpoint.pt` 自动恢复；
- config 或 manifest hash 不匹配会立即失败，而不会混合实验。

当前 runner 是单进程、单 GPU 训练，不是 DDP；不要直接改成 `torchrun` 后仍把结果视为同一 frozen protocol。正式复现应只暴露一张支持 bf16 的 GPU，例如 `export CUDA_VISIBLE_DEVICES=0`。

也可以用一个或多个精确、互不重叠的 run IDs：

```bash
python -m neurips_permutations.experiments \
  --config configs/henry_permutation.toml \
  --run \
  --only transformer-tasks16-seed17
```

不要让两个 controllers 同时运行同一 run ID，因为 run directory 没有跨进程 lock。

每个 run 只保留一个滚动 `checkpoint.pt`；每 1,000 steps 的历史 checkpoint 不会全部保留。因此这套 artifacts 支持恢复和最终审计，但不能事后重建完整 learning curve 或重新选择 best intermediate checkpoint。

### 13.5 Status 与严格审计

```bash
python -m neurips_permutations.experiments \
  --config configs/henry_permutation.toml \
  --status

python -m neurips_permutations.audit \
  --config configs/henry_permutation.toml
```

## 14. Repository 与 artifact policy

GitHub public repository 保存：

- source code；
- frozen configs；
- tests 和 CI；
- protocol、experiment 和 results documents。

GitHub 不保存：

- 1.29 GB v2 production dataset；
- 1,139,175,228-byte v3 production dataset；
- 1.15 GB formal checkpoints；
- pilot/checkpoint runtime directories。

本机路径：

```text
/home/yangx/neurips/data/permutation-10m-v2
/home/yangx/neurips/data/permutation-10m-v3
/home/yangx/neurips/runs/henry-permutation
```

如需共享这些 artifacts，应使用 object storage、dataset hosting、GitHub Release assets 或专门的 model registry，而不是普通 Git blobs。

## 15. 已知限制与下一步

`permutation-20/v3` 的 manifest、split manifests 和 full verification 已完成，schema-aware nested runner 的 30-run plan/dry-run 也已复核且没有启动训练。论文主实验的下一优先级是补充 E4/S4/A4 category orchestration，再决定并启动两套 revised matrices。旧 v2 的 30 个模型和下列分析仍作为 baseline/appendix 保留，不会被删除或重命名成 v3 结果。

1. **Independent test**：冻结 evaluator 后，只在 shard 099 上执行一次最终测试，并用 explicit greedy decoding 对 sequence metric 做实现级一致性检查。
2. **Few-shot generalization**：在每个 fixed holdout 上使用 Henry 建议的 20 samples 和低 learning rate fine-tune 30 个 base models；可再增加 5-shot 与 100-shot curves。
3. **Random-init baseline**：使用完全相同 architecture、seed、shots、steps 和 optimizer budget，从随机初始化训练，比较 adaptation gain。
4. **Linear probes**：优先在 `<ONE_END>` 等 task token 出现前的位置抽取逐层 hidden states，避免 answer leakage，也减少未 grounding holdout token 的影响。
5. **Representation geometry**：比较 layerwise CKA/SVCCA、Procrustes、effective rank、clustering 或 representational similarity。
6. **Multiple task orders**：当前三个 seeds 只覆盖 initialization/shuffle variation；task subset order 只抽样了一次，因此误差条不包含“选择了哪些 tasks”的方差。
7. **Fixed-budget interpretation**：task 数增加时，每个 task 的 exposure 从约 1.28M 降到 80k；当前设计同时改变 diversity 与 per-task data，16-task 回落不能直接归因于 interference。
8. **Metric granularity**：token accuracy 包含 delimiters、copy tokens 和 EOS；每 task 只有 47–160 个 validation examples，输出长度与准确率量化粒度差异很大，应同时报告逐任务 exact accuracy。
9. **Distribution scope**：validation 与 train 都是 `n=2–30` 的同分布新 shards；尚未测试 `n>30` size extrapolation、组合分布迁移或 cross-representation transfer。
10. **Architecture matching**：Transformer 有 3.463M registered parameters，MLP 有 2.930M，前者多约 18.2%；但 MLP 中 1,047,552 个 strict upper-triangular token-mixing parameters 会被 mask 且不参与 forward，所以 nominal count 也不能视为 active-capacity matching。两种 architecture 没有分别进行完整 hyperparameter tuning，因此只能描述差异，不能作强因果结论。
11. **Statistical scope**：只有三个 seeds，未做 task-level bootstrap 或 significance test；MLP parity 等非零值会被单个 seed 驱动。
12. **Opaque holdout tokens**：hard zero-shot 无法从 token 本身推断未见 operation 的含义；需要共享语义、task descriptions、cross-representation combinations 或 few-shot supervision。
13. **Determinism**：seed、data order 和 sharding 是确定的，但没有启用 `torch.use_deterministic_algorithms`；跨 GPU、CUDA 或 PyTorch 版本不承诺 bitwise-identical weights。
14. **Environment provenance**：checkpoint/marker 没有内嵌 Git commit、Python、Torch、CUDA 或 driver version；本文件记录了本次环境，但未来协议应把它们写入 marker。
15. **4×8 representation grid**：若要回答第二份附件的 representation transfer 问题，需要让 cycle、Lehmer 和 inversion-vector 也作为 primary inputs，再训练指定 11 个 combinations 并测试其余 21 个。

## 16. 关键文件索引

- [README.md](README.md)：快速开始。
- [PROTOCOL.md](PROTOCOL.md)：20-task 数学与数据协议。
- [EXPERIMENTS.md](EXPERIMENTS.md)：Henry nested matrix 概览。
- [TRAINING_RESULTS.md](TRAINING_RESULTS.md)：最终 validation 与 generalization 数表。
- [configs/henry_permutation.toml](configs/henry_permutation.toml)：冻结实验配置。
- [configs/henry_permutation_revised.toml](configs/henry_permutation_revised.toml)：Henry 反馈后的 v3 设计（尚未训练）。
- [generate.py](src/neurips_permutations/generate.py)：数据生成。
- [verify.py](src/neurips_permutations/verify.py)：独立数据验证。
- [passage.py](src/neurips_permutations/passage.py)：tokenizer 与 Passage Math grammar。
- [models.py](src/neurips_permutations/models.py)：Transformer 与 causal MLP。
- [training.py](src/neurips_permutations/training.py)：streaming training 和 checkpoint/resume。
- [experiments.py](src/neurips_permutations/experiments.py)：30-run orchestration。
- [audit.py](src/neurips_permutations/audit.py)：严格完训审计。
