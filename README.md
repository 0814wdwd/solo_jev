# Structure-Aware Relational Analysis for Decision Models

Extending SOLO's data reordering and structural-reuse ideas to decision-model inference
over large tables.

Running a hundred thousand semantic judgements over one table: efficiency depends not
only on how fast the model is, but on how the data is organised, shared and fed to it.

SOLO-Jev takes SOLO as its methodological basis and organises row/column reordering,
task-aware encoding, request packing and decision calibration into a single pipeline for
table inference. It exploits the repeated structure in relational data to reduce input
and request overhead, while using key-field retention and quality evaluation to control
the effect of compression and batching on the judgements.

In one scan of nearly a hundred thousand rows against a public API, the system processed
99,400 rows in 1,657 requests, taking 543 seconds. That result comes from a complete run
of a single predicate on the flight table; the configuration, quality metrics and
comparison basis are given below.

**Structure-aware planning · Task-aware compression · Budget-constrained packing ·
Decision-quality evaluation**

Project stage: research prototype. This repository is aimed at method reproduction,
execution optimisation and experimental evaluation.

---

## Background

SOLO studies a problem that is easy to overlook: for the same relational table,
different data arrangements lead to different model execution costs.

In conventional LLM serving, SOLO reorders a table's rows and columns so that adjacent
requests share a longer prefix, increasing prefix-cache reuse. What it exploits is not a
change in the text content, but the repeated structure of the relational data itself.

SOLO-Jev extends this idea to decision models. Jev can handle multiple questions against
one shared state, so the same relational data need not be split into independent per-row
requests; it can instead be organised into input blocks that share information.

Under the OpenRouter/Jev configuration tested, the experiments observed no cross-request
prefix-cache benefit, so this project optimises around information reuse within a
request, rather than relying on server-side cache acceleration.

### The structural statistic

A key statistic in SOLO is G^(k): the number of distinct value combinations formed by the
first k columns under a given column order. It characterises the shareable prefix
structure in the table.

Under the corresponding sorting and elision encoding, this structure also determines how
many times a field has to be written out explicitly; its normalised form G^(k)/N
characterises the explicit-write ratio. Row/column planning therefore becomes more than a
change of input order: it can also be used to analyse representation cost, and to
determine which fields need particular protection after compression.

SOLO provides the methodological basis for organising reusable structure; SOLO-Jev adds
input representation, request execution and result calibration around the decision model.

---

## Pipeline

SOLO-Jev treats a table scan as an execution pipeline that needs planning and evaluation,
not merely a repeated model call for every row.

```
relational table + predicate definitions
        ↓
field projection and SOLO row/column planning
        ↓
task-aware encoding: reuse repeated information, keep key fields explicit
        ↓
budget-constrained blocking and request packing
        ↓
Jev shared-state decision inference
        ↓
per-predicate thresholds, probability calibration and result evaluation
```

### Structure-aware planning

Using the value distribution and prefix-combination structure of the relational table,
plan the row and column order so that reusable information forms a contiguous shared
structure in the encoding.

The planner integer-encodes each column in lexical value order and selects the
smallest shared unsigned dtype (`uint8`, `uint16`, `uint32` or `uint64`) needed
by the column cardinalities. Candidate columns are evaluated by scanning
contiguous prefix groups with a reusable value-indexed stamp array. Only the
chosen column updates the groups and row permutation, using stable counting
partitions. Full-table greedy planning reuses that permutation; sampled planning
partitions all rows separately under the selected column order. Both preserve
the original stable string-lexicographic order and serialization.

After value encoding, exact greedy planning takes O(M²N) time and O(N + M)
auxiliary space, in addition to the O(NM) encoded table. Fixed-order row
partitioning takes O(MN) time. These bounds cover the integer grouping stage;
encoding still reads the original strings and sorts each column's distinct
values. Numba compiles and caches the scans, so the first call for a new array
signature also includes compilation or cache-loading overhead.

The accompanying cost model measures the actually rendered input rather than merely
comparing raw string lengths. In the reported validation settings on the flight, movies
and synthetic low-cardinality tables, the error between predicted cost and actual billing
is 0.02–1.45%.

### Task-aware compression

The goal of compression is not only to reduce tokens, but to let the model locate the
evidence for the current row's judgement.

SOLO-Jev distinguishes inputs by the field dependencies a predicate declares: repeated
non-critical content can be shared; the fields required for the judgement are kept
explicit on every row through pinning; and field labels further let the model identify
which column a value belongs to, rather than relying on positional counting alone.

This design addresses two problems at once: whether the required information is present,
and whether it can be identified correctly. In the taxi_sum diagnostic experiment on the
flight table, adding labels to already-pinned fields raised balanced accuracy from 88.8%
to 98.7%, with input tokens about 41% of the row_kv encoding at the same block size.

### Budget-constrained packing

Combine multiple rows and their questions into requests that satisfy the input budget,
amortising the per-request fixed overhead. In the original experimental configuration,
each request carries about 260 fixed billed tokens, so per-row requests pay that cost
repeatedly.

Block size affects both throughput and judgement quality. The system's experimental
pipeline supports comparing different block sizes rather than equating "filling the
context" with the optimal configuration. The near-hundred-thousand-row run below uses
60-row blocks; larger blocks are used in other throughput and accuracy experiments.

### Decision-quality evaluation

The model's probability and the final verdict are two different output layers. SOLO-Jev
evaluates the decision threshold separately per predicate, and checks the effect of the
threshold and of probability calibration on held-out data.

Across 84 experimental cells on the flight table, fitting the threshold against balanced
accuracy raised overall balanced accuracy from 86.6% to 89.3%. In a separate set of
probability-calibration experiments, Platt scaling reduced held-out ECE from 0.097 to
0.025, with a corresponding AUC of 0.954.

Threshold and probability calibration are evaluable configuration items: the benefit
varies by predicate and needs to be confirmed on held-out data.

---

## Results

The following results answer three separate questions: execution efficiency, input
representation and task quality. The configurations across the tables are not identical,
and the best value in each should not be read as achieved simultaneously under one
configuration.

### Execution at near-hundred-thousand-row scale

The subject is one predicate on the flight table, executed in full through the public API
with 60-row blocks.

| item | value |
|---|---|
| rows | 99,400 |
| requests | 1,657 |
| billed input tokens | 3,929,704 (39.5 per row) |
| recorded cost | $0.1650 |
| wall clock | 543.0 s (183 rows/s) |
| balanced accuracy at 0.5 | 100.0% |
| rows without an answer | 0 |

Compared with one request per row, the number of requests is reduced by about 60x.
Against a separately measured per-row baseline throughput of 3.1 rows/s, the throughput
ratio is about 59x. The comparison basis is an independently measured per-row processing
rate; what was executed in full at the 99,400-row scale is the packed scheme.

This run validates the execution behaviour of one predicate at near-hundred-thousand-row
scale; it does not imply that all semantic tasks reach the same accuracy. The cost is the
actual record of the original experiment, not a commitment about current service pricing.

### Input encodings

On two real table structures, flight and movies, the experiments compared different input
encodings. The token ratios below are relative to row_kv on the corresponding table, and
the accuracies are the balanced accuracies reported in the original experiments.

| encoding | flight accuracy | flight tokens | movies accuracy | movies tokens |
|---|---|---|---|---|
| `row_kv` | 96.8% | 100% | 98.5% | 100% |
| `csv_block` | 95.8% | 54% | 98.1% | 74% |
| `csv_rle` | 88.2% | 41% | 97.0% | 65% |
| `csv_rle` + pinning | 94.9% | 41% | 98.6% | 65% |

This set of results supports using task-aware retention to improve judgement quality
after compression, but the size of the saving depends on the table structure. A 0.1-point
difference on the movies table should not be interpreted as a stable accuracy improvement.

The row_kv here also uses blocked input; it is not the baseline that processes one row per
request. The two kinds of comparison answer different questions.

In a separate ablation on the flight table, after batching, reducing repeated column
labels and run-length compression, SOLO reordering further reduced token overhead from
19.4% to 17.6% of the per-row baseline, i.e. about 9% less on top of the existing
compression. The benefit of the full pipeline comes from a combination of stages; the
incremental benefit of reordering is characterised separately by this ablation.

### Structural diagnosis

When a field is rarely written out explicitly, the model has to recover its value across
rows. Structural diagnostic experiments show that this kind of compression can damage
judgement quality, and that explicitly retaining task-relevant fields improves the result
significantly.

| predicate (movies) | explicit-write ratio | `csv_rle` | `csv_rle` + pinning |
|---|---|---|---|
| `lang_en` | 23% | 93.1% | 100.0% |
| `is_action` | 2% | 62.8% | 100.0% |

The movies experiment above supplies the flag column required for the judgement as input,
in order to examine the representation; it differs from the semantic experiment in the
next section, which hides the genre label and judges the genre from the title alone.

Across 52 paired experimental cells over two real table structures and two sampling
settings, the original experiments observed no case where pinning caused more than a
one-point loss of accuracy. This supports treating it as the encoding configuration to
evaluate first, but does not constitute a lossless guarantee for arbitrary tasks.

---

## Applicable tasks

SOLO-Jev targets per-row analysis that requires model knowledge or semantic judgement —
for example judging a film's genre from its title, judging which queue a support ticket
belongs to, or checking the semantic consistency of a text record. Of these, this document
provides measured results for film-genre judgement; the other examples describe task
shapes.

The movies experiment hides the genre flag, supplies only the title to the model, and
compares against a TF-IDF classifier over the same titles:

| predicate | base rate | TF-IDF | Jev |
|---|---|---|---|
| is it an action film | 0.443 | 52.9% | 83.2% |
| is it a comedy | 0.335 | 58.6% | 90.9% |
| is it a drama | 0.338 | 56.5% | 79.0% |
| is it a horror film | 0.044 | 50.0% | 90.6% |

The TF-IDF and Jev columns report balanced accuracy; the average input overhead for this
set of experiments is 34 tokens per row. It is used to show the decision backend's ability
on tasks that require additional knowledge, not to measure SOLO reordering's contribution
on its own.

Deterministic conditions should still be handled by SQL or local code. For example
`ArrDelay > DepDelay` and `revenue > budget` can already be computed directly from the
input columns. This project uses such predicates for structural, encoding and
numeric-judgement diagnostics because they provide unambiguous ground truth; they are not
application scenarios that require calling a model.

---

## Installation and reproduction

This project is positioned as a research prototype, containing offline planning and
encoding modules, real-API probe scripts and result-analysis entry points. The offline
installation and reproduction flow is listed first; for the availability of the unified
scan interface and online calibration, refer to the implementation of the version in use.

From the repository root:

```
python -m pip install -e ".[dev]"
python -m pytest tests/ -q
```

The offline parts use numpy, numba and tiktoken, and need no API key or network
access once the dependencies and tokenizer assets are installed.

```
python3 -m jev_solo.bench_offline --csv table.csv --rows 5000
```

The planner-only benchmark compares stable outputs, elapsed time and traced
allocation peaks with the sorting-based implementation from commit `65b0a21`:

```
python3 probes/bench_integer_planning.py --out results/integer_planning.json
```

It runs one process with a 2 GiB address-space limit, using 20,000 rows and 20
columns by default. Tables are capped at 20,000 rows and 40 columns. Steady-state
times exclude the separately recorded first call; the process peak RSS also
includes imports and compilation/cache loading. These synthetic measurements
cover planning and materialization, not model inference.

```
# decision threshold: fitted per experimental configuration, averaged over several splits
python3 -m jev_solo.threshold

# probability calibration: ECE, temperature scaling and Platt calibration
python3 -m jev_solo.calibration

# cross-table structural comparison
python3 -m jev_solo.cross_table

# check local classifier baselines
python3 probes/baseline_classifiers.py
```

The result-analysis entry points require the corresponding experimental records; the
commands themselves do not replace data preparation or online measurement.

### Defining your own table

Define a `TableSpec` in `jev_solo/datasets.py`, configuring the data file, the input
projection and the predicates, and provide a `truth` function for evaluation.

When the ground-truth label lives in a column the model should not see, use `truth_cols`
to separate the evaluation label from the model input, avoiding label leakage.

### Connecting to a model

Configure the key in `.env.local` following `.env.local.example`. The original experiments
called `typesafe/jev-1.13` through OpenRouter; the connection is managed by
`probes/jev_client.py`.

`probes/mock_server.py` provides a key-free protocol mock service that can be used to
check the request and response flow; it is not for validating a real model's accuracy or
throughput.

---

## Repository layout

```
jev_solo/
  datasets.py          tables, field projections and predicate definitions
  tokens.py            token measurement and rendered-field weights
  objective.py         SOLO structural statistics and the analytic cost model
  plan.py              row/column planning implementations
  encodings.py         state encodings and key-field retention
  pack.py              request packing under the input budget
  bench_offline.py     offline cost evaluation
  threshold.py         decision-threshold analysis
  calibration.py       probability-calibration analysis
  analyze_accuracy.py  accuracy-cost comparison and mechanism diagnosis
  cross_table.py       cross-table structural evaluation

probes/                real API, protocol mock and experimental probes
results/               raw experimental records
tests/                 serialization and related regression tests
```

These modules cover structural planning, input representation, execution budget and result
evaluation respectively.

---

## Limitations

The current evidence covers two real table structures, flight and movies, plus a synthetic
table used to control cardinality. The main online results come from the same Jev backend
accessed through OpenRouter; they cannot yet establish that the findings hold for all
decision models, data distributions or tasks.

Compression and batching involve a quality trade-off. Pinning and field labels improve the
identifiability of the input, but do not eliminate all multi-row judgement error. In one
setting of `delay_gt`, a 30-row block achieved 96.5% balanced accuracy at 56.2 tokens per
row, while single-row requests achieved 100.0% at 338.0 tokens per row.

Evaluation must account for model variability. Repeated requests, question order and
row-ID changes can all affect the result; in the original experiments, differences smaller
than about 1.5 points on some fitted metrics approach the observed noise level.
Configuration choices should therefore combine repeated measurement, held-out evaluation
and the task's tolerance for error, rather than comparing single best values.

Use of an external API must also comply with data-access permissions and privacy
requirements. Sensitive data should not be sent to an unapproved service merely because it
can be compressed and packed.

---

## Relation to SOLO

The methodological basis of this project comes from SOLO:

> Prefix-Cache-Aware Data Reordering for LLM-Augmented Database Analytics.

SOLO-Jev focuses on the extended implementation and experimental evaluation of this
data-organisation idea in a decision-model execution setting. The original SOLO paper code
and this repository are maintained separately.

When citing this project, please also cite the original SOLO research and the repository
version used.
