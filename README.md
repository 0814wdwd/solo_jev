# jev-solo

Run a per-row judgement over a large relational table on a **decision model**, without
paying to write the same text down a million times — and without silently trading
away the answers' correctness to do it.

Status: early (v0.1.0). The offline half — encodings, the cost model, planning,
packing — is installable and tested. The end-to-end pipeline is not packaged yet; see
[Next](#next). Everything below is measured against the real API; nothing is quoted
from a vendor page unless it says so.

```bash
pip install -e ".[dev]"     # offline parts need only numpy + tiktoken
python -m pytest tests/ -q
```

---

## What this is, in plain terms

You have a table with hundreds of thousands of rows, and you want to ask every row a
question that SQL cannot answer: *is this record internally consistent? does this row
contradict itself? is this complaint about billing?* SQL does not judge, so a model
has to look at each row.

The naive way is one request per row. A hundred thousand rows is a hundred thousand
requests — slow, and enormously repetitive, because those requests are nearly
identical to one another.

**Jev** (TypeSafe AI, released September 2026) is a new kind of model aimed exactly at
this. It does not write text. You hand it a blob of data and a list of typed
questions, and it hands back numbers — probabilities and typed answers — evaluating
every question in one parallel pass. It is roughly two orders of magnitude cheaper
and faster than a frontier LLM at repetitive decision work, and **output tokens are
free**: you pay only for what you send in.

Think of it as a judge with a checklist rather than a writer. You pass it a page of
records and a list of yes/no questions; it passes back a score sheet.

**SOLO** ([ICML 2026](#citing)) attacked the same repetition on conventional LLM
serving. Those systems cache the *beginning* of a prompt they just processed, so if
the next prompt starts with the same text the model can skip re-reading it. SOLO
reorders a table's rows and columns so neighbouring prompts share the longest possible
common beginning, and the cache hits far more often. Like sorting a stack of forms so
consecutive forms have identical top halves, letting the clerk say "same as above".

### How the three relate — which is not what we expected

Jev looks tailor-made for SOLO's problem, so the obvious move is to port SOLO to Jev.
We measured, and the port does not work the way you would guess. That finding is what
this project is:

1. **SOLO's engine does not exist on Jev.** There is no prefix cache at all. Four
   conditions — identical state, edited tail, edited head, brand new state — eight
   repeats each: the bill and the latency are indistinguishable (4323 vs 4324 tokens,
   291 vs 288 ms). Sorting rows to hit a cache is pointless when there is no cache.

2. **The waste did not disappear; it moved.** Jev charges for what you send, and lets
   you put many rows and many questions into one request. So the waste is no longer
   "recomputing a prefix" — it is "writing the same thing down over and over inside
   the request you send". Same disease, different organ.

3. **SOLO's mathematics survived the move and turned out to be the right tool.** Its
   core quantity — how many distinct value-combinations the first *k* columns form —
   is exactly the number of times a value still has to be written down if repeats may
   be implied. The reordering machinery transfers; we verified the resulting formula
   predicts the real bill to within **0.5%**.

**Jev killed SOLO's mechanism, inherited SOLO's problem, and SOLO's mathematics turned
out to fit the new form of that problem.** This project is what you build once you
know that.

---

## What it does

**1. Pack.** Stop sending one request per row. Every request carries ~260 tokens of
fixed overhead no matter how small it is, and that is billed per request. Packing a
scan into budget-feasible requests turned 5,000 requests into 27.

**2. Compress, carefully.** Inside a request, stop repeating what you need not repeat:
write column names once, and where a value repeats down a sorted column, leave it
blank to mean "same as above". SOLO's reordering earns its place here — it maximises
how much can be left blank.

**3. Read the output properly.** The model returns probabilities, not verdicts. Its
probabilities are systematically shifted, so the obvious "above 0.5 means yes" gives
wrong answers. Fitting the cutoff per predicate and recalibrating the probabilities
fixes that at zero extra API cost.

---

## The core finding

**Compression and correctness fight each other, in a precise and predictable way.**

If a column's repeated values are elided, the model has to look far up the page to
recover the value — and it gets it wrong. The predictor is a column's **emit
fraction**: how often its value is actually written down, which for sorted rows is
exactly SOLO's G^(k)/N.

- emitted on **100%** of rows → compression costs nothing
- emitted on **23%** → accuracy starts to slip
- emitted on **2%** → balanced accuracy collapses from ~100% to **60-63%**, barely
  better than guessing

The better a column compresses, the less often its value sits near the row that needs
it. Compression and failure are the same phenomenon.

**The fix is cheap and absolute: never elide the columns the question actually reads.**
Compress everything else. It costs **under 1% more tokens** and restores accuracy in
full — 62.8% → 100% on our worst case.

> You may abbreviate the parts of a form nobody will check. Not the box the inspector
> is reading.

Verified on two unrelated schemas, both sampling regimes, **52 paired cells, harmful in
zero of them.** So the library should pin by default, and a caller never has to reason
about emit fraction to be safe.

---

## Why you would use it

**Throughput, not cost — this is the real pitch.** The money is irrelevant: scanning
100k rows costs about $5.17 naively and $0.97 optimised, and nobody starts a project to
save $4. But the provider caps you at **1,200 requests/minute**. One request per row
means a 100k-row table spends **83 minutes** just clearing the request quota. Packed,
it is **under 2 minutes** — roughly **50x more throughput against a hard external limit
you cannot buy your way past**. (Projected from 5,000-row measurements; not yet run at
100k.)

**It makes aggressive compression safe.** Without the pinning rule, compression wrecks
accuracy *silently* — and silence is the danger, because nothing errors out. You get
back a set of perfectly plausible probabilities that happen to be wrong. This gives a
rule that prevents it and a diagnostic that predicts it before you spend anything.

**It behaves predictably on tables you have never measured.** Pinning cuts
cross-schema accuracy spread from 6.5–8.8 points down to 2.0–3.7. For a default
shipped to other people that is worth more than the average gain.

**Free accuracy from reading the output correctly.** +2.7 points overall and +5.7 on
selective predicates, with zero extra API calls. It also documents the trap: tune the
cutoff for raw *accuracy* and it collapses to "always no" on 45% of runs for selective
predicates — excellent on an accuracy metric, useless in practice.

**The probabilities are usable** after an affine recalibration (ECE 0.10 → 0.025, AUC
0.954), which keeps selectivity estimation on the table. This part is preliminary.

### Results worth knowing even if you never use this

Three things we tried that did **not** work, so nobody has to repeat them:

- **Splitting one question into several does not help.** A published result gets 62.6%
  → 95% on phishing that way. On relational predicates it does not reproduce: the
  ensemble is no better than asking once, and costs 27% more. Its apparent gain was its
  combiner quietly fixing calibration — which a fitted threshold does better and free.
- **Having the model read values into bands and computing in code is worse**: −6 points
  and +66% tokens. Ordinal bands are lossy exactly where a comparison is decided.
- **The cheapest-looking encoding is unusable.** Transposing into column-major with run
  lengths wins on tokens and scores near chance (64.1% balanced accuracy). Cheap tokens
  are worthless if the answers are noise.

---

## Quickstart

Offline — no API key, no network. Costs a table's worth of input tokens under the
measured billing model and tells you what a scan would cost:

```bash
python3 -m jev_solo.bench_offline \
  --csv /path/to/table.csv --rows 5000 \
  --encodings row_kv,csv_block,csv_rle,columnar_rle,factored_rle \
  --out results/mytable.json

python3 -m jev_solo.rescore --bench results/mytable.json \
                            --tokenize results/or_tokenize.json
```

Against the real API — put a key in `.env.local` (see `.env.local.example`):

```bash
# reachable through OpenRouter as typesafe/jev-1.13
python3 probes/probe_accuracy_scaled.py --table movies --random-sample
python3 -m jev_solo.cross_table          # is the rule schema-specific?
python3 -m jev_solo.threshold            # offline; fitted decision thresholds
python3 -m jev_solo.calibration          # ECE, noise floor, Platt refit
```

Probes can also run with **no key at all** against `probes/mock_server.py`, a stdlib
mock of the wire protocol, or against a self-hosted stand-in (`scripts/setup_jeff.sh`).

**Bring your own table** by adding a `TableSpec` to `jev_solo/datasets.py`: the file,
the projection, and predicates whose ground truth is computed from the table itself.
That is the interface — the probes take `--table` and never hardcode a schema.

---

## What we measured

All figures below come from **784 calls, $0.55 total**, against `typesafe/jev-1.13`
via OpenRouter. Accuracy figures use balanced accuracy with thresholds fitted per cell
on held-out halves, averaged over 10–20 train/test splits — a single split moves a cell
by 1–2 points, enough to flip conclusions, so single-split numbers are not reported.

### The billing model

```
billed  ~=  260 per request  +  slope_enc x state_tokens  +  (9 + instruction) per question
```

- **260 tokens of fixed overhead per request**, measured near zero rather than
  extrapolated (a naive regression intercept says 424, which is wrong).
- **9.00 tokens per extra question**, exactly linear over n = 1…32, plus the
  instruction's own tokens. Keep per-row predicates terse.
- `slope_enc` is per encoding (1.29–1.50 against cl100k), R² = 0.998.
- `cost == input_tokens/1e6 × $0.042` exactly; output free. Vendor pricing confirmed.
- Warm latency **239–316 ms**, flat in state size to 31k tokens. A first-ever call took
  31 s; that is provisioning, not steady state.
- **State as a JSON object costs 2.8×** the same content as a string. Send strings.
- Documented budgets: 64k tokens per request, 32k for state plus the longest question.

### No cross-request cache, and no question-count ceiling

Cache: billed tokens 4323 vs 4324 (ratio 1.000), latency 291 vs 288 ms, across the four
conditions above. Neither channel shows a hit on either layer.

Questions: from 1 to 512 nouls against a fixed 120-row state, latency goes 298 → 420 ms
(1.41×) while per-decision latency falls 298 → **0.82 ms**, a 363× improvement. Pack as
many rows per request as the 32k state budget allows.

### Where the savings come from (flight, 5,000 rows × 110 cols)

Baseline is one request per row: 5,000 requests, 6,167,773 billed tokens, $0.2590 — of
which **21.1% is pure per-request overhead**.

| step | % of baseline | cumulative |
|---|---|---|
| batching rows at all | 79.7% | 1.25× |
| + dropping per-row column labels | 24.4% | 4.1× |
| + run-length compression | 19.4% | 5.2× |
| + reordering | 17.6% | **5.70×** |

Requests: 5,000 → 29 (172× fewer). **Reordering — SOLO's actual contribution — is the
smallest of the three effects, about 1.1×.** We would rather say that than oversell the
lineage.

### Accuracy vs cost

7 predicates × 7 encodings × 360 rows per cell, two redundancy regimes (consecutive
rows from the sorted file; random rows):

| encoding | flight | movies | tokens (flight / movies) | note |
|---|---|---|---|---|
| `row_kv` (SOLO's prompt shape) | 96.8% | 98.5% | 100% / 100% | accuracy ceiling |
| `csv_block` | 95.8% | 98.1% | 54% / 74% | no ditto, no risk |
| **`csv_rle` + pin** | **94.9%** | **98.6%** | **41% / 65%** | **recommended default** |
| `columnar_rle` + pin | 94.4% | 96.4% | 45% / 69% | best worst case (87.4%) |
| `factored_rle` | 91.5% | 98.0% | 67% / 79% | |
| `csv_rle` (no pin) | 88.2% | 97.0% | 41% / 65% | unsafe without pin |
| `columnar_rle` | 63.9% | 67.2% | 42% / 66% | **unusable** |

`csv_rle`+pin is Pareto-optimal on both tables and on movies edges out `row_kv`
outright. **The policy transfers; the size of the saving does not** — 41% of `row_kv`
tokens on flight, 65% on movies, because fewer columns and wider values mean per-row
labels are a smaller share of the bill. Ship the encoding default, measure the saving.

### The mechanism, with a control that could have falsified it

Pinning can only matter where columns are actually being elided, so it must be a large
win at low emit fraction and a no-op at high. On each table independently:

| table | emit < 25% | emit ≥ 75% |
|---|---|---|
| flight | 75.3 → 98.9 (**+23.6**) | 92.5 → 93.6 (+1.1) |
| movies | 90.7 → 96.0 (**+5.3**) | 99.5 → 99.6 (+0.2) |

Per predicate the effect is sharper than the bucket means: `is_action` on movies, whose
flag is emitted on 2% of rows, goes **62.8% → 100%**. `columnar_rle`+pin gains at
*both* emit levels on both tables, which correctly separates its defect — the
transposition, not the ditto.

### The decision threshold

Over all 84 flight cells:

| rule | balanced acc | accuracy | cells collapsing to one class |
|---|---|---|---|
| `p ≥ 0.5` | 86.6% | 88.6% | 2/84 |
| fitted for **accuracy** | 87.0% | 92.7% | **4/84** |
| fitted for **balanced accuracy** | **89.3%** | 89.1% | **0/84** |

On the 14 cells with base rate < 0.15 — the normal case for a `WHERE` clause — the
balanced objective is what rescues it: 75.3% → 81.0% with no collapses, against 73.5%
and three collapses for the accuracy objective. Fitted thresholds span **0.09 to 0.99**
(mean 0.65), so no single global cutoff serves a workload, and 0.5 is nowhere near the
centre.

The worst case makes the trap concrete. `state_eq` in the high-redundancy regime, base
rate 0.128, averaged over 20 splits: fitting for accuracy gives the table's best
accuracy (88.2%) and its worst classifier — it answers "no" to everything on **45%** of
splits. Fitting for balanced accuracy: 71.5%, never collapsing.

### Calibration

Pooled over 12,600 non-degenerate predictions: **ECE 0.100 against a resampling noise
floor of 0.007**, and the model **over-predicts** (mean 0.357 vs base rate 0.263). That
independently reproduces a published third-party figure (0.107 against a 0.024 floor) on
a different task.

Miscalibration tracks compression more steeply than accuracy does — `row_kv` 0.063 →
`columnar_rle` 0.197, with pinning recovering much of it (→ 0.113).

Temperature alone barely helps (T = 0.70, ECE 0.097 → 0.075), because what is wrong is a
*bias* and no temperature moves a bias. Adding the intercept fixes it: Platt scaling
`a = 1.35, b = −1.10` takes held-out ECE **0.097 → 0.025** with **AUC 0.954**. The
ranking is sound; the mapping is skewed.

---

## End to end, both arms measured

Everything above measures a component. This is the whole path — table, pack, encode,
send, threshold, verdicts — with a per-row baseline run under the *same* client
rate-limit policy, so the comparison is observed rather than computed. flight, 5,000
rows per predicate, baseline sampled across the sorted order:

| predicate | arm | requests → 100k rows | rows/s | bal@0.5 | bal@fitted |
|---|---|---|---|---|---|
| `delay_gt` | baseline | 100,000 | 3.3 | **100.0%** | 100.0% |
| `delay_gt` | **packed** | **840** | **320** | 81.0% | **95.4%** |
| `weekend` | baseline | 100,000 | 3.4 | 100.0% | 100.0% |
| `weekend` | **packed** | **840** | **310** | **100.0%** | **100.0%** |
| `state_eq` | baseline | 100,000 | 3.5 | 100.0% | 100.0% |
| `state_eq` | **packed** | **840** | **323** | 96.7% | **99.3%** |

Scaling each arm's measured per-row rate to 100k rows: **~98× faster, 6.4× fewer
tokens, 119× fewer requests**, $2.05 → $0.32.

**A correction the component numbers could not have caught.** The per-row baseline
scores **100% on all three predicates** — better than any blocked encoding. The
earlier frontier compared encodings at a fixed 120-row block on *both* sides, so it
understated the gap to what one-row-per-request actually achieves. The honest
end-to-end trade is **~98× throughput for 0 to 4.6 points of balanced accuracy,
depending on the predicate**.

### Two defects the end-to-end run exposed

The first version of this run scored `delay_gt` at 66.2% / 90.8%, against a baseline
of 100%. That gap was not a property of the model. It was two defects in our own
encoding, and both are now fixed — the numbers in the table above are the fixed ones.

**Pinning guaranteed presence, not identification.** A dittoed row reads

```
r2,,,,,,,,,,,,,,,,,2,-5.00,3,2023-01-03,N605LR,-8.00
```

so locating `DepDelay` means counting seventeen commas — the same positional reading
that makes `columnar_rle` unusable. Pinning had put the value on every row without
making it findable. Labelling the pinned cells (`DepDelay=-5.00` in the CSV slot)
removes the counting, and at a 120-row block on flight it is worth:

| predicate | `csv_rle`+pin | **+labelled** | `row_kv` (ceiling) | tokens vs `row_kv` |
|---|---|---|---|---|
| `taxi_sum` @0.5 | 88.8% | **98.7%** | 99.0% | 41% |
| `taxi_sum` @fitted | 91.3% | **98.9%** | 99.1% | |
| `delay_gt` @0.5 | 79.0% | **85.6%** | 91.0% | 36% |
| `delay_gt` @fitted | 92.4% | **94.1%** | 96.2% | |
| `state_eq` @fitted | 98.5% | **99.8%** | 99.2% | 34% |

`taxi_sum` gains 9.9 points and reaches the `row_kv` ceiling at 41% of its tokens;
`state_eq` passes it. Labelling costs about 10% more tokens than bare pinning.
It is on by default (`label_pinned=True`).

**The packer filled the token budget because nothing told it not to** — ~555 rows per
request. Sweeping block size (flight, 1,200 rows):

| block | rows/s | `delay_gt` @0.5 | @fitted | `state_eq` @fitted | $/100k rows |
|---|---|---|---|---|---|
| 15 | 48 | 86.2% | **93.0%** | 98.8% | $0.43 |
| 60 | 187 | 81.7% | 92.6% | **99.9%** | $0.34 |
| **120** | **317** | 78.9% | **92.8%** | 99.4% | **$0.32** |
| 240 | 561 | 74.2% | 89.9% | 96.7% | $0.32 |
| 600 | 751 | 71.3% | 89.5% | 97.3% | $0.32 |

Two things fall out. **Most of the decay is calibration drift, not comprehension
loss**: at 0.5 accuracy slides steadily with block size (86.2 → 71.3), while with a
fitted threshold it is flat from 15 to 120 (93.0 → 92.8) and only breaks after 240.
And **past ~120 rows the bill stops moving** — tokens fall 2% from block 120 to 600
while accuracy drops 3.3 points. Filling the budget buys throughput, not money, so
the default is a 120-row cap and filling it is something a latency-bound caller asks
for.

## Defects this project found in itself

Worth stating plainly, because the whole pitch is "compression that does not silently
break correctness", and all three were found by auditing rather than by anything
failing:

- **CSV values were never escaped.** A value containing a comma added a field and
  shifted every later column out of alignment with the header. On flight that was
  *every row* (`OriginCityName` = "Hartford, CT"). Re-running the affected
  measurements after the fix moved balanced accuracy by at most 1.8 points, and the
  unaffected control (`row_kv`) moved 0.5 on its own, so **the conclusions did not
  change** — the model turned out to be robust to the misalignment. The defect was
  still real, and on a table with a different column order it would not have been
  harmless.
- **An empty cell was indistinguishable from "same as above".** A genuine NULL was
  silently read as the previous row's value. flight carries 34 empty `DepDelay` and 40
  empty `ArrDelay` per 2000 rows. Fixed with an explicit `\N` sentinel.
- **`columnar_rle` never emitted per-row ids at all**, despite announcing them. The
  test suite found the structural cause of a result we had only observed empirically:
  it scores near chance because the model cannot address a row, and pinning rescues it
  because pinning restores per-row ids.

The lesson is the obvious one: this project had no tests until after all of those
measurements were taken. It has them now (`tests/`), and CI runs the offline half on
every push.

## Limits

- **Two tables, one planner, one model, through a proxy.** flight and movies agree on
  the policy but not the saving. Two schemas show the policy is not a flight artifact;
  they cannot predict a third. All runs use `solo_greedy` as the planner and reach Jev
  through OpenRouter.
- **The throughput headline is a projection** from 5,000-row measurements, not a 100k
  measurement.
- **Costs rest on a fitted slope** with up to 8.2% residual, so cost differences under
  ~10% between encodings are not resolvable.
- **Jev is weak on signed numeric comparison** — `delay_gt` reaches 98.0% only with a
  fitted threshold, and relational predicates are mostly numbers and dates.
- **The cache result is a negative one through a proxy.** A positive finding there would
  have been confounded by OpenRouter; a negative one is not, but a TypeSafe-direct key
  would close it properly.
- **Calibration is measured, not shipped.** The Platt layer is not implemented.

---

## Layout

```
jev_solo/            the framework (offline parts need no key, no network)
  datasets.py        tables and predicates: the public interface
  tokens.py          pluggable token counters; rendered-field weights
  encodings.py       six state encodings, incl. pin= on the compressed ones
  objective.py       G^(k), the analytic cost, and its validation
  plan.py            column planners: default, random, ndv, solo_greedy, token_greedy*
  pack.py            packing sorted rows into 64k/32k-feasible requests
  bench_offline.py   what would this table cost?
  rescore.py         the same under the measured billing model
  threshold.py       fitted decision thresholds, averaged over splits
  calibration.py     ECE with a noise floor, temperature and Platt refits
  analyze_accuracy.py  accuracy-cost frontier and the mechanism test
  cross_table.py     is the rule schema-specific?

probes/              against the real API, a mock, or a self-hosted stand-in
  jev_client.py      wire schema; picks TypeSafe or OpenRouter from --base-url
  mock_server.py     stdlib mock of the protocol — validates probes with no key
  probe_accuracy_scaled.py   accuracy by encoding, any table via --table
  probe_pinned.py    emit fraction as mechanism, and pinning
  probe_decompose.py question ensembles and extract-then-compute, with controls
  probe_cache.py / probe_parallel.py / probe_tokenize.py / probe_overhead.py
  probe_latency_shape.py / probe_openrouter_smoke.py

tests/               regression tests for the serialization layer
third_party/jeff/    MIT self-hosted stand-in, cloned on demand by setup_jeff.sh
results/             every raw measurement behind the numbers above, tracked in git
```

The SOLO paper's own code is **not** in this repository — it lives separately
because its handover documents carry live cluster credentials.

---

## Next

1. **Package the pipeline.** The capability is spread across probe scripts; a user
   cannot `import` and go. Target shape: an analyzer that takes a table and predicates,
   packs, encodes with pinning on by default, fits thresholds, and reports the measured
   saving for *that* table rather than a promised one.
2. **Run the whole scan at 100k rows.** Both arms are now measured at 5k and scaled;
   the packed arm should be run at 100k for real, which costs about $0.30.
3. **Sweep block size on more predicates and the second table.** The 120-row default
   rests on two predicates of one table.
4. **Robustness and determinism suites**: NULLs, unicode, oversized single rows,
   duplicate rows, and whether the same request twice returns the same answers.
5. **Ship the threshold layer**, the highest-value and cheapest piece — the rule is
   settled (fit per predicate against balanced accuracy, never likelihood).
6. **Implement Platt recalibration** per (question type, encoding), then measure
   selectivity error end to end.
7. **A third and fourth schema**, a second planner, and a second model (`jeff` is
   wired up and has never been run). Two tables settle "not an artifact"; they do not
   settle "general".
8. **Baselines.** The first question any user asks is "why not a fine-tuned
   classifier, or an LLM with structured output, or a SQL heuristic?" We cannot
   currently answer it.
9. `probe_cache.py` with a TypeSafe-direct key.

---

## Citing

The reordering machinery comes from:

> Yingze Li et al. *Prefix-Cache-Aware Data Reordering for LLM-Augmented Database
> Analytics.* ICML 2026 (submission #10824).

Note honestly that this project reuses SOLO's combinatorial core while its central
mechanism — cross-request prefix-cache reuse — does not exist on Jev.

## Sources for the vendor-documented limits

Jev 1.13, `POST /v1/systemone`, $0.042/M input tokens (output free), 250k tokens/s and
1,200 req/min (dynamic), 64k per request / 32k for state plus the longest question, text
only, no documented caching, no batch endpoint — <https://docs.typesafe.ai/models> and
<https://docs.typesafe.ai/introduction>. Through OpenRouter the model id is
`typesafe/jev-1.13` on `POST /alpha/decisions` (`typesafe/jev-latest` does not exist
there). The wire schema is pinned from `third_party/jeff/src/jeff/core/schemas.py`,
whose docstring states it matches `typesafe_sdk/_schemas/models.py`.

---

## Internal notes (HIT group)

- **Do not run inference on the 210 code machine.** `/home/ubuntu/lyz/资源总览.md`
  reserves it for code and infra, and its disk is full — which is why the GLiFormer
  weights for `jeff` were never downloaded here. Use xtra3090 or the DCU node.
- Jev cannot be self-hosted: no weights, no on-prem, no paper. Open stand-ins are
  `jeff` (vendored), Laya and OpenJev.
- `jeff` defaults cap `JEFF_MAX_QUESTIONS=64` and `JEFF_MAX_STATE_CHARS=20000`, far
  below Jev's budgets. Raise both or it rejects exactly the large blocks this is about.
- DMV / OP_DTL / GA / Food live on the DCU and 3090 nodes, not here; only flight and
  movies are local, which is why the cross-schema check uses those two.
- Never send `/home/ubuntu/ycr/` clinical data to any external API.
