# vllm-jev and Open-Jev-2B

SOLO's first additional open JEV framework is
[mode-io/vllm-jev](https://github.com/mode-io/vllm-jev), initially with
[ZefanCai/Open-Jev-2B](https://huggingface.co/ZefanCai/Open-Jev-2B). Both are
Apache-2.0 licensed. This pairing was selected because vllm-jev serves the model
natively through vLLM and exposes an explicit cross-request prefix-cache
namespace, which is the interface SOLO needs after arranging fields and rows.

## Why this target came first

| Candidate | Runtime shape | Fit with SOLO prefix reuse | Decision |
| --- | --- | --- | --- |
| vllm-jev + Open-Jev-2B | Native vLLM decision serving; 2B checkpoint | Explicit `cache_salt` and cached-token counters | Implemented first |
| Open-Jev compatible HTTP servers | OpenAI-style candidate scoring | Useful decision API, but no equivalent cache namespace in the reviewed interface | Later adapter |
| lev | Qwen LoRA with its own System One server | No exposed cross-request vLLM cache control | Defer |
| jeff / Von | Encoder-style classifiers | No autoregressive KV/prefix-cache path for SOLO to enlarge | Not the first target |

One vllm-jev adapter may later cover more checkpoints supported by the framework.
The cross-request online cache described here is currently provided for repeated
Open-Jev-2B text requests on Linux.

## Start the server

vllm-jev 0.3.2 requires Python 3.12 and pins vLLM 0.29.0, so install it in a
separate environment from the AutoTrust JEV-9B reference server:

```bash
git clone https://github.com/mode-io/vllm-jev.git
cd vllm-jev
git checkout ac00233ac1a0d2fd9c49d235b286ffeebca9f634
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install .

VLLM_JEV_ONLINE_PREFIX_CACHE=1 \
  vllm-jev serve ZefanCai/Open-Jev-2B --prefix-match-unit 16
```

The reviewed revision above identifies itself as v0.3.2 and pins the model and
foundation revisions during checkpoint preparation. The default endpoint is
`http://127.0.0.1:8795`. The two cache settings in the last command are required
for the framework's online prefix-cache path.

## Use it from SOLO

Install this project where the input data and optimizer run:

```python
from solo_layout import DecisionEngine, VllmJevBackend

backend = VllmJevBackend("http://127.0.0.1:8795")
with DecisionEngine(backend=backend, concurrency=4) as engine:
    result = engine.scan(
        records,
        "Does this record require action?",
        method="solo",
        cache_salt="batch-2026-10-07-001",
    )
print(result.to_pandas())
```

Use the same salt only for records intended to share one cache namespace. The
adapter deliberately calls `/plugins/vllm-jev/choice`: vllm-jev's
`/v1/systemone` route creates a fresh salt per request and therefore cannot
preserve cross-record reuse. `DecisionEngine.compare()` generates a fresh salt
for each method and repeat automatically.

The state remains a string containing the complete SOLO serialization. Parsing
it into a JSON object at the adapter boundary could reorder fields and destroy
the prefix layout. Binary decisions use the ordered candidates `no, yes`; score
decisions use `0` through `5`; categorical choices preserve the user-supplied
option order.

## Live RTX 4090 validation

The adapter was exercised end to end on one RTX 4090 24 GB with vllm-jev 0.3.2
at commit `ac00233ac1a0d2fd9c49d235b286ffeebca9f634`, vLLM 0.29.0,
Open-Jev-2B BF16, concurrency 4, online prefix caching enabled and
`--prefix-match-unit 16`.

The controlled workload contains 64 complete records. Every record carries the
same long refund policy, one of four customer contexts, a unique request and
purchase ID, and an evidence field that determines the binary answer. The
original layout starts with the unique ID; SOLO chooses
`policy → evidence → customer_context → request_id → purchase`. Each result is
the median of three interleaved cold-namespace trials, including layout planning
and first cache population.

| Prefix reads | Layout | Batch time | Rows/s | Cached input | Accuracy |
| --- | --- | ---: | ---: | ---: | ---: |
| Enabled | Original | 5.448 s | 11.75 | 45.78% | 100% |
| Enabled | SOLO | **3.254 s** | **19.67** | **91.59%** | 100% |
| Disabled | Original | 9.618 s | 6.65 | 0% | 100% |
| Disabled | SOLO | 9.374 s | 6.83 | 0% | 100% |

With cache reads enabled, SOLO is **1.67×** faster than the original layout and
doubles the reported cached-input fraction. With cache reads disabled, the two
layouts are within 2.6%, isolating the gain to prefix reuse rather than field
movement alone. Both enabled-layout runs sent 380,288 prompt tokens and agreed
on all 64 decisions.

[Benchmark script](../scripts/benchmark_vllm_jev.py) ·
[cache-enabled raw trials](../validation/vllm-jev-benchmark.json) ·
[cache-disabled raw trials](../validation/vllm-jev-benchmark-no-cache.json) ·
[machine-readable environment record](../validation/vllm-jev-live.json)

This is a mechanism-focused synthetic workload, separate from the public
ContractNLI profiling and from the existing AutoTrust JEV-9B benchmark numbers.
The vllm-jev online cache is currently an upstream experimental feature and is
disabled by default.
