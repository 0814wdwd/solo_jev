# Deploy JEV-9B with vLLM

The client and GPU server install separately. The reference configuration runs
AutoTrust JEV-9B BF16 on one RTX 4090 with 24 GB VRAM, a 16K context limit and four
concurrent sequences. Its measured driver is 595.71.05. Other hardware, driver
versions and context/concurrency settings require their own validation.

## Model source and request path

The upstream repository is **[autotrust/JEV-9B on Hugging Face](https://huggingface.co/autotrust/JEV-9B)**,
which hosts the model weights, decision adapter, inference code and model card.
AutoTrust built this independent open model on
[Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B), using TypeSafe Jev 1.13
output distributions as distillation targets. Model credit belongs to AutoTrust
and Qwen; SOLO supplies the layout optimization and structured-data client.

Our integration follows the upstream
[vLLM quickstart](https://huggingface.co/autotrust/JEV-9B#quickstart-with-vllm-recommended):

1. [Download](../deploy/download.py) the text backbone, `adapter_vllm/`, tokenizer
   and calibration files from [revision
   `b63f651ce8ed64481d3f5e73ecdb05f740042f01`](https://huggingface.co/autotrust/JEV-9B/tree/b63f651ce8ed64481d3f5e73ecdb05f740042f01), fixed in
   [deployment.json](../deploy/deployment.json).
2. [Start vLLM](../deploy/serve.sh) with the `jev-decision` LoRA adapter,
   `--enable-prefix-caching` and `--mamba-cache-mode align`.
3. [JevBackend](../src/solo_layout/backend.py) builds the upstream `bare-v1`
   decision prompt and calls `/v1/completions` with `max_tokens=1` and the allowed
   option token IDs. It adds the head bias from `adapter_vllm/decision_head.json`,
   divides by the per-kind temperature from `calibration.json`, and applies
   softmax. `DecisionEngine` then restores results to input row order.

The benchmark uses this text decision path. The upstream repository also
contains generation and vision examples, which are separate from these scans.

## One-command setup and startup

From the source checkout on a Linux NVIDIA GPU host with Python 3.10+ and a
working CUDA driver:

```bash
bash deploy/jev9b.sh up
```

The command creates an isolated server environment, installs pinned dependencies,
applies the documented vLLM compatibility patch, downloads the pinned model and
adapter, starts the server, then waits for the decision adapter to become ready.
Allow roughly **35 GB of free disk** for weights and runtime dependencies, plus
headroom for download caches. Download time depends on connectivity; model files
alone are about 18 GB.

Default paths:

| Item | Location |
| --- | --- |
| Server environment | `deploy/.venv/` |
| Model and adapter | `deploy/models/JEV-9B/` |
| Process state, logs, runtime package lock | `deploy/runtime/` |
| API | `http://127.0.0.1:8000` |
| Decision adapter name | `jev-decision` |

The helper's lifecycle and compatibility patch have local automated checks.
The new `start` entry point and repeated `up` command were also verified on the
RTX 4090 using the existing environment and downloaded weights. The installed
client passed six input checks and all three decision heads against that service.
See the [live deployment record](../validation/deployment-live.json) and
[client results](../validation/live-api-v020.json). A fresh-machine installation
from an empty environment has not been rerun with this helper.

## Reuse an environment or downloaded weights

```bash
export JEV_MODEL_DIR=/path/to/JEV-9B
export JEV_VENV=/path/to/vllm-environment
bash deploy/jev9b.sh doctor
# For an already installed compatible runtime:
bash deploy/jev9b.sh start
```

`setup` installs the pinned requirements into the selected environment. Use a
dedicated environment if its existing packages must remain untouched. Downloads
resume and reuse the selected model directory. Standard `HF_TOKEN` and
`HTTPS_PROXY` environment settings are respected; configure your hosting
provider's network acceleration before running setup if needed.

Available commands:

```bash
bash deploy/jev9b.sh setup   # Install and download only
bash deploy/jev9b.sh start   # Start and wait for the model
bash deploy/jev9b.sh status  # Inspect the owned process and model readiness
bash deploy/jev9b.sh doctor  # Inspect GPU, files, environment and endpoint
bash deploy/jev9b.sh stop    # Stop the process group owned by this checkout
```

By default the endpoint is bound to localhost. For a client on another machine,
an SSH tunnel preserves that default:

```bash
ssh -L 8000:127.0.0.1:8000 your-gpu-host
```

Install the client where your data lives:

```bash
python -m pip install ".[pandas,hub]"
python examples/scan_json.py
```

Check real input formats, output alignment and all three decision heads against
the running service:

```bash
python scripts/check_live.py --model-dir /path/to/JEV-9B --out live-check.json
```

If the client can read the model metadata locally, pass
`--model-dir /path/to/JEV-9B` to the example to avoid metadata downloads.

## Serving parameters

`deploy/deployment.json` records the pinned reference configuration.
`deploy/serve.sh` launches the model with LoRA decision adapter support, one-token
log probabilities, prompt-token cache details and automatic prefix caching.
Defaults include:

| Environment variable | Default |
| --- | ---: |
| `JEV_HOST` | `127.0.0.1` |
| `JEV_PORT` | `8000` |
| `JEV_MAX_MODEL_LEN` | `16384` |
| `JEV_MAX_NUM_SEQS` | `4` |
| `JEV_BATCH_TOKENS` | `2048` |
| `JEV_GPU_MEMORY_UTILIZATION` | `0.90` |
| `JEV_EAGER` | `1` |
| `JEV_PREFIX_CACHING` | `1` |
| `JEV_PER_REQUEST_METRICS` | `0` |
| `JEV_START_TIMEOUT` | `600` seconds |

The tested hybrid-model prefix-cache mode is `align`. Changing runtime flags can
change memory use and throughput; keep them fixed within a layout comparison.
Clients use the served adapter name `jev-decision`, not the base-model name.

Set `JEV_PER_REQUEST_METRICS=1` for diagnostic runs that need vLLM queue and
first-token intervals. Keep a separate metrics-disabled run to quantify its CPU
overhead. The cache/layout causal comparison requires two real server starts:

```bash
bash deploy/jev9b.sh stop
JEV_PREFIX_CACHING=1 JEV_PER_REQUEST_METRICS=1 bash deploy/jev9b.sh start
# run the cache-enabled report, then stop the server

bash deploy/jev9b.sh stop
JEV_PREFIX_CACHING=0 JEV_PER_REQUEST_METRICS=1 bash deploy/jev9b.sh start
# run the cache-disabled report
```

A fresh `cache_salt` isolates matches between trials but does not disable APC or
isolate GPU contention. Do not treat namespace isolation as the cache-off arm.

## Runtime compatibility patch

The pinned vLLM runtime may attempt to import an SM100-only MiniMax warmup kernel
before checking the GPU architecture. The included patch moves that existing
hardware guard ahead of the import. It does not alter the Qwen inference kernels.
It checks the expected source/version, is idempotent, preserves a backup and
records hashes. An unexpected source layout raises an error rather than
silently applying a new patch.

## Troubleshooting

| Symptom | First check |
| --- | --- |
| Download stalls | Hugging Face reachability, provider acceleration, proxy and disk space |
| CUDA unavailable | `nvidia-smi`, driver compatibility, selected Python environment |
| Server runs out of memory | Other GPU processes, context limit, sequence count, memory fraction |
| Adapter missing | `status`, server log, model directory and pinned download completeness |
| Context-length error | Demo's maximum prompt length and server `JEV_MAX_MODEL_LEN` |
| No cache benefit | Shared token prefixes, block size, fresh-scan overhead, cache eviction and concurrency |

Do not run two performance experiments concurrently on the same GPU. Cache salts
isolate prefix matches, not GPU resources.
