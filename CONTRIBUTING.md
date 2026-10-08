# Contributing

SOLO — System One Layout Optimizer is a research prototype with a small Python client and an optional
GPU backend. Install the client from a source checkout:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,demo,plot]'
python -m pytest
python -m build
```

The offline tests need no model weights, API key, GPU or network access after
dependencies are installed. Live inference is an explicit, separate check.
Deployment scripts live in `deploy/`; they are shipped in the source archive,
and model weights and GPU libraries are excluded from the client wheel.

Keep the public interface small. New input formats should preserve complete
records, nested array order, duplicate rows and original output alignment. Any
change to the planner should be checked against an independent greedy oracle,
including stable ties and sampling. The inherited integer grouping kernel and
its provenance are documented in `planner-provenance.json`.

For performance changes, record the dataset seed and hash, row/column counts,
field lengths, exact serving configuration, repetitions and raw observations.
Use an independent cache salt for every measured scan, include the initial cache
fill, and report decision quality alongside throughput. Describe synthetic
distributions as synthetic; do not present a selected demo as a general
production speedup. Keep all baselines in a published comparison.

For a pull request, describe the concrete behavior that changes, how it was
checked, and any compatibility limitation. Do not include model weights,
credentials, `.env` files, generated runtime logs or machine-specific endpoints.
Benchmark records should use reproducible configuration identifiers instead.
