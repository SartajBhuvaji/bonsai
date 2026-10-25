# bonsai

Pruning and distilling Qwen3-30B-A3B, a 30B-parameter Mixture-of-Experts
model, into a smaller model specialized for one customer's actual workload,
following NVIDIA's Minitron recipe (structured pruning + knowledge
distillation). TypeScript code generation stands in here for "a customer
whose traffic concentrates on one pattern." The reference papers this is
built from are in [docs/papers](docs/papers).

## Result

The best candidate is **47% smaller** (16.03B vs. 30.50B parameters), holds
**80.7% of baseline accuracy** on the full 159-problem HumanEval-TS
benchmark, and **beats the unpruned model's own measured serving throughput
by 17-21%** on real vLLM and SGLang benchmarks, not a parameter-count
estimate.

| Candidate | ~params | Zero-shot loss | KD val (step) | Full pass@1 | 95% CI |
| --- | --- | --- | --- | --- | --- |
| baseline (unpruned) | 30.50B | 0.9670 | N/A | 0.717 (114/159) | ±6.9pp |
| depth0/experts112 | 5.15B | 2.567 | 0.569 @ 499 | 0.082 (13/159) | ±4.3pp |
| depth0/experts64 | 16.03B | 0.970 | 0.145 @ 39 | 0.579 (92/159) | ±7.6pp |
| depth0/experts32 | 23.28B | 0.889 | 0.070 @ 29 | 0.604 (96/159) | ±7.5pp |
| depth0/experts16 | 26.91B | 0.874 | 0.047 @ 29 | 0.648 (103/159) | ±7.3pp |
| **depth0/experts8** | 28.72B | 0.869 | 0.042 @ 29 | **0.704 (112/159)** | ±7.0pp |
| depth16/experts0 | 20.56B | 1.157 | 0.207 @ 49 | 0.283 (45/159) | ±6.9pp |
| depth24/experts0 | 15.58B | 2.109 | 0.335 @ 499 | 0.107 (17/159) | ±4.8pp |

Seven single-axis candidates, fixed-loss short KD, full 159-problem
HumanEval-TS against the unpruned baseline. Experts-only pruning (depth0
rows) holds up far better than depth-only pruning (experts0 rows) at
comparable or even larger parameter counts: depth16/experts0 (20.56B)
scores worse than depth0/experts64 (16.03B), despite 4.5B more parameters.

This table isolates the two pruning axes; thirteen candidates in total
(including combined depth+experts cuts) were pruned, distilled, and
evaluated on the full 159-problem suite, not a 40-problem sample -- this
project found that sampling systematically overstates accuracy, by 22% to
226% depending on the candidate. Experts-only pruning dominates depth-only
and combined depth+experts pruning at every matched parameter budget.

**The core contribution:** Minitron's recipe prunes dense transformers along
three axes (depth, width, attention heads). A Mixture-of-Experts model has
no width axis to prune -- every MLP parameter lives inside an expert. This
project adapts Minitron's activation-based importance-scoring philosophy to
expert pruning directly: a gate-weighted expert-output-norm criterion that
two other groups (REAP, NVIDIA's own Minitron-MoE extension) independently
converged on around the same time, validated here against the model's
previously-reported "Super Experts."

The paper's appendix is a real debugging record, not just a results
summary: two knowledge-distillation loss bugs that caused repetition
collapse, a baseline-denominator mismatch that inflated an early result,
and a fabricated benchmark rerun caught by diffing raw per-request output
files against the previous commit before trusting the number.

**Paper:** ["Mixture-of-Experts Pruning for Optimizing Customer-Specific
Inference"](docs/latex/main.pdf). **Checkpoints:** [Hugging Face
collection](https://huggingface.co/collections/sartajbhuvaji/bonsai-6adaa6c9472b58d28f07d796).

## Requirements

Python 3.11 or newer.

## Install

```bash
git clone git@github.com:SartajBhuvaji/bonsai.git
cd bonsai
make install
```

That does an editable install with the dev and notebook extras and installs
the pre-commit hooks.

## Development

Checks are defined in `tox.ini`. CI runs the same environments, so anything that
passes locally passes there.

```bash
make check   # lint + types + tests + notebooks on the current interpreter
make fmt     # reformat and autofix
make cov     # tests with a coverage report
tox          # every supported interpreter (3.11, 3.12, 3.13)
```

| Command | What it runs |
| --- | --- |
| `tox -e lint` | `ruff check` and `ruff format --check` (covers notebook cells too) |
| `tox -e fmt` | `ruff format` and `ruff check --fix` |
| `tox -e type` | `mypy` in strict mode |
| `tox -e py` | `pytest` |
| `tox -e cov` | `pytest` with coverage |
| `tox -e notebooks` | checks committed notebooks still carry their outputs |

CI never rewrites files. `tox -e lint` reports formatting problems, `tox -e fmt`
fixes them.

## Notebooks

Where a step produces something worth looking at, it gets a notebook under
[notebooks/](notebooks) alongside the code that does the work. See
[notebooks/README.md](notebooks/README.md) for the convention: logic lives in
`src/bonsai/`, notebooks call it and keep only the one-time or visual code.

## Compute

Everything past baseline eval needs a rented GPU. A DigitalOcean GPU
Droplet, on demand. Two ways to get the box:

**Already have one** (created by hand from the DO console/CLI):

```bash
cp .env.example .env                        # HF_TOKEN
SSH_KEY=~/.ssh/id_ed25519 WITH_EVAL=1 ./tools/bootstrap_do_gpu.sh <droplet-ip>
ssh -i ~/.ssh/id_ed25519 root@<droplet-ip>
./tools/gpu_env.sh                          # provenance block for whatever ran there
```

**Provision one from the CLI** instead, with `launch_do.sh` (needs
`DIGITALOCEAN_ACCESS_TOKEN` in `.env`):

```bash
cp .env.example .env                        # HF_TOKEN, DIGITALOCEAN_ACCESS_TOKEN
./tools/launch_do.sh sizes                   # see where the default GPU size has capacity
REGION=<region-from-above> ./tools/launch_do.sh launch
# or add BOOTSTRAP=1 to also run bootstrap_do_gpu.sh once the Droplet is up
./tools/launch_do.sh terminate <droplet-id>  # tear down when done -- this bills hourly
```

Either way, `bootstrap_do_gpu.sh` syncs this repo to the box over rsync,
copies `.env`, and runs `setup_instance_do.sh` remotely (venv, deps, confirms
GPU + Qwen3-MoE import) in one shot; `WITH_EVAL=1` also installs the eval
extra plus Node.js and `tsc` for MultiPL-E's TypeScript tasks.

`do_status.sh` catches a Droplet left running: `./tools/do_status.sh status`
lists every bonsai-tagged Droplet with elapsed time and running cost,
`./tools/do_status.sh killall` tears all of them down at once (asks for
confirmation unless `FORCE=1`).

## Releases

Versioning is driven by PR labels. Every PR carries exactly one of `patch`,
`minor` or `major`, and CI fails until it does. On merge, the release workflow
bumps the version in `pyproject.toml`, updates `CHANGELOG.md`, tags `vX.Y.Z` and
publishes a GitHub release. See [CONTRIBUTING.md](CONTRIBUTING.md).

## Layout

```
src/bonsai/               the package: config loading, experiment tracking
src/bonsai/architecture/  model inspection and diagramming (no pruning logic yet)
notebooks/                explainer notebooks -- see notebooks/README.md
configs/                  TOML configs loaded by bonsai.config
tests/                    pytest suite
tools/                    provisioning and CI-check scripts, not part of the package
docs/                     documentation
docs/papers/              reference PDFs, pinned to the version we worked from
.github/                  CI, release automation, Dependabot
tox.ini                   lint, type, test and notebook-check environments
```

## License

MIT. See [LICENSE](LICENSE).
