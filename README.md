# bonsai

Pruning and distilling a Mixture-of-Experts model into a smaller,
TypeScript-specialized one, following NVIDIA's Minitron recipe (structured
pruning + knowledge distillation). The papers this is built from are in
[docs/papers](docs/papers).

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

Everything past baseline eval needs a rented GPU (see `plan.md`'s sizing table).
A DigitalOcean GPU Droplet, on demand. Two ways to get the box:

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
