# MNIST + SPLL pipeline

This pipeline is split into distinct stage files plus one orchestration entrypoint.

## Files

- `train_mnist.py`: trains the MNIST classifier and exports the model plus the fixed dataset split manifest.
- `compile_spll.py`: generates and compiles the SPLL programs for every configured `(term_count, cutoff)` target.
- `stage_experiments.py`: samples MNIST digit-addition experiments from the fixed inference split and saves them.
- `infer_experiments.py`: runs posterior inference for the staged experiments and saves broad raw run data as JSON.
- `visualize_results.py`: computes summaries, tables, and plots from the saved raw inference JSON.
- `run_spll_pipeline.py`: orchestration entrypoint that takes a stage name as input.
- `run_spll_sum_experiments.py`: backward-compatible wrapper around the new stage-based pipeline.
- `mnist_spll_pipeline_core.py`: shared pipeline helpers.
- `mnist_spll_common.py`: shared model/config utilities.
- `mnist_spll_config.yaml`: joint config for all stages.

## Approximation reminder

In this repo, `-k/--topKCutoff` is **not** a literal top-k class count. It is a **probability cutoff in the range 0..1** that prunes low-probability branches during inference. Exact inference is represented by `null` in `approximation_thresholds`.

## Pipeline order

The intended stage order is:

```text
train -> compile spll -> stage experiments -> inference -> visualisation
```

## Important Apple Silicon caveat

Use explicit interpreter paths. Do **not** rely on a shell alias like `python -> /opt/homebrew/bin/python@3.11`.

Recommended rule:

- native arm64 stages: use `python3` or the arm venv interpreter explicitly
- Rosetta/x86_64 compile: use the x86 venv interpreter explicitly

Examples in this README therefore use:

- `./.venv-train-arm64/bin/python`
- `./.venv-spll-x86/bin/python`

## Why the compile stage needs Python packages too

The current orchestrator imports all stage modules at startup. Because of that, the compile stage currently needs the shared Python dependencies too, even though the actual SPLL compilation work is done through Stack.

That means the Rosetta/x86 compile env needs these Python packages installed as well:

- `numpy`
- `PyYAML`
- `Pillow`
- `matplotlib`
- `torch`
- `torchvision`

`stack` is still required separately for the actual SPLL compile step.

## Package installation

Run these from `mnist_spll_pipeline/`.

### Native arm64 env

```bash
python3 -m venv --copies .venv-train-arm64
source .venv-train-arm64/bin/activate
./.venv-train-arm64/bin/python -m pip install --upgrade pip setuptools wheel
./.venv-train-arm64/bin/python -m pip install numpy PyYAML Pillow matplotlib
./.venv-train-arm64/bin/python -m pip install torch torchvision
```

### Rosetta / x86_64 env

Open a clean Rosetta shell first:

```bash
arch -x86_64 zsh -f
cd /Users/lisztomaniacae/IdeaProjects/spll-approx-comp/mnist_spll_pipeline
arch -x86_64 /usr/local/bin/python3.11 -m venv --copies .venv-spll-x86
source .venv-spll-x86/bin/activate
./.venv-spll-x86/bin/python -m pip install --upgrade pip setuptools wheel
./.venv-spll-x86/bin/python -m pip install numpy PyYAML Pillow matplotlib
./.venv-spll-x86/bin/python -m pip install torch torchvision
```

If `.venv-spll-x86` already exists and only packages are missing, just activate it and run the three `pip install` lines.

## Verify both interpreters

### Native arm64

```bash
./.venv-train-arm64/bin/python -c "import platform, torch; print(platform.machine(), torch.__version__)"
```

Expected machine value: `arm64`

### Rosetta / x86_64

```bash
./.venv-spll-x86/bin/python -c "import platform, torch; print(platform.machine(), torch.__version__)"
```

Expected machine value: `x86_64`

## Commands

From this folder:

```bash
cd /Users/lisztomaniacae/IdeaProjects/spll-approx-comp/mnist_spll_pipeline
```

### Train

Run in native arm64:

```bash
source .venv-train-arm64/bin/activate
./.venv-train-arm64/bin/python run_spll_pipeline.py --config mnist_spll_config.yaml train
```

### Compile SPLL

Run in Rosetta/x86_64:

```bash
arch -x86_64 zsh -f
cd /Users/lisztomaniacae/IdeaProjects/spll-approx-comp/mnist_spll_pipeline
source .venv-spll-x86/bin/activate
./.venv-spll-x86/bin/python run_spll_pipeline.py --config mnist_spll_config.yaml compile
```

This compiles every configured `(term_count, cutoff)` pair for the range defined by `terms_per_sum_min` and `terms_per_sum_max`.

### Stage experiments

Run in native arm64:

```bash
source .venv-train-arm64/bin/activate
./.venv-train-arm64/bin/python run_spll_pipeline.py --config mnist_spll_config.yaml stage
```

### Inference

Run in native arm64:

```bash
source .venv-train-arm64/bin/activate
./.venv-train-arm64/bin/python run_spll_pipeline.py --config mnist_spll_config.yaml infer
```

This does **not** compute summary metrics yet. It writes broad raw run records to JSON, including:

- experiment metadata
- image paths
- true labels and true sums
- candidate sums
- raw posterior values for every candidate
- runtime per run
- compiled program path used for the run

### Visualisation

Run in native arm64:

```bash
source .venv-train-arm64/bin/activate
./.venv-train-arm64/bin/python run_spll_pipeline.py --config mnist_spll_config.yaml visualize
```

This stage reads the raw inference JSON and computes derived outputs on demand:

- `detailed_results.csv`
- `summary_results.csv`
- `summary_results.json`
- plots under `visualization/plots`

### Run all stages

```bash
./.venv-train-arm64/bin/python run_spll_pipeline.py --config mnist_spll_config.yaml all
```

On Apple Silicon, running all stages in one command is less robust than the stage-by-stage split because compile lives in the Rosetta env.

## Raw and derived outputs

The pipeline writes its stage artifacts under:

```text
outputs/spll_experiments/
```

Important files:

- `compile_manifest.json`
- `staged_experiments.json`
- `inference_manifest.json`
- `inference_runs.json`
- `visualization/detailed_results.csv`
- `visualization/summary_results.csv`
- `visualization/summary_results.json`
- `visualization/plots/*.png`

## Progress output

You still get progress bars for:

- compilation targets
- staged experiments
- loading compiled Python targets
- inference runs
- per-run posterior candidate sums

Disable all bars with:

```yaml
inference:
  show_progress: false
```

Disable only the inner posterior bar with:

```yaml
inference:
  show_inner_progress: false
```

## Notes

- `repo_root` in the YAML should point to your local NeST checkout.
- The compile stage expects `stack` to be available on the shell `PATH`.
- The infer stage does not need Stack, but it will fail if the compiled outputs are missing.
- The default `terms_per_sum_max` should stay conservative because exact inference gets expensive fast.

## Fast fix for your current compile error

If the x86 venv already exists and the current failure is only missing packages, run:

```bash
cd /Users/lisztomaniacae/IdeaProjects/spll-approx-comp/mnist_spll_pipeline
source .venv-spll-x86/bin/activate
./.venv-spll-x86/bin/python -m pip install --upgrade pip setuptools wheel
./.venv-spll-x86/bin/python -m pip install numpy PyYAML Pillow matplotlib
./.venv-spll-x86/bin/python -m pip install torch torchvision
./.venv-spll-x86/bin/python run_spll_pipeline.py --config mnist_spll_config.yaml compile
```
