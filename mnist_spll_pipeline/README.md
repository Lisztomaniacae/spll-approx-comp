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


## Living knowledge base

Detailed architecture notes, workflow caveats, output schemas, and future patch discipline live in [`KNOWLEDGE_BASE.md`](KNOWLEDGE_BASE.md). Any patch that changes the MNIST SPLL pipeline should update that file in the same patch.

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

- native arm64 stages: use the arm venv interpreter explicitly
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
- raw branch counts for every candidate sum
- runtime per run
- compiled program path used for the run

### Visualisation

Run in native arm64:

```bash
source .venv-train-arm64/bin/activate
./.venv-train-arm64/bin/python run_spll_pipeline.py --config mnist_spll_config.yaml visualize
```

This stage reads the raw inference JSON and writes a fuller bundle:

- tables under `visualization/tables`
- main-text figures under `visualization/figures/main_text`
- appendix/supporting figures under `visualization/figures/appendix`
- appendix heatmaps under `visualization/figures/appendix/heatmaps`

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
- `visualization/tables/detailed_results.csv`
- `visualization/tables/summary_results.csv`
- `visualization/tables/summary_results.json`
- `visualization/tables/overhead_exact_vs_zero_summary.csv`
- `visualization/tables/model_accuracy_targets.csv`
- `visualization/figures/main_text/*.png`
- `visualization/figures/appendix/*.png`
- `visualization/figures/appendix/heatmaps/*.png`

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
- Branch counts are only available if the SPLL targets were compiled with branch counting enabled.
- If you turn branch counting on after compiling, you must re-run the compile stage so the generated `program.py` files expose that data.

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

---

# Pipeline II: training through generated SPLL inference

Pipeline II tests whether SPLL approximation changes the **training process** itself. Unlike the inference-evaluation pipeline above, this pipeline trains an MNIST base model from **sum labels only** by calling the generated SPLL Python artifact inside every optimizer step.

## Pipeline II files

- `run_spll_training_pipeline.py`: stage dispatcher for Pipeline II.
- `mnist_spll_training_config.yaml`: default Pipeline II config.
- `mnist_spll_training_smoke_config.yaml`: small layered smoke-test override using one-level `extends`.
- `prepare_spll_training.py`: creates the balanced split manifest, compact schedule manifests/previews, and shared initial checkpoints.
- `compile_spll_training.py`: writes SPLL sum programs and compiles exact/approximate generated Python artifacts.
- `train_spll_generated.py`: trains through the generated SPLL artifacts with a differentiable `readMNist` callback.
- `visualize_spll_training.py`: writes milestone tables, per-arity grouped milestone bar charts, and training plots.
- `spll_training_core.py`: Pipeline II helpers for paths, schedules, generated-artifact calls, training guards, and split handling.

## Pipeline II design summary

Pipeline II uses only the official MNIST training partition. It materializes one global equal-count-per-digit 80/20 split:

- 80% source pool for sum-supervised training cases;
- 20% held-out uniform digit validation split;
- the official MNIST test partition is reserved/unused.

Training supervision is only the true sum. Digit labels are used for balanced split construction, but milestones are now task-level full-posterior sum accuracy, not held-out digit accuracy. The training stage supports SPLL sum mini-batches through `training.sum_batch_size`. The default config uses `sum_batch_size: 100`, so one optimizer update accumulates 100 generated SPLL true-sum queries before validation:

```text
for each case in the sum batch:
    p_true_i = generated_spll.main.forward(true_sum_i, *global_indices_i)
loss = mean_i(-log(p_true_i + epsilon))
```

The generated SPLL artifact still receives scalar global MNIST indices and is still the source of truth for exact/approximate pruning. To avoid repeating the CNN forward for every scalar SPLL call, the training loop first runs the current differentiable base model once over all unique images in the batch, installs a temporary `readMNist(index)` lookup backed by those softmax rows, and then calls the generated artifact once per sum case. The stored probability tensors remain attached to the autograd graph, so gradients from the mean batch loss flow back to the CNN.

`training.max_steps`, `validation.interval_steps`, milestone steps, and the `train_trace.csv` `step` column are measured in **sum cases seen**, not optimizer updates. The trace also records `optimizer_update`, `batch_size`, `branch_count_mean`, and `branch_count_total`. Set `training.sum_batch_size: 1` to recover the old one-case-per-update behavior.

Validation can run asynchronously via `validation.async.enabled`. In that mode the trainer snapshots the current model and optimizer state at each validation interval, sends the snapshot to a separate validator process, and immediately continues training. The validator enumerates the full generated-SPLL posterior over candidate sums `0..9*n_terms` for each held-out sum case, predicts the argmax sum, and reports `sum_posterior_accuracy`. When a result crosses a milestone, the milestone checkpoint is written from the validated snapshot; when the highest milestone is reached, the trainer stops the next time it polls the validator result. This avoids blocking every training batch on validation, but the stop can be slightly delayed if validation is slower than training. The default async validator uses `device: cpu` and `max_pending_jobs: 1` to avoid fighting the MPS/GPU training process.

## Pipeline II stage order

```text
prepare -> compile -> train -> visualize
```

`all` exists, but on Apple Silicon it is usually safer to run stages explicitly because `compile` should run under Rosetta/x86 while `prepare`, `train`, and `visualize` should run in the native arm64 environment.

## Pipeline II commands

From `mnist_spll_pipeline/`.

### Smoke test prepare

Run in native arm64:

```bash
source .venv-train-arm64/bin/activate
./.venv-train-arm64/bin/python run_spll_training_pipeline.py --config mnist_spll_training_smoke_config.yaml prepare
```

### Smoke test compile

Run in Rosetta/x86_64:

```bash
arch -x86_64 zsh -f
cd /Users/lisztomaniacae/IdeaProjects/spll-approx-comp/mnist_spll_pipeline
source .venv-spll-x86/bin/activate
./.venv-spll-x86/bin/python run_spll_training_pipeline.py --config mnist_spll_training_smoke_config.yaml compile
```

### Smoke test train and visualize

Run in native arm64:

```bash
source .venv-train-arm64/bin/activate
./.venv-train-arm64/bin/python run_spll_training_pipeline.py --config mnist_spll_training_smoke_config.yaml train
./.venv-train-arm64/bin/python run_spll_training_pipeline.py --config mnist_spll_training_smoke_config.yaml visualize
```

### Default Pipeline II run

Use the same stage split, replacing the config path:

```bash
./.venv-train-arm64/bin/python run_spll_training_pipeline.py --config mnist_spll_training_config.yaml prepare
```

```bash
arch -x86_64 zsh -f
source .venv-spll-x86/bin/activate
./.venv-spll-x86/bin/python run_spll_training_pipeline.py --config mnist_spll_training_config.yaml compile
```

```bash
./.venv-train-arm64/bin/python run_spll_training_pipeline.py --config mnist_spll_training_config.yaml train
./.venv-train-arm64/bin/python run_spll_training_pipeline.py --config mnist_spll_training_config.yaml visualize
```

## Pipeline II outputs

Default output root:

```text
outputs/spll_training/
```

Smoke-test output root:

```text
outputs/spll_training_smoke/
```

Important artifacts:

```text
outputs/spll_training/
  config_used.yaml
  data_split_manifest.json
  schedules/
    seed_42_terms_02_schedule_manifest.json
    previews/seed_42_terms_02_preview.jsonl
  initial_checkpoints/
    seed_42_terms_02.pt
  generated/
    spll_programs/sum_terms_02.spll
    compiled_python/terms_02/<mode>/program.py
  runs/
    seed_42_terms_02_exact/
      train_trace.csv
      validation_trace.csv
      milestones.json
      checkpoints/milestone_0p10.pt
      checkpoints/final.pt
  visualization/
    tables/milestone_summary.csv
    tables/milestone_aggregate_summary.csv
    figures/main_text/terms_02_steps_to_sum_posterior_milestone.png
    figures/main_text/terms_02_time_to_sum_posterior_milestone.png
    figures/main_text/*.png
    figures/appendix/*.png
```

The `steps_to_sum_posterior_milestone` and `time_to_sum_posterior_milestone` figures are grouped bar charts: milestones are on the x-axis, the metric is on the y-axis, and each inference mode keeps the same color across milestone and trace figures. Bars show the mean over reached seeds, and error bars use the configured across-seed uncertainty interval. Missing or censored milestones remain explicit in the summary tables and plot footnote.

By default, Pipeline II visualisation uses one sample standard deviation across seeds for milestone error bars and smoothed trace uncertainty bands:

```yaml
visualization:
  trace_smoothing_window_points: 100
  uncertainty_interval: std   # std | sem | ci95 | none
  min_uncertainty_samples: 2
  show_milestone_error_bars: true
  show_trace_uncertainty_bands: true
  show_raw_trace_uncertainty_bands: false
  trace_band_alpha: 0.16
```

Smoothed trace figures first smooth each seed independently, then plot the across-seed mean and uncertainty band. Raw appendix traces keep uncertainty bands disabled by default so the raw noise remains inspectable rather than hidden under wide fills.

## Pipeline II safety checks

The training stage is intentionally strict. It aborts if:

- prepared split/schedule/checkpoint artifacts are missing;
- compiled generated Python artifacts are missing;
- the generated SPLL probability is detached or not a torch tensor;
- the preflight call does not produce finite nonzero gradients;
- true-sum probability or loss becomes `NaN`, `inf`, or negative;
- gradients contain `NaN` or `inf`.

If compiled artifacts are missing, `train` refuses to compile automatically and prints the Rosetta/x86 compile command instead.

## Pipeline II config inheritance

`mnist_spll_common.load_config(...)` supports one `extends` level. This is used by `mnist_spll_training_smoke_config.yaml`.

Merge semantics:

- dictionaries deep-merge;
- lists replace completely;
- nested `extends` is rejected;
- `config_used.yaml` stores the fully resolved config.

Whenever `mnist_spll_training_config.yaml` changes, the smoke override must be checked and updated in the same patch so it remains a fast representative of the default config.
