# MNIST + SPLL Pipeline

This folder contains the Python-side experiment code for the thesis comparison between **exact SPLL inference** and **approximate SPLL inference** on MNIST digit addition.

There are two separate pipelines:

| Pipeline | Entrypoint | Purpose | Stage order |
|---|---|---|---|
| Pipeline I: inference evaluation | `run_spll_pipeline.py` | Train/evaluate MNIST models, then compare exact vs approximate SPLL inference at test time. | `train -> compile -> stage -> infer -> visualize` |
| Pipeline II: generated-SPLL training | `run_spll_training_pipeline.py` | Train an MNIST model from sum labels by calling generated SPLL inference inside the optimizer loop. | `prepare -> compile -> train -> visualize` |

For detailed architecture notes, schema details, caveats, and patch discipline, read [`KNOWLEDGE_BASE.md`](KNOWLEDGE_BASE.md). Keep this README as an operator guide, but do not remove either pipeline from it.

---

## 1. Setup on Apple Silicon

Use explicit interpreter paths. Do not rely on a shell alias such as `python -> /opt/homebrew/bin/python@3.11`.

Run all commands below from `mnist_spll_pipeline/` unless stated otherwise.

### Native arm64 environment

Use this environment for training, staging, inference, and visualization.

```bash
python3 -m venv --copies .venv-train-arm64
./.venv-train-arm64/bin/python -m pip install --upgrade pip setuptools wheel
./.venv-train-arm64/bin/python -m pip install numpy PyYAML Pillow matplotlib torch torchvision
```

Verify:

```bash
./.venv-train-arm64/bin/python -c "import platform, torch; print(platform.machine(), torch.__version__)"
```

Expected machine value: `arm64`.

### Rosetta/x86_64 compile environment

Use this environment for SPLL compilation because the Haskell/SPLL stack is normally run under Rosetta.

```bash
arch -x86_64 zsh -f
cd /path/to/spll-approx-comp/mnist_spll_pipeline
arch -x86_64 /usr/local/bin/python3.11 -m venv --copies .venv-spll-x86
./.venv-spll-x86/bin/python -m pip install --upgrade pip setuptools wheel
./.venv-spll-x86/bin/python -m pip install numpy PyYAML Pillow matplotlib torch torchvision
```

Verify:

```bash
./.venv-spll-x86/bin/python -c "import platform, torch; print(platform.machine(), torch.__version__)"
```

Expected machine value: `x86_64`.

`stack` must also be available in the Rosetta shell.

---

## 2. Pipeline I: inference evaluation

Pipeline I evaluates trained MNIST models under exact and approximate SPLL inference. It intentionally uses a custom split over the combined official MNIST train+test pool: the configured ratios create train, validation/model-selection, and inference subsets from that pool. Do not describe Pipeline I outputs as official MNIST-test results unless the split logic is changed.

### Files

| File | Role |
|---|---|
| `mnist_spll_config.yaml` | Main config. |
| `run_spll_pipeline.py` | Stage dispatcher. |
| `train_mnist.py` | Trains MNIST model variants. |
| `compile_spll.py` | Generates and compiles SPLL inference artifacts. |
| `stage_experiments.py` | Samples fixed digit-addition experiments. |
| `infer_experiments.py` | Runs posterior inference and writes raw JSON. |
| `visualize_results.py` | Writes derived tables and plots. |
| `mnist_spll_pipeline_core.py` | Shared Pipeline I helpers. |

`run_spll_sum_experiments.py` is a legacy wrapper. Prefer `run_spll_pipeline.py`.

### Commands

Train in native arm64:

```bash
./.venv-train-arm64/bin/python run_spll_pipeline.py --config mnist_spll_config.yaml train
```

Compile in Rosetta/x86_64:

```bash
arch -x86_64 zsh -f
cd /path/to/spll-approx-comp/mnist_spll_pipeline
./.venv-spll-x86/bin/python run_spll_pipeline.py --config mnist_spll_config.yaml compile
```

Stage experiments in native arm64:

```bash
./.venv-train-arm64/bin/python run_spll_pipeline.py --config mnist_spll_config.yaml stage
```

Run inference in native arm64:

```bash
./.venv-train-arm64/bin/python run_spll_pipeline.py --config mnist_spll_config.yaml infer
```

Visualize in native arm64:

```bash
./.venv-train-arm64/bin/python run_spll_pipeline.py --config mnist_spll_config.yaml visualize
```

`all` exists, but the stage-by-stage workflow is safer on Apple Silicon because `compile` normally runs in the x86_64 environment.

### Main outputs

Default output root:

```text
outputs/spll_experiments/
```

Important artifacts:

| Path | Produced by | Contents |
|---|---|---|
| `compile_manifest.json` | `compile` | Compiled SPLL target inventory. |
| `staged_experiments.json` | `stage` | Fixed digit-addition inputs. |
| `inference_manifest.json` | `infer` | Inference run metadata. |
| `inference_runs.json` | `infer` | Raw posterior/runtime/branch-count records. |
| `visualization/tables/*.csv` | `visualize` | Derived result tables. |
| `visualization/figures/**/*.png` | `visualize` | Main and appendix figures. |

Treat `inference_runs.json` as the main empirical artifact. Visualization should derive results from it rather than rerunning inference.

---

## 3. Pipeline II: training through generated SPLL inference

Pipeline II tests whether SPLL approximation changes the **training process itself**. It trains an MNIST base model from **sum labels only** by calling the generated SPLL Python artifact inside each optimizer step.

### Files

| File | Role |
|---|---|
| `mnist_spll_training_config.yaml` | Default Pipeline II config. |
| `mnist_spll_training_smoke_config.yaml` | Small smoke-test override. |
| `run_spll_training_pipeline.py` | Stage dispatcher. |
| `prepare_spll_training.py` | Builds split/schedule manifests and initial checkpoints. |
| `compile_spll_training.py` | Generates and compiles exact/approximate SPLL training artifacts. |
| `train_spll_generated.py` | Trains through generated SPLL artifacts. |
| `visualize_spll_training.py` | Writes milestone tables and training plots. |
| `spll_training_core.py` | Shared Pipeline II helpers. |

### Design summary

Pipeline II uses the official MNIST training partition only. It builds a balanced split, creates sum-supervised schedules, compiles generated SPLL programs, then trains with loss based on the generated true-sum probability:

```text
loss = mean_i(-log(p_true_sum_i + epsilon))
```
my 
The generated SPLL artifact remains the source of truth for exact/approximate pruning. The training loop batches CNN evaluation outside the scalar generated-SPLL calls so gradients still flow back to the CNN.

Milestones and trace `step` values are measured in **sum cases seen**, not optimizer updates. `training.sum_batch_size: 1` recovers the old one-case-per-update behavior.

Validation can run asynchronously if enabled in the config. Milestone checkpoints are written from validated snapshots.

### Smoke-test commands

Prepare in native arm64:

```bash
./.venv-train-arm64/bin/python run_spll_training_pipeline.py --config mnist_spll_training_smoke_config.yaml prepare
```

Compile in Rosetta/x86_64:

```bash
arch -x86_64 zsh -f
cd /path/to/spll-approx-comp/mnist_spll_pipeline
./.venv-spll-x86/bin/python run_spll_training_pipeline.py --config mnist_spll_training_smoke_config.yaml compile
```

Train and visualize in native arm64:

```bash
./.venv-train-arm64/bin/python run_spll_training_pipeline.py --config mnist_spll_training_smoke_config.yaml train
./.venv-train-arm64/bin/python run_spll_training_pipeline.py --config mnist_spll_training_smoke_config.yaml visualize
```

### Default-run commands

Use the same stage split with the default config:

```bash
./.venv-train-arm64/bin/python run_spll_training_pipeline.py --config mnist_spll_training_config.yaml prepare
```

```bash
arch -x86_64 zsh -f
cd /path/to/spll-approx-comp/mnist_spll_pipeline
./.venv-spll-x86/bin/python run_spll_training_pipeline.py --config mnist_spll_training_config.yaml compile
```

```bash
./.venv-train-arm64/bin/python run_spll_training_pipeline.py --config mnist_spll_training_config.yaml train
./.venv-train-arm64/bin/python run_spll_training_pipeline.py --config mnist_spll_training_config.yaml visualize
```

### Main outputs

Default output root:

```text
outputs/spll_training/
```

Smoke-test output root:

```text
outputs/spll_training_smoke/
```

Important artifacts:

| Path | Contents |
|---|---|
| `config_used.yaml` | Fully resolved config for the run. |
| `data_split_manifest.json` | Balanced source/validation split. |
| `schedules/*.json` and `schedules/previews/*.jsonl` | Sum-supervised training schedules. |
| `initial_checkpoints/*.pt` | Shared initial model checkpoints. |
| `generated/spll_programs/*.spll` | Generated SPLL sum programs. |
| `generated/compiled_python/**/program.py` | Generated Python inference artifacts. |
| `runs/*/train_trace.csv` | Training trace. |
| `runs/*/validation_trace.csv` | Validation trace. |
| `runs/*/milestones.json` | Reached or censored milestones. |
| `runs/*/checkpoints/*.pt` | Milestone/final checkpoints. |
| `visualization/tables/*.csv` | Milestone and aggregate tables. |
| `visualization/figures/**/*.png` | Training and milestone figures. |

The training stage is intentionally strict: it refuses to run if prepared or compiled artifacts are missing, if generated probabilities are detached/non-finite, or if gradients become invalid.

---

## 4. Important config semantics

### Approximation thresholds

Approximation settings are represented by cutoff values in the config.

| YAML value | Meaning |
|---|---|
| `null` | exact baseline; no SPLL pruning flag is passed |
| `0.0` | approximate code path with zero cutoff; useful as overhead baseline, but not identical to exact |
| positive number, e.g. `0.01` | approximate inference with pruning threshold passed via `-k` |

### Inference-time `readMNist` caching

Pipeline I has a config switch for the cache around generated-SPLL neural calls:

```yaml
inference:
  read_mnist_cache_policy: uncached
  read_mnist_warmup_calls: 0
```

Allowed cache values:

| Value | Meaning |
|---|---|
| `uncached` | Default thesis timing policy. Every generated-SPLL `readMNist` call runs the MNIST model; no prediction cache is installed around inference. |
| `run_scoped_no_cross_run_cache` | Sensitivity/debug policy. Use a fresh cache for each timed full-posterior or true-candidate query. This avoids repeated CNN calls inside one generated-SPLL query without sharing cache state between exact and cutoff runs. |
| `precomputed_per_measurement` | Inference-isolation policy. Before each timed full-posterior or true-candidate query, compute neural probabilities for that query's image paths into a fresh lookup table, then time only the generated-SPLL query. Precompute time is logged separately and never shared between exact and cutoff runs. |

`read_mnist_warmup_calls` defaults to `0` and should stay `0` for thesis-facing runtime benchmarks. A positive value is only a diagnostic option for intentionally excluding one-off cold-start effects; it runs untimed base `readMNist` calls before a model variant starts measured inference, without installing any inference cache.

Aliases such as `run_scoped`, `cached`, `precomputed`, `none`, or `off` are accepted, but prefer the canonical values above in committed configs.

`-k/--topKCutoff` is a probability cutoff in `[0, 1]`. It is **not** a literal top-k class count.

### Diagnosing cutoff `0.0` timing artifacts

Use `diagnose_cutoff_zero_mwe.py` when the approximate `0.0` path appears faster than exact even though branch counts and probabilities should be almost identical. The script runs the exact and `cutoff_0p0` generated Python artifacts repeatedly, writes per-call timings, and can also write `cProfile` and `dis.dis` reports.

First isolate generated-Python/interpreter overhead with a fake list-valued `readMNist`:

```bash
./.venv-train-arm64/bin/python diagnose_cutoff_zero_mwe.py \
  --config mnist_spll_config.yaml \
  --n-terms 2 \
  --read-mnist-source uniform-list \
  --query full-posterior \
  --repeats 100 \
  --sequence exact,exact,cutoff,cutoff \
  --disassemble \
  --profile-repeats 1000
```

Then add PyTorch tensor dispatch without the real CNN:

```bash
./.venv-train-arm64/bin/python diagnose_cutoff_zero_mwe.py \
  --config mnist_spll_config.yaml \
  --n-terms 2 \
  --read-mnist-source uniform-tensor \
  --query full-posterior \
  --repeats 100 \
  --sequence exact,cutoff,exact,cutoff
```

Finally test the real model and staged images:

```bash
./.venv-train-arm64/bin/python diagnose_cutoff_zero_mwe.py \
  --config mnist_spll_config.yaml \
  --n-terms 2 \
  --experiment-id 1 \
  --model-id acc90 \
  --read-mnist-source real \
  --query full-posterior \
  --repeats 100 \
  --sequence exact,cutoff,exact,cutoff
```

Outputs are written under:

```text
outputs/spll_experiments/diagnostics/cutoff_zero_mwe/<timestamp>/
```

The most useful files are `timings.csv`, `block_summary.csv`, `summary.json`, optional `profile_*.txt`, and optional `dis_*_forward.txt`.

### Config inheritance

`mnist_spll_common.load_config(...)` supports one `extends` level. This is used by `mnist_spll_training_smoke_config.yaml`.

Merge semantics:

- dictionaries deep-merge;
- lists replace completely;
- nested `extends` is rejected;
- `config_used.yaml` stores the fully resolved config.

Whenever `mnist_spll_training_config.yaml` changes, check the smoke override in the same patch.

### Progress bars

Progress bars can be disabled in the config:

```yaml
inference:
  show_progress: false
  show_inner_progress: false
```

---

## 5. Notes for future changes

- Update [`KNOWLEDGE_BASE.md`](KNOWLEDGE_BASE.md) in the same patch whenever behavior, workflow, schemas, commands, caveats, or architecture change.
- Keep exact and approximate inference comparable. Avoid process-global caching or other run-order effects that can make one cutoff look faster for external reasons.
- Keep both Pipeline I and Pipeline II visible in this README. Detailed internals belong in the knowledge base, not here.
