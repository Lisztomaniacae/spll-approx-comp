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
| `run_spll_pipeline.py` | Lazy stage dispatcher. |
| `train_mnist.py` | Trains MNIST model variants. |
| `compile_spll.py` | Generates and compiles SPLL inference artifacts. |
| `stage_experiments.py` | Samples fixed digit-addition experiments. |
| `infer_experiments.py`, `inference_engine.py` | Run posterior inference and write raw JSON. |
| `visualize_results.py` | Coordinates derived tables and plots. |
| `pipeline1_config.py`, `pipeline1_models.py`, `pipeline1_data.py` | Pipeline I configuration, model-variant, and data boundaries. |
| `spll_artifacts.py` | SPLL source generation, compilation, loading, and generated-value extraction. |
| `pipeline1_analysis.py`, `pipeline1_plotting.py` | Metric derivation and figure construction. |
| `pipeline_support.py`, `mnist_model.py` | Lightweight shared infrastructure and torch-dependent MNIST model code. |
| `mnist_spll_common.py`, `mnist_spll_pipeline_core.py` | Compatibility facades for historical scripts and the unchanged diagnostic utility. |

The broken legacy wrapper `run_spll_sum_experiments.py` was removed. Use `run_spll_pipeline.py`.

### Commands

Train in native arm64:

```bash
./.venv-train-arm64/bin/python run_spll_pipeline.py --config mnist_spll_config.yaml train
```

Compile in Rosetta/x86_64:

```bash
arch -x86_64 zsh -f
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
| `visualization/figures/**/*.png` | `visualize` | Main and appendix figures. The standard runtime centerpiece remains `main_text/runtime_accuracy_tradeoff_by_terms.png`; `main_text/mnist_lookup_accuracy_tradeoff_by_terms.png` reports mean generated `readMNist` calls against retained accuracy; biased runtime counterparts are isolated in `main_text/runtime_accuracy_tradeoff_biased_models_by_terms.png`. |

Treat `inference_runs.json` as the main empirical artifact. Visualization should derive results from it rather than rerunning inference.

The default Pipeline I config trains three standard models and three label-biased counterparts at the same 50%, 70%, and 90% validation targets. YAML merge anchors keep each biased model's architecture and training budget identical to its standard counterpart; only the training-label distribution changes. The biased runs are tagged with `visualization_group: biased_tradeoff`, so every existing plot continues to use only standard models and only the dedicated biased runtime–accuracy tradeoff figure consumes the biased rows.

---

## 3. Pipeline II: training through generated SPLL inference

Pipeline II tests whether SPLL approximation changes the **training process itself**. It trains an MNIST base model from **sum labels only** by calling the generated SPLL Python artifact inside each optimizer step.

### Files

| File | Role |
|---|---|
| `mnist_spll_training_config.yaml` | Default Pipeline II config. |
| `mnist_spll_training_smoke_config.yaml` | Small smoke-test override. |
| `run_spll_training_pipeline.py` | Lazy stage dispatcher. |
| `prepare_spll_training.py` | Builds split/schedule manifests and initial checkpoints. |
| `compile_spll_training.py` | Generates and compiles exact/approximate SPLL training artifacts. |
| `train_spll_generated.py` | Coordinates the active fixed-budget training path. |
| `visualize_spll_training.py` | Coordinates tables and figures. |
| `pipeline2_config.py`, `pipeline2_data.py`, `pipeline2_artifacts.py` | Pipeline II configuration, dataset/schedule, and generated-artifact boundaries. |
| `pipeline2_runtime.py` | Differentiable direct-uncached `readMNist` runtime primitives. |
| `pipeline2_checkpoint_transfer.py` | Aggregate exact-checkpoint selection and approximate transfer runs. |
| `pipeline2_analysis.py`, `pipeline2_plotting.py` | Training-result aggregation and figure construction. |

### Design summary

Pipeline II uses the official MNIST training partition only. It builds a balanced split, creates sum-supervised schedules, compiles generated SPLL programs, then trains with loss based on the generated true-sum probability:

```text
loss = mean_i(-log(p_true_sum_i + epsilon))
```

The generated SPLL artifact remains the source of truth for exact/approximate pruning. Pipeline II deliberately uses a direct, uncached `readMNist` callback: every generated `readMNist(index)` invocation reloads that MNIST sample and executes a fresh CNN forward pass. There is no per-batch probability table, tensor LRU, memoization, or model-output lookup. This makes pruning-induced reductions in generated neural calls part of the measured training cost instead of hiding them behind precomputation.

Pipeline II supports only exact inference and fixed numeric cutoffs. Adaptive posterior-mass cutoff selection is intentionally not part of Pipeline II; that mechanism remains available in Pipeline I.

Trace `step` values are measured in **training iterations / sum cases seen**. The redesigned benchmark keeps `training.sum_batch_size: 1`, so one step is one iteration. Pure exact and pure approximate runs train for the configured fixed `max_steps`; there is no validation-driven early stopping.

Checkpoints are now computed after all pure exact seeds finish. The pipeline first averages the exact training `p(true_sum)` trace across seeds, then applies a strict full-window rolling mean over `checkpointing.rolling_window_updates` (default 100), and then finds crossings of `checkpointing.posterior_thresholds`. The first 100-point rolling value is emitted only after 100 updates; shorter prefix windows are not used. These aggregate posterior checkpoints become the exact anchors for checkpoint-transfer approximate runs. Held-out full-posterior validation remains available as old diagnostic plumbing, but it is disabled in the default redesigned benchmark and is not used for checkpointing or plots.

Checkpoint-transfer approximate runs load the per-seed exact step checkpoint at each aggregate anchor step and reuse the complete deterministic sum-case interval `s_A+1..s_B` that the aggregate exact curve used between anchors. Reaching or exceeding the next exact checkpoint's displayed rolling posterior value is recorded as diagnostic metadata, but it does not stop the continuation early. Therefore every green segment ends at the next anchor's training iteration, including when approximation rises above the target posterior before that iteration. In trajectory figures, exact aggregate posterior checkpoints are shown as distinct purple X markers on the displayed exact curve, and the green curve is drawn as separate pieces so segment restarts are visible. Checkpoint **steps** remain defined by `checkpointing.rolling_window_updates`, but marker heights and the visual green anchor are sampled from the currently displayed exact curve. Consequently, changing `visualization.trace_smoothing_window_points` cannot detach the X markers from the blue line.

Pipeline II thesis-facing combined trace figures use a shared figure-level legend, no gray subtitle under the title, no legend title, and slim lines for the pure exact, pure approximate, and approximate-continuation curves. Loss plots use data-driven logarithmic y-limits with padding; do not reintroduce fixed loss limits because low-loss checkpoints and continuation segments can be clipped. All thesis figures are exported as 300 dpi PNGs and should be sized as if they will be embedded at approximately A4 page width. Keep these conventions in sync with `plot_palette.py`, `pipeline1_plotting.py`, and `pipeline2_plotting.py` when changing the plots.

Posterior-checkpoint bar figures show a mode/checkpoint pair only when **every configured seed** reached that target and supplied the plotted metric values. A target reached by only a subset of seeds remains visible in the CSV/JSON aggregate tables through `reached_seed_count`, but its bars are omitted instead of displaying a successful-subset mean as though the run had finished.

### Smoke-test commands

Prepare in native arm64:

```bash
./.venv-train-arm64/bin/python run_spll_training_pipeline.py --config mnist_spll_training_smoke_config.yaml prepare
```

Compile in Rosetta/x86_64:

```bash
arch -x86_64 zsh -f
./.venv-spll-x86/bin/python run_spll_training_pipeline.py --config mnist_spll_training_smoke_config.yaml compile
```

Train and visualize in native arm64:

```bash
./.venv-train-arm64/bin/python run_spll_training_pipeline.py --config mnist_spll_training_smoke_config.yaml train
./.venv-train-arm64/bin/python run_spll_training_pipeline.py --config mnist_spll_training_smoke_config.yaml visualize
```

If the normal training stage completed but the checkpoint-transfer substage needs
to be rerun, use the recovery entry point without retraining the pure runs:

```bash
./.venv-train-arm64/bin/python run_spll_training_pipeline.py --config mnist_spll_training_smoke_config.yaml checkpoint-transfer
```

### Default-run commands

Use the same stage split with the default config:

```bash
./.venv-train-arm64/bin/python run_spll_training_pipeline.py --config mnist_spll_training_config.yaml prepare
```

```bash
arch -x86_64 zsh -f
./.venv-spll-x86/bin/python run_spll_training_pipeline.py --config mnist_spll_training_config.yaml compile
```

```bash
./.venv-train-arm64/bin/python run_spll_training_pipeline.py --config mnist_spll_training_config.yaml train
./.venv-train-arm64/bin/python run_spll_training_pipeline.py --config mnist_spll_training_config.yaml visualize
```

Recovery-only checkpoint-transfer run:

```bash
./.venv-train-arm64/bin/python run_spll_training_pipeline.py --config mnist_spll_training_config.yaml checkpoint-transfer
```

### Main outputs

Default output root:

```text
outputs/spll_training_direct_uncached/
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
| `aggregate_checkpoints/terms_*_exact_posterior_checkpoints.json` | Aggregate exact rolling-mean posterior checkpoints used as transfer anchors. |
| `runs/*/milestones.json` | Backward-compatible alias for posterior checkpoint crossings. |
| `runs/*/checkpoints/steps/step_*.pt`, `posterior_*.pt`, and `final.pt` | Exact step snapshots for aggregate-anchor transfer plus per-run posterior/final snapshots. |
| `visualization/tables/*.csv` | Milestone, aggregate, run-summary, and checkpoint-transfer tables. |
| `visualization/figures/**/*.png` | Fixed-budget loss/posterior, checkpoint-transfer, and MNIST-model-evaluation figures. |

The training stage is intentionally strict: it refuses to run if prepared or compiled artifacts are missing, if generated probabilities are detached/non-finite, or if gradients become invalid.

---

## 4. Important config semantics

### Approximation thresholds

Pipeline I accepts both fixed cutoff values and adaptive posterior-mass thresholds in `inference.approximation_thresholds`.

| YAML value | Meaning |
|---|---|
| `null` | exact baseline; no SPLL pruning flag is passed |
| `0.0` | approximate code path with zero cutoff; useful as overhead baseline, but not identical to exact |
| positive number, e.g. `0.01` | approximate inference with pruning threshold passed via `-k` |
| mapping with `top_k_cutoff: auto` | compile a separate adaptive approximate artifact named `cutoff_topk` with seed cutoff `0.0`, then tune the generated module's mutable `TOP_K_CUTOFF` to capture `posterior_mass_target` posterior mass |

Default adaptive setting:

```yaml
inference:
  adaptive_top_k:
    posterior_mass_target: 0.8
    probe_experiments: 20
    max_iterations: 14
    tolerance: 0.02
    min_cutoff: 0.0
    max_cutoff: 0.5
  approximation_thresholds:
    - null
    - 0.01
    - name: approx_mass_0p8
      top_k_cutoff: auto
      adaptive_top_k: true
      posterior_mass_target: 0.8
```

The adaptive search is deterministic bounded bisection over cutoff values in `[0.0, 0.5]`, not simulated annealing. Pipeline I tunes once per model, term count, cutoff mode, and adaptive threshold label, using the first configured staged probe experiments for that term count. It logs `runtime_top_k_cutoff`, `mean_surviving_posterior_mass`, `adaptive_cutoff_search_runtime_sec`, and visualizable bisection evaluations under `visualization/tables/adaptive_topk_search_trace.*`; timed posterior inference remains separate from cutoff-search runtime. The adaptive top-k search figures use a single shared legend across term-count panels instead of repeating the same legend inside each subplot. Adaptive runs use their own `cutoff_topk` artifact per term count, and fixed-cutoff runs still reset `TOP_K_CUTOFF` before every measured posterior so runtime cutoff state cannot leak across measurements.

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

Every measured full-posterior and true-candidate query records two separate counters:

- `read_mnist_lookup_calls`: generated `readMNist` invocations. This is the compiler/inference lookup count used by `mnist_lookup_accuracy_tradeoff_by_terms.png`.
- `read_mnist_model_evaluations`: underlying MNIST-model evaluations. For `uncached` and run-scoped caching this is the cache-miss count; for `precomputed_per_measurement` it is the number of precomputed unique images.

Do not interpret lookup reduction as neural-compute reduction under `precomputed_per_measurement`: approximation can reduce generated lookup accesses while the policy still evaluates every unique input image before the timed query.

Aliases such as `run_scoped`, `cached`, `precomputed`, `none`, or `off` are accepted, but prefer the canonical values above in committed configs.

`-k/--topKCutoff` is a probability cutoff in `[0, 1]`. It is **not** a literal top-k class count.

### Diagnosing cutoff `0.0` timing artifacts

Use `diagnose_cutoff_zero_mwe.py` when the approximate `0.0` path appears faster than exact even though branch counts and probabilities should be almost identical. The script runs the exact and `cutoff_0p0` generated Python artifacts repeatedly, writes per-call timings, decomposes each measured call into timed `readMNist` backend time versus residual generated-SPLL/inference time, and can also write `cProfile` and `dis.dis` reports.

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

The most useful files are `timings.csv`, `block_summary.csv`, `adjacent_time_savings.csv`, `all_pair_time_savings.csv`, `summary.json`, optional `profile_*.txt`, and optional `dis_*_forward.txt`.

`timings.csv` includes `read_mnist_calls`, `read_mnist_unique_images`, `read_mnist_runtime_sec`, `read_mnist_avg_call_sec`, `inference_runtime_sec`, `read_mnist_time_share`, and `inference_time_share`. `block_summary.csv` aggregates the same split, and `summary.json` adds adjacent-pair savings fields such as `inference_only_exact_over_cutoff_speedup` and `max_inference_only_savable_share_of_exact_total`.

### Config inheritance

`pipeline_support.load_config(...)` supports one `extends` level. This is used by `mnist_spll_training_smoke_config.yaml`.

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

Run the focused regression suite from the repository root after changing pipeline code:

```bash
PYTHONPATH=mnist_spll_pipeline python -m unittest discover -s mnist_spll_pipeline/tests -v
```

The tests cover deterministic source/case generation, config and path contracts, Pipeline I cache/adaptive behavior, Pipeline II direct-uncached model-call invariants, strict rolling windows, checkpoint selection/schema, analysis aggregation, lightweight imports, the diagnostic compatibility facades, and a tiny fixed-budget Pipeline II training run.

- Update [`KNOWLEDGE_BASE.md`](KNOWLEDGE_BASE.md) in the same patch whenever behavior, workflow, schemas, commands, caveats, or architecture change.
- Keep exact and approximate inference comparable. Avoid process-global caching or other run-order effects that can make one cutoff look faster for external reasons.
- Keep both Pipeline I and Pipeline II visible in this README. Detailed internals belong in the knowledge base, not here.
