# MNIST + SPLL pipeline

This bundle contains:

- `train_mnist.py`: trains a configurable CNN on a 70/20/10 split of the full MNIST pool and exports `mnist.pt` only if the configured test-accuracy threshold is reached.
- `run_spll_sum_experiments.py`: stages MNIST sum tasks, compiles SPLL programs, and runs exact plus approximate inference. It now supports separate commands for each stage.
- `mnist_spll_common.py`: shared model/config utilities.
- `mnist_spll_config.yaml`: the joint config for both scripts.

## What the second script actually approximates

In this repo, `-k/--topKCutoff` is **not** a literal top-k class count. It is a **probability cutoff in the range 0..1** that prunes low-probability branches during inference. Exact inference is represented by `null` in `approximation_thresholds`.

## Recommended Apple Silicon workflow

Use three separate steps:

1. **Train** in a native arm64 environment so PyTorch can use **MPS**.
2. **Stage + compile** under Rosetta/x86_64 so Stack can build the SPLL programs with the repo's older GHC toolchain.
3. **Infer** either
   - under Rosetta too, or
   - back in your native arm64 env to let `readMNist` use **MPS**, as long as the compiled Python artifacts already exist.

The split works because the compile step produces generated Python files (`program.py` + `pythonLib.py`). The inference step only imports those files and injects your current-process `readMNist` function.

## Commands

From this folder:

```bash
cd /Users/lisztomaniacae/IdeaProjects/spll-approx-comp/mnist_spll_pipeline
```

Use these environments:

- native Apple Silicon / MPS: `source .venv-train-arm64/bin/activate`
- Rosetta / x86_64 compile env: `source .venv-spll-x86/bin/activate`


### Train

Run in your native arm64 environment:

```bash
source .venv-train-arm64/bin/activate
python3 train_mnist.py --config mnist_spll_config.yaml
```

### Stage experiments

This samples the MNIST addition tasks once and writes `sampled_experiments.json` plus the `.spll` source files.

Run in your native arm64 environment:

```bash
source .venv-train-arm64/bin/activate
python3 run_spll_sum_experiments.py --config mnist_spll_config.yaml stage
```

### Compile only

Run this in your Rosetta/x86_64 environment:

```bash
arch -x86_64 zsh
cd /Users/lisztomaniacae/IdeaProjects/spll-approx-comp/mnist_spll_pipeline
source .venv-spll-x86/bin/activate
python run_spll_sum_experiments.py --config mnist_spll_config.yaml compile
```

This reads the already staged experiments and compiles only the required `(term_count, cutoff)` targets. Existing compiled outputs are reused unless `force_recompile: true`.

### Infer only

Run this after compilation. This command **does not compile**; it requires the generated `program.py` files to already exist.

Run in your native arm64 environment to let PyTorch use MPS:

```bash
source .venv-train-arm64/bin/activate
python3 run_spll_sum_experiments.py --config mnist_spll_config.yaml infer
```

This is the command you can rerun from your native arm64/MPS environment after compiling under Rosetta.

### One-shot run

For the old behavior:

```bash
source .venv-train-arm64/bin/activate
python run_spll_sum_experiments.py --config mnist_spll_config.yaml all
```

Use this only if you want one command to do everything. On Apple Silicon, the explicit `stage -> compile -> infer` split is usually safer.

If you omit the command, `all` is used by default.

## Progress output

The script now prints explicit stage markers and progress bars for:

- **Staging**: progress over sampled MNIST sum experiments
- **Compile**: progress over all generated `(term_count, cutoff)` compilation targets
- **Load compiled**: progress while importing cached compiled Python targets
- **Inference**: progress over all `(experiment, cutoff)` posterior runs, with per-run timing
- **Posterior**: progress inside each individual inference run over candidate sums `0..9*n_terms`

You can disable all bars with:

```yaml
inference:
  show_progress: false
```

Or keep the outer bars but disable the per-run posterior bar with:

```yaml
inference:
  show_inner_progress: false
```

## Config knobs relevant to the split

```yaml
inference:
  force_recompile: false
  stack_arch: x86_64
```

- `force_recompile: false` means existing compiled `program.py` files are reused.
- `stack_arch: x86_64` makes the compile step call `stack --arch x86_64 ...`, which is useful on Apple Silicon for this repo's GHC 8.10.4 setup.

## Notes

- `repo_root` in the YAML should point to your local NeST checkout.
- The compile command expects `stack` to be available on your shell PATH.
- The infer command does **not** need Stack, but it will fail if compiled outputs are missing.
- The default `terms_per_sum_max: 4` is intentionally conservative because exact inference gets expensive as the number of summed digits grows.

## Handy activation snippets

Switch back to native arm64 shell work:

```bash
deactivate
source .venv-train-arm64/bin/activate
```

Open a Rosetta shell for compilation:

```bash
arch -x86_64 zsh
cd /Users/lisztomaniacae/IdeaProjects/spll-approx-comp/mnist_spll_pipeline
source .venv-spll-x86/bin/activate
```

Leave the Rosetta env again:

```bash
deactivate
exit
```
