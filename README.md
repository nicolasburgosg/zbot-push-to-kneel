<div align="center">
<h1>K-Scale Z-Bot Benchmark</h1>
<p>Train and deploy your own z-bot controller in 700 lines of Python</p>
<h3>
  <a href="https://url.kscale.dev/leaderboard">Leaderboard</a> ·
  <a href="https://url.kscale.dev/docs">Documentation</a> ·
  <a href="https://github.com/kscalelabs/ksim/tree/master/examples">K-Sim Examples</a>
</h3>
</div>

## CPU Training (Laptop-Optimized)

This fork includes laptop-optimized defaults for training on CPU (tested on M2 Air). Key changes from the upstream template:

| Parameter | GPU Default | Laptop Default |
|-----------|------------|----------------|
| `num_envs` | 4096 | 8 |
| `batch_size` | 256 | 4 |
| `hidden_size` | 128 | 64 |
| `depth` | 5 | 3 |
| `num_mixtures` | 5 | 3 |

Model size reduced from ~1.1M to ~195K parameters. Each training step takes ~1-2 minutes on M2 Air.

```bash
# Laptop training (uses defaults)
python -m train

# GPU training (if available)
python -m train num_envs=256 batch_size=64 hidden_size=128 depth=5 num_mixtures=5
```

### Additional Changes

- **Local `MixtureOfGaussians`**: Re-implemented since it was removed in ksim 0.2.10+
- **Custom reward classes**: `NaiveForwardReward`, `LateralVelocityPenalty`, etc.
- **Bug fix**: `mode` property access in action distribution

## Getting Started

1. Read through the [ksim examples](https://github.com/kscalelabs/ksim/tree/master/examples)
2. Create a new repository from this template by clicking [here](https://github.com/new?template_name=kscale-zbot-benchmark&template_owner=kscalelabs)
3. Make sure you have installed `git-lfs`:

```bash
sudo apt install git-lfs  # Ubuntu
brew install git-lfs  # MacOS
```

4. Clone the new repository you create from this template:

```bash
git clone git@github.com:<YOUR USERNAME>/kscale-zbot-benchmark.git
cd kscale-zbot-benchmark
```

5. Create a new Python environment (we require Python 3.11 or later)
6. Install the package with its dependencies:

```bash
pip install -r requirements.txt
pip install 'jax[cuda12]'  # If using GPU machine, install Jax CUDA libraries
```

7. Train a policy:

```bash
python -m train
```

8. Convert the checkpoint to a `kinfer` model:

```bash
python -m convert /path/to/ckpt.bin /path/to/model.kinfer
```

9. Visualize the converted model:

```bash
kinfer-sim assets/model.kinfer kbot --save-video assets/video.mp4
```

## Troubleshooting

If you encounter issues, please consult the [ksim documentation](https://docs.kscale.dev/docs/ksim#/) or reach out to us on [Discord](https://url.kscale.dev/docs).

## Tips and Tricks

To see all the available command line arguments, use the command:

```bash
python -m train --help
```

To visualize running your model without using `kos-sim`, use the command:

```bash
python -m train run_mode=view
```

This repository contains a pre-trained checkpoint of a model which has been learned to be robust to pushes, which is useful for both jump-starting model training and understanding the codebase. To initialize training from this checkpoint, use the command:

```bash
python -m train load_from_ckpt_path=assets/ckpt.bin
```

You can visualize the pre-trained model by combining these two commands:

```bash
python -m train load_from_ckpt_path=assets/ckpt.bin run_mode=view
```