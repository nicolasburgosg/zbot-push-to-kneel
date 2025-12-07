# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Zeroth-Fall is a K-Scale Z-Bot reinforcement learning training framework for the Zeroth-01 biped robot. It uses PPO (Proximal Policy Optimization) with JAX/KSIM for training neural network controllers with push recovery and controlled kneel capabilities.

## Common Commands

### Environment Setup
```bash
# Activate conda environment
source /opt/homebrew/Caskroom/miniconda/base/etc/profile.d/conda.sh && conda activate zbot

# Or use venv
pip install -r requirements.txt
pip install 'jax[cuda12]'  # GPU support
```

### Development
```bash
make install          # Install dependencies
make install-dev      # Install dev tools (black, ruff, mypy)
make format           # Format code with black and ruff
make static-checks    # Run linting and type checks
```

### Training
```bash
python -m train                                      # Train with laptop defaults (8 envs, small model)
python -m train run_mode=view                        # Train with visualization
python -m train load_from_ckpt_path=path/to/ckpt.bin # Resume from checkpoint
python -m train --help                               # Show all arguments

# GPU production training (if available)
python -m train num_envs=256 batch_size=64 hidden_size=128 depth=5 num_mixtures=5
```

### Deployment
```bash
python -m convert /path/to/ckpt.bin /path/to/model.kinfer
kinfer-sim assets/model.kinfer kbot --save-video assets/video.mp4
```

## Architecture

### Model Structure
Default config is optimized for laptop development (M2 Air):
- **Actor**: 3-layer GRU RNN outputting mixture-of-gaussians (3 components per joint)
- **Critic**: 3-layer GRU RNN outputting scalar value
- **Hidden size**: 64 (use 128 for production GPU training)
- **Actions**: 20 joint position targets
- **Actor Input**: 50 dims (joint pos/vel, IMU orientation, commands)
- **Critic Input**: 484 dims (actor input + COM, forces, base pose)

For production training on GPU, use: `hidden_size=128, depth=5, num_mixtures=5`

### Task Components (train.py)
The `ZbotWalkingTask` class defines:
- **Observations**: Joint states, IMU, base pose, foot contacts (dict-based API)
- **Rewards**: Velocity tracking, orientation, energy penalties, stability
- **Physics Randomizers**: Friction, mass, damping, joint offsets
- **Events**: Push disturbances via `LinearPushEvent`
- **Resets**: Joint position/velocity randomization

### Key Custom Classes
| Class | Purpose |
|-------|---------|
| `MixtureOfGaussians` | Action distribution (local implementation) |
| `NaiveForwardReward` | Simple forward velocity reward |
| `FeetAirtimeReward` | Encourage proper gait timing |
| `FeetechActuators` | Servo simulation with torque limits |

### Joint Configuration
20 joints total: 6 per leg (hip yaw/roll/pitch, knee, ankle pitch/roll), 4 per arm (shoulder pitch/roll, elbow, gripper)

## KSIM 0.2.10+ API Notes

The template was updated for ksim 0.2.10. Key patterns:
- Methods like `get_observations()`, `get_rewards()`, `get_events()` return `dict[str, T]` not lists
- Observation keys must match expected names (e.g., `"joint_position_observation"`)
- Use `ksim.AdditiveGaussianNoise(std=...)` for noise, not float values
- `LinearPushEvent` replaces `PushEvent` with `linvel` parameter
- `BadZTermination` uses `min_z`/`max_z` instead of `unhealthy_z_lower`/`upper`

## Code Style
- Black formatter (120 char line length)
- Ruff linter with strict rules
- Mypy with strict mode
- Google-style docstrings
- JAX functional patterns with `@jax.jit`
- Equinox modules (`eqx.Module`)
- Frozen attrs dataclasses (`@attrs.define(frozen=True)`)

## Development Approach

This is a **portfolio project** for robotics masters applications. The goal is demonstrating:
1. **Engineering quality** - task design, state machine, reward shaping
2. **Working pipeline** - training runs, checkpoints, evaluation harness
3. **Documentation** - clear writeup of design decisions and results

NOT massive scale training. The laptop-optimized config proves the pipeline works and shows how reward/observation changes affect behavior.

## Current Development Status

Working on push-recovery + controlled kneel controller:
1. **RECOVER mode**: Stay upright after pushes via stepping/ankle strategies
2. **KNEEL mode**: Lower COM safely when recovery is impossible
3. Mode switching via deterministic state machine

See `design.md` for full implementation plan.

---

## Session Summary (2025-12-06)

### What Was Accomplished

1. **Laptop-Optimized Config Applied**
   - Changed `train.py` defaults: `hidden_size=64`, `depth=3`, `num_mixtures=3`
   - Updated main block: `num_envs=8`, `batch_size=4`, `num_passes=2`, `rollout_length_seconds=2.0`
   - Reduced physics iterations (4 instead of 8) for faster compile
   - Model size reduced from ~1.1M to ~195K parameters

2. **Bug Fix: `mode()` Property Call**
   - Fixed `train.py:1929`: Changed `action_dist_j.mode()` to `action_dist_j.mode`
   - The `MixtureOfGaussians.mode` is a `@property`, not a method

3. **Training Pipeline Verified Working**
   - `run_mode=view` works - MuJoCo viewer opens, robot stays upright
   - `run_mode=train` (default) works - logs to `zbot_walking_task/run_XXX/`
   - Each training step takes ~1-2 minutes on M2 Air CPU

4. **Documentation Updated**
   - `CLAUDE.md`: Added portfolio focus section, updated model structure
   - `design.md`: Added portfolio context, updated training commands, revised success criteria

### Current State

- **Active training run**: `zbot_walking_task/run_028` (background process c9f445)
- **Model**: 195,061 parameters (laptop-optimized)
- **Config**: 8 envs, batch_size=4, 3-layer GRU, 3 mixtures

### Key Files Modified

| File | Changes |
|------|---------|
| `train.py:1233-1253` | Config defaults (hidden_size=64, depth=3, num_mixtures=3) |
| `train.py:1937-1958` | Main block (num_envs=8, batch_size=4, shorter rollouts) |
| `train.py:1929` | Bug fix: `mode` property access |

### How to Resume Work

```bash
# Activate environment
source /opt/homebrew/Caskroom/miniconda/base/etc/profile.d/conda.sh && conda activate zbot

# Check if training is still running
ps aux | grep "python -m train"

# View training logs
tail -f zbot_walking_task/run_028/logs.txt

# Start TensorBoard
tensorboard --logdir=zbot_walking_task/run_028/tensorboard

# Start new training run
python -m train

# View trained policy
python -m train run_mode=view load_from_ckpt_path=zbot_walking_task/run_028/checkpoints/ckpt.X.bin
```

### Remaining Work (from design.md)

1. **Evaluation harness**: Implement `eval/push_battery.py`
2. **State machine**: Implement `state_machine/mode_switch.py` for RECOVER→KNEEL
3. **Custom task**: Create `tasks/push_to_kneel_task.py` with mode input
4. **Documentation**: Create `docs/results.md` with plots and videos

### Known Issues

- JAX CPU AOT loader warnings (harmless, just cache mismatch)
- `equinox.static_field` deprecation warnings (cosmetic)
- `warp` module not found (optional GPU acceleration, not needed)
