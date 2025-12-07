# Training Results

## Run Summary

| Run | Steps | Duration | Final Reward | Notes |
|-----|-------|----------|--------------|-------|
| run_028 | 518 | ~4.5 hrs | 2.2 | Laptop CPU (M2 Air) |

## Learning Curves

### Total Reward
![Total Reward](../results/plots/reward_total.png)

Total reward increased from ~1.0 to ~2.2 over 518 training steps, indicating successful learning.

### Forward Velocity
![Forward Velocity](../results/plots/reward_forward.png)

Forward velocity reward improved from ~0.1 to ~0.7, showing the robot learned to walk forward.

### Individual Reward Components

| Reward | Start | End | Trend |
|--------|-------|-----|-------|
| `reward/_total` | 1.0 | 2.2 | Improving |
| `reward/forward` | 0.1 | 0.7 | Improving |
| `reward/arm_pose` | -0.3 | -0.13 | Improving |
| `reward/feet_airtime` | -0.12 | -0.07 | Improving |
| `reward/feet_too_close` | -0.11 | -0.01 | Improving |
| `reward/feet_orient` | 0.046 | 0.022 | Oscillating |

## Demo Videos

### Walking Demo
![Walking Demo](../results/demo.gif)

The trained policy shows basic forward walking with reasonable gait timing.

## Observations

### What Worked
- Forward velocity reward scaling (5.0) was effective
- Feet airtime reward (2.5) encouraged alternating gait
- Arm pose penalty (-2.0) kept arms stable

### What Could Improve
- Feet orientation dropped mid-training (may need more weight)
- Lateral velocity still present (penalty could be stronger)
- More training steps would likely improve stability

## Checkpoint

Best checkpoint saved at: `results/ckpt.bin` (step 518)

To view:
```bash
python -m train run_mode=view load_from_ckpt_path=results/ckpt.bin
```

## Training Configuration

```yaml
num_envs: 8
batch_size: 4
hidden_size: 64
depth: 3
num_mixtures: 3
learning_rate: 3e-4
rollout_length_seconds: 2.0
```
