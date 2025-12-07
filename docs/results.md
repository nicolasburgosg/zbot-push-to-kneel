# Training Results

## Run Summary

| Run | Steps | Duration | Final Reward | Notes |
|-----|-------|----------|--------------|-------|
| run_028 | ~550 | ~5 hrs | 2.2-2.4 | Laptop CPU (M2 Air) |

## Learning Curves

### Total Reward
![Total Reward](../results/plots/reward_total.jpeg)

Total reward started around 2.0, dipped during early exploration, then climbed steadily to plateau around 2.2-2.4 by step 550. The plateau suggests the policy has converged.

### Forward Velocity
![Forward Velocity](../results/plots/results_forward.jpeg)

Forward velocity reward improved from ~0.64 to ~0.78, with oscillations showing the robot exploring different gaits. The sustained high values indicate successful forward locomotion.

### Individual Reward Components

| Reward | Start | End | Trend |
|--------|-------|-----|-------|
| `reward/_total` | 2.0 | 2.2-2.4 | Converged |
| `reward/forward` | 0.64 | 0.78 | Improving |
| `reward/arm_pose` | -0.28 | -0.08 | Improving |
| `reward/feet_airtime` | -0.10 | -0.06 | Improving |
| `reward/lateral_vel_penalty` | 0.5 | 0.7 | Improving |
| `reward/stay_alive` | ~-0.01 | ~-0.006 | Stable |

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
