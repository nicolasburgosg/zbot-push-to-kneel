# Zeroth-01 Push-Recovery + Controlled Kneel Implementation Plan

## Portfolio Context

This is a **portfolio project** for robotics masters applications. Development is on a MacBook Air M2 (24GB RAM, CPU-only JAX). The goal is NOT massive-scale training, but demonstrating:

1. **Engineering quality** - task design, state machine, reward shaping
2. **Working pipeline** - training runs that show learning progress
3. **Evaluation harness** - metrics, push battery, video recordings
4. **Documentation** - clear design decisions and honest discussion of constraints

The laptop-optimized config (hidden_size=64, depth=3, num_mixtures=3, num_envs=8) proves the pipeline works. GPU training is optional for "polishing" results.

---

## Project Overview
Train and deploy a controller for Zeroth-01 that:
1. **RECOVER mode**: Stays upright after pushes via stepping/ankle strategies
2. **KNEEL mode**: Lowers COM safely when recovery is impossible

Architecture: **Explicit phase flag** with **deterministic state machine** for mode switching.

---
## Phase 0: Environment Setup

### 0.1 Install Prerequisites
```bash
# Install Homebrew packages
brew install git-lfs

# Initialize git-lfs
git-lfs install
```

### 0.2 Create Conda Environment
```bash
# Create environment with Python 3.11
conda create -n zbot python=3.11 -y
conda activate zbot
```

**Note**: JAX and MuJoCo will be installed via `requirements.txt` in Phase 1.

### 0.3 Verify MuJoCo Works
```python
# Quick test script
import mujoco
print(f"MuJoCo version: {mujoco.__version__}")
```

---

## Phase 1: Repository Setup

### 1.1 Create Repo from Template
1. Go to https://github.com/kscalelabs/ksim-gym-zbot
2. Click "Use this template" → Create new repository
3. Name: `zbot-push-to-kneel` (or similar)

### 1.2 Clone and Configure
```bash
# Clone into a subdirectory (safer than cloning into `.`)
cd /Users/nicolasburgos/Desktop/Projects
git clone git@github.com:<username>/zbot-push-to-kneel.git zeroth-fall
cd zeroth-fall

# Pull LFS assets (checkpoints, models)
git lfs pull

# Add upstream for future updates
git remote add upstream https://github.com/kscalelabs/ksim-gym-zbot.git
```

### 1.3 Install Dependencies (order matters)
```bash
# Install requirements first
pip install -r requirements.txt

# Then install JAX CPU build explicitly (avoids macOS issues)
pip install -U "jax[cpu]"
```

### 1.4 Project Structure (files to add)
```
zeroth-fall/
├── tasks/
│   └── push_to_kneel_task.py      # Custom RL task with mode input
├── configs/
│   └── push_to_kneel.yaml         # Hyperparameters, rewards, curriculum
├── eval/
│   ├── push_battery.py            # Deterministic push test suite
│   └── metrics.py                 # Success rate, peak accel, etc.
├── scripts/
│   ├── train_recover.sh           # Train RECOVER mode
│   ├── train_kneel.sh             # Train KNEEL mode
│   └── export_kinfer.sh           # Convert checkpoint to .kinfer
├── state_machine/
│   └── mode_switch.py             # Deterministic RECOVER→KNEEL logic
└── docs/
    ├── design.md                  # This design document
    ├── results.md                 # Videos, plots, metrics
    └── safety.md                  # Limits + test procedures
```

---

## Phase 2: Baseline Verification

### 2.1 Run Pretrained Checkpoint
```bash
# View the pretrained push-robust policy (run_mode=view for visualization)
python -m train load_from_ckpt_path=assets/ckpt.bin run_mode=view
```

### 2.2 Verify Simulation Renders
- Confirm MuJoCo viewer opens
- Confirm robot model loads correctly
- Test basic push responses

### 2.3 Understand Existing Task Structure
- Read the default task implementation
- Identify observation space (joint states, IMU, base pose)
- Identify action space (joint position targets)
- Note existing reward components

### 2.4 Scripted Kneel Baseline (Debug Reference)
Before training an RL kneel policy, implement a pure PD trajectory to a kneel pose. This serves as:
- A sanity check that the kneel pose is reachable
- A baseline to beat on smoothness/impact metrics
- A debugging tool when RL training goes wrong

```python
# scripts/scripted_kneel.py
def scripted_kneel(env, duration=2.0, dt=0.02):
    """
    Simple PD trajectory from standing to kneel pose.
    No learning—just interpolate joint targets.
    """
    standing_pose = env.get_default_pose()
    kneel_pose = KNEEL_JOINT_TARGETS  # from Phase 3.3

    steps = int(duration / dt)
    for i in range(steps):
        alpha = i / steps  # 0 → 1
        target = {}
        for joint in standing_pose:
            target[joint] = (1 - alpha) * standing_pose[joint] + alpha * kneel_pose[joint]
        env.set_joint_targets(target)
        env.step()

    return env.get_metrics()  # peak accel, contact impulse, etc.
```

Use this to establish baseline metrics:
- Peak base acceleration during scripted kneel
- Time to stable kneel
- Any torso/head contacts (should be zero if pose is valid)

---

## Phase 3: Custom Task Implementation

### 3.1 Observation Space Extension
Add `mode` input to policy observations:
```python
# One-hot encoding: [is_recover, is_kneel]
# OR scalar: 0.0 = RECOVER, 1.0 = KNEEL
mode_obs = jnp.array([1.0, 0.0])  # RECOVER mode
```

Full observation vector:
- Joint positions (16 joints)
- Joint velocities (16 joints)
- Base orientation (quaternion or euler)
- Base angular velocity
- IMU acceleration
- **Mode flag** (new)

### 3.2 Reward Components

#### RECOVER Mode Rewards
| Component | Description | Weight |
|-----------|-------------|--------|
| `upright_bonus` | Low roll/pitch angles | +2.0 |
| `stable_height` | Base height near nominal | +1.0 |
| `feet_contact` | Both feet on ground | +0.5 |
| `energy_penalty` | Minimize torque squared | -0.01 |
| `action_smoothness` | Penalize action jerk | -0.1 |
| `survival_bonus` | Per-step bonus for not falling | +0.1 |

#### KNEEL Mode Rewards
| Component | Description | Weight |
|-----------|-------------|--------|
| `com_lowering` | Reward decreasing COM height | +2.0 |
| `controlled_descent` | Low base linear acceleration | +1.5 |
| `low_angular_rate` | Low base angular velocity | +1.0 |
| `no_torso_contact` | Strong penalty for torso/head ground contact | -50.0 |
| `target_kneel_pose` | Reward approaching target joint angles | +1.0 |
| `feet_planted` | Feet remain in contact | +0.5 |
| `action_smoothness` | Penalize action jerk | -0.2 |

### 3.3 Target Kneel Pose
Define a safe kneel configuration:
```python
KNEEL_JOINT_TARGETS = {
    'hip_pitch': -1.2,    # radians, bent forward
    'knee': 2.0,          # bent
    'ankle_pitch': -0.8,  # compensate
    # ... other joints for arm bracing if applicable
}
```

### 3.4 Contact Rules
**Allowed contacts** (no penalty):
- Feet (always)
- Knees/shins (in KNEEL mode)

**Forbidden contacts** (strong penalty / termination):
- Torso
- Head
- Hands (unless intentionally bracing—future work)

### 3.5 Episode Termination
- **RECOVER mode**: Terminate if torso/head contacts ground OR timeout
- **KNEEL mode**: Terminate if torso/head contacts ground (fail) OR stable kneel achieved (success)

**Important**: Knee/shin contact is expected and allowed in KNEEL mode. Don't penalize it.

---

## Phase 4: State Machine (Mode Switching)

### 4.1 Deterministic Switch Logic
```python
class ModeStateMachine:
    def __init__(self):
        self.mode = "RECOVER"
        self.tilt_threshold = 0.5       # radians
        self.angvel_threshold = 2.0     # rad/s
        self.hysteresis_steps = 10      # frames to confirm
        self.trigger_count = 0

    def update(self, roll, pitch, angular_velocity):
        tilt = max(abs(roll), abs(pitch))
        angvel_mag = np.linalg.norm(angular_velocity)

        if self.mode == "RECOVER":
            if tilt > self.tilt_threshold or angvel_mag > self.angvel_threshold:
                self.trigger_count += 1
                if self.trigger_count >= self.hysteresis_steps:
                    self.mode = "KNEEL"
            else:
                self.trigger_count = 0

        # Once in KNEEL, stay in KNEEL until stable
        elif self.mode == "KNEEL":
            if self._is_stable_kneel():
                self.mode = "STABLE"

        return self.mode
```

### 4.2 Training with Mode Flag
During training:
- **Phase A**: Train RECOVER with `mode=0`, random pushes
- **Phase B**: Train KNEEL with `mode=1`, start from unstable states
- **Phase C (optional)**: Combined training with state machine triggering mode switches

### 4.3 Deployment Strategy Options
To avoid observation vector size mismatch on hardware:

**Option 1: Single policy with mode flag**
- Train one policy that accepts mode as input
- Implement same observation wrapper on deployment side (KOS)
- Requires modifying the inference observation construction

**Option 2: Two separate policies (recommended for simplicity)**
- Train `recover.kinfer` and `kneel.kinfer` separately
- State machine selects which `.kinfer` model runs
- No observation mismatch—each model has its native input size
- Combine only at evaluation/deployment via external switching

---

## Phase 5: Training Curriculum

### 5.1 Push Curriculum (K-Sim PushEvent format)
K-Sim's `PushEvent` uses force components (linear + angular) and interval timing. Define pushes as:
```yaml
curriculum:
  stage_1:  # Easy
    push_force:
      # Linear forces (Newtons)
      x_range: [-20, 20]      # front/back
      y_range: [-10, 10]      # left/right
      z_range: [0, 0]         # up/down
      # Angular forces (explicitly 0 for now)
      x_angular_range: [0, 0]
      y_angular_range: [0, 0]
      z_angular_range: [0, 0]
    interval_range: [2.0, 4.0]  # seconds between pushes
  stage_2:  # Medium
    push_force:
      x_range: [-40, 40]
      y_range: [-30, 30]
      z_range: [0, 0]
      x_angular_range: [0, 0]
      y_angular_range: [0, 0]
      z_angular_range: [0, 0]
    interval_range: [1.5, 3.0]
  stage_3:  # Hard
    push_force:
      x_range: [-60, 60]
      y_range: [-50, 50]
      z_range: [-10, 10]
      x_angular_range: [0, 0]
      y_angular_range: [0, 0]
      z_angular_range: [0, 0]
    interval_range: [1.0, 2.5]
  stage_4:  # Kneel triggers
    push_force:
      x_range: [-100, 100]    # Large enough to trigger kneel
      y_range: [-80, 80]
      z_range: [-20, 20]
      x_angular_range: [0, 0]
      y_angular_range: [0, 0]
      z_angular_range: [0, 0]
    interval_range: [1.0, 2.0]
```

**Note**: Angular forces set to 0 initially. Add angular perturbations later if needed for robustness.

### 5.2 Training Commands (Laptop-Optimized)

Default `train.py` is already configured for M2 Air (num_envs=8, batch_size=4, small model). Just run:

```bash
# Basic training - uses laptop defaults
python -m train

# With visualization (watch robot learn in real-time)
python -m train run_mode=view

# Resume from checkpoint
python -m train load_from_ckpt_path=checkpoints/latest.bin
```

For custom task development (future phases):
```bash
# Stage 1: RECOVER mode only
python -m train task=push_to_kneel mode=recover

# Stage 2: KNEEL mode (start from unstable states)
python -m train task=push_to_kneel mode=kneel \
    load_from_ckpt_path=checkpoints/recover.bin
```

**GPU Training (Optional)**: If rented GPU is available for final polish:
```bash
python -m train num_envs=256 batch_size=64 hidden_size=128 depth=5 num_mixtures=5
```

---

## Phase 6: Evaluation Harness

### 6.1 Push Battery Definition (force component format)
```python
# Each push: (fx, fy, fz, apply_at_time_sec)
# Forces in Newtons, matching K-Sim PushEvent style
PUSH_BATTERY = [
    # Easy recovers
    {"force": (20, 0, 0), "time": 0.5},     # Front push
    {"force": (-25, 0, 0), "time": 0.5},    # Back push
    {"force": (0, 20, 0), "time": 0.5},     # Left push
    {"force": (0, -20, 0), "time": 0.5},    # Right push

    # Medium pushes
    {"force": (50, 0, 0), "time": 0.5},     # Stronger front
    {"force": (35, 25, 0), "time": 0.5},    # Diagonal front-left
    {"force": (-45, 20, 0), "time": 0.5},   # Diagonal back-left

    # Large pushes (should trigger kneel)
    {"force": (90, 0, 0), "time": 0.5},     # Large front
    {"force": (-100, 0, 0), "time": 0.5},   # Large back
    {"force": (70, 50, 0), "time": 0.5},    # Large diagonal
]
```

### 6.2 Metrics to Track
```python
@dataclass
class EvalMetrics:
    recovery_rate: float          # % pushes recovered
    kneel_success_rate: float     # % large pushes → safe kneel
    peak_base_acceleration: float # max m/s² during descent
    peak_contact_impulse: float   # max N·s at ground contact
    time_to_kneel: float          # seconds from trigger to stable
    action_jerk: float            # smoothness proxy
```

### 6.3 Automated Video Recording
```bash
python eval/push_battery.py --checkpoint path/to/ckpt.bin \
    --record-video --output-dir results/videos/
```

---

## Phase 7: Domain Randomization

### 7.1 Randomization Parameters
```yaml
domain_randomization:
  friction:
    range: [0.5, 1.5]
  mass_multiplier:
    range: [0.9, 1.1]
  sensor_noise:
    imu_accel_std: 0.1
    imu_gyro_std: 0.02
    joint_pos_std: 0.01
  actuator_gains:
    kp_multiplier: [0.8, 1.2]
    kd_multiplier: [0.8, 1.2]
  push_timing:
    offset_range: [-0.2, 0.2]  # seconds
```

---

## Phase 8: Export and Packaging

### 8.1 Convert to Kinfer (official positional args)
```bash
# Convert checkpoint to .kinfer format (positional args, per official template)
python -m convert checkpoints/final.bin deploy/policy.kinfer

# For two-policy approach:
python -m convert checkpoints/recover.bin deploy/recover.kinfer
python -m convert checkpoints/kneel.bin deploy/kneel.kinfer
```

### 8.2 Visualize in Simulation (kinfer-sim)
```bash
# Use kinfer-sim to visualize the exported model
kinfer-sim deploy/policy.kinfer kbot

# Save video for documentation
kinfer-sim deploy/policy.kinfer kbot --save-video results/videos/policy_demo.mp4
```

---

## Phase 9: Documentation (Portfolio Artifacts)

### 9.1 README Structure
```markdown
# ZBot Push-to-Kneel Controller

## Problem
Humanoid robots fall catastrophically. This project trains a controller that...

## Approach
- Explicit phase flag (RECOVER/KNEEL modes)
- Deterministic state machine for safety-critical switching
- Curriculum learning for progressive difficulty

## Results
- Recovery rate: X%
- Kneel success rate: Y%
- Peak impact reduction: Z% vs baseline

## Videos
[Embedded GIFs/links]

## Reproduce
1. Clone repo
2. Install deps
3. Run training
4. Run evaluation
```

### 9.2 Technical Report Sections
1. Simulation model details (MJCF/URDF source, actuator calibration)
2. Reward engineering decisions
3. Domain randomization choices
4. State machine tuning
5. Sim-to-real gap considerations
6. Safety limits for deployment

---

## Immediate Next Steps (Today)

1. **Install prerequisites**: Homebrew git-lfs, conda environment
2. **Create repo from template**: Use GitHub's template feature
3. **Clone and install**: Get dependencies working
4. **Verify baseline**: Run `python -m train load_from_ckpt_path=assets/ckpt.bin run_mode=view`
5. **Read existing task code**: Understand observation/action spaces
6. **Implement scripted kneel baseline**: PD trajectory to validate kneel pose is reachable

---

## Success Criteria

### Portfolio Success (Primary Goal)
| Artifact | Requirement |
|----------|-------------|
| Working training pipeline | Produces checkpoints, shows learning curves |
| Evaluation harness | `push_battery.py` runs, records videos |
| Documentation | README, design.md, results.md with plots |
| Visualization | `run_mode=view` shows robot behavior |

### Performance Targets (Stretch Goals)
| Metric | Target |
|--------|--------|
| Recovery rate (small pushes) | > 70% |
| Kneel success rate (large pushes) | > 60% |
| Peak base acceleration (kneel) | < 20 m/s² |
| No torso/head contact | > 90% |

**Note**: These are stretch goals. The portfolio value comes from demonstrating the engineering, not perfect metrics. Honest discussion of limitations is a positive signal.

---

## Files to Create/Modify

| File | Purpose |
|------|---------|
| `scripts/scripted_kneel.py` | PD baseline kneel trajectory (debug reference) |
| `tasks/push_to_kneel_task.py` | Custom RL task with mode input |
| `configs/push_to_kneel.yaml` | Hyperparameters and rewards |
| `state_machine/mode_switch.py` | Deterministic RECOVER→KNEEL logic |
| `eval/push_battery.py` | Evaluation harness |
| `eval/metrics.py` | Metric computation |
| `scripts/train_recover.sh` | Training script for RECOVER |
| `scripts/train_kneel.sh` | Training script for KNEEL |
| `scripts/export_kinfer.sh` | Checkpoint → kinfer conversion |
| `docs/design.md` | Design document |
| `docs/results.md` | Results and videos |
| `docs/safety.md` | Safety limits |

---

## Reference Links

- [Zeroth-01 Bot meta repo](https://github.com/zeroth-robotics/zeroth-bot)
- [ksim-gym-zbot (training base)](https://github.com/kscalelabs/ksim-gym-zbot)
- [kos-zbot (deployment)](https://github.com/kscalelabs/kos-zbot)
- [K-Sim PyPI](https://pypi.org/project/ksim/)
- [K-Scale Docs](https://docs.kscale.dev)

---

## Hardware Context (for later)

From the official Zeroth-01 Bill of Materials:
- **Actuators**: STS3250 serial bus servos (x16)
- **Controller**: Milk-V (RISC-V Linux SBC)
- **Servo driver board**: Waveshare Bus Servo Adapter
- **IMU path**: Waveshare LCD IMU RP2040 board + Adafruit 9-DOF IMU
- **Camera**: Milk-V CAM-GC2083
- **Battery + power conversion**

CAD lives in Onshape (linked from official docs).

---

## Real Robot Deployment (Phase 5 - Future)

### KOS Bring-up Sequence
```bash
# Start robot-specific service (loads metadata, gains, limits)
kos zbot

# Run inference with safety scaling
kos zbot infer --model ./policy.kinfer --action-scale 0.2 --episode-length 60
```

### Safety Constraints (Non-negotiable)
Before running any learned policy:
1. Enforce joint limits + velocity limits (lean on robot metadata mode)
2. Start with tethering / soft surface
3. Start with "kneel only" scripted trajectory at low speed
4. Only then try RL policy with small action scale (--action-scale 0.1 to start)
5. Gradually increase action scale as confidence builds

---

## Appendix: Source of Truth Document

This implementation plan is grounded in the official Zeroth-01 Bot repo, K-Scale docs, and K-Sim tooling. The project goal is to build a portfolio-grade end-to-end robotics project: simulation → policy training → deployment on real hardware.

The controller performs:
1. **Push recovery**: Robot remains upright after disturbances
2. **Controlled kneel**: If disturbance is too large, robot transitions into a stable kneel (lowering COM, managing ground contact timing, minimizing impact) instead of face-planting

This maps to the longer-term biomechatronics direction: active eccentric support (braking/softening descent) rather than full-body lifting.
