# Limitations and Future Work

This is a simulation-only project. Real-world deployment was not attempted and would require expertise I don't yet have.

## What This Project Demonstrates

- PPO training pipeline with JAX/KSIM
- Reward shaping for locomotion
- Laptop-scale RL experimentation
- Documentation and reproducibility

## What This Project Does NOT Cover

- **Sim-to-real transfer**: Real actuators have delays, friction, and dynamics that differ from simulation
- **Hardware safety**: Joint limits, torque limits, e-stops, tethering protocols
- **Failure modes**: What happens when the policy outputs unsafe commands
- **Sensor noise**: Real IMUs and encoders have different noise characteristics

## If I Were to Deploy This

These are questions I'd need to answer (and hope to learn in a robotics program):

1. How do I validate joint limits match the real robot?
2. What action scaling is safe to start with?
3. How do I implement a safety watchdog?
4. What's the proper bring-up procedure for a learned policy?
5. How do I characterize the sim-to-real gap for this platform?

## References I'd Consult

- K-Scale deployment docs for Zeroth-01
- Literature on sim-to-real for legged robots (e.g., ANYmal, MIT Cheetah)
- Domain randomization best practices

---

*This section exists to acknowledge what I don't know, not to claim expertise I don't have.*
