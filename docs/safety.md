# Safety Considerations

## Simulation vs Reality Gap

This policy was trained entirely in simulation. Before deploying to real hardware:

1. **Start with low action scale** (`--action-scale 0.1`)
2. **Use tethering** or soft landing surface
3. **Verify joint limits** match real robot
4. **Test with gentle pushes first**

## Joint Limits

The policy respects joint limits defined in the URDF. Verify these match your robot:

| Joint | Min (rad) | Max (rad) |
|-------|-----------|-----------|
| Hip Yaw | -0.5 | 0.5 |
| Hip Roll | -0.5 | 0.5 |
| Hip Pitch | -1.5 | 1.0 |
| Knee | 0.0 | 2.5 |
| Ankle Pitch | -1.0 | 1.0 |
| Ankle Roll | -0.3 | 0.3 |

## Termination Conditions

The policy was trained with these safety terminations:
- **Bad Z**: Terminate if base height < 0.05m or > 0.5m
- **Not Upright**: Terminate if tilt > 60 degrees
- **Episode Length**: Max 80 seconds

## Deployment Checklist

- [ ] Verify URDF joint limits match hardware
- [ ] Test with `action-scale 0.1` first
- [ ] Have e-stop ready
- [ ] Use soft surface for initial tests
- [ ] Monitor motor temperatures
- [ ] Check for joint velocity limits

## Known Limitations

1. **No push recovery mode yet**: Current policy only does forward walking
2. **No kneel mode yet**: Controlled descent not implemented
3. **Sim-to-real gap**: Real actuators have delays, friction differs
4. **No terrain handling**: Trained on flat ground only

## Emergency Procedures

If robot becomes unstable:
1. Press e-stop immediately
2. Support robot manually if safe
3. Review logs for cause
4. Reduce action scale before retrying

## Contact

For safety concerns or questions about deployment, open an issue on GitHub.
