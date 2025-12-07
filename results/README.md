# Results

This folder contains curated training outputs:

- `ckpt.bin` - Best trained checkpoint (step 518)
- `demo.gif` - Demo animation (to be recorded)
- `plots/` - TensorBoard plot exports

## Recording a demo GIF

```bash
# Option 1: Use MuJoCo viewer + screen recording
python -m train run_mode=view load_from_ckpt_path=results/ckpt.bin
# Then use QuickTime or similar to screen record

# Option 2: Use kinfer-sim with video export
python -m convert results/ckpt.bin results/model.kinfer
kinfer-sim results/model.kinfer kbot --save-video results/demo.mp4
# Convert to GIF: ffmpeg -i demo.mp4 -vf "fps=15,scale=480:-1" demo.gif
```
