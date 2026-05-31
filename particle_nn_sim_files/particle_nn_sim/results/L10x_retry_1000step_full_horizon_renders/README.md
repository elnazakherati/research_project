# L10x Retry 1000-Step Full-Horizon Renders

These videos use the saved best checkpoint from the L10x retry run.

- Rollout horizon: 1000 simulation steps
- Simulation duration: 10.0 seconds (`dt=0.01`)
- Video frames: 1001 frames
- Playback: 100 fps, about 10 seconds total
- Episodes: held-out test episodes 18432 and 18433

The older stride-5 render folder showed `step 200/200` in the video text because it only rendered every fifth frame. It still represented a 1000-step rollout, but the label was confusing. The old folders were moved to `results/_archive_l10x_confusing_renders/`.
