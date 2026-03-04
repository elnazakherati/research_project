# particle_nn_sim

A small project template for learning a neural network simulator for 2D elastic particle collisions (N=2).

## Files
- `particle_nn_sim/simulator.py`: physics simulator (ParticleSim2D)
- `particle_nn_sim/data.py`: sampling, collision flags, episode collection, XY builders (absolute + residual)
- `particle_nn_sim/models.py`: MLP baseline + tiny InteractionNet2
- `particle_nn_sim/train.py`: standardization, dataset, train/eval loops (with optional collision weighting)
- `particle_nn_sim/rollout_eval.py`: NN rollout (absolute/residual), error plot, side-by-side animation
- `run_experiment.py`: example end-to-end script

## Quick start
```bash
python run_experiment.py
```

If you're in a notebook, add the folder to your path:
```python
import sys
sys.path.append(".")
```
