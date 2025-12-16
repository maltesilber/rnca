from flax import nnx
from orbax import checkpoint as ocp

import numpy as np
from PIL import Image
import matplotlib.cm as cm

# checkpointing
def save_ckpt(ckpt_dir, model, epoch, metric, mode='min'):
    ckpt_dir.mkdir(exist_ok=True)
    options = ocp.CheckpointManagerOptions(best_fn=lambda x: x['val'], best_mode=mode, max_to_keep=1)
    with ocp.CheckpointManager(ckpt_dir.absolute(), options=options) as mngr:
        _, state = nnx.split(model)
        mngr.save(epoch, args=ocp.args.StandardSave(state), metrics={'val': metric})
        mngr.wait_until_finished()

def load_ckpt(model, ckpt_dir) -> nnx.Module:
    with ocp.CheckpointManager(ckpt_dir.absolute()) as mngr:
        graph_def, state = nnx.split(model)
        print(f'Restoring from {mngr.latest_step()}')
        state = mngr.restore(mngr.latest_step(), args=ocp.args.StandardRestore(state))
        model = nnx.merge(graph_def, state)
    return model

# visualization
def make_grid(videos, nrow=4, padding=0, pad_value=0):
    N, T, H, W, C = videos.shape
    ncol = nrow
    assert N == nrow * ncol
    Hpad = H + padding
    Wpad = W + padding
    grid_h = nrow * Hpad + padding
    grid_w = ncol * Wpad + padding
    grid = np.full((T, grid_h, grid_w, C), pad_value, dtype=videos.dtype)
    idx = 0
    for r in range(nrow):
        for c in range(ncol):
            y = padding + r * Hpad
            x = padding + c * Wpad
            grid[:, y:y+H, x:x+W, :] = videos[idx]
            idx += 1
    return grid

def visualize_trajectory(state_traj, gt):
    bs, steps, H, W, _ = state_traj.shape
    
    gt = gt[:, None, ...]
    traj_grid = make_grid(state_traj[..., -1:], nrow=int(np.sqrt(bs)), padding=2, pad_value=0.5)
    frames = []
    for step in range(steps):
        state = np.clip(traj_grid[step].squeeze(-1), 0, 1)
        frame = (cm.get_cmap("inferno")(state)[..., :3] * 255).astype(np.uint8)
        frames.append(Image.fromarray(frame, mode='RGB'))
    return frames