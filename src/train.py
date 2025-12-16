"""
Training script for Refinement Neural Cellular Automata (RNCA).
Trains RNCA models for segmentation mask repair using pool-based training.
"""

from dataclasses import dataclass
from pathlib import Path
import shutil

import jax
import numpy as np
import matplotlib.pyplot as plt
from flax import nnx
import optax
from tqdm.auto import tqdm
import grain.python as pygrain

from dataset import Dataset, init_pool, random_batch_update, update_pool
from nca import RNCA, loss_fun, mse
from utils import save_ckpt, load_ckpt, visualize_trajectory

@nnx.jit(static_argnames=('num_steps', 'batch_size'))
def train_step(key, model, optimizer, pool, new_states, num_steps, batch_size):
    key_update, key_step = jax.random.split(key)
    batch, pool_idxs = random_batch_update(key_update, pool, new_states, batch_size)
    
    grad_fun = nnx.value_and_grad(loss_fun, has_aux=True)
    (loss, states), grad = grad_fun(model, batch, key_step, num_steps=num_steps)
    optimizer.update(model, grad)
    
    new_states = {'X': batch['X'], 'Y': batch['Y'], 'S': states, 'idx': batch['idx']}
    pool = update_pool(pool, pool_idxs, new_states)
    return loss, pool


@nnx.jit(static_argnames=('num_steps',))
def eval_step(model, batch, num_steps):
    
    S_traj = model.repair(batch['S'], batch['X'], num_steps=num_steps)
    return mse(S_traj[:, -1], batch['Y'])


def make_loader(data_dir, train: bool, state_channels, batch_size, *, shuffle, num_epochs, seed=None,
):
    ds = Dataset(data_dir, train=train, state_channels=state_channels)
    sampler = pygrain.IndexSampler(
        num_records=len(ds),
        shuffle=shuffle,
        num_epochs=num_epochs,
        seed=seed,
    )
    loader = pygrain.DataLoader(
        data_source=ds,
        sampler=sampler,
        operations=[pygrain.Batch(batch_size)],
    )
    return ds, loader

def train(cfg):
    key = jax.random.key(cfg.seed)

    ckpt_dir = Path(cfg.ckpt_dir)
    if ckpt_dir.exists():
        shutil.rmtree(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    ds_train, loader_train = make_loader(
        cfg.data_dir, train=True, state_channels=cfg.state_channels, batch_size=cfg.replace_n, shuffle=True, num_epochs=cfg.replace_n, seed=cfg.seed,
    )

    _, loader_val = make_loader(
        cfg.data_dir, train=False, state_channels=cfg.state_channels, batch_size=cfg.batch_size, shuffle=False, num_epochs=1,
    )
    
    key, key_pool, key_model = jax.random.split(key, 3)
    pool = init_pool(ds_train, cfg.pool_size, key_pool)
    
    model = RNCA(
        img_channels=ds_train[0]['X'].shape[-1], 
        state_channels=cfg.state_channels, 
        img_kernel_size=cfg.img_kernel_size,
        hidden_dim=cfg.hidden_dim,
        dropout_rate=cfg.dropout_rate,
        alive_threshold=cfg.alive_threshold,
        rngs=nnx.Rngs(key_model))
    
    opt = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(cfg.learning_rate)
    )
    optimizer = nnx.Optimizer(model, opt, wrt=nnx.Param)
    
    metrics = {'train_loss': [], 'val_loss': []}
    for epoch in (pbar := tqdm(range(cfg.epochs), ncols=100)):
        losses = []
        model.train()
        for i, new_states in enumerate(loader_train):
            step_key = jax.random.fold_in(key, epoch * len(ds_train) + i)
            loss, pool = train_step(
                step_key, model, optimizer, pool, new_states, 
                cfg.nca_steps, cfg.batch_size
            )
            losses.append(loss)
        train_loss = float(np.mean(losses))
        metrics['train_loss'].append(train_loss)
        
        model.eval()
        val_losses = [eval_step(model, val_batch, 256) for val_batch in loader_val]
        val_loss = float(np.mean(val_losses))
        metrics['val_loss'].append(val_loss)
        
        pbar.set_postfix({'loss': f'{train_loss:.3e}|{val_loss:.3e}'})
        tqdm.write(f"Epoch {epoch}: train={train_loss:.3e}, val={val_loss:.3e}")
        
        save_ckpt(ckpt_dir, model, epoch, val_loss, mode='min')
        
        plt.figure()
        plt.plot(metrics['train_loss'], label='train', marker='o')
        plt.plot(metrics['val_loss'], label='val', marker='s')
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(ckpt_dir / 'loss.png', dpi=150)
        plt.close()
    
    model = load_ckpt(model, ckpt_dir)
    model.eval()
    for i, batch in enumerate(tqdm(loader_val, desc="Visualizing")):
        traj = model.repair(batch['S'], batch['X'], num_steps=512)
        frames = visualize_trajectory(traj, batch['Y'])
        frames[0].save(
            ckpt_dir / f'trajectory_{i}.gif',
            save_all=True,append_images=[frames[0]]*30 + frames, duration=60, loop=0
        )
    return model, metrics