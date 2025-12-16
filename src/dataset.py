import pathlib
from functools import partial
import random
import yaml

import jax
import jax.numpy as jnp
import grain.python as pygrain

from PIL import Image
import numpy as np

class Dataset(pygrain.RandomAccessDataSource):
    def __init__(self, data_dir, train=True, state_channels=16):
        self.split = 'train' if train else 'val'
        self.data_dir = pathlib.Path(data_dir) / self.split
        self.entries = [i.name for i in (self.data_dir / 'images').glob('*.png')]
        self.state_channels = state_channels
        
    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        base_name = self.entries[idx]
        img_path = self.data_dir / 'images' / base_name
        label_path = self.data_dir / 'labels' / base_name
        states_path = self.data_dir / 'states' / base_name    
        img = np.asarray(Image.open(img_path)).astype(np.float32) / 255.0
        label = (np.asarray(Image.open(label_path)) > 128).astype(np.float32)
        state_vis = (np.asarray(Image.open(states_path))).astype(np.float32) / 255.0
        state_vis = np.clip(state_vis, 0.1, 1.0) # clip non alive cells 

        H, W = img.shape[:2]
        if len(img.shape) == 2:
            img = img[..., None]

        state = np.zeros((H, W, self.state_channels), dtype=np.float32)
        state[..., -1] = state_vis
        batch = {'X': img, 'Y': label[..., None], 'S': state, 'idx': idx}
        return jax.tree.map(lambda x: jnp.array(x), batch) 

def init_pool(dataset, pool_size, key):
    indices = jax.random.choice(key, len(dataset), shape=(min(pool_size, len(dataset)),), replace=False)
    return jax.tree.map(lambda *x: jnp.stack(x), *[dataset[i] for i in indices])


@partial(jax.jit, static_argnames=('batch_size',))
def sample_pool(key, pool, *, batch_size):
    pool_size = pool['X'].shape[0]
    idxs = jax.random.choice(key, pool_size, shape=(batch_size,), replace=False)
    batch = jax.tree.map(lambda x: x[idxs], pool)
    return idxs, batch

@partial(jax.jit, static_argnames=('batch_size',))
def random_batch_update(key, pool, entry, batch_size=1):
    key_sample, key_idx = jax.random.split(key)
    pool_idxs, batch = sample_pool(key_sample, pool, batch_size=batch_size)
    idxs = jax.random.choice(key_idx, batch_size, shape=(len(entry['X']),), replace=False)
    new_batch = jax.tree.map(lambda x, y: x.at[idxs].set(y), batch, entry)
    return new_batch, pool_idxs

@jax.jit
def update_pool(pool_states, idxs, new_states):
    return jax.tree.map(lambda x, y: x.at[idxs].set(y), pool_states, new_states)

