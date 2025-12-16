"""
NCA implementation to repair segmentation masks.

Inspired by implementation of CAX https://github.com/maxencefaldor/cax
"""
from functools import partial

import jax
import jax.numpy as jnp
from flax import nnx

def identity_kernel(ndim):
	kernel = jnp.zeros((3,) * ndim)
	center_idx = (1,) * ndim
	kernel = kernel.at[center_idx].set(1.0)
	return jnp.expand_dims(kernel, axis=-1)

def grad_kernel(ndim, normalize=True):
	grad = jnp.array([-1, 0, 1])
	smooth = jnp.array([1, 2, 1])
	kernels = []
	for i in range(ndim):
		kernel = jnp.ones([3] * ndim)
		for j in range(ndim):
			axis_kernel = smooth if i != j else grad
			kernel = kernel * axis_kernel.reshape([-1 if k == j else 1 for k in range(ndim)])
		kernels.append(kernel)
	if normalize:
		kernels = [kernel / jnp.sum(jnp.abs(kernel)) for kernel in kernels]
	return jnp.stack(kernels, axis=-1)

class RNCA(nnx.Module):
    def __init__(
        self,
        img_channels: int,
        state_channels: int = 16,
        img_kernel_size: tuple = (3, 3),
        hidden_dim: int = 128,
        dropout_rate: float = 0.5,
        alive_threshold: float = 0.1,    
        *,
        rngs: nnx.Rngs = nnx.Rngs(0),
    ):
        self.alive_threshold = alive_threshold

        kernel = jnp.concatenate([identity_kernel(ndim=2), grad_kernel(ndim=2)], axis=-1)
        signal_size = state_channels * kernel.shape[-1]

        perceive_kernel = jnp.expand_dims(jnp.concatenate([kernel] * state_channels, axis=-1), axis=-2)
        self.perceive = nnx.Conv(state_channels, signal_size, (3, 3), feature_group_count=state_channels, use_bias=False, rngs=rngs)
        self.perceive.kernel = nnx.Variable(perceive_kernel)
        self.cond_embedder = nnx.Conv(img_channels, signal_size, img_kernel_size, use_bias=False, rngs=rngs)
        
        self.alive_pool = partial(nnx.max_pool, window_shape=(3, 3), strides=(1, 1), padding='SAME')
        
        self.update = nnx.Sequential(
            lambda x_emb, signal: jnp.concatenate([x_emb, signal], axis=-1),
            nnx.Conv(signal_size*2, hidden_dim, (1, 1), rngs=rngs),
            nnx.relu,
            nnx.Conv(hidden_dim, img_channels, (1, 1), kernel_init=nnx.initializers.zeros, rngs=rngs),
            nnx.Dropout(dropout_rate, deterministic=False, rngs=rngs),
        )

    def get_alive_mask(self, state):
        alpha_channel = state[..., -1:]
        pooled = self.alive_pool(alpha_channel)
        return pooled > self.alive_threshold

    def __call__(self, state, x, *, num_steps):
        x_emb = self.cond_embedder(x)
        def step_fn(carry, _):
            model, state = carry
            signal = model.perceive(state)
            alive_mask = model.get_alive_mask(state)
            delta = model.update(signal, x_emb)
            new_state = state + delta
            alive_mask &= model.get_alive_mask(new_state)
            new_state = alive_mask * new_state
            return (model, new_state), state

        _, states = nnx.scan(step_fn, in_axes=(nnx.Carry, None), length=num_steps)((self, state), None)
        return states

    @nnx.jit
    def render(self, state):
        return state[..., -1:] > self.alive_threshold

    @nnx.jit(static_argnames=('num_steps',))
    def repair(self, state, x, *, num_steps):
        batch_size = x.shape[0]
        state_axes = nnx.StateAxes({nnx.RngState: 0, ...: None})
        call_wrapper = lambda f, s, y: f(s, y, num_steps=num_steps)
        call_wrapper = nnx.vmap(call_wrapper, in_axes=(state_axes, 0, 0))
        states = nnx.split_rngs(splits=batch_size)(call_wrapper)(self, state, x)
        return states

def mse(state, x):
    return jnp.mean((state[..., -1:] - x) ** 2)

def loss_fun(model, batch, key, num_steps=128):
    batch_size = batch['X'].shape[0]
    states = model.repair(batch['S'], batch['X'], num_steps=num_steps)
    idx = jax.random.randint(key, (batch_size,), num_steps // 2, num_steps)
    states = states[jnp.arange(batch_size), idx]
    loss = mse(states, batch['Y'])
    return loss, states

