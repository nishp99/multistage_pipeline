"""Full-batch tied linear autoencoder used for trace calibration."""

from functools import partial
import jax
import jax.numpy as jnp
import numpy as np
import optax
from .projector import gaussian_data


@partial(jax.jit, static_argnames=("samples", "dimension", "rank", "steps"))
def fit_encoder(
    key, samples, dimension, rank, delta=0.1, steps=40000, rate=5e-4, penalty=1e-4
):
    data_key, init = jax.random.split(key)
    x = gaussian_data(data_key, samples, dimension, rank, delta, jnp.float32)
    w = jax.random.normal(
        jax.random.split(init)[0], (rank, dimension), dtype=jnp.float32
    )
    w /= jnp.sqrt(jnp.float32(dimension))
    optimizer = optax.adam(rate)

    def loss(w):
        return jnp.mean(((x @ w.T) @ w - x) ** 2) + penalty * jnp.sum(w * w)

    def update(_, carry):
        w, state = carry
        changes, state = optimizer.update(jax.grad(loss)(w), state)
        return optax.apply_updates(w, changes), state

    return jax.lax.fori_loop(0, steps, update, (w, optimizer.init(w)))[0]


def encoder_basis(encoder):
    return np.linalg.qr(np.asarray(encoder, dtype=np.float64).T, mode="reduced")[0]
