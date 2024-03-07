import os

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax import random

jax.config.update("jax_enable_x64", True)
# jax.config.update("jax_platform_name", "cpu")
# jax.config.update("jax_disable_jit", True)


def main() -> None:
    seed = 0
    key = random.PRNGKey(seed)

    Kmax = 5
    alpha = 1

    key, beta_key = random.split(key)

    betas = random.beta(beta_key, 1, alpha, shape=[Kmax])
    betas = betas.at[-1].set(1)

    beta_prod = jnp.cumprod(1 - betas)
    weights = betas
    weights = weights.at[1:].set(betas[1:] * beta_prod[:-1])

    key, normal_key = random.split(key)
    deltas = random.normal(normal_key, shape=[Kmax])

    deltas = np.array(deltas)
    weights = np.array(weights)

    _, ax = plt.subplots()

    ax.ecdf(deltas, weights, color="C1")
    ax.vlines(deltas, 0, weights, color="C0")

    ax.set_ylim(bottom=0)

    plt.show()


if __name__ == "__main__":
    main()
