import os

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax import random
from jax.typing import ArrayLike
from matplotlib import cm

from IPython import embed  # noqa: F401

jax.config.update("jax_enable_x64", True)
# jax.config.update("jax_platform_name", "cpu")
# jax.config.update("jax_disable_jit", True)

KeyArray = ArrayLike


def dp_prior_draws(
    key: KeyArray, Kmax: int, alpha: int, num_cdfs: int = 1
) -> tuple[ArrayLike, ArrayLike]:
    """
    A truncated Dirichlet process prior based on Jordan & Teh (2015), Eq. (46).
    """

    key, beta_key = random.split(key)
    betas = random.beta(beta_key, 1, alpha, shape=[num_cdfs, Kmax])
    betas = betas.at[:, -1].set(1)

    beta_prod = jnp.cumprod(1 - betas, axis=1)
    weights = betas
    weights = weights.at[:, 1:].set(betas[:, 1:] * beta_prod[:, :-1])

    key, normal_key = random.split(key)
    deltas = random.normal(normal_key, shape=[num_cdfs, Kmax])

    return deltas, weights


def main() -> None:
    seed = 0
    key = random.PRNGKey(seed)

    num_cdfs = 100

    Kmax = 50
    alpha = 10

    deltas, weights = dp_prior_draws(key, Kmax, alpha, num_cdfs)

    deltas = np.asarray(deltas)
    weights = np.asarray(weights)

    print(f"Integral of CDFs: {np.sum(weights, axis=1)}")

    _, ax = plt.subplots()

    for n, (d, w) in enumerate(zip(deltas, weights, strict=False)):
        ax.ecdf(d, w, color=cm.Oranges(0.2 + 0.6 * (n / (num_cdfs - 1))), alpha=0.7)
        ax.vlines(d, 0, w, color=cm.Blues(0.2 + 0.6 * (n / (num_cdfs - 1))), alpha=0.7)
    # ax.ecdf(deltas, weights, color="C1")
    # ax.vlines(deltas, 0, weights, color="C0")

    ax.set_ylim(bottom=0)

    plt.show()


if __name__ == "__main__":
    main()
