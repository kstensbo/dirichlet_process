import argparse
import os

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax import random
from jax.typing import ArrayLike
from matplotlib import colormaps

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


def dp_posterior_draws(
    key: KeyArray, observations: ArrayLike, Kmax: int, alpha: int, num_cdfs: int = 1
) -> tuple[ArrayLike, ArrayLike]:
    """
    A truncated Dirichlet process posterior based on Jordan & Teh (2015), Eq. (55).
    """

    N = observations.shape[0]
    posterior_alpha = alpha + N

    key, subkey = random.split(key)

    # Sampling locations from the posterior:
    sample_probabilities = jnp.array([alpha / (alpha + N)] + N * [1 / (alpha + N)])
    deltas = random.choice(
        subkey,
        jnp.array([jnp.nan, *observations]),
        shape=[num_cdfs, Kmax],
        p=sample_probabilities,
    )
    # num_base_measure_samples = int(jnp.isnan(deltas).sum())
    base_deltas = random.normal(
        subkey,
        # shape=[num_base_measure_samples],
        shape=[num_cdfs, Kmax],
    )

    # deltas = deltas.at[jnp.argwhere(jnp.isnan(deltas))].set(base_deltas)
    deltas = jnp.where(jnp.isnan(deltas), base_deltas, deltas)

    key, beta_key = random.split(key)
    betas = random.beta(beta_key, 1, posterior_alpha, shape=[num_cdfs, Kmax])
    betas = betas.at[:, -1].set(1)

    beta_prod = jnp.cumprod(1 - betas, axis=1)
    weights = betas
    weights = weights.at[:, 1:].set(betas[:, 1:] * beta_prod[:, :-1])

    return deltas, weights


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument(
        "--true_mean", type=float, default=0, help="Mean of true distribution."
    )
    parser.add_argument(
        "--true_var", type=float, default=1, help="Variance of true distribution."
    )
    parser.add_argument(
        "--num_samples", type=int, default=10, help="Number of observed samples."
    )
    parser.add_argument(
        "--num_cdfs", type=int, default=100, help="Number of drawn CDFs."
    )
    parser.add_argument("--Kmax", type=int, default=50, help="Number of DP steps.")
    parser.add_argument(
        "--alpha", type=int, default=1, help="DP concentration parameter."
    )
    parser.add_argument(
        "--dist",
        type=str,
        choices=["normal", "beta"],
        default="normal",
        help="Type of true distribution.",
    )
    args = parser.parse_args()

    key = random.PRNGKey(args.seed)
    key, subkey = random.split(key)

    # True data distribution and samples:
    samples = (
        jnp.sqrt(args.true_var) * random.normal(subkey, shape=(args.num_samples,))
        + args.true_mean
    )

    key, prior_key, posterior_key = random.split(key, num=3)
    prior_deltas, prior_weights = dp_prior_draws(
        prior_key, args.Kmax, args.alpha, args.num_cdfs
    )
    posterior_deltas, posterior_weights = dp_posterior_draws(
        posterior_key, samples, args.Kmax, args.alpha, args.num_cdfs
    )

    prior_deltas = np.asarray(prior_deltas)
    prior_weights = np.asarray(prior_weights)
    posterior_deltas = np.asarray(posterior_deltas)
    posterior_weights = np.asarray(posterior_weights)

    # print(f"Integral of CDFs: {np.sum(prior_weights, axis=1)}")

    true_cdf_x = jnp.linspace(-5, 5, 200)
    true_cdf = jax.scipy.stats.norm.cdf(
        true_cdf_x, loc=args.true_mean, scale=jnp.sqrt(args.true_var)
    )

    ## The following is definitely not correct for the variance. Should somehow be
    ## integrated over the variance.
    ## Eq. (50), Jordan & Teh (2015)
    # dp_mean = jax.scipy.stats.norm.cdf(true_cdf_x, loc=0, scale=1)
    ## Eq. (51), Jordan & Teh (2015)
    # dp_var = dp_mean * (1 - dp_mean) / (1 + alpha)

    _, ax = plt.subplots()

    for n, (d, w) in enumerate(zip(prior_deltas, prior_weights, strict=False)):
        ax.ecdf(
            d,
            w,
            color=colormaps["Blues"](0.2 + 0.6 * (n / (args.num_cdfs - 1))),
            alpha=0.05 + 1 / args.num_cdfs,
        )
        # ax.vlines(
        #     d,
        #     0,
        #     w,
        #     color=colormaps["Blues"](0.2 + 0.6 * (n / (num_cdfs - 1))),
        #     alpha=0.2,
        # )

    for n, (d, w) in enumerate(zip(posterior_deltas, posterior_weights, strict=False)):
        ax.ecdf(
            d,
            w,
            color=colormaps["YlOrBr"](0.2 + 0.5 * (n / (args.num_cdfs - 1))),
            alpha=0.05 + 1 / args.num_cdfs,
        )
        # ax.vlines(
        #     d,
        #     0,
        #     w,
        #     color=colormaps["YlOrBr"](0.2 + 0.6 * (n / (num_cdfs - 1))),
        #     alpha=0.2,
        # )

    ax.ecdf(
        np.ravel(prior_deltas),
        np.ravel(prior_weights),
        color="C0",
        label="Mean prior CDF.",
    )

    ax.ecdf(
        np.ravel(posterior_deltas),
        np.ravel(posterior_weights),
        color="C1",
        label="Mean posterior CDF.",
    )

    ax.plot(true_cdf_x, true_cdf, color=colormaps["PuRd"](0.7))
    # ax.plot(true_cdf_x, true_cdf, color=colormaps["Set1"](0))

    ## Analytic mean and standard deviations of the prior DP:
    # ax.fill_between(
    #    true_cdf_x,
    #    dp_mean - jnp.sqrt(dp_var),
    #    dp_mean + jnp.sqrt(dp_var),
    #    color="C0",
    #    alpha=0.2,
    #    edgecolor="none",
    # )
    # ax.plot(true_cdf_x, dp_mean, color="C0", alpha=0.7)

    for i in range(args.num_samples):
        ax.axvline(
            samples[i],
            0,
            0.03,
            linewidth=2,
            # color=colormaps["YlGn"](0.4 + 0.4 * i / (num_samples - 1)),
            color=colormaps["RdPu"](0.4 + 0.4 * i / (args.num_samples - 1)),
            alpha=0.7,
        )

    # ax.ecdf(deltas, weights, color="C1")
    # ax.vlines(deltas, 0, weights, color="C0")

    ax.set_ylim(bottom=0)
    ax.set_xlim(left=true_cdf_x.min(), right=true_cdf_x.max())

    plt.show()


if __name__ == "__main__":
    main()
