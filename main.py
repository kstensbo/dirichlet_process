import os

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"

import jax

jax.config.update("jax_enable_x64", True)
# jax.config.update("jax_platform_name", "cpu")
# jax.config.update("jax_disable_jit", True)


def main() -> None:
    print(jax.devices())


if __name__ == "__main__":
    main()
