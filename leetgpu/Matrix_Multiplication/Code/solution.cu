import jax
import jax.numpy as jnp


# A: (M, K), B: (K, N)
@jax.jit
def solve(A: jax.Array, B: jax.Array, M: int, N: int, K: int) -> jax.Array:
    # return output matrix directly
    return jnp.matmul(A, B)