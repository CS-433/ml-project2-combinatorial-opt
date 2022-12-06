    def gen_samples2(self, thetas: np.ndarray, size: int):
    # print(rng_key)
        samples = np.zeros((size, len(thetas)))
        for idx, theta in enumerate(thetas):
            samples[:, idx] =  jax.random.bernoulli(rng_key,jnp.cos(theta)**2,(size,))

        return samples
