import numpy as np

class MeanFieldModel:
    def __init__(self, dim, seed):
        """
        params:
            dim: dimnsiom of the problem (dim v_mat = (dim, dim), dim thetas = (dim, 1)
            seed: random seed
        """
        self.dim = dim
        self.seed = np.random.seed(seed)
        self.grad_max = 100
        self.batch_size = 100

    def gen_samples(self, thetas: np.ndarray, size: int, useTheta: bool) -> np.ndarray:
        """
        Generates sample states 
        params: 
            thetas: ndarray, dim thetas = (self.dim, 1)
            size: int, sample size
        
        returns:
            samples: ndarray dim samples = (size, dim)
        """
        samples = np.ones((size, len(thetas)), dtype=int)
        if useTheta:
            for idx, theta in enumerate(thetas):
                samples[:, idx] *= (-1)**(np.random.rand(size) > np.cos(theta)**2)
        else:
            for idx, theta in enumerate(thetas):
                samples[:, idx] *= (-1)**(np.random.rand(size) > 0.5)

        return samples

    def energy(self, s) -> float:
        """
        Returns the energy of the state, s

        params:
            s: sample state, dim s = (self.dim, 1)
        returns:
            the energy of the state: int
        """
        #Simple 1D Ising
        return np.sum(s*np.roll(s, 1, 0))

    def loss(self, thetas):
        """
        Loss function

        params:
            thetas: ndarray, dim thetas = (dim, 1) the parameters to optimize
        returns:
            the value of the loss function
        """
        tot_energy = 0
        samples = self.gen_samples(thetas, self.batch_size, True)
        for idx in range(samples.shape[0]):
            tot_energy += self.energy(samples[idx, :])
        return tot_energy / samples.shape[0]

    def probability(self, samples: np.ndarray, thetas: np.ndarray):
        """
        The probability of a given state

        params:
            samples: ndarray, dim samples = (size, self.dim)
            thetas: ndarray, dim thetas = (self.dim, 1)
        returns:
            probability of the state: int
        """
        return np.prod(
            np.cos(thetas[np.where(samples==1)])**2
            ) * np.prod(1 - np.cos(thetas[np.where(samples==-1)])**2)

    def grad_loss(self, thetas):
        """
        Returns the gradient of the loss function

        params:
            thetas: ndarray dim thetas = (self.dim, 1)
        returns: 
            the gradient of the loss function: ndarray dim gradient = (self.dim, 1)
        """
        samples = self.gen_samples(thetas, self.batch_size, False)
        return np.array(
            [np.sum( 
                [2 * np.clip(
                    -np.tan(thetas[i]) if s[i]==1 else 1/np.tan(thetas[i]), 
                    -self.grad_max, 
                    self.grad_max
                    ) * 
                self.energy(s) * 
                self.probability(s, thetas) for s in samples]) 
            for i in range(self.dim)]
            ) / samples.shape[0]

    def gradient_descent(self, param, nbr_steps, learning_rate):
        """
        Runs gradient descent
        """
        loss = []
        for i in range(nbr_steps):
            grad_loss = self.grad_loss(param).reshape((self.dim,1))
            param -= learning_rate * grad_loss
            param = param % (np.pi)

        return param, loss

def main():
    dim = 8
    seed = 12
    learning_rate = 1
    nbr_steps = 100

    thetas = np.pi*np.random.rand(dim,1)
    m = MeanFieldModel(dim, seed)
    final_thetas, loss = m.gradient_descent(thetas, nbr_steps, learning_rate)

    print(f"Theta: {np.round(final_thetas.T/np.pi, 2)}*pi")
    print(f"Loss: {m.loss(final_thetas)}")

if __name__ == "__main__":
    main()