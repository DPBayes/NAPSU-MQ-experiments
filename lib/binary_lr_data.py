import torch
import itertools

class BinaryLogisticRegressionDataGenerator:
    """
    Generator for d-dimensional binary datasets where all variables but the 
    last are generated with independent coinflips, and the last variable 
    is generated from logistic regression on the others.
    """
    def __init__(self, true_params):
        """Initialise the generator.

        Args:
            true_params (torch.tensor): Coefficients for the logistic regression.
        """
        self.true_params = true_params
        self.d = true_params.shape[0] + 1
        self.x_values = self.compute_x_values()
        self.values_by_feature = {i: [0, 1] for i in range(self.d)}

    def generate_data(self, n, generator=None):
        """Generate the data d-dimensional binary data.

        Args:
            n (int): Number of datapoints to generate.
            generator (Pytorch RNG, optional): Random number generator. Defaults to None.

        Returns:
            torch.tensor: The generated data.
        """
        x = torch.bernoulli(torch.full((n, self.d - 1), 0.5), generator=generator)
        alpha = x @ self.true_params.view((-1, 1))
        probs = torch.special.expit(alpha)
        y = torch.bernoulli(probs, generator=generator)
        return torch.concat((x, y), dim=1)
    
    def compute_x_values(self):
        """Enumerate all possible datapoints.

        Returns:
            torch.tensor: 2-d array enumerating all possible datapoints.
        """
        x_values = torch.zeros((2**self.d, self.d))
        for i, val in enumerate(itertools.product(range(2), repeat=self.d)):
            x_values[i, :] = torch.tensor(val)
        return x_values
