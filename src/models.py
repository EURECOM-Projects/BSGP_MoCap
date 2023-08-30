import gpflow
from .kernels import RBFAID
import numpy as np

class VariationalGaussianProcess():
    def __init__(self, data, D, T, kernel_computation_type):
        self.labels = data['labels']
        self.model = gpflow.models.VGP((data['Y'], data['labels']), likelihood=gpflow.likelihoods.Bernoulli(), kernel=RBFAID(variance=0.1, randomized=False, d=D, T=T, kernel_computation_type=kernel_computation_type))

    def train(self, maxiter):
        opt = gpflow.optimizers.Scipy()
        opt.minimize(self.model.training_loss, 
                    variables=self.model.trainable_variables,
                    options={"disp": 50, "maxiter": maxiter},
                    method="L-BFGS-B",)

    def compute_accuracy(self, Y, labels):
        Fmean, _ = self.model.predict_f(Y)
        Psamples = self.model.likelihood.invlink(Fmean)
        Y_pred = np.array([int(i) for i in Psamples >= 0.5]).reshape(-1,1)
        return np.sum(Y_pred  ==  labels) / Y_pred.shape[0]