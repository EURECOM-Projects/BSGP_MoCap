import numpy as np
from scipy.stats import bernoulli as np_bern
from .dgp_model import DGP
from scipy.stats import norm
from scipy.special import logsumexp
from .kernels import SquaredExponential as BgpSE
from .kernels import FullPrecisionRBF as BgpFullRBF
from .likelihoods import Bernoulli
import tensorflow as tf

PRIORS = ['uniform', 'normal', 'determinantal', 'strauss']
import logging
logger = logging.getLogger(__name__)

class Model(object):
    def __init__(self, prior_type, output_dim=None):
        class ARGS:
            num_inducing = 100
            iterations = 10000
            minibatch_size = 100
            window_size = 64
            num_posterior_samples = 100
            posterior_sample_spacing = 50
            full_cov = True
            n_layers = 1
            prior_type = None
            logdir = '/tmp/'
            precise_kernel = False # LRBF-MOD
            prior_precision_type = None # LRBF-MOD
            prior_laplace_b = None # LRBF-MOD

        self.ARGS = ARGS
        self.model = None
        self.output_dim = output_dim
        self.global_step = 0
        if prior_type in PRIORS:
            self.ARGS.prior_type = prior_type
        else:
            raise Exception("Invalid prior type")

    def _fit(self, X, Y, lik, Xtest, Ytest, Ystd, **kwargs):
        if len(Y.shape) == 1:
            Y = Y[:, None]

        kerns = []
        if not self.model:
            for i in range(self.ARGS.n_layers):
                output_dim = 196 if i >= 1 and X.shape[1] > 700 else X.shape[1]
                prior_precision_info = {'type': self.ARGS.prior_precision_type, 'hparams': self.ARGS.prior_laplace_b}
                kernel = BgpFullRBF(variance=0.1, randomized=False, d=29, T=100, prior_precision_info=prior_precision_info) if self.ARGS.precise_kernel else BgpSE(output_dim, ARD=True, lengthscales=float(min(X.shape[1], output_dim))**0.5) 
                kerns.append(kernel)

            mb_size = self.ARGS.minibatch_size if X.shape[0] > self.ARGS.minibatch_size else X.shape[0]

            self.model = DGP(X, Y, self.ARGS.num_inducing, kerns, self.ARGS.precise_kernel, lik,
                             minibatch_size=mb_size,
                             window_size=self.ARGS.window_size,
                             full_cov=self.ARGS.full_cov,
                             prior_type=self.ARGS.prior_type, 
                             prior_precision_type=self.ARGS.prior_precision_type,
                             prior_laplace_b = self.ARGS.prior_laplace_b,
                             output_dim=self.output_dim,
                             **kwargs)
            print(self.model)

        self.model.reset(X, Y)
        # writer = SummaryWriter(self.ARGS.logdir, flush_secs=1)
        try:
            for _ in range(self.ARGS.iterations):
                self.global_step += 1
                # self.model.rewhiten()
                self.model.sghmc_step()
                if self.ARGS.prior_type == "determinantal":
                    self.model.reset_Lm()
                self.model.train_hypers() if hasattr(self.model, 'hyper_train_op') else None
                if _ % 250 == 1:
                    marginal_ll = self.model.print_sample_performance()
                    # writer.add_scalar('optimisation/marginal_likelihood', marginal_ll*len(X), self.global_step)
                    print('TRAIN | iter = %6d      sample marginal LL = %5.2f' % (_, marginal_ll))
                    # Test with previous samples with Xtest and Ytest are both not None
                    if not (Xtest is None or Ytest is None or Ystd is None):
                        ms, vs = self.model.predict_y(Xtest, len(self.model.window), posterior=False)
                        logps = norm.logpdf(np.repeat(Ytest[None, :, :]*Ystd, len(self.model.window), axis=0), ms*Ystd, np.sqrt(vs)*Ystd)
                        mnll = -np.mean(logsumexp(logps, axis=0) - np.log(len(self.model.window)))
                        # writer.add_scalar('test/predictive_nloglikelihood', mnll, self.global_step)
                        print('TEST  | iter = %6d       MNLL = %5.2f' % (_, mnll))

            self.model.collect_samples(self.ARGS.num_posterior_samples, self.ARGS.posterior_sample_spacing)
            self.posterior_samples_kern_L = [list(self.model.posterior_samples[i].values())[-2].tolist() for i in range(self.ARGS.num_posterior_samples)] # LRBF-MOD
            self.posterior_samples_kern_logvar = [list(self.model.posterior_samples[i].values())[-1].tolist() for i in range(self.ARGS.num_posterior_samples)] # LRBF-MOD
            self.posterior_samples_U = [np.squeeze(list(self.model.posterior_samples[i].values())[0]).tolist() for i in range(self.ARGS.num_posterior_samples)] # LRBF-MOD
            self.posterior_samples_Z = [np.squeeze(list(self.model.posterior_samples[i].values())[1]).tolist() for i in range(self.ARGS.num_posterior_samples)] # LRBF-MOD
            
        except KeyboardInterrupt:  # pragma: no cover
            self.model.collect_samples(self.ARGS.num_posterior_samples, self.ARGS.posterior_sample_spacing)
            pass

    def _predict(self, Xs, S):
        ms, vs = [], []
        n = max(len(Xs) / 10000, 1) 
        for xs in np.array_split(Xs, n):
            m, v = self.model.predict_y(xs, S)
            ms.append(m)
            vs.append(v)

        return np.concatenate(ms, 1), np.concatenate(vs, 1)  


class ClassificationModel(Model):
    def __init__(self, prior_type, output_dim=None):
        super().__init__(prior_type, output_dim)

    def fit(self, X, Y, Xtest=None, Ytest=None, Ystd=None, **kwargs):
        lik = Bernoulli()
        return self._fit(X, Y, lik, Xtest, Ytest, Ystd, **kwargs)

    def predict(self, Xs):
        ms, vs = self._predict(Xs, self.ARGS.num_posterior_samples)
        m = np.average(ms, 0)
        v = np.average(vs + ms**2, 0) - m**2
        return m, v

    def calculate_density(self, Xs, Ys, ymean=0., ystd=1.):
        # cross-entropy mnll for classification
        ms, vs = self._predict(Xs, self.ARGS.num_posterior_samples)
        Ys_ = np.repeat(Ys[None, :, :], self.ARGS.num_posterior_samples, axis=0)
        return np.mean(Ys_ * np.log(ms) + (1 - Ys_) * np.log(1 - ms))
    
    def calculate_accuracy(self, Xs, Ys, ymean=0., ystd=1.):
        ms, vs = self._predict(Xs, self.ARGS.num_posterior_samples)
        Y_pred = ms >= 0.5
        return np.sum(np.repeat(Ys[None, :, :], self.ARGS.num_posterior_samples, axis=0) == Y_pred) / (Ys.shape[0]*self.ARGS.num_posterior_samples)

    def sample(self, Xs, S):
        ms, vs = self._predict(Xs, S)
        return ms + vs**0.5 * np.random.randn(*ms.shape)

