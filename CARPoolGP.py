import numpy as np
import jax.numpy as jnp
from jax.scipy import linalg
from tinygp import kernels
import jax


@jax.jit
@jax.value_and_grad
def loss(params, kernel_func, X, Y, jitter=None, off_diag=None, kernel_M=None):
    """
    Return the loss and gradient of the loss for gradient descent
    """
    # Build Kernel with current parameter values
    kernel = kernel_func(params)
    N = len(X)
    # Add noise
    if jitter is None:
        cov = kernel(X, X) + jnp.exp(params['log_jitter']) * jnp.eye(N)
    else:
        p = 20 / (1 + jnp.exp(-params["log_pl"])) - 10
        MK = Mkernel(jnp.exp(-p))
        # Off diagonal elements are weighed by linear exponential
        M = off_diag*MK(X, X) * jnp.exp(params['log_jitter']) + jnp.eye(N)*jnp.exp(params['log_jitter'])
        cov = kernel(X, X) + M
    # Compute liklihood
    alpha, scale_tril = decomp(cov, Y, 0)
    L = log_liklihood(scale_tril, alpha)

    return -L


def predict(Y, cov, cov_new, mu_y):
    """
    mean = Ks C^{-1} (Y-\mu_Y) + \mu_Y
    cov = Kss - Ks C^{-1}Ks^T

    Ks = covariance of new thetas with old thetas,
    Kss= covariance of new thetas
    C  = covariance from GP
    x  = Value of params
    gp_mean = mean function

    returns mean and cov
    """
    ltn = cov_new.shape[0] - cov.shape[0]
    mean = cov_new[:ltn, ltn:] @ np.linalg.inv(cov)@(Y - mu_y) + mu_y
    cov = cov_new[:ltn, :ltn] - \
        cov_new[:ltn, ltn:] @ np.linalg.inv(cov) @  cov_new[:ltn, ltn:].T
    return mean, cov


@jax.jit
def decomp(cov, Q, mean):
    scale_tril = linalg.cholesky(cov, lower=True)
    alpha = linalg.solve_triangular(scale_tril, Q-mean, lower=True)
    return alpha, scale_tril


@jax.jit
def invdecomp(cov, Q, mean):
    scale_tril = linalg.cholesky(cov, lower=True)
    alpha = linalg.solve_triangular(scale_tril, Q-mean, lower=True, trans=1)
    return alpha, scale_tril


@jax.jit
def log_liklihood(scale_tril, alpha):
    return -0.5 * jnp.sum(jnp.square(alpha)) - \
        jnp.sum(jnp.log(jnp.diag(scale_tril))) + \
        0.5 * scale_tril.shape[0] * np.log(2 * np.pi)

class GPMixture(kernels.Kernel):
    """
    Custom kernel for carpool with \Delta P parameter
    """
    def __init__(self, amp, tau, delta_p=0):
        self.amp = jnp.atleast_1d(amp)
        self.tau = jnp.atleast_1d(tau)
        self.delta_p = jnp.atleast_1d(delta_p)

    def evaluate(self, X1, X2):
        x = jnp.atleast_1d(jnp.sqrt((X2 - X1 + self.delta_p)**2))
        return jnp.sum(self.amp * jnp.exp(-x**2*self.tau))


def build_I(N):
    """
    Create an off diagonal identity matrix for M matrix
    """
    off_diag = np.zeros((2*N, 2*N))
    for i in range(N):
        off_diag[i, N+i] = 1
    for i in range(N):
        off_diag[N+i, i] = 1
    return off_diag


class Mkernel(kernels.Kernel):
    """
    linear exponential kernel only applied to the off diagonals. This encodes
    the level of correlation between surrogates and HR sims. The value is
    transformed using a sigmoid function to bind it between -1 and 1
    """
    def __init__(self, p):
        self.p = jnp.atleast_1d(p)

    def evaluate(self, X1, X2):
        x = jnp.atleast_1d(jnp.sqrt((X2 - X1)**2))
        return jnp.sum(jnp.exp(-self.p * x))


def get_GPMixture(params):
    try:
        K = GPMixture(
            jnp.exp(params["log_amp"]),
            jnp.exp(params["log_tau"]),
            jnp.exp(params["log_p"]))
    except:
        K = GPMixture(
            jnp.exp(params["log_amp"]),
            jnp.exp(params["log_tau"]))
    return K


def get_kernelM(params):
    K = Mkernel(params["log_pl"])
    return K
