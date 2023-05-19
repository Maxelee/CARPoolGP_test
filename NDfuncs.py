import numpy as np
from tinygp import kernels, GaussianProcess, transforms
from tinygp.kernels import ExpSquared
import jax
from jax.scipy import linalg
import jax.numpy as jnp
import optax
from tqdm.notebook import trange, tqdm
from tinygp.noise import Diagonal
from tqdm import notebook
jax.config.update("jax_enable_x64", True)
import pickle
import pandas as pd
from CARPoolGP import *

def generate_theta(param_df,N, seed):
    from scipy.stats import qmc
    lbs = list(param_df.min())
    ubs = list(param_df.max())
    sampler = qmc.Sobol(d=7, scramble=True, seed=seed)
    sample = sampler.random_base2(m=N)
    theta = qmc.scale(sample, lbs, ubs).T
    return theta

def generate_model(theta, uncorr_gp, Y_mean, seed=1992, sigma=0.5):
    T = uncorr_gp.predict(Y_mean, theta.T)
    np.random.seed(seed)
    data = np.random.normal(T, sigma, theta.shape[1])
    intrinsic_noise = T - data
    return data, intrinsic_noise

def generate_surrogates(theta, intrinsic_noise,uncorr_gp, Y_mean, Groups=2**4):
    ## Different split approach
    from sklearn.cluster import AgglomerativeClustering
    clustering = AgglomerativeClustering(int(Groups)).fit(theta.T)
    labels = clustering.labels_
    unique_ls, counts = np.unique(labels, return_counts=True)
    theta_R = np.vstack([[np.mean(theta[:, labels==i], axis=1)]*c for i, c in zip(unique_ls, counts)])
    noises = np.concatenate([intrinsic_noise[labels==i] for i, c in zip(unique_ls, counts)])
    surrogate_raw = uncorr_gp.predict(Y_mean, theta_R)
    surrogate_data = surrogate_raw + noises 
    return theta_R, surrogate_data, surrogate_raw

def fit_CP(theta, theta_R, data, surrogate_data, return_all=False, iters=500):
    paramsCP = {
        "log_amp":jnp.ones(7),
        "log_tau":jnp.ones(7),
        "log_jitter":-1.0,
        "log_pl":np.zeros(7)}

    kernel_func =  jax.tree_util.Partial(get_GPMixture)

    theta_QR = np.concatenate((theta.T, theta_R))
    QR = np.concatenate((data, surrogate_data))
    I = build_I(len(theta.T))

    # # If you are getting nans, try making the LR smaller
    opt = optax.sgd(learning_rate=1e-3)
    opt_state = opt.init(paramsCP)
    ls = []
    ps = []
    for i in trange(iters):
        loss_val, grads = loss(paramsCP, kernel_func, theta_QR, QR, jitter=0, off_diag=I)
        updates, opt_state = opt.update(grads, opt_state)
        paramsCP = optax.apply_updates(paramsCP, updates)
        ps.append(paramsCP.values())
        ls.append(loss_val)
    if return_all:
        return paramsCP, ls, ps
    else:
        return paramsCP
    
def predict_CP(paramsCP, theta, theta_R, data, surrogate_data, test_theta):
    theta_QR = np.concatenate((theta.T, theta_R))
    QR = np.concatenate((data, surrogate_data))
    I = build_I(len(theta.T))
    
    # Create the kernel
    kernel = get_GPMixture(paramsCP)
    p = 20 / (1 + np.exp(-paramsCP["log_pl"])) - 10
    M = Mkernel(np.exp(-p))

    # Generate the covariance matrices from the kernel
    #r = 1/ (1 + np.exp(-paramsCP["pl"]))
    noise =I * M(theta_QR, theta_QR) * np.exp(paramsCP['log_jitter']) + np.eye(2*len(theta.T))*jnp.exp(paramsCP['log_jitter'])
    cov = kernel(theta_QR, theta_QR)+noise
    pred_cov = kernel(np.concatenate((test_theta.T, theta_QR)), np.concatenate((test_theta.T, theta_QR)))

    # Predict the mean and cov
    pred_meanCP, pred_varCP = predict(QR, cov, pred_cov, 0)
    return pred_meanCP, pred_varCP


def fit_doubleGP(theta, data, iters=300, return_all=False):

    paramsGP = {
        "log_amp":1.0,
        "log_tau":1.0,
        "log_jitter":0.5}

    kernel_func =  jax.tree_util.Partial(get_exp2kernel)

    opt = optax.sgd(learning_rate=1e-3)
    opt_state = opt.init(paramsGP)
    ls = []
    ps = []
    for i in trange(iters):
        loss_val, grads = loss(paramsGP, kernel_func, theta.T, data)
        updates, opt_state = opt.update(grads, opt_state)
        paramsGP = optax.apply_updates(paramsGP, updates)
        ps.append(paramsGP.values())
        ls.append(loss_val)
    if return_all:
        return paramsGP, ls, ps
    else:
        return paramsGP
    
def predict_GP(paramsGP, theta, data, test_theta):
    # Create the kernel
    kernel = get_exp2kernel(paramsGP)
    noise = np.eye(len(theta.T))*jnp.exp(paramsGP['log_jitter'])
    cov = kernel(theta.T, theta.T)+noise
    pred_cov = kernel(np.concatenate((test_theta.T, theta.T)), np.concatenate((test_theta.T, theta.T)))

    # Predict the mean and cov
    pred_meanGP, pred_varGP = predict(data, cov, pred_cov, 0)
    return pred_meanGP, pred_varGP


import numpy as np
from scipy.optimize import curve_fit

def gaussian(x, amplitude, mean, standard_deviation):
    """Gaussian function"""
    return 1/np.sqrt(2*np.pi)/standard_deviation * np.exp(-(x - mean)**2 / (2 * standard_deviation**2))

def fit_gaussian(x, y):
    """Fit a Gaussian curve to the given data"""
    # Initial guess for the parameters
    initial_params = [np.max(y), np.mean(x), np.std(x)]

    # Perform the curve fit
    optimized_params, _ = curve_fit(gaussian, x, y, p0=initial_params)

    # Extract the optimized parameters
    amplitude, mean, standard_deviation = optimized_params

    return amplitude, mean, standard_deviation

def CompareCARPool(param_df, uncorr_gp, Y_mean, Groups, test_theta=None, N_CP=9, seed=1993, iters=300):
    
    theta = generate_theta(param_df, N=N_CP, seed=seed)
    
    data, intrinsic_noise = generate_model(theta, 
                                           uncorr_gp, 
                                           Y_mean, 
                                           seed=seed+1)
    
    theta_R, surrogate_data, surrogate_raw = generate_surrogates(
                                                    theta, 
                                                    intrinsic_noise, 
                                                    Y_mean, 
                                                    Groups=Groups)
    
    paramsCP = fit_CP(theta, theta_R, 
                      data, surrogate_data,iters=iters)
    
    double_theta = generate_theta(param_df, 
                                  N=N_CP+1, 
                                  seed=seed)
    
    double_data, double_noise = generate_model(double_theta, 
                                               uncorr_gp, 
                                               Y_mean, 
                                               seed=seed-1)
    
    paramsGP = fit_doubleGP(double_theta, double_data, iters=iters)
    
    if not test_theta:
        test_theta=np.array(param_df)
        test_data = uncorr_gp.predict(Y_mean, test_theta)
        
    pred_meanCP, pred_varCP = predict_CP(paramsCP, theta, theta_R, data, surrogate_data, test_theta.T)
    pred_meanGP, pred_varGP = predict_GP(paramsGP, double_theta, double_data,  test_theta.T)
    
    means = [pred_meanCP, pred_meanGP]
    var   = [pred_varCP, pred_varGP]
    params = [paramsCP, paramsGP]
    
    return means, var, params

def get_exp2kernel(params):
    return jnp.exp(params["log_amp"]) * transforms.Linear(jnp.exp(-params["log_tau"]), ExpSquared())