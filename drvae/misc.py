import numpy as np
import pyprind
from sklearn.metrics import roc_auc_score, precision_score, \
                            recall_score, f1_score, r2_score, \
                            average_precision_score
import pandas as pd

def beat_recon_rmse(x, recon_x):
    return np.sqrt(np.mean((x-recon_x)**2))

def beat_recon_mae(x, recon_x):
    return np.mean(np.abs(x-recon_x))

def beat_recon_bootstrap(x, recon_x, error_type="rmse", num_samples=100):
    if error_type == "rmse":
        err = np.sqrt(np.mean((x-recon_x)**2, axis=-1))
    elif error_type == "mae":
        err = np.mean(np.abs(x-recon_x), axis=-1)

    samps = []
    for _ in range(num_samples):
        idx = np.random.choice(len(err), size=len(err), replace=True)
        eboot = err[idx]
        samps.append(np.mean(eboot))

    return np.array(samps)

def bootstrap_auc(ytrue, ypred, fun=roc_auc_score, num_samples=100):
    # make nan safe
    nan_idx = np.isnan(ytrue)
    ytrue = ytrue[~nan_idx]
    ypred = ypred[~nan_idx]

    samps = []
    for _ in range(num_samples):
        idx = np.random.choice(len(ytrue), size=len(ytrue), replace=True)
        auc = fun(ytrue[idx], ypred[idx])
        #auc = np.max([auc, 1-auc])
        samps.append(auc)

    return np.array(samps)


def bootstrap_average_precision_score(ytrue, ypred, num_samples=100):
    return bootstrap_auc(ytrue, ypred,
        fun=average_precision_score, num_samples=num_samples)


def bootstrap_auc_comparison(ytrue, ypreda, ypredb, num_samples=100):
    samps_a, samps_b, diff = [], [], []
    for _ in range(num_samples):
        idx = np.random.choice(len(ytrue), size=len(ytrue), replace=True)
        auc_a = roc_auc_score(ytrue[idx], ypreda[idx])
        auc_b = roc_auc_score(ytrue[idx], ypredb[idx])
        samps_a.append(auc_a)
        samps_b.append(auc_b)
        diff.append(auc_a-auc_b)
    return samps_a, samps_b, diff


def bootstrap_prec_recall_f1(ytrue, ypred, num_samples=100):
    psamps, rsamps, fsamps = [], [], []
    for _ in range(num_samples):
        idx = np.random.choice(len(ytrue), size=len(ytrue), replace=True)
        psamps.append( precision_score(ytrue[idx], ypred[idx]) )
        rsamps.append( recall_score(ytrue[idx], ypred[idx]) )
        fsamps.append( f1_score(ytrue[idx], ypred[idx]) )

    return np.array(psamps), np.array(rsamps), np.array(fsamps)


def bootstrap_corr(x, y, num_samples=100):
    samps = []
    for _ in range(num_samples):
        idx = np.random.choice(len(x), size=len(x), replace=True)
        samps.append(np.corrcoef(x[idx], y[idx])[0, 1])
    return np.array(samps)


def bootstrap_summary(y, fun=np.mean, num_samples=1000):
    samps = []
    for _ in range(num_samples):
        idx = np.random.choice(len(y), size=len(y), replace=True)
        samps.append(fun(y[idx]))
    return np.array(samps)


def bootstrap_r2(ytrue, ypred, num_samples=1000):
    samps = []
    for _ in range(num_samples):
        idx = np.random.choice(len(ytrue), size=len(ytrue), replace=True)
        samps.append(r2_score(ytrue[idx], ypred[idx]))
    return np.array(samps)
