# CLI and Script Args
import argparse, os, shutil, time, sys, pyprind, pickle
parser = argparse.ArgumentParser(description='Create EKG Dataset for prediction')
parser.add_argument("--training-outcome", default='gaussian', help='what to predict')
parser.add_argument("--training", action="store_true")
parser.add_argument("--evaluating", action="store_true")
args, _ = parser.parse_known_args()

# our code
import pandas as pd
import numpy as np
import numpy.random as npr

#############################
# Experiment Parameters     #
#############################
from drvae.model.mlp import sigmoid
import matplotlib.pyplot as plt; plt.ion()
import seaborn as sns

#import vae_util as vu
from drvae import misc
from drvae.model.vae import LinearCycleVAE, BeatMlpCycleVAE
import torch
from torch import nn
output_dir = os.path.join("./synthetic-output", args.training_outcome)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# experiment parameter variables
# Latent Space Size
K  = 10
num_discrim_factors = 1
obs_dim        = 50
num_obs        = 10000
num_epochs     = 200
num_mlp_hidden = 50

# betas halved
betas = [100]
for _ in range(6):
    betas.append( betas[-1] / 2 )
#betas = [10000, 1000, 100, 10, 5, 1]

# variances
sig2  = np.linspace(.1, 1, obs_dim)

covariance_type="full"
def K_squared_exp(x1, x2, sig2, Sinv):
     """ K(x1,x2) = sig2*exp(-1/2 (x1-x2) Sinv (x1-x2)) """
     dist = x1[:,None] - x2[None,:]
     qterm = np.sum(np.matmul(dist, Sinv)*dist, axis=-1)
     return sig2*np.exp(-.5*qterm)

def sample_gp(xpts, Kc, eps=None):
     eps = npr.randn(Kc.shape[0]) if eps is None else eps
     return np.dot(eps, Kc.T)


def create_dataset():
    N, D = num_obs, obs_dim

    if covariance_type == "independent":
        X     = npr.randn(N, 1, D) * np.sqrt(sig2)
        Xval  = npr.randn(N, 1, D) * np.sqrt(sig2)
        Xtest = npr.randn(N, 1, D)* np.sqrt(sig2)

    elif covariance_type == "full":
        xgrid = np.linspace(-2, 2, D)[:,None]
        Sigma = K_squared_exp(xgrid, xgrid, 1, 15*np.eye(1)) + np.eye(len(xgrid))*1e-4

        # scale variances so the first dimension is small, last is big
        sig = np.sqrt(sig2)
        scale = np.dot(sig[:,None], sig[None,:])
        Sigma *= scale

        #Sigma   = np.dot(K_squared_exp(xgrid, xgrid, 1, .5*np.eye(1)), np.diag(sig2))
        Sigma_c = np.linalg.cholesky(Sigma)
        X     = np.dot(npr.randn(N, D), Sigma_c.T)[:,None,:]
        Xval  = np.dot(npr.randn(N, D), Sigma_c.T)[:,None,:]
        Xtest = np.dot(npr.randn(N, D), Sigma_c.T)[:,None,:]

    # create classifier dataset
    mlp_model = nn.Sequential(nn.Linear(D, num_mlp_hidden),
                              nn.Tanh(),
                              nn.Linear(num_mlp_hidden, 1))
    n_adjusted = 0
    for n, p in mlp_model.named_parameters():
        print(n)
        if n == "0.weight":
            print(p.shape)
            p.data[:, 0 ].normal_(std=4.)
            p.data[:, num_discrim_factors:] = 0.
            n_adjusted += 1
        elif 'weight' in n:
            p.data.normal_(std=.25)
        elif n == '4.bias':
            p.data.normal_(std=.1)

    assert n_adjusted == 1, "DIDN'T FIX"
    mlp_model.eval()
    print("ORACLE RMSE: ", np.mean(np.sqrt(sig2[:-K])))
    return (X, Xval, Xtest), Sigma, mlp_model


def fit_vae(Xdata, mlp_model, train_vae=False):

    # intermediate output dir
    vae_output_dir = os.path.join(output_dir, "vae-output")
    if not os.path.exists(vae_output_dir):
        os.makedirs(vae_output_dir)

    from drvae.model.vae import BeatConvVAE, BeatMlpVAE, BeatMlpCondVAE, LinearVAE
    lr = 2e-2
    n_channels = 1
    n_samples = 50
    resdict = {}
    if train_vae:
        mlp_vae = LinearVAE(n_channels = n_channels,
                             n_samples  = n_samples,
                             hdims      = [500],
                             latent_dim = K)
        mlp_vae.discrim_model = [mlp_model]
        print("Mlp vae # params: ", mlp_vae.num_params())

        mlp_vae.fit(Xdata, Xdata,
                   epochs=num_epochs,
                    log_interval=None,
                    epoch_log_interval = 25,
                    lr=lr,
                    output_dir = vae_output_dir)
        resdict['mlp_vae'] = mlp_vae

    # Fit Beat Cycle VAE 1
    for beta in betas:
        print("-------- beta: %2.1f ---------"%beta)
        mlp_cycle_vae = LinearCycleVAE(n_channels=n_channels,
                                       n_samples=n_samples,
                                       hdims      = [500],
                                       latent_dim = K)
        mlp_cycle_vae.set_discrim_model(mlp_model, discrim_beta=beta)
        mlp_cycle_vae.fit(Xdata, Xdata,
                          epochs=num_epochs,
                          log_interval=None,
                          epoch_log_interval = 25,
                          lr=lr,
                          output_dir = vae_output_dir)
        resdict['beta-%s'%str(beta)] = mlp_cycle_vae

    return resdict


def plot_mlp_function(mlp_model):
    mlp_model.cpu()

    pm = np.sqrt(sig2[0]) * 2.5
    xg = torch.linspace(-pm, pm, 100)
    xgrid = xg.view(-1, 1).repeat(1, obs_dim)
    lnps = mlp_model(xgrid).detach()
    print(" lnp lo/hi ", lnps.min(), lnps.max())

    sns.set_context("paper")
    sns.set_style("white")
    #import matplotlib.pyplot as plt; plt.ion()
    fig, ax = plt.figure(figsize=(6,3)), plt.gca()
    ax.plot(xg.numpy(), lnps.numpy().squeeze())
    ax.set_xlabel("$\mathbf{x}$")
    ax.set_ylabel("$m(\mathbf{x})$")
    fig.savefig(os.path.join(output_dir, "mlp_function.pdf"),
                bbox_inches='tight')
    plt.close("all")


def plot_reconstructions(mod, Xdata):
    xval = Xdata[2]
    bs = vu.batch_forward(mod, xval)
    xr = torch.cat([ b[0] for b in bs ], dim=0).numpy()
    fig, ax = plt.figure(figsize=(8,6)), plt.gca()

    idx = 0
    ax.plot(np.arange(obs_dim), xval[idx,0,:], label="data")
    ax.plot(np.arange(obs_dim), xr[idx,0,:], label="recon")
    fig.savefig(os.path.join(output_dir, "data-recon.pdf"),
                bbox_inches='tight')
    plt.close("all")


def make_beta_vs_rmse_plot(resdict, Xdata, mlp_model):
    """ creates a plot that shows reconstruction error on X-axis,
    small multiples of increasing beta """

    xval = Xdata[2]
    ndim = xval.shape[-1]

    modnames = ['mlp_vae'] + ['beta-%s'%str(b) for b in sorted(betas)]
    prettynames = ["VAE"] + [r"$\beta=%2.2f$"%b for b in sorted(betas)]

    # track average errors and discrim-dim errors
    discrim_error = []
    average_error = []
    logit_error = []
    prob_error = []

    ncol = int(len(resdict) / 2)
    sns.set_context("paper")
    fig, axarr   = plt.subplots(2, ncol, figsize=(ncol*3, 5))
    rfig, raxarr = plt.subplots(2, ncol, figsize=(ncol*3, 5))
    for ax, rax, k, pk in zip(axarr.flatten(), raxarr.flatten(),
                              modnames, prettynames):
        print(k)
        mod = resdict[k]
        ax.set_title(pk)

        # reconstruct test data
        bs = vu.batch_forward(mod, xval)
        xr = torch.cat([ b[0] for b in bs ], dim=0).numpy()

        # show residual of XR
        res_std = np.std(xval - xr, 0)
        ax.plot(np.arange(ndim), res_std.squeeze(), '--o')

        xvar = np.var(xval, 0).squeeze()
        res_var = np.var(xval-xr, 0).squeeze()
        r2 = 1 - res_var / xvar
        rax.plot(np.arange(ndim), r2, '-.')
        rax.set_title(pk)
        rax.set_ylim( rax.get_ylim()[0], 1. )

        # error on discrimination dimensions
        derr = np.sqrt(np.var(xr[:,0,:num_discrim_factors] -
                              xval[:,0,:num_discrim_factors]))
        discrim_error.append(derr)

        # total error
        d2 = np.sqrt( np.var(xr - xval) )
        average_error.append(d2)

        # logit error
        mlp_model.cpu()
        zdata = mlp_model(torch.FloatTensor(xval)).detach().numpy()
        zrecon = mlp_model(torch.FloatTensor(xr)).detach().numpy()
        lerr = np.sqrt(np.var(zdata-zrecon))
        logit_error.append(lerr)
        perr = np.sqrt(np.var(sigmoid(zdata) - sigmoid(zrecon)))
        prob_error.append(perr)

    # align maximum
    ymax = np.max([ ax.get_ylim()[1] for ax in axarr.flatten() ])
    for ax in axarr.flatten():
        ax.set_ylim(0, ymax)

    ymin = np.min([ rax.get_ylim()[0] for ax in raxarr.flatten() ])
    for rax in raxarr.flatten():
        rax.set_ylim(ymin, 1.)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "beta-rmse.pdf"), bbox_inches='tight')

    rfig.tight_layout()
    rfig.savefig(os.path.join(output_dir, "beta-r2.pdf"), bbox_inches='tight')

    # create error plots
    fig, ax = plt.figure(figsize=(6, 3)), plt.gca()
    xgrid = np.arange(0, len(prettynames))
    ax.plot(xgrid, discrim_error, '-s', label=r"$\mathbf{x}$-discrim rmse")
    ax.plot(xgrid, average_error, '-o', label=r"$\mathbf{x}$-avg rmse")
    ax.plot(xgrid, logit_error, '-v',
            label=r"logit $p(\mathbf{y} | \mathbf{x})$ rmse")
    ax.set_ylabel("RMSE")
    ax.set_xticks(xgrid)
    ax.set_xticklabels(prettynames, rotation=45)
    ax.legend()
    fig.savefig(os.path.join(output_dir, "beta-summary.pdf"), bbox_inches='tight')

    sns.set_context("talk")
    fig, ax = plt.figure(figsize=(6, 3)), plt.gca()
    xgrid = np.arange(0, len(prettynames))
    ax.plot(xgrid, average_error, '-o', label=r"$\mathbf{x}$-avg rmse")
    ax.plot(xgrid, prob_error, '-v',
            label=r"$p(\mathbf{y} | \mathbf{x})$ rmse")
    ax.set_ylabel("RMSE")
    ax.set_xticks(xgrid)
    ax.set_xticklabels(prettynames, rotation=45)
    ax.legend()
    fig.savefig(os.path.join(output_dir, "xave-rmse-vs-prob-rmse.pdf"), bbox_inches='tight')

    # cleanup
    plt.close("all")


def plot_example_data(Xdata, Sigma):

    sns.set_context("talk")
    sns.set_style("white")
    num_to_plot = 7
    Xs =  Xdata[0][:num_to_plot,0,:]
    D  = Xs.shape[1]
    dgrid = np.arange(1, D+1)

    print(num_to_plot)
    fig, ax = plt.figure(figsize=(6, 3)), plt.gca()
    sc = ax.scatter(np.ones(num_to_plot), Xs[:,0], s=25, c=Xs[:,0], cmap='coolwarm')
    cs = sc.to_rgba(Xs[:,0])

    for n in range(num_to_plot):
        cc = (cs[n][0], cs[n][1], cs[n][2])
        ax.plot(dgrid, Xs[n], ".-", alpha=.95, color=cc, markersize=2)

    ax.set_xlabel(r"dimension $i$")
    ax.set_ylabel(r"$x_i$ value")
    ax.set_xlim(0, D+1)
    fig.savefig(os.path.join(output_dir, "example-data.pdf"), bbox_inches='tight')

    # plot eigenvalues of covariance
    w = np.sort(np.linalg.eigvalsh(Sigma))[::-1]
    frac_var = w / np.sum(w)
    print("K = %d explains %2.2f percent of variance"%(K, np.cumsum(frac_var)[K]))
    fig, ax = plt.figure(figsize=(6, 3)), plt.gca()
    ax.plot(np.arange(D), frac_var, "-.")
    ax.set_xlabel("eigen value")
    ax.set_ylabel("Fraction of variance explained")
    fig.savefig(os.path.join(output_dir, "data-spectra.png"), bbox_inches='tight', dpi=250)
    plt.close("all")





if __name__=="__main__":

    import torch
    torch.manual_seed(0)
    Xdata, Sigma, mlp_model = create_dataset()
    plot_example_data(Xdata, Sigma)

    # plot MLP function used
    plot_mlp_function(mlp_model)

    # run vae, fit true vae
    zdiscrim = mlp_model.forward(torch.FloatTensor(Xdata[1]))
    x = torch.randn(*Xdata[0].shape)
    z0 = mlp_model(x.view(-1, x.shape[-1]))
    x[:,0,1:] = 0.
    z1 = mlp_model(x.view(-1, x.shape[-1]))
    zdiscrim = mlp_model.forward(torch.FloatTensor(Xdata[0])).detach().numpy()

    if args.training:
        resdict = fit_vae(Xdata, mlp_model, train_vae=True)
        with open(os.path.join(output_dir, "resdict.pkl"), 'wb') as f:
            pickle.dump(resdict, f)

        # verify pca has same behavior --- check
        from sklearn.decomposition import PCA
        mod = PCA(n_components = K)
        mod.fit(Xdata[0].squeeze())
        ztest = mod.transform(Xdata[2].squeeze())
        mtest = np.dot(ztest, mod.components_)
        delta = mtest - Xdata[2].squeeze()
        np.std(delta, 0)

    ################################################
    # Summarize Zdiscrim variation explained
    ################################################
    with open(os.path.join(output_dir, "resdict.pkl"), 'rb') as f:
        resdict = pickle.load(f)

    def compute_zdiscrim_r2(mod):
        # reconstruct test data
        xval = Xdata[2]
        bs = vu.batch_forward(mod.cuda(), xval)
        xr = torch.cat([ b[0] for b in bs ], dim=0).numpy()

        # show residual of XR
        res_std = np.std(xval - xr, 0)
        xval_std = np.std(xval, 0)
        print("r-error", 1 - (res_std / xval_std))

        # compare total variation of Z discrim
        mlp_model.cuda()
        zv = mlp_model.forward(torch.FloatTensor(xval).cuda()).cpu().detach().numpy()
        zrd = mlp_model.forward(torch.FloatTensor(xr).cuda()).cpu().detach().numpy()
        zres_std = np.std(zv - zrd)
        zv_std = np.std(zv)
        print("r-error z", 1-zres_std/zv_std)

    for mname in resdict.keys():
        print(" model ", mname)
        compute_zdiscrim_r2(resdict[mname])

    mod = resdict[mname]

    ###################################
    # Summarize reconstructed vectors
    ###################################
    make_beta_vs_rmse_plot(resdict, Xdata, mlp_model)
