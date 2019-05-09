#CLI and Script Args
import torch
import argparse, os, shutil, time, sys, pyprind, pickle
parser = argparse.ArgumentParser(description="toy mnist experiment for DR-VAE")
parser.add_argument("--training-outcome", default='gaussian', help='what to predict')
parser.add_argument("--training", action="store_true")
parser.add_argument("--evaluating", action="store_true")
parser.add_argument("--batch-size", type=int, default=128,
                    help="batch size for training")
args, _ = parser.parse_known_args()
args.cuda = torch.cuda.is_available()


#############################
# Experiment Setup          #
#############################
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt; plt.ion()
import seaborn as sns
import pandas as pd
import numpy as np
import numpy.random as npr
from drvae import misc
from drvae.model.mlp import sigmoid
from drvae.model.vae import LinearCycleVAE, BeatMlpCycleVAE
from torch import nn
output_dir = os.path.join("./mnist-output", args.training_outcome)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


#############################
# Load Digit Data           #
#############################
npr.seed(42)
from data_mnist import load_mnist, plot_images
N, train_img, train_lab, test_img, test_lab = load_mnist()

# create the confounded dataset
def make_confounded_data(classa=[0, 1, 2], classb=[3, 4, 5], alpha=.75):
    # divvy into two classes
    aidx = np.isin(train_lab, classa)
    bidx = np.isin(train_lab, classb)

    # mask for extra pixels
    cmask = np.zeros((28, 28))
    cmask[1:5, 1:5] = 1.


    # add +'s to the first class
    xa = train_img[aidx]
    xa = np.reshape(xa, (-1, 28, 28))
    xa[:,1:4,2] = alpha/2
    xa[:,2,1:4] = alpha/2
    xa[:,2,2]   = alpha
    xa = np.reshape(xa, (xa.shape[0], -1))
    xb = train_img[bidx]

    # train images
    X = np.row_stack([xa, xb])
    Y = np.concatenate([np.zeros(len(xa)), np.ones(len(xb))])

    # make confounded classifier --- it pays attention only to the
    # cross idx
    num_mlp_hidden = 5
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
    return X, Y, mlp_model

X, Y, mlp_model = make_confounded_data(alpha=.5)



# split data --- train/test
def split_data(X, Y, frac_train=.7, frac_val=.15):
    rs = npr.RandomState(0)
    idx = rs.permutation(len(X))
    ntrain = int(frac_train*len(idx))
    nval   = int(frac_val*len(idx))
    ntest  = len(idx)-ntrain-nval
    idx_train = idx[:ntrain]
    idx_val   = idx[ntrain:(ntrain+nval)]
    idx_test  = idx[(ntrain+nval):]
    return (X[idx_train], X[idx_val], X[idx_test]), \
           (Y[idx_train], Y[idx_val], Y[idx_test])

Xdata, Ydata = split_data(X, Y)

# plot examples
xa = X[Y==0.][:10,:]
xb = X[Y==1.][:10,:]
fig, ax = plt.figure(figsize=(8,6)), plt.gca()
plot_images(xa, ax=ax)
fig.savefig(os.path.join(output_dir, "example-confounded-a.png"), bbox_inches='tight')
fig, ax = plt.figure(figsize=(8,6)), plt.gca()
plot_images(xb, ax=ax)
fig.savefig(os.path.join(output_dir, "example-confounded-b.png"), bbox_inches='tight')

# model imports
from drvae.model.mlp import BeatMlpClassifier
from drvae.model.vae import BeatMlpVAE, BeatMlpCycleVAE


#################
# Train Models  #
#################

if args.training:

    #
    # Train MLP Model
    #
    mlp_model = BeatMlpClassifier(data_dim=784,
                                  n_outputs=1,
                                  hdims=[50, 50, 50],
                                  dropout_p=.5)
    mlp_model.fit(Xdata, Ydata, epochs=20)
    mlp_model.save(os.path.join(output_dir, 'discrim_model.pkl'))


    #
    # Train VAE
    #
    vae_output_dir = os.path.join(output_dir, "vae-output")
    if not os.path.exists(vae_output_dir):
        os.makedirs(vae_output_dir)
    vae_kwargs = {'hdims'      : [500],
                  'n_samples'  : 784,
                  'n_channels' : 1,
                  'latent_dim' : 20,
                  'loglike_function': 'bernoulli'}

    num_epochs=200
    lr = 1e-2
    opt_kwargs = {'lr': 1e-3,
                  'num_epochs': 100,
                  'log_interval': None,
                  'epoch_log_interval': 5}

    mlp_vae = BeatMlpVAE(**vae_kwargs)
    mlp_vae.fit(Xdata, Xdata,
                epochs=num_epochs,
                log_interval=None,
                epoch_log_interval = 5,
                lr=lr,
                output_dir = vae_output_dir)
    mlp_vae.save(os.path.join(vae_output_dir, "vae.pkl"))

    #
    # Fit CVAE with varying betas
    #
    cvae_output_dir = os.path.join(output_dir, "cvae-output")
    if not os.path.exists(cvae_output_dir):
        os.makedirs(cvae_output_dir)

    betas = [.0001, .0002, .0004, .001, .005]
    for beta in betas:
        print("-------- beta: %2.6f ---------"%beta)
        odir = os.path.join(cvae_output_dir, 'beta-%2.4f'%beta)
        if not os.path.exists(odir):
            os.makedirs(odir)

        mlp_cycle_vae = BeatMlpCycleVAE(**vae_kwargs)
        mlp_cycle_vae.set_discrim_model(mlp_model, discrim_beta=beta)
        mlp_cycle_vae.fit(Xdata, Xdata,
                          epochs=num_epochs,
                          log_interval=None,
                          epoch_log_interval=5,
                          lr=lr,
                          output_dir = odir)
        mlp_cycle_vae.save(os.path.join(odir, "cvae.pkl"))


#########################
# Evaluating + Plots    #
#########################

