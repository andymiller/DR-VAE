"""
VAEs and variants of VAEs (for generic data)

Models for joint generative models over some attribute (discrete/binary)
and the continuous beat signal
"""
import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
import pyprind
from drvae.model import base, train
import drvae.misc
ln2pi = np.log(2*np.pi)


######################
# VAE Model Classes  #
######################
class VAE(base.Model):
    def __init__(self, **kwargs):
        super(VAE, self).__init__(**kwargs)
        ll_name = kwargs.get("loglike_function", "gaussian")
        if ll_name == "gaussian":
            self.ll_fun = recon_loglike_function
        elif ll_name == "bernoulli":
            self.ll_fun = binary_recon_loglike_function

        print(ll_name)
        print(self.ll_fun)

    def forward(self, x, use_mean=False):
        mu, logvar = self.encode(x)
        if use_mean:
            z = mu
        else:
            z = self.reparameterize(mu, logvar)
        return self.decode(z), z, mu, logvar

    def reparameterize(self, **kwargs):
        raise NotImplementedError

    def sample_and_decode(self, mu, logvar):
        z = self.reparameterize(mu, logvar)
        return self.decode(z)

    def fit(self, Xdata, Ydata, **kwargs):
        Xtrain, Xval, Xtest = Xdata
        Ytrain, Yval, Ytest = Ydata
        self.fit_res = train.fit_vae(self, Xtrain, Xval, Xtest,
                                           Ytrain, Yval, Ytest, **kwargs)
        return self.fit_res

    def reconstruction_error(self, X):
        dset = torch.utils.data.TensorDataset(
            torch.FloatTensor(X), torch.zeros(X.shape[0], 1))
        loader = torch.utils.data.DataLoader(dset, batch_size=512)
        errors = []
        for bi, (data, _) in enumerate(pyprind.prog_bar(loader)):
            data = Variable(data).cuda()
            rX = self.forward(data)[0]
            err = torch.abs(data - rX).mean(-1).mean(-1)
            errors.append(err.cpu())
        return torch.cat(errors)

    def lossfun(self, data, recon_data, target, mu, logvar):
        recon_ll = self.ll_fun(
            recon_data.view(recon_data.shape[0], -1),
            data.view(data.shape[0], -1))
        kl_to_prior = kldiv_to_std_normal(mu, logvar)
        return -torch.mean(recon_ll - kl_to_prior)


class BeatConvVAE(VAE):
    """ Convolutional VAE for a single Beat ... conv - deconv model
    """
    def __init__(self, **kwargs):
        super(BeatConvVAE, self).__init__(**kwargs)
        n_channels      = kwargs.get("n_channels")  # 3 for long leads
        n_samples       = kwargs.get("n_samples")   # 70-ish for single beats
        self.latent_dim = kwargs.get("latent_dim")
        self.data_shape = (n_channels, n_samples)
        self.verbose    = kwargs.get("verbose", False)
        kernel_size     = kwargs.get("kernel_size", 8)

        self.encode_net = conv.BeatConvMLP(n_channels=n_channels,
                                           n_samples=n_samples,
                                           n_outputs=50)
        self.erelu        = nn.ReLU()
        self.encode_mu    = nn.Linear(50, self.latent_dim)
        self.encode_lnvar = nn.Linear(50, self.latent_dim)
        self.encode_drop  = nn.Dropout()

        self.decode_net = conv.BeatDeconvMLP(n_channels=n_channels,
                                             n_samples=n_samples,
                                             n_latents=self.latent_dim)

    def decode(self, z):
        return self.decode_net(z)

    def encode(self, x):
        h1 = self.encode_net(x).view(x.size(0), -1)
        h1 = self.erelu(h1)
        return self.encode_mu(h1), self.encode_lnvar(h1)

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = Variable(std.data.new(std.size()).normal_())
        return eps.mul(std).add_(mu)


class LinearVAE(VAE):
    def __init__(self, **kwargs):
        super(LinearVAE, self).__init__(**kwargs)
        n_channels      = kwargs.get("n_channels")  # 3 for long leads
        n_samples       = kwargs.get("n_samples")   # 70-ish for single beats
        self.latent_dim = kwargs.get("latent_dim")
        self.data_dim   = n_channels * n_samples
        self.data_shape = (n_channels, n_samples)
        self.verbose    = kwargs.get("verbose", False)

        # construct generative network for beats
        self.decode_net = nn.Linear(self.latent_dim, self.data_dim)

        # encode network
        self.encode_mu = nn.Linear(self.data_dim, self.latent_dim)
        self.encode_lnvar = nn.Linear(self.data_dim, self.latent_dim)

        # z parameters --- add mean
        self.init_params()

    def decode(self, z):
        return self.decode_net(z).view(-1, *self.data_shape)

    def encode(self, x):
        x = x.view(-1, self.data_dim)
        return self.encode_mu(x), self.encode_lnvar(x)

    def reparameterize(self, mu, logvar, training=False):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x, use_mean=False):
        mu, logvar = self.encode(x.view(-1, self.data_dim))
        if use_mean:
            z = mu
        else:
            z = self.reparameterize(mu, logvar)
        return self.decode(z), z, mu, logvar


class LinearCycleVAE(LinearVAE):
    """ Reconstructs a high-dimensional observation using an additional
    reconstruction penalty """
    def set_discrim_model(self, discrim_model, discrim_beta):
        self.discrim_model = [discrim_model]
        self.discrim_beta = discrim_beta

    def lossfun(self, data, recon_data, target, mu, logvar):
        # vae ELBO loss
        vae_loss = super(LinearCycleVAE, self).lossfun(
            data, recon_data, target, mu, logvar)

        if self.discrim_beta == 0:
            return vae_loss

        # discrim reconstruction loss
        zdiscrim_data  = self.discrim_model[0](data)
        zdiscrim_recon = self.discrim_model[0](recon_data)
        disc_loss = self.discrim_beta * \
            torch.sum((zdiscrim_data-zdiscrim_recon)**2)
        #print(disc_loss, vae_loss)
        assert ~np.isnan(vae_loss.clone().detach().cpu())
        assert ~np.isnan(disc_loss.clone().detach().cpu())
        return vae_loss + disc_loss


class BeatMlpVAE(VAE):

    def __init__(self, **kwargs):
        super(BeatMlpVAE, self).__init__(**kwargs)
        n_channels      = kwargs.get("n_channels")  # 3 for long leads
        n_samples       = kwargs.get("n_samples")   # 70-ish for single beats
        self.latent_dim = kwargs.get("latent_dim")
        self.hdims      = kwargs.get("hdims", [500])
        self.verbose    = kwargs.get("verbose", False)
        self.data_dim   = n_channels * n_samples
        self.data_shape = (n_channels, n_samples)

        # construct generative network for beats
        sizes = [self.latent_dim] + self.hdims
        modules = []
        for din, dout in zip(sizes[:-1], sizes[1:]):
            modules.append(nn.Linear(din, dout))
            modules.append(nn.ReLU())
            modules.append(nn.Dropout())

        modules.append(nn.Linear(sizes[-1], self.data_dim))
        modules.append(nn.Tanh())
        self.decode_net = nn.Sequential(*modules)

        # encoder network guts (reverses the generative process)
        rsizes = self.hdims[::-1]
        emodules = [nn.Linear(self.data_dim, sizes[-1])]
        for din, dout in zip(rsizes[:-1], rsizes[1:]):
            emodules.append(nn.Linear(din, dout))
            emodules.append(nn.ReLU())
            emodules.append(nn.Dropout())

        self.encode_net   = nn.Sequential(*emodules)
        self.encode_mu    = nn.Linear(rsizes[-1], self.latent_dim)
        self.encode_lnvar = nn.Linear(rsizes[-1], self.latent_dim)
        self.encode_drop  = nn.Dropout()

        # z parameters --- add mean
        self.init_params()

    def decode(self, z):
        return self.decode_net(z).view(-1, *self.data_shape)

    def encode(self, x):
        h1 = self.encode_net(x.view(-1, self.data_dim))
        return self.encode_mu(h1), self.encode_lnvar(h1)

    def reparameterize(self, mu, logvar, training=False):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x, use_mean=False):
        mu, logvar = self.encode(x.view(-1, self.data_dim))
        if use_mean:
            z = mu
        else:
            z = self.reparameterize(mu, logvar)
        return self.decode(z), z, mu, logvar


class BeatMlpCycleVAE(BeatMlpVAE):
    """ Reconstructs a high-dimensional observation using an additional
    reconstruction penalty """
    def set_discrim_model(self, discrim_model, discrim_beta):
        self.discrim_model = [discrim_model]
        self.discrim_beta = discrim_beta

    #TODO add cycle loss to the existing loss
    def lossfun(self, data, recon_data, target, mu, logvar):
        # vae ELBO loss
        vae_loss = super(BeatMlpCycleVAE, self).lossfun(
            data, recon_data, target, mu, logvar)

        if self.discrim_beta == 0:
            return vae_loss

        # discrim reconstruction loss
        zdiscrim_data  = self.discrim_model[0](data)
        zdiscrim_recon = self.discrim_model[0](recon_data)
        disc_loss = self.discrim_beta * \
            torch.sum((zdiscrim_data-zdiscrim_recon)**2)

        assert ~np.isnan(vae_loss.clone().detach().cpu())
        assert ~np.isnan(disc_loss.clone().detach().cpu())
        return vae_loss + disc_loss


class BeatMlpCondVAE(VAE):
    """ Reconstructs a high-dimensional observation given a conditioning
    latent variable (could be a sample) """
    def __init__(self, **kwargs):
        super(BeatMlpCondVAE, self).__init__(**kwargs)
        n_channels      = kwargs.get("n_channels")  # 3 for long leads
        n_samples       = kwargs.get("n_samples")   # 70-ish for single beats
        self.latent_dim = kwargs.get("latent_dim")
        self.cond_dim   = kwargs.get("cond_dim")
        unsupervised_dropout_p = kwargs.get("unsupervised_dropout_p", .25)
        self.hdims      = kwargs.get("hdims", [500])
        self.data_dim   = n_channels * n_samples
        self.data_shape = (n_channels, n_samples)

        # construct generative network for beats: dim(zfull) => data_dim
        self.zu_dropout = nn.Dropout(p=unsupervised_dropout_p)
        sizes   = [self.latent_dim + self.cond_dim] + self.hdims
        modules = []
        for din, dout in zip(sizes[:-1], sizes[1:]):
            modules.append(nn.Linear(din, dout))
            modules.append(nn.ReLU())
            modules.append(nn.Dropout(p=.25))

        self.decode_net = nn.Sequential(*modules)
        self.decode_final = nn.Linear(sizes[-1]+self.latent_dim+self.cond_dim, self.data_dim)

        # encoder network guts (reverses the generative process)
        rsizes = self.hdims[::-1]
        emodules = [nn.Linear(self.data_dim, sizes[-1])]
        for din, dout in zip(rsizes[:-1], rsizes[1:]):
            emodules.append(nn.Linear(din, dout))
            emodules.append(nn.ReLU())
            emodules.append(nn.Dropout(p=.25))

        self.encode_net   = nn.Sequential(*emodules)
        self.encode_mu    = nn.Linear(rsizes[-1], self.latent_dim)
        self.encode_lnvar = nn.Linear(rsizes[-1], self.latent_dim)
        self.encode_drop  = nn.Dropout()

        # z parameters --- add mean
        self.init_params()

    def set_discrim_model(self, discrim_model):
        self.discrim_model = [discrim_model]

    def decode(self, z):
        h = self.decode_net(z)
        h = self.decode_final(torch.cat([z, h], 1))
        return h.view(-1, *self.data_shape)

    def encode(self, x):
        h1 = self.encode_net(x)
        return self.encode_mu(h1), self.encode_lnvar(h1)

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = Variable(std.data.new(std.size()).normal_())
        return eps.mul(std).add_(mu)

    def sample_and_decode(self, mu, logvar):
        z = self.reparameterize(mu, logvar)
        return self.decode(z)

    def forward(self, x, use_mean=False):
        mu, logvar = self.encode(x.view(-1, self.data_dim))
        if use_mean:
            z = mu
        else:
            z = self.reparameterize(mu, logvar)
        zcond = self.discrim_model[0](x)
        # do dropout on the unsupervised part to encourage use
        # of the conditional code
        zfull = torch.cat([zcond, self.zu_dropout(z)], 1)
        return self.decode(zfull), zfull, mu, logvar


###################
# Loss functions  #
###################

def recon_loglike_function(recon_x, x, noise_var=.01*.01):
    num_obs_per_batch = x.shape[1]
    ln_noise_var = np.log(noise_var)
    diff = x - recon_x
    ll   = -(.5/noise_var) * (diff*diff).sum(1) \
           -(.5*ln2pi + .5*ln_noise_var) * num_obs_per_batch
    return ll

def binary_recon_loglike_function(recon_x, x):
    ll = F.binary_cross_entropy_with_logits(
        recon_x, x.view(-1, 784), reduction='none').sum(dim=-1)
    return -1.*ll

def kldiv_to_std_normal(mu, logvar):
    # KL(q(z) || p(z)) where q(z) is the recognition network normal
    KL_q_to_prior = .5*torch.sum(logvar.exp() - logvar + mu.pow(2) - 1, dim=1)
    return KL_q_to_prior


##################
# Models Modules #
##################

class MvnVAE(VAE):

    def __init__(self, latent_dim, data_dim, hdims=[500]):
        super(MvnVAE, self).__init__()
        self.latent_dim = latent_dim
        self.data_dim   = data_dim

        # construct generative network for beats
        sizes = [latent_dim] + hdims
        modules = []
        for din, dout in zip(sizes[:-1], sizes[1:]):
            modules.append(nn.Linear(din, dout))
            modules.append(nn.ReLU())
            modules.append(nn.Dropout())

        modules.append(nn.Linear(sizes[-1], data_dim))
        modules.append(nn.Tanh())
        self.decode_net = nn.Sequential(*modules)

        # encoder network guts (reverses the generative process)
        rsizes = hdims[::-1]
        emodules = [nn.Linear(data_dim, sizes[-1])]
        for din, dout in zip(rsizes[:-1], rsizes[1:]):
            emodules.append(nn.Linear(din, dout))
            emodules.append(nn.ReLU())
            emodules.append(nn.Dropout())

        self.encode_net   = nn.Sequential(*emodules)
        self.encode_mu    = nn.Linear(rsizes[-1], latent_dim)
        self.encode_lnvar = nn.Linear(rsizes[-1], latent_dim)
        self.encode_drop  = nn.Dropout()

        # z parameters --- add mean
        #self.zmean = nn.Parameter(torch.randn(latent_dim)*.01)
        #self.class_layer = nn.Linear(latent_dim, 1)
        #self.class_layer = nn.Linear(1, 1)
        self.init_params()

    def name(self):
        return "model-mvn-vae"

    def decode(self, z):
        return self.decode_net(z)
        #.view(-1, self.num_channels, self.num_samples)

    def encode(self, x):
        h1 = self.encode_net(x)
        return self.encode_mu(h1), self.encode_lnvar(h1)

    def sample_and_decode(self, mu, logvar):
        z = self.reparameterize(mu, logvar)
        return self.decode(z)

    def reparameterize(self, mu, logvar, training=False):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x, use_mean=False):
        mu, logvar = self.encode(x.view(-1, self.data_dim))
        if use_mean:
            z = mu
        else:
            z = self.reparameterize(mu, logvar)
        return self.decode(z), z, mu, logvar

    def lossfun(self, data, recon_data, target, mu, logvar):
        recon_ll = recon_loglike_function(recon_data, data) #) / data.size(1)
        kl_to_prior = kldiv_to_std_normal(mu, logvar) #a) / data.size(1)
        return -torch.mean(recon_ll - kl_to_prior)

    def kl_q_to_prior():
        pass


def decode_batch_list(mod, zmat, batch_size=256):
    data = torch.utils.data.TensorDataset(
        torch.FloatTensor(zmat), torch.FloatTensor(zmat))
    loader = torch.utils.data.DataLoader(data,
        batch_size=batch_size, shuffle=False, pin_memory=True)
    batch_res = []
    if torch.cuda.is_available():
        mod.cuda()
        do_cuda = True
    for batch_idx, (data, target) in enumerate(pyprind.prog_bar(loader)):
        data, target = Variable(data), Variable(target)
        if do_cuda:
            data, target = data.cuda(), target.cuda()
            data, target = data.contiguous(), target.contiguous()

        res = mod.decode(data)
        batch_res.append(res.data.cpu())

    return torch.cat(batch_res, dim=0)
