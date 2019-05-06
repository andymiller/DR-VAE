""" MLP and fitting functions """
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from sklearn.metrics import roc_auc_score, r2_score, f1_score
from drvae.model import base
from drvae.model.base import Model

# nn transformations
def logit(p):    return np.log(p) - np.log(1.-p)
def sigmoid(a):  return 1. / (1. + np.exp(-a))
def relu(x):     return np.maximum(0, x)
def softplus(a): return np.logaddexp(a, 0)


######################
# Beat MLP Models    #
######################

class BeatMlpMixed(Model):
    def __init__(self, **kwargs):
        super(BeatMlpMixed, self).__init__(**kwargs)
        self.bin_dims = kwargs.get("binary_dims")
        self.cont_dims = kwargs.get("continuous_dims")
        self.net = MLP(**kwargs)
        self.cont_loss = base.NanMSELoss(reduction='mean')
        self.bin_loss  = base.NanBCEWithLogitsLoss(reduction='mean')
        self.is_continuous = True

    def lossfun(self, data, target):
        logit = self.forward(data)
        bin_loss  = self.bin_loss(logit[:, self.bin_dims], target[:, self.bin_dims])
        cont_loss = self.cont_loss(logit[:, self.cont_dims], target[:, self.cont_dims])
        print("bin/cont: ", bin_loss.mean()[0,0], cont_loss.mean()[0,0])
        return torch.mean(bin_loss + cont_loss), logit

    def forward(self, data):
        return self.net(data)

    def fit(self, Xdata, Ydata, **kwargs):
        Xtrain, Xval, Xtest = Xdata
        Ytrain, Yval, Ytest = Ydata
        self.fit_res = base.fit_mlp(self, Xtrain, Xval, Xtest,
                                    Ytrain, Yval, Ytest,
                                    **kwargs)
        return self.fit_res


class BeatMlpClassifier(Model):
    def __init__(self, **kwargs):
        super(BeatMlpClassifier, self).__init__(**kwargs)
        self.net = MLP(**kwargs)
        #self.loss = nn.BCEWithLogitsLoss(reduction='mean')
        self.loss = base.NanBCEWithLogitsLoss(reduction='mean')
        self.is_continuous = False

    def lossfun(self, data, target):
        logit = self.forward(data)
        pred_loss = self.loss(logit, target)
        return torch.mean(pred_loss), logit

    def forward(self, data):
        return self.net(data)

    def fit(self, Xdata, Ydata, **kwargs):
        Xtrain, Xval, Xtest = Xdata
        Ytrain, Yval, Ytest = Ydata
        self.fit_res = base.fit_mlp(self, Xtrain, Xval, Xtest,
                                    Ytrain, Yval, Ytest,
                                    **kwargs)
        return self.fit_res


class BeatDeepWideMlpClassifierFixed(BeatMlpClassifier):
    def __init__(self, **kwargs):
        super(BeatDeepWideMlpClassifierFixed, self).__init__(**kwargs)
        self.dim_wide = kwargs.get("dim_wide")

    def forward(self, data):
        """ this module takes in a batch_sz x (C x T + dim_wide) """
        last_out = data[:, -self.dim_wide:]
        ekg_data = data[:, :-self.dim_wide].contiguous()
        return self.net(ekg_data) + last_out


class BeatDeepWideMlpClassifier(BeatMlpClassifier):
    """ wide and deep EKG beat mlp.  Expects the last `dim_wide` dimensions
        to be linearly added into the final prediction.
    """
    def __init__(self, **kwargs):
        # first, make the EKGResnet have 100 outputs
        self.h_dim     = kwargs.get("h_dim")
        self.n_outputs = kwargs.get("total_n_outputs")
        self.dim_wide  = kwargs.get("dim_wide")
        kwargs['n_outputs'] = self.h_dim
        super(BeatDeepWideMlpClassifier, self).__init__(**kwargs)
        self.wide_out = nn.Linear(self.h_dim + self.dim_wide, self.n_outputs, bias=True)

    def forward(self, data):
        """ this module takes in a batch_sz x (C x T + dim_wide) """
        # ekg transform
        last_out = data[:, -self.dim_wide:]
        ekg_data = data[:, :-self.dim_wide].contiguous()
        # wide + EKG representation
        zout = torch.cat([self.net(ekg_data), last_out], 1)
        return self.wide_out(zout)


class BeatMlpRegression(Model):
    def __init__(self, **kwargs):
        super(BeatMlpRegression, self).__init__(**kwargs)
        self.net = MLP(**kwargs)
        #self.loss = nn.MSELoss(reduction='mean')
        self.loss = base.NanMSELoss(reduction='mean')
        self.is_continuous = True

    def lossfun(self, data, target):
        logit = self.forward(data)
        pred_loss = self.loss(logit, target)
        return torch.mean(pred_loss), logit

    def forward(self, data):
        return self.net(data)

    def fit(self, Xdata, Ydata, **kwargs):
        Xtrain, Xval, Xtest = Xdata
        Ytrain, Yval, Ytest = Ydata
        self.fit_res = base.fit_mlp(self, Xtrain, Xval, Xtest,
                                    Ytrain, Yval, Ytest,
                                    **kwargs)
        return self.fit_res


class BeatDeepWideMlpRegression(BeatMlpRegression):
    """ wide and deep EKG beat mlp.  Expects the last `dim_wide` dimensions
        to be linearly added into the final prediction.
    """
    def __init__(self, **kwargs):
        self.h_dim     = kwargs.get("h_dim")
        self.n_outputs = kwargs.get("total_n_outputs")
        self.dim_wide  = kwargs.get("dim_wide")
        kwargs['n_outputs'] = self.h_dim
        super(BeatDeepWideMlpRegression, self).__init__(**kwargs)
        self.wide_out = nn.Linear(self.h_dim + self.dim_wide, self.n_outputs, bias=True)

    def forward(self, data):
        """ this module takes in a batch_sz x (C x T + dim_wide) """
        # ekg transform
        last_out = data[:, -self.dim_wide:]
        ekg_data = data[:, :-self.dim_wide].contiguous()
        # wide + EKG representation
        zout = torch.cat([self.net(ekg_data), last_out], 1)
        return self.wide_out(zout)


class BeatGroupedRegression(BeatMlpRegression):
    """
      Regress some target on a high-dimensional EKG Beat and side information
      vector.  Side information vector modulates the weights of the last
      layer for the EKG --- so if P-dimensional side info
      S = [race_white, race_black, ..., ...] then the final predictor looks like

          yhat = (S_{1xP} * W_{PxH}) * mlp(EKG)_{H} + bias

      args:
        - h_dim           : dimension of last hidden layer in Beat MLP
        - total_n_outputs : dimensionality of target variable
        - dim_wide        : dimension of side information vector we condition on
    """
    def __init__(self, **kwargs):
        self.h_dim = kwargs.get("h_dim")
        self.n_outputs = kwargs.get("total_n_outputs")
        self.dim_wide = kwargs.get("dim_wide")
        self.include_last_bias = kwargs.get("include_last_bias", False)
        kwargs['n_outputs'] = self.h_dim
        super(BeatGroupedRegression, self).__init__(**kwargs)
        self.side_info_mat = nn.Linear(self.dim_wide, self.h_dim+1, bias=False)

        #if self.include_last_bias:
        #self.last_bias = nn.Parameter(torch.FloatTensor([0.]), requires_grad=True)
        #self.last_bias = nn.Parameter(taorch.randn(1, self.dim_wideFloatTensor([0.]), requires_grad=True)

    def forward(self, data):
        side_info = data[:, -self.dim_wide:]
        ekg_data  = data[:, :-self.dim_wide].contiguous()
        last_w = self.side_info_mat(side_info)
        last_h = self.net(ekg_data)
        zout  = torch.sum(last_w[:,:-1]*last_h, dim=1, keepdim=True) + last_w[:,-1][:, None]
        return zout


class MLP(Model):
    def __init__(self, **kwargs):
        super(MLP, self).__init__(**kwargs)
        self.data_dim   = kwargs.get("data_dim")
        self.output_dim = kwargs.get("n_outputs")
        self.hdims      = kwargs.get("hdims", [50,50,50])
        self.dropout_p  = kwargs.get("dropout_p", .5)

        # compute log Pr(Y | h_last)
        # construct generative network for beats
        sizes = [self.data_dim] + self.hdims
        modules = []
        for din, dout in zip(sizes[:-1], sizes[1:]):
            modules.append(nn.Linear(din, dout))
            modules.append(nn.ReLU())
            modules.append(nn.Dropout(self.dropout_p))

        modules.append(nn.Linear(sizes[-1], self.output_dim))
        self.forward_net = nn.Sequential(*modules)
        self.init_params()
        self.is_binary=True
        self.is_continuous=False

    def forward(self, x):
        return self.forward_net(x.view(x.shape[0], -1))
