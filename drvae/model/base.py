import torch
import torch.nn as nn
from torch import nn
from torch.autograd import Variable
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from sklearn.metrics import roc_auc_score, r2_score, f1_score
import pandas as pd

def roc_auc_score_nan(y, p):
    nan_idx = pd.isnull(y) | pd.isnull(p)
    return roc_auc_score(y[~nan_idx], p[~nan_idx])

def f1_score_nan(y, p):
    nan_idx = np.isnan(y)
    return f1_score(y[~nan_idx], p[~nan_idx])

def r2_score_nan(y, p):
    nan_idx = pd.isnull(y) | pd.isnull(p)
    return r2_score(y[~nan_idx], p[~nan_idx])

import numpy as np
import pandas as pd
import pyprind, copy

class Model(nn.Module):
    """ base model class w/ some helper functions for training/manipulating
    parameters, and saving
    """

    def __init__(self, **kwargs):
        super(Model, self).__init__()
        self.kwargs = kwargs
        self.fit_res = None

    def save(self, filename):
        torch.save({'state_dict'  : self.state_dict(),
                    'kwargs'      : self.kwargs,
                    'fit_res'     : self.fit_res,
                    'model_class' : type(self)},
                    f=filename)

    def fit(self, data):
        raise NotImplementedError

    def lossfun(self, data, target):
        raise NotImplementedError

    def init_params(self):
        for p in self.parameters():
            if p.requires_grad==True:
                p.data.uniform_(-.05, .05)

    def fix_params(self):
        for p in self.parameters():
            p.requires_grad = False

    def free_params(self):
        for p in self.parameters():
            p.requires_grad = True

    def num_params(self):
        return np.sum([p.numel() for p in self.parameters()])


def load_model(fname):
    model_dict = torch.load(fname)
    mod = model_dict['model_class'](**model_dict['kwargs'])
    mod.load_state_dict(model_dict['state_dict'])
    mod.fit_res = model_dict['fit_res']
    return mod


class MaskedBCELoss(nn.Module):
    """ BCELoss that accounts for NaNs (given by mask) """
    def __init__(self, reduction='mean'):
        super(MaskedBCELoss, self).__init__()
        self.bce_loss = nn.BCELoss(reduction=reduction)

    def forward(self, output, target, mask):
        """ masked binary cross entropy loss
        Args:
          - output: batch_size x D float tensor with values in [0, 1]
          - target: batch_size x D float tensor with values in {0, 1}
          - mask  : batch_size x D byte tensor with 1 = not nan (include in loss)
        """
        tvec = target.view(-1)
        ovec = output.view(-1)
        mvec = mask.view(-1)

        # grab valid --- return bce loss
        tvalid = tvec.masked_select(mvec)
        ovalid = ovec.masked_select(mvec)
        return self.bce_loss(ovalid, tvalid)


def isnan(x):
    return x != x


class NanBCEWithLogitsLoss(nn.Module):
    """ BCELoss that accounts for NaNs (given by mask) """
    def __init__(self, reduction='mean'):
        super(NanBCEWithLogitsLoss, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='mean')

    def forward(self, output, target):
        """ masked binary cross entropy loss
        Args:
          - output: batch_size x D float tensor with values in [0, 1]
          - target: batch_size x D float tensor with values in {0, 1}
          - mask  : batch_size x D byte tensor with 1 = not nan (include in loss)
        """
        tvec = target.view(-1)
        ovec = output.view(-1)
        mvec = Variable(~isnan(tvec).data)

        # grab valid --- return bce loss
        tvalid = tvec.masked_select(mvec)
        ovalid = ovec.masked_select(mvec)
        return self.bce_loss(ovalid, tvalid)


class NanMSELoss(nn.Module):
    """ BCELoss that accounts for NaNs (given by mask) """
    def __init__(self, reduction='mean'):
        super(NanMSELoss, self).__init__()
        self.mse_loss = nn.MSELoss(reduction='mean')

    def forward(self, output, target):
        """ masked binary cross entropy loss
        Args:
          - output: batch_size x D float tensor with values in [0, 1]
          - target: batch_size x D float tensor with values in {0, 1}
          - mask  : batch_size x D byte tensor with 1 = not nan (include in loss)
        """
        tvec = target.view(-1)
        ovec = output.view(-1)
        mvec = Variable(~isnan(tvec).data)

        # grab valid --- return bce loss
        tvalid = tvec.masked_select(mvec)
        ovalid = ovec.masked_select(mvec)
        return self.mse_loss(ovalid, tvalid)


##############################
# standard multi-layer MLP   #
##############################

def fit_mlp(model, Xtrain, Xval, Xtest, Ytrain, Yval, Ytest, **kwargs):
    # args
    #kwargs = {}
    do_cuda       = kwargs.get("do_cuda", torch.cuda.is_available())
    batch_size    = kwargs.get("batch_size", 256)
    epochs        = kwargs.get("epochs", 50)
    output_dir    = kwargs.get("output_dir", "./")
    weight_decay  = kwargs.get("weight_decay", 1e-5)
    learning_rate = kwargs.get("learning_rate", 1e-3)
    opt_type      = kwargs.get("optimizer", "adam")
    class_weights = kwargs.get("class_weights", False)
    side_dim      = kwargs.get("side_dim", None)
    wide_weights  = kwargs.get("wide_weights", None)
    log_interval  = kwargs.get("log_interval", False)
    lr_reduce_interval = kwargs.get("lr_reduce_interval", 25)
    epoch_log_interval = kwargs.get("epoch_log_interval", 1)
    #Wtrain, Wval, Wtest = kwargs.get("Wdata", (None, None, None))
    print("-------------------")
    print("fitting mlp: ", kwargs)

    # set up data
    #kwargs = {'num_workers': 1, 'pin_memory': True} if do_cuda else {}
    def prep_loader(X, Y, shuffle=False, weighted_sampler=False):
        data = torch.utils.data.TensorDataset(
            torch.FloatTensor(X), torch.FloatTensor(Y))

        if weighted_sampler==True:
            class_sample_count = pd.value_counts(Y.squeeze())
            weights = np.zeros(len(Y.squeeze()))
            weights[ Y[:,0] == 0. ] = 1./class_sample_count.loc[0.]
            weights[ Y[:,0] == 1. ] = 1./class_sample_count.loc[1.]
            sample_weights = torch.DoubleTensor(weights)
            sampler = torch.utils.data.sampler.WeightedRandomSampler(sample_weights, len(sample_weights))
            shuffle = False
        else:
            sampler = None

        loader = torch.utils.data.DataLoader(data,
            batch_size=batch_size, shuffle=shuffle, pin_memory=True, sampler=sampler)
        return data, loader

    train_data, train_loader = prep_loader(Xtrain, Ytrain,
        shuffle=True, weighted_sampler=class_weights)
    val_data, val_loader     = prep_loader(Xval, Yval, weighted_sampler=class_weights)
    test_data, test_loader   = prep_loader(Xtest, Ytest)

    # set up optimizer
    plist = list(filter(lambda p: p.requires_grad, model.parameters()))
    if opt_type=="adam":
        optimizer = optim.Adam(plist, lr=learning_rate,
                               weight_decay=weight_decay)
    else:
        optimizer = optim.SGD(plist, lr=learning_rate,
                              weight_decay=weight_decay)

    # use GPU
    if do_cuda:
        model.cuda()
        model.is_cuda = True

    best_val_loss = np.inf
    best_val_state = None
    prev_val_loss = np.inf
    train_loss = []
    val_loss   = []
    if model.is_continuous:
        print("{:10}  {:10}  {:10}  {:10}  {:10}  ". \
            format("Epoch", "train-loss", "val-loss", "train-r2", "val-r2"))
    else:
        print("{:10}  {:10}  {:10}  {:10}  {:10}  {:10}  {:10}". \
            format("Epoch", "train-loss", "val-loss", "train-auc", "val-auc",
                            "train-f1", "val-f1"))
    for epoch in range(1, epochs + 1):
        tloss, ttrue, tpred, tauc, tf1 = run_epoch(epoch, model, train_loader, optimizer,
            do_cuda, only_compute_loss=False, log_interval=log_interval)
        vloss, vtrue, vpred, vauc, vf1 = run_epoch(epoch, model, val_loader, optimizer,
            do_cuda, only_compute_loss=True, log_interval=log_interval)
        print("{:10}  {:10}  {:10}  {:10}  {:10}  {:10}  {:10}". \
            format(epoch, "%2.6f"%tloss, "%2.6f"%vloss,
                           tauc, vauc, tf1, vf1))

        train_loss.append(tloss)
        val_loss.append(vloss)
        if vloss < best_val_loss:
            print("  (updating best loss)")
            best_val_loss = vloss
            best_val_state = copy.deepcopy(model.state_dict())

        # update learning rate if we're not doing better
        if epoch % lr_reduce_interval == 0:
            print("... reducing learning rate!")
            for param_group in optimizer.param_groups:
                param_group['lr'] *= .5

    # load in best state by validation loss
    model.load_state_dict(best_val_state)
    model.eval()

    # transform train/val/test into logit zs
    ztrain = forward_batches(model, train_loader)
    zval   = forward_batches(model, val_loader)
    ztest  = forward_batches(model, test_loader)
    resdict = {'train_elbo'  : train_loss,
               'val_elbo'    : val_loss,
               'ztrain'      : ztrain,
               'zval'        : zval,
               'ztest'       : ztest,
               'fit_kwargs'  : kwargs}
    return resdict


def forward_batches(model, data_loader):
    batch_res = []
    do_cuda = False
    if torch.cuda.is_available():
        model.cuda()
        do_cuda = True

    for batch_idx, (data, target) in enumerate(pyprind.prog_bar(data_loader)):
        data, target = Variable(data), Variable(target)
        if do_cuda:
            data, target = data.cuda(), target.cuda()
            data, target = data.contiguous(), target.contiguous()

        res = model.forward(data)
        batch_res.append(res.data.cpu())

    return torch.cat(batch_res, dim=0)


def forward_X(mod, X, batch_size=256):
    batch_res = forward_batch_list(mod, X, batch_size)
    return torch.cat(batch_res, dim=0)


def forward_batch_list(mod, X, batch_size=256, argnum=0):
    data = torch.utils.data.TensorDataset(
        torch.FloatTensor(X), torch.FloatTensor(X))
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

        res = mod.forward(data)
        if isinstance(res, tuple):
            res = res[argnum]

        batch_res.append(res.data.cpu())

    return torch.cat(batch_res, dim=0)


def run_epoch(epoch, model, data_loader, optimizer, do_cuda,
              only_compute_loss = False,
              log_interval = 20,
              num_samples  = 1):
    binary_loss = nn.BCEWithLogitsLoss()
    if only_compute_loss:
        model.eval()
    else:
        model.train()

    # iterate over batches
    total_loss = 0
    trues, preds = [], []
    for batch_idx, (data, target) in enumerate(data_loader):
        data, target = Variable(data), Variable(target)
        if do_cuda:
            data, target = data.cuda(), target.cuda()
            data, target = data.contiguous(), target.contiguous()

        # set up optimizer
        if not only_compute_loss:
            optimizer.zero_grad()

        # push data through model (make sure the recon batch batches data)
        loss, logitpreds = model.lossfun(data, target)

        # backprop
        if not only_compute_loss:
            loss.backward()
            optimizer.step()

        # track pred probs
        if model.is_continuous:
            tprobs = logitpreds.data.cpu().numpy()
        else:
            tprobs = torch.sigmoid(logitpreds).data.cpu().numpy()
        preds.append(tprobs)
        trues.append(target.data.cpu().numpy())
        total_loss += loss.item()
        if (log_interval!=False) and (batch_idx % log_interval == 0):
            print('{pre} Epoch: {ep} [{cb}/{tb} ({frac:.0f}%)]\tLoss: {loss:.6f}'.format(
                pre = "  Val" if only_compute_loss else "  Train",
                ep  = epoch,
                cb  = batch_idx*data_loader.batch_size,
                tb  = len(data_loader.dataset),
                frac = 100. * batch_idx / len(data_loader),
                loss = total_loss / (batch_idx+1)))

    total_loss /= len(data_loader)
    trues, preds = np.concatenate(trues)[:,None], np.row_stack(preds)

    # compute auc
    if model.is_continuous:
        target_aucs = np.array([ r2_score_nan(t, p)
                                 for t,p in zip(trues[:,:,0].T, preds.T)])
        print(" r2 score: ", r2_score_nan(trues.squeeze(), preds.squeeze()))
        target_f1 = "n/a"
    else:
        target_aucs = np.array([ roc_auc_score_nan(t, p)
                                 for t,p in zip(trues.T, preds.T)])
        target_f1s  = np.array([ f1_score_nan(t, np.array(p>.5, dtype=np.float))
                                 for t, p in zip(trues.T, preds.T) ])
        if len(target_f1s) > 1:
          target_f1 = "[%2.3f - %2.3f]"%(np.min(target_f1s), np.max(target_f1s))
        else:
          target_f1 = "%2.5f"%target_f1s[0]

    #target_auc = target_aucs.mean()
    if len(target_aucs) > 1:
      #target_auc = "[%2.3f - %2.3f]"%(np.min(target_aucs), np.max(target_aucs))
      #target_auc = "[%2.3f - %2.3f]"%(np.min(target_aucs), np.max(target_aucs))
      target_auc = "%2.3f, %2.3f, ..."%(target_aucs[0], target_aucs[1])
    else:
      target_auc = "%2.5f"%target_aucs[0]

    return total_loss, trues, preds, target_auc, target_f1
