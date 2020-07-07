import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from geomloss import SamplesLoss

from src.config import *


class NNRougeRegModel(nn.Module):

    def __init__(self, config):
        super(NNRougeRegModel, self).__init__()
        self.config = config
        self.sinkhorn = SamplesLoss(loss='sinkhorn', p=self.config['p'], blur=self.config['blur'], scaling=self.config['scaling'])
        self.layer = nn.Linear(self.config['D_in'], self.config['D_out'])

    def transform(self, x):
        return F.relu(self.layer(x))

    def predict(self, d, s, h, hs):
        return self.sinkhorn(h, self.transform(d), hs, self.transform(s))

    def forward(self, sent):
        return torch.norm(self.transform(sent), p=2, dim=1)


class NNWAvgPRModel(nn.Module):

    def __init__(self, config):
        super(NNWAvgPRModel, self).__init__()
        self.config = config
        self.layer1 = nn.Linear(self.config['D_in'], self.config['D_out'])
        self.layer2 = nn.Linear(self.config['H'], 1)
        self.sigm = nn.Sigmoid()

    def transform(self, x):
        return F.relu(self.layer1(x))

    def predict(self, d, s, m):
        x = torch.cat((self.transform(d), self.transform(s)), axis=2)
        z = self.layer2(x)
        return torch.stack([ torch.sum(z[i].masked_select(m[i])) for i in range(z.shape[0]) ])

    def forward(self, d, s1, s2, m1, m2):
        score1 = self.predict(d, s1, m1)
        score2 = self.predict(d, s2, m2)
        return self.sigm(self.config['scaling_factor'] * (score1 - score2))


class LinSinkhornRegModel(nn.Module):
    
    def __init__(self, config):
        super(LinSinkhornRegModel, self).__init__()
        self.config = config
        self.M = nn.Parameter(torch.randn(self.config['D_in'], self.config['D_out']))
        self.sinkhorn = SamplesLoss(loss='sinkhorn', p=self.config['p'], blur=self.config['blur'], scaling=self.config['scaling'])
    
    def transform(self, x):
        return torch.mm(x, self.M.unsqueeze(0).expand(x.shape[0], -1, -1))

    def predict(self, d, s, h, ds):
        return self.sinkhorn(h, self.transform(d), hs, self.transform(s))

    def forward(self, d, s, h, hs):
        return torch.exp(-self.predict(d, s, h, hs))


class LinSinkhornPRModel(nn.Module):

    def __init__(self, config):
        super(LinSinkhornPRModel, self).__init__()
        self.config = config
        self.M = nn.Parameter(torch.randn(self.config['D_in'], self.config['D_out']))
        self.sinkhorn = SamplesLoss(loss='sinkhorn', p=self.config['p'], blur=self.config['blur'], scaling=self.config['scaling'])
        self.sigm = nn.Sigmoid()
    
    def transform(self, x):
        return torch.bmm(x, self.M.unsqueeze(0).expand(x.shape[0], -1, -1))

    def predict(self, d, s, h, hs):
        return self.sinkhorn(h, self.transform(d), hs, self.transform(s))

    def forward(self, d, s1, s2, h, h1, h2):
        dist1 = self.predict(d, s1, h, h1)
        dist2 = self.predict(d, s2, h, h2)
        return self.sigm(self.config['scaling_factor'] * (dist2 - dist1))


class NNSinkhornPRModel(nn.Module):

    def __init__(self, config):
        super(NNSinkhornPRModel, self).__init__()
        self.config = config
        self.sinkhorn = SamplesLoss(loss='sinkhorn', p=self.config['p'], blur=self.config['blur'], scaling=self.config['scaling'])
        self.layer = nn.Linear(self.config['D_in'], self.config['D_out'])
        self.sigm = nn.Sigmoid()
    
    def transform(self, x):
        return F.relu(self.layer(x))
    
    def predict(self, d, s, h, hs):
        return self.sinkhorn(h, self.transform(d), hs, self.transform(s))

    def forward(self, d, s1, s2, h, h1, h2):
        dist1 = self.predict(d, s1, h, h1)
        dist2 = self.predict(d, s2, h, h2)
        return self.sigm(self.config['scaling_factor'] * (dist2 - dist1))
