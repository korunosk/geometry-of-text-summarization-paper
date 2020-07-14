import torch
import torch.nn as nn
import torch.nn.functional as F
from geomloss import SamplesLoss


class NNRougeRegModel(nn.Module):

    def __init__(self, config):
        super(NNRougeRegModel, self).__init__()
        self.config = config
        self.layer = nn.Linear(self.config['D'], self.config['D'])
        self.sinkhorn = SamplesLoss(loss='sinkhorn', p=self.config['p'], blur=self.config['blur'], scaling=self.config['scaling'])

    def transform(self, x):
        return F.relu(self.layer(x))

    def predict(self, d, si, h, hi):
        return self.sinkhorn(h, self.transform(d), hi, self.transform(si))

    def forward(self, e):
        return torch.norm(self.transform(e), p=2, dim=1)


class NNWAvgPRModel(nn.Module):

    def __init__(self, config):
        super(NNWAvgPRModel, self).__init__()
        self.config = config
        self.layer1 = nn.Linear(self.config['D'], self.config['D'])
        self.layer2 = nn.Linear(self.config['H'], 1)
        self.sigm = nn.Sigmoid()

    def transform(self, x):
        return F.relu(self.layer1(x))

    def predict(self, d, s, m):
        x = torch.cat((self.transform(d), self.transform(s)), axis=2)
        z = self.layer2(x).squeeze()
        return torch.stack([ torch.sum(z[i].masked_select(m[i])) for i in range(z.shape[0]) ])

    def forward(self, d, si, sj, mi, mj):
        score1 = self.predict(d, si, mi)
        score2 = self.predict(d, sj, mj)
        return self.sigm(self.config['scaling_factor'] * (score1 - score2))


class LinSinkhornRegModel(nn.Module):
    
    def __init__(self, config):
        super(LinSinkhornRegModel, self).__init__()
        self.config = config
        self.layer = nn.Linear(self.config['D'], self.config['D'], bias=False)
        self.sinkhorn = SamplesLoss(loss='sinkhorn', p=self.config['p'], blur=self.config['blur'], scaling=self.config['scaling'])
    
    def transform(self, x):
        return self.layer(x)

    def predict(self, d, si, h, hi):
        return self.sinkhorn(h, self.transform(d), hi, self.transform(si))

    def forward(self, d, si, h, hi):
        return torch.exp(-self.predict(d, si, h, hi))


class LinSinkhornPRModel(nn.Module):

    def __init__(self, config):
        super(LinSinkhornPRModel, self).__init__()
        self.config = config
        self.layer = nn.Linear(self.config['D'], self.config['D'], bias=False)
        self.sinkhorn = SamplesLoss(loss='sinkhorn', p=self.config['p'], blur=self.config['blur'], scaling=self.config['scaling'])
        self.sigm = nn.Sigmoid()
    
    def transform(self, x):
        return self.layer(x)

    def predict(self, d, si, h, hi):
        return self.sinkhorn(h, self.transform(d), hi, self.transform(si))

    def forward(self, d, si, sj, h, hi, hj):
        dist1 = self.predict(d, si, h, hi)
        dist2 = self.predict(d, sj, h, hj)
        return self.sigm(self.config['scaling_factor'] * (dist2 - dist1))


class NNSinkhornPRModel(nn.Module):

    def __init__(self, config):
        super(NNSinkhornPRModel, self).__init__()
        self.config = config
        self.layer = nn.Linear(self.config['D'], self.config['D'])
        self.sinkhorn = SamplesLoss(loss='sinkhorn', p=self.config['p'], blur=self.config['blur'], scaling=self.config['scaling'])
        self.sigm = nn.Sigmoid()
    
    def transform(self, x):
        return F.relu(self.layer(x))
    
    def predict(self, d, si, h, hi):
        return self.sinkhorn(h, self.transform(d), hi, self.transform(si))

    def forward(self, d, si, sj, h, hi, hj):
        dist1 = self.predict(d, si, h, hi)
        dist2 = self.predict(d, sj, h, hj)
        return self.sigm(self.config['scaling_factor'] * (dist2 - dist1))
