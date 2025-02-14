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

    def predict(self, d, si, m, mi):
        x = torch.cat((self.transform(d), self.transform(si)), axis=2)
        z = self.layer2(x).squeeze()
        return torch.stack([ torch.sum(z[i].masked_select(mi[i])) for i in range(z.shape[0]) ])

    def forward(self, d, si, sj, mi, mj):
        score1 = self.predict(d, si, None, mi)
        score2 = self.predict(d, sj, None, mj)
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


# Conditional Models

class AvgModel(nn.Module):

    def __init__(self, D):
        super(AvgModel, self).__init__()
        self.D = D
        self.layer = nn.Linear(self.D, self.D ** 2)
    
    def forward(self, avg, *args):
        if avg: # If already averaged (CondNNWAvgPRModel)
            d, = args
            dm = d[:,0,:] # The first element at dim 1 is repeated
        else:   # If not already averaged (CondLinSinkhornPRModel)
            d, h = args
            dm = torch.sum(d * h.unsqueeze(2), axis=1)
        M = F.relu(self.layer(dm))
        return M.reshape(-1, self.D, self.D)


class CondLinSinkhornPRModel(nn.Module):

    def __init__(self, config):
        super(CondLinSinkhornPRModel, self).__init__()
        self.config = config
        self.model = AvgModel(self.config['D'])
        self.sinkhorn = SamplesLoss(loss='sinkhorn', p=self.config['p'], blur=self.config['blur'], scaling=self.config['scaling'])
        self.sigm = nn.Sigmoid()

    def generate_transformation(self, d, h):
        return self.model(False, d, h)
    
    def transform(self, x, M):
        return torch.bmm(x, M)

    def predict(self, d, si, h, hi, M=None):
        if M is None:
            M = self.generate_transformation(d, h)
        return self.sinkhorn(h, self.transform(d, M), hi, self.transform(si, M))

    def forward(self, d, si, sj, h, hi, hj):
        M = self.generate_transformation(d, h)
        dist1 = self.predict(d, si, h, hi, M)
        dist2 = self.predict(d, sj, h, hj, M)
        return self.sigm(self.config['scaling_factor'] * (dist2 - dist1))


class CondNNWAvgPRModel(nn.Module):

    def __init__(self, config):
        super(CondNNWAvgPRModel, self).__init__()
        self.config = config
        self.model = AvgModel(self.config['D'])
        self.layer = nn.Linear(self.config['H'], 1)
        self.sigm = nn.Sigmoid()
    
    def generate_transformation(self, d):
        return self.model(True, d)

    def transform(self, x, M):
        return torch.bmm(x, M)

    def predict(self, d, si, m, mi, M=None):
        if M is None:
            M = self.generate_transformation(d)
        x = torch.cat((self.transform(d, M), self.transform(si, M)), axis=2)
        z = self.layer(x).squeeze()
        return torch.stack([ torch.sum(z[i].masked_select(mi[i])) for i in range(z.shape[0]) ])

    def forward(self, d, si, sj, mi, mj):
        M = self.generate_transformation(d)
        score1 = self.predict(d, si, None, mi)
        score2 = self.predict(d, sj, None, mj)
        return self.sigm(self.config['scaling_factor'] * (score1 - score2))
