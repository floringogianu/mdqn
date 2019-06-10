""" Model defintions. """
from copy import deepcopy

import torch
from torch import nn


def init_weights(module):
    """ Callback for resetting a module's weights to Xavier Uniform and
        biases to zero.
    """
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        module.bias.data.zero_()
    elif isinstance(module, nn.Conv2d):
        nn.init.xavier_uniform_(module.weight)
        module.bias.data.zero_()


class MiniGridDropnet(nn.Module):
    def __init__(
        self, in_channels, action_no, hidden_size=64, p=0.1, mc_samples=10
    ):
        super(MiniGridDropnet, self).__init__()
        self.in_channels = in_channels
        self.mc_samples = mc_samples

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=p),
            nn.Conv2d(16, 16, kernel_size=2),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=p),
            nn.Conv2d(16, 16, kernel_size=2),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Dropout(p=p),
            nn.Linear(144, hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(p=p),
            nn.Linear(hidden_size, action_no),
        )
        self.reset_parameters()

    def forward(self, x):
        assert (
            x.dtype == torch.uint8
        ), "The model expects states of type ByteTensor"
        x = x.float().div_(255)
        if x.ndimension() == 5:
            x = x.view(x.shape[0], x.shape[1] * x.shape[2], 7, 7)
        return self.head(self.features(x).view(x.size(0), -1))
    
    def var(self, x):
        with torch.no_grad():
            ys = torch.stack([self(x) for _ in range(self.mc_samples)], 0)
            return ys.var(0)

    def reset_parameters(self):
        """ Reinitializez parameters to Xavier Uniform for all layers and
            0 bias.
        """
        self.apply(init_weights)


class MiniGridNet(nn.Module):
    def __init__(self, in_channels, action_no, hidden_size=64):
        super(MiniGridNet, self).__init__()
        self.in_channels = in_channels

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=2),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Linear(144, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, action_no),
        )
        self.reset_parameters()

    def forward(self, x):
        assert (
            x.dtype == torch.uint8
        ), "The model expects states of type ByteTensor"
        x = x.float().div_(255)
        if x.ndimension() == 5:
            x = x.view(x.shape[0], x.shape[1] * x.shape[2], 7, 7)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.head(x)
        return x

    def reset_parameters(self):
        """ Reinitializez parameters to Xavier Uniform for all layers and
            0 bias.
        """
        self.apply(init_weights)


class BootstrappedEstimator(nn.Module):
    """ Implements an ensemble of models.
    """

    def __init__(self, proto_model, B=20):
        """BootstrappedEstimator constructor.

        Args:
            proto_model (torch.nn.Model): Model to be ensembled.
        """
        super(BootstrappedEstimator, self).__init__()
        self.__ensemble = nn.ModuleList(
            [deepcopy(proto_model) for _ in range(B)]
        )
        for model in self.__ensemble:
            model.reset_parameters()
        self.__bno = B

    def forward(self, x, mid=None):
        """ In training mode, when `mid` is provided, do an inference step
            through the ensemble component indicated by `mid`. Otherwise it
            returns the mean of the predictions of the ensemble.

        Args:
            x (torch.tensor): input of the model
            mid (int): id of the component in the ensemble to train on `x`.

        Returns:
            torch.tensor: the mean of the ensemble predictions.
        """
        if mid is not None:
            return self.__ensemble[mid](x)
        return torch.stack([model(x) for model in self.__ensemble], 0)

    def var(self, x, action=None):
        """ Returns the variance (uncertainty) of the ensemble's prediction
            given `x`.

        Args:
            x (torch.tensor): Input data
            action (int): Action index. Used for returning the uncertainty of a
                given action in state `x`.

        Returns:
            var: the uncertainty of the ensemble when predicting `f(x)`.
        """
        with torch.no_grad():
            ys = [model(x) for model in self.__ensemble]

        if action is not None:
            return torch.stack(ys, 0).var(0)[0][action]
        return torch.stack(ys, 0).var(0)

    def parameters(self, recurse=True):
        """ Groups the ensemble parameters so that the optimizer can keep
            separate statistics for each model in the ensemble.

        Returns:
            iterator: a group of parameters.
        """
        return [{"params": model.parameters()} for model in self.__ensemble]

    def __iter__(self):
        return iter(self.__ensemble)

    def __len__(self):
        return len(self.__ensemble)

    def __str__(self):
        return f"BootstrappedEstimator(N={len(self)})"
