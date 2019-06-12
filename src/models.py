""" Model defintions. """
from copy import deepcopy

import torch
from torch import nn
import torch.nn.functional as F


class ByteToFloat(nn.Module):
    """ Converts ByteTensor to FloatTensor and rescales.
    """

    def forward(self, x):
        assert (
            x.dtype == torch.uint8
        ), "The model expects states of type ByteTensor."
        return x.float().div_(255)


def get_feature_extractor(in_channels, map_no=16):
    return nn.Sequential(
        ByteToFloat(),
        nn.Conv2d(in_channels, 16, kernel_size=3),
        nn.ReLU(inplace=True),
        nn.Conv2d(16, map_no, kernel_size=2),
        nn.ReLU(inplace=True),
        nn.Conv2d(map_no, map_no, kernel_size=2),
        nn.ReLU(inplace=True),
    )


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


class MiniGridFF(nn.Module):
    def __init__(self, in_channels, action_no, hidden_size=64):
        super(MiniGridFF, self).__init__()
        self.lin0 = nn.Linear(in_channels * 7 * 7, hidden_size)
        self.lin1 = nn.Linear(hidden_size, action_no)
    
    def forward(self, x):
        assert (
            x.dtype == torch.uint8
        ), "The model expects states of type ByteTensor."
        x = x.float().div_(255)
        x = x.view(x.shape[0], -1)
        return self.lin1(F.relu(self.lin0(x)))


class MiniGridNet(nn.Module):
    def __init__(self, in_channels, action_no, hidden_size=64, map_no=16):
        super(MiniGridNet, self).__init__()
        self.in_channels = in_channels

        self.features = get_feature_extractor(in_channels, map_no=map_no)
        self.head = nn.Sequential(
            nn.Linear(9 * map_no, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, action_no),
        )
        self.reset_parameters()

    def forward(self, x):
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


class BootstrappedEstimator(nn.Module):
    """ Implements an ensemble of models.
    """

    def __init__(self, proto_model, B=20, full=True):
        """BootstrappedEstimator constructor.

        Args:
            proto_model (torch.nn.Model): Model to be ensembled.
        """
        super(BootstrappedEstimator, self).__init__()
        if full:
            self.__features = None
            self.__ensemble = nn.ModuleList(
                [deepcopy(proto_model) for _ in range(B)]
            )
        else:
            self.__features = deepcopy(proto_model.features)
            self.__ensemble = nn.ModuleList(
                [deepcopy(proto_model.head) for _ in range(B)]
            )
        self.__bno = B
        self.reset_parameters()

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
        if x.ndimension() == 5:
            x = x.view(x.shape[0], x.shape[1] * x.shape[2], 7, 7)

        if x.ndimension() == 4 and self.__features is not None:
            x = self.__features(x)
            x = x.view(x.size(0), -1)
        elif x.ndimension() != 2 and self.__features is not None:
            raise RuntimeError(f"Received a strange input: {x.shape}")

        if mid is not None:
            return self.__ensemble[mid](x)
        return torch.stack([model(x) for model in self.__ensemble], 0)

    def feature_extractor(self, x):
        return self.__features(x)

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
        if x.ndimension() == 5:
            x = x.view(x.shape[0], x.shape[1] * x.shape[2], 7, 7)

        if x.ndimension() == 4 and self.__features is not None:
            x = self.__features(x)
            x = x.view(x.size(0), -1)
        elif x.ndimension() != 2 and self.__features is not None:
            raise RuntimeError(f"Received a strange input: {x.shape}")

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

    def reset_parameters(self):
        """ Reinitializez parameters to Xavier Uniform for all layers and
            0 bias.
        """
        self.apply(init_weights)

    @property
    def has_feature_extractor(self):
        return self.__features is not None

    def __iter__(self):
        return iter(self.__ensemble)

    def __len__(self):
        return len(self.__ensemble)

    def __str__(self):
        return f"BootstrappedEstimator(N={len(self)}, f={self.__ensemble[0]})"
