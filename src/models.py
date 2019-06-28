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
    def __init__(self, in_channels, action_no, hidden_size=64, support=None):
        super(MiniGridFF, self).__init__()
        self.bin_no = 1
        if support is not None:
            self.min, self.max, self.bin_no = support
            self.support = torch.linspace(*support)
        self.action_no = action_no
        self.lin0 = nn.Linear(in_channels * 7 * 7, hidden_size)
        self.lin1 = nn.Linear(hidden_size, action_no * self.bin_no)
        assert self.bin_no > 0, "No of bins can't be smaller than 1"
        self.reset_parameters()

    def forward(self, x, probs=False):
        assert (
            x.dtype == torch.uint8
        ), "The model expects states of type ByteTensor."
        x = x.float().div_(255)
        x = x.view(x.shape[0], -1)
        y = self.lin1(F.relu(self.lin0(x)))
        if self.bin_no > 1:
            # distributional RL
            logits = y.view(x.shape[0], self.action_no, self.bin_no)
            qs_probs = torch.softmax(logits, dim=2)
            if probs:
                return qs_probs
            return torch.mul(qs_probs, self.support.expand_as(qs_probs)).sum(2)
        # simply return the Q-values
        return y

    def reset_parameters(self):
        """ Reinitializez parameters to Xavier Uniform for all layers and
            0 bias.
        """
        self.apply(init_weights)

    def cuda(self, device=None):
        try:
            self.support = self.support.cuda(device)
        except AttributeError:
            pass
        return super().cuda(device)

    def cpu(self):
        try:
            self.support = self.support.cpu()
        except AttributeError:
            pass
        return super().cpu()


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

    def feature_extractor(self, x):
        return self.__features(x)

    def forward(self, x, mid=None, **kwargs):
        """ In training mode, when `mid` is provided, do an inference step
            through the ensemble component indicated by `mid`. Otherwise it
            returns the mean of the predictions of the ensemble.

        Args:
            x (torch.tensor): input of the model
            mid (int): id of the component in the ensemble to train on `x`.

        Returns:
            torch.tensor: the mean of the ensemble predictions.
        """
        x = self.__prep_inputs(x)

        if mid is not None:
            return self.__ensemble[mid](x, **kwargs)
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
        x = self.__prep_inputs(x)

        with torch.no_grad():
            ys = [model(x) for model in self.__ensemble]

        if action is not None:
            return torch.stack(ys, 0).var(0)[0][action]
        return torch.stack(ys, 0).var(0)

    def entropy_decrease(self, x):
        x = self.__prep_inputs(x)

        with torch.no_grad():
            ys = torch.stack(
                [model(x, probs=True) for model in self.__ensemble], 0
            )

        ys_mc = ys.mean(0)
        entropy = -(ys_mc * torch.log(ys_mc)).sum(2)
        exp_entropy = -(ys * torch.log(ys)).sum(3).mean(0)
        return entropy - exp_entropy

    def __prep_inputs(self, x):
        if x.ndimension() == 5:
            x = x.view(x.shape[0], x.shape[1] * x.shape[2], 7, 7)

        if x.ndimension() == 4 and self.__features is not None:
            x = self.__features(x)
            x = x.view(x.size(0), -1)
        elif x.ndimension() != 2 and self.__features is not None:
            raise RuntimeError(f"Received a strange input: {x.shape}")
        return x

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
    def support(self):
        return self.__ensemble[0].support

    @property
    def has_feature_extractor(self):
        return self.__features is not None

    def __iter__(self):
        return iter(self.__ensemble)

    def __len__(self):
        return len(self.__ensemble)

    def __str__(self):
        return f"BootstrappedEstimator(N={len(self)}, f={self.__ensemble[0]})"
