""" Model defintions. """
from copy import deepcopy

import torch
import torch.distributions as D
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

    def __init__(self, proto_model, B=20, beta=0):
        """BootstrappedEstimator constructor.

        Args:
            proto_model (torch.nn.Model): Model to be ensembled.
            B (int, optional): Defaults to 20. Size of the ensemble
            beta (int, optional): Defaults to 0. The scale of the prior
                function. If beta=0 there is no prior.
            vote (bool, optional): Defaults to False. The prediction is given
                by the majority agreeing on the optimal action.
        """
        super(BootstrappedEstimator, self).__init__()
        self.__ensemble = [deepcopy(proto_model) for _ in range(B)]
        self.__bno = B
        self.__beta = beta
        self.__prior_fns = []
        self.__priors = []

        for model in self.__ensemble:
            model.reset_parameters()

        if beta:
            self.__prior_fns = [deepcopy(model) for model in self.__ensemble]
            for model, prior_fn in zip(self.__ensemble, self.__prior_fns):
                # set some priors based on the ensemble initialization
                loc = model.weight.data.clone()
                scale = torch.full_like(loc, 0.1 * self.__beta)
                self.__priors.append(D.Normal(loc, scale))
                # we won't be training the prior functions
                prior_fn.weight.requires_grad = False

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
            y = self.__ensemble[mid](x)
            if self.__priors:
                self.__prior_fns[mid].weight.data = self.__priors[mid].sample()
                y += self.__prior_fns[mid](x)
            return y

        if self.__priors:
            for prior, prior_fn in zip(self.__priors, self.__prior_fns):
                prior_fn.weight.data = prior.sample()
            ys = [
                m(x) + p(x) for m, p in zip(self.__ensemble, self.__prior_fns)
            ]
        else:
            ys = [model(x) for model in self.__ensemble]

        ys = torch.stack(ys, 0)
        # return (ys.mean(0), self.__agreed_q_vals(ys))
        return ys.mean(0)

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

        if action is None:
            return torch.stack(ys, 0).var(0)
        return torch.stack(ys, 0).var(0)[0][action]

    def parameters(self, recurse=True):
        """ Groups the ensemble parameters so that the optimizer can keep
            separate statistics for each model in the ensemble.

        Returns:
            iterator: a group of parameters.
        """
        return [{"params": model.parameters()} for model in self.__ensemble]

    def __agreed_q_vals(self, ys):
        bno, state_no = self.__bno, ys.shape[1]
        max_vals, max_idxs = ys.max(2)
        min_vals, _ = ys.min(2)

        # count the votes
        vote_cnt = max_idxs.sum(0).float()

        # the agreed wining action for each state
        winning_acts = vote_cnt > torch.zeros_like(vote_cnt).fill_(bno / 2)

        # mask according to the agreed wining action
        mask = torch.where(winning_acts.byte(), max_idxs, 1 - max_idxs).byte()

        qvals = torch.zeros(state_no, 2)
        for i, argmax in enumerate(winning_acts):
            max_val = (max_vals[:, i].masked_select(mask[:, i])).mean()
            min_val = (min_vals[:, i].masked_select(mask[:, i])).mean()

            qvals[i][argmax.item()] = max_val
            qvals[i][1 - argmax.item()] = min_val
        return qvals

    def __iter__(self):
        return iter(self.__ensemble)

    def __len__(self):
        return len(self.__ensemble)

    def __str__(self):
        return f"BootstrappedEstimator(N={len(self)}, beta={self.__beta})"
