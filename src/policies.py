""" DQN Policy routines for learning from bootstrapped data.
"""
from functools import partial
from typing import NamedTuple
import torch
from numpy import random
import wintermute as wt
from src.models import BootstrappedEstimator, MiniGridNet


class EpsilonGreedyOutput(NamedTuple):
    """ The output of the epsilon greedy policy. """

    action: int
    q_value: float
    full: object


class BayesianDQNLoss(NamedTuple):
    r""" DQNLoss but for ensembles. """

    loss: torch.Tensor  # (batch_size * 1) tensor of losses.
    mc_sample_losses: list  # list of DQNLosses for each MC sample.


class DropPE(wt.EpsilonGreedyPolicy):
    """ Policy-Evaluation for Dropout-SVI estimators.

        It has two exploration modes:

        - thompson-sampling
        - epsilon-greedy
        - a combination of the two because there are no asserts in place to
        avoid this :).
    """

    def __init__(self, estimator, action_space, epsilon, thompson=False):
        super().__init__(estimator, action_space, epsilon)
        self._thompson = thompson

    def act(self, x):
        self.policy.estimator.train(mode=self._thompson)
        pi = super().act(x)
        self.policy.estimator.train(mode=True)
        return pi

    def var(self, states, actions=None):
        """ Return the variance of the Q-values.
        """
        with torch.no_grad():
            qval_vars = self.policy.estimator.var(states)
        if actions is not None:
            return qval_vars.gather(1, actions)
        return qval_vars

    def __str__(self):
        return f"DropPolicyEvaluation(thompson={self._thompson})"


class DropPI(wt.DQNPolicyImprovement):
    """ Policy Improvement for Dropout-SVI estimator.
    """

    def __call__(self, batch, cb=None):
        batch = [el.to(self.device) for el in batch]

        dqn_loss = wt.get_dqn_loss(
            batch,
            self.estimator,
            self.gamma,
            target_estimator=self.target_estimator,
            is_double=self.is_double,
            loss_fn=self.loss_fn,
        )

        loss = BayesianDQNLoss(loss=dqn_loss.loss, mc_sample_losses=[dqn_loss])

        if cb:
            loss = cb(loss)
        else:
            loss = loss.loss.mean()

        loss.backward()
        self.update_estimator()


class BootstrappedPE:
    """ Implements the policy evaluation step for bootstrapped estimators
        (ensembles). It has two behaviours:

        1) The action is the argmax of the Q-values obtained by averaging
        over the ensemble.
        2) The action is the one picked by the majority of the ensemble
        components (`vote`==True).
    """

    def __init__(self, estimator, action_no, epsilon, vote=True):
        self.__estimator = estimator
        self.__vote = vote
        self.__ensemble_sz = len(self.__estimator)
        self.policy = self
        self.__device = wt.get_estimator_device(estimator)

        self.action_no = action_no
        self.epsilon = epsilon
        try:
            epsilon = next(self.epsilon)
        except TypeError:
            self.epsilon = wt.get_epsilon_schedule(**self.epsilon)
            epsilon = next(self.epsilon)

        # TODO: Most likely this policy evaluation step is not working
        # with batches. Need to test.

        # self._get_variance = eval(var_method + "_variance")
        self.act = self.__most_voted if vote else self.__mean_value

    def __mean_value(self, state):
        state = state.to(self.__device)
        ensemble_qvals = self.__estimator(state).squeeze(1)
        qvals = ensemble_qvals.mean(0)

        epsilon = next(self.epsilon)
        if epsilon > random.uniform():
            action = random.randint(0, self.action_no)
            qval = qvals[action]
        else:
            qval, argmax_a = qvals.max(0)
            action = argmax_a.item()

        # variance = self._get_variance(ensemble_qvals, action)
        return EpsilonGreedyOutput(
            action=action, q_value=qval.item(), full=ensemble_qvals
        )

    def __most_voted(self, state):
        state = state.to(self.__device)
        ensemble_qvals = self.__estimator(state).squeeze(1)
        act_no = ensemble_qvals.shape[1]
        qvals, argmaxs = ensemble_qvals.max(1)
        votes = torch.zeros(act_no, dtype=argmaxs.dtype, device=argmaxs.device)
        votes.put_(argmaxs, torch.ones_like(argmaxs), accumulate=True)

        epsilon = next(self.epsilon)
        if epsilon > random.uniform():
            action = random.randint(0, self.action_no)
            qval = ensemble_qvals[:, action].mean()
        else:
            action = votes.argmax().item()
            qval = qvals[argmaxs == action].mean()

        # variance = self._get_variance(ensemble_qvals, action, votes)
        return EpsilonGreedyOutput(
            action=action, q_value=qval, full=ensemble_qvals
        )

    def var(self, states, actions=None):
        """ Return the variance of the Q-values.
        """
        with torch.no_grad():
            qval_vars = self.estimator.var(states)
        if actions is not None:
            return qval_vars.gather(1, actions)
        return qval_vars
    
    def entropy_decrease(self, states, actions=None):
        """ Return H - E_w[H]
        """
        with torch.no_grad():
            entropy_diffs = self.estimator.entropy_decrease(states)
        if actions is not None:
            return entropy_diffs.gather(1, actions)
        return entropy_diffs

    def __call__(self, state):
        return self.act(state)

    def __str__(self):
        cls_name = self.__class__.__name__
        act_slct = "vote" if self.__vote else "mean"
        return f"{cls_name}(strategy={act_slct})"

    @property
    def estimator(self):
        """ Return estimator. """
        return self.__estimator


def split_batch(batch, boot_masks):
    """ Split batch in mini-batches for each ensemble component. because now
    state and state_ have differen dimensions we cannot do: batches =
    [[el[bm] for el in batch] for bm in boot_masks] instead we mask the
    bootmask too... :(

    Args:
        batch (list<torch.tensor>): The classic batch.
        boot_masks (torch.tensor): A K*batch_size mask telling which transition
            is used by each ensemble component for learning from.

    Returns:
        [list]: A list of mini-batches.
    """

    batches = []
    for mid, bmask in enumerate(boot_masks):
        if bmask.sum() > 0:
            batches.append(
                (
                    mid,
                    [
                        batch[0][bmask],
                        batch[1][bmask],
                        batch[2][bmask],
                        batch[3][bmask[batch[4].squeeze()]],
                        batch[4][bmask],
                    ],
                    bmask[batch[4].squeeze()],
                )
            )
    return batches


class BootstrappedPI:
    r""" Object doing DQN Policy improvement step with a Bootstrapped
    Ensemble estimator.
    """

    def __init__(self, delegate, categorical=False):
        self.__delegate = delegate
        self.categorical = categorical
        self.__get_loss = (
            self.__get_categorical if categorical else self.__get_dqn_loss
        )

    def __call__(self, batch, cb=None):

        batch = wt.to_device(batch, self.device)
        batch, boot_masks = batch
        bsz, bsz_ = batch[0].shape[0], batch[3].shape[0]

        # pass through the feature extractor and replace states
        # with features. Also pass next_states once more if Double-DQN.
        if self.estimator.has_feature_extractor:
            online = self.estimator.feature_extractor
            target = self.target_estimator.feature_extractor
            batch[0] = online(batch[0]).view(bsz, -1)
            # if self.is_double:
            #     with torch.no_grad():
            #         features_ = online(batch[3])
            batch[3] = target(batch[3]).view(bsz_, -1)

        # split batch in smaller batches for each ensemble component.
        batches = split_batch(batch, boot_masks)

        dqn_loss = self.__get_loss(batches, boot_masks)

        if cb:
            loss = cb(dqn_loss)
        else:
            loss = dqn_loss.loss.mean()

        loss.backward()
        self.update_estimator()

    def __get_categorical(self, batches, boot_masks):
        bsz = boot_masks.shape[1]

        # Gather the losses for each batch and ensemble component. We use
        # partial application to set which ensemble component gets trained.
        dqn_losses = [
            wt.get_categorical_loss(
                batch_,
                partial(self.estimator, mid=mid),
                self.gamma,
                self.support,
                target_estimator=partial(self.target_estimator, mid=mid),
            )
            for mid, batch_, next_state_mask in batches
        ]

        # sum up the losses of a given transition across ensemble components
        dqn_loss = torch.zeros((bsz, 1), device=dqn_losses[0].loss.device)
        counts = torch.ones((bsz, 1), device=dqn_losses[0].loss.device)
        for loss, (mid, _, _) in zip(dqn_losses, batches):
            counts += boot_masks[mid].unsqueeze(1).float()
            dqn_loss[boot_masks[mid]] += loss.loss

        dqn_loss /= counts
        return BayesianDQNLoss(loss=dqn_loss, mc_sample_losses=dqn_losses)

    def __get_dqn_loss(self, batches, boot_masks):
        bsz = boot_masks.shape[1]

        # Gather the losses for each batch and ensemble component. We use
        # partial application to set which ensemble component gets trained.
        dqn_losses = [
            wt.get_dqn_loss(
                batch_,
                partial(self.estimator, mid=mid),
                self.gamma,
                target_estimator=partial(self.target_estimator, mid=mid),
                is_double=self.is_double,
                loss_fn=torch.nn.MSELoss(reduction="none"),
            )
            for mid, batch_, next_state_mask in batches
        ]

        # sum up the losses of a given transition across ensemble components
        dqn_loss = torch.zeros((bsz, 1), device=dqn_losses[0].loss.device)
        for loss, (mid, _, _) in zip(dqn_losses, batches):
            dqn_loss[boot_masks[mid]] += loss.loss

        # TODO: gradient rescalling!!!
        return BayesianDQNLoss(loss=dqn_loss, mc_sample_losses=dqn_losses)

    def __getattr__(self, name):
        return getattr(self.__delegate, name)


def main():
    B, bsz = 11, 7
    prototype = MiniGridNet(6, 4)
    ensemble = BootstrappedEstimator(prototype, B=B)
    policy = wt.EpsilonGreedyPolicy(
        ensemble, 4, {"start": 1.0, "end": 0.1, "steps": 1000}
    )

    x = torch.randint(0, 255, (1, 6, 7, 7)).byte()
    print(ensemble(x))
    print(policy(x))
    print(ensemble.var(x, 2))

    policy_improvement = BootstrappedPI(
        wt.DQNPolicyImprovement(
            ensemble,
            torch.optim.Adam(ensemble.parameters(), lr=0.00235),
            0.92,
            is_double=True,
        )
    )

    probs = torch.empty(B, bsz).fill_(0.5)
    batch = [
        [
            torch.randint(0, 255, (bsz, 6, 7, 7)).byte(),
            torch.randint(0, 4, (bsz, 1)),
            torch.rand((bsz, 1)),
            torch.randint(0, 255, (bsz - 5, 6, 7, 7)).byte(),
            torch.tensor([[1, 0, 0, 0, 1, 0, 0]]).t().byte(),
        ],
        torch.bernoulli(probs).byte(),
    ]

    policy_improvement(batch)


if __name__ == "__main__":
    main()
