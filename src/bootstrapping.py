""" DQN Policy routines for learning from bootstrapped data.
"""
from functools import partial
from typing import NamedTuple
import torch
from numpy import random
from wintermute.policy_evaluation import EpsilonGreedyPolicy
from wintermute.policy_improvement import DQNPolicyImprovement, get_dqn_loss
from wintermute.policy_evaluation import get_epsilon_schedule
from src.models import BootstrappedEstimator, MiniGridNet


class EpsilonGreedyOutput(NamedTuple):
    """ The output of the epsilon greedy policy. """

    action: int
    q_value: float
    full: object


class EnsembleDQNLoss(NamedTuple):
    r""" DQNLoss but for ensembles. """

    loss: torch.Tensor
    variance: torch.Tensor
    component_losses: list  # list of DQNLoss objects


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
        params = estimator.parameters()
        try:
            self.__device = next(params).device
        except TypeError:
            self.__device = next(params[0]["params"]).device

        self.action_no = action_no
        self.epsilon = epsilon
        try:
            epsilon = next(self.epsilon)
        except TypeError:
            self.epsilon = get_epsilon_schedule(**self.epsilon)
            epsilon = next(self.epsilon)

        # TODO: Most likely this policy evaluation step is not working
        # with batches. Need to test.

        # self._get_variance = eval(var_method + "_variance")
        self.act = self.__voting_action if vote else self.__mean_action

    def __mean_action(self, state):
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

    def __voting_action(self, state):
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


class BootstrappedDQNPolicyImprovement(DQNPolicyImprovement):
    r""" Object doing DQN Policy improvement step with a Bootstrapped
    Ensemble estimator.
    """

    def __call__(self, batch, cb=None):
        dqn_loss = self.__get_dqn_loss(batch)

        if cb:
            loss = cb(dqn_loss)
        else:
            loss = dqn_loss.loss.mean()

        loss.backward()
        self.update_estimator()

    def __get_dqn_loss(self, batch):
        batch, boot_masks = batch
        bsz = batch[0].shape[0]
        batch = [el.to(self.device) for el in batch]
        boot_masks.to(self.device)
        # split batch in mini-batches for each ensemble component.
        # because now state and state_ have differen dimensions we cannot do:
        # batches = [[el[bm] for el in batch] for bm in boot_masks]
        # instead we mask the bootmask too... :(
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

        # Gather the losses for each batch and ensemble component. We use
        # partial application to set which ensemble component gets trained.
        dqn_losses = [
            get_dqn_loss(
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
        return EnsembleDQNLoss(
            loss=dqn_loss,
            # variance=self.estimator.var(batch[0]).gather(1, batch[1]),
            variance=None,
            component_losses=dqn_losses,
        )


def main():
    B, bsz = 11, 7
    prototype = MiniGridNet(6, 4)
    ensemble = BootstrappedEstimator(prototype, B=B)
    policy = EpsilonGreedyPolicy(
        ensemble, 4, {"start": 1.0, "end": 0.1, "steps": 1000}
    )

    x = torch.randint(0, 255, (1, 6, 7, 7)).byte()
    print(ensemble(x))
    print(policy(x))
    print(ensemble.var(x, 2))

    policy_improvement = BootstrappedDQNPolicyImprovement(
        ensemble,
        torch.optim.Adam(ensemble.parameters(), lr=0.00235),
        0.92,
        is_double=True,
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
