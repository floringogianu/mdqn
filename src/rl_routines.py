""" RL primitives.
"""
from functools import partial
import torch
from wintermute.utils import CategoricalLoss, DQNLoss


def priority_update(mem, idxs, weights, dqn_loss, other_prio=None):
    """ Callback for updating priorities in the proportional-based experience
    replay and for computing the importance sampling corrected loss.
    """
    losses = dqn_loss.loss

    # Prioritize by variance
    if other_prio is not None:
        # print("var: \n", variances.mean())
        mem.update(idxs, [x.item() for x in other_prio.detach()])
        return (losses * weights.to(losses.device).view_as(losses)).mean()

    # Prioritize by |TD-err| of a Monte-Carlo sampling of the posterior
    if hasattr(dqn_loss, "mc_sample_losses"):
        # take |td-error| of each mc sample loss and average them
        with torch.no_grad():
            td_errors = torch.stack(
                [
                    (mcsl.qsa_targets - mcsl.qsa).detach().abs()
                    for mcsl in dqn_loss.mc_sample_losses
                ],
                0,
            ).mean(0)
            print(td_errors)
    # Prioritize by |TD-err|
    elif isinstance(dqn_loss, CategoricalLoss):
        td_errors = losses.detach()
    elif isinstance(dqn_loss, DQNLoss):
        with torch.no_grad():
            td_errors = (dqn_loss.qsa_targets - dqn_loss.qsa).detach().abs()
    else:
        raise (
            "Weird corner case: the loss is neither a Categorical, a DQN,"
            + " or something else..."
        )

    # print(f"tde: {td_errors.shape}\n", td_errors.squeeze())
    mem.update(idxs, [td.item() for td in td_errors])
    return (losses * weights.to(losses.device).view_as(losses)).mean()


class DQNPolicy:
    """ A wrapper over the three main components of the DQN algorithm. This
    basically makes a DQN Agent.
    """

    def __init__(  # pylint: disable=bad-continuation
        self,
        policy_evaluation,
        policy_improvement,
        experience_replay,
        priority="uni",
    ):
        """ This class simplifies the interaction with the three components
        of the DQN agent.

        Args:
            policy_evaluation (EpsilonGreedyPolicy): A behaviour.
            policy_improvement (DQNPolicyImprovement): An improvement routine.
            experience_replay (ExperienceReplay): A memory buffer.
        """
        self.policy_evaluation = policy_evaluation
        self.policy_improvement = policy_improvement
        self.experience_replay = experience_replay
        self.__priority = priority
        self.__step_cnt = 0

    def act(self, state):
        """ Take an action, increment the no of steps taken in the environmnt.
        """
        self.__step_cnt += 1
        with torch.no_grad():
            return self.policy_evaluation(state)

    def learn(self):
        """ Learn from a batch of experiences sampled from experience replay.
        """
        batch = self.experience_replay.sample()

        if self.__priority == "tde":
            # this is a prioritized sampler using |TD-error|
            batch, idxs, weights = batch
            clbk = partial(
                priority_update, self.experience_replay, idxs, weights
            )
        elif self.__priority == "var":
            # this is a prioritized sampler using sigma**2(s, a)
            batch, idxs, weights = batch
            if len(batch) == 2:
                # the experience replay works with bootstrapped data
                # and batch = [[transitions...], boot_mask]
                batch_, _ = batch
            with torch.no_grad():
                variances = self.policy_evaluation.var(batch_[0], batch_[1])
            clbk = partial(
                priority_update,
                self.experience_replay,
                idxs,
                weights,
                other_prio=variances,
            )
        elif self.__priority == "bal":
            # this is a prioritized sampler using BALD: H - E[H]
            batch, idxs, weights = batch
            if len(batch) == 2:
                # the experience replay works with bootstrapped data
                # and batch = [[transitions...], boot_mask]
                batch_, _ = batch
            with torch.no_grad():
                entropy_diff = self.policy_evaluation.entropy_decrease(
                    batch_[0], batch_[1]
                )
            clbk = partial(
                priority_update,
                self.experience_replay,
                idxs,
                weights,
                other_prio=entropy_diff,
            )
        else:
            # this is uniform sampling
            clbk = None

        self.policy_improvement(batch, cb=clbk)

    def push(self, transition):
        """ Push `s,a,r,s_,d` transition in experience replay.
        """
        self.experience_replay.push(transition)

    @property
    def steps(self):
        """ Return steps taken in the environment.
        """
        return self.__step_cnt

    @property
    def estimator(self):
        """ Returns reference to the underlying estimator.
        """
        return self.policy_improvement.estimator

    def __str__(self):
        return "\nDQNPolicy(\n  | {0}\n  | {1}\n  | {2}\n  | {3}\n)".format(
            self.policy_evaluation,
            self.policy_improvement,
            self.experience_replay,
            f"priority={self.__priority}",
        )


class Episode:
    """ An iterator accepting an environment and a policy, that returns
    experience tuples.
    """

    def __init__(self, env, policy, with_pi=False):
        self.env = env
        self.policy = policy
        self.__with_pi = with_pi
        self.__state, self.__done = self.env.reset(), False
        self.__R = 0
        self.__step_cnt = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.__done:
            raise StopIteration

        with torch.no_grad():
            pi = self.policy.act(self.__state)
        _state, _action = self.__state.clone(), pi.action
        self.__state, reward, self.__done, _ = self.env.step(pi.action)

        self.__R += reward
        self.__step_cnt += 1
        if self.__with_pi:
            return (_state, _action, reward, self.__state, self.__done), pi
        return _state, _action, reward, self.__state, self.__done

    @property
    def total_reward(self):
        """ Return the expected return.
        """
        return self.__R

    @property
    def steps(self):
        """ Return steps taken in the environment.
        """
        return self.__step_cnt
