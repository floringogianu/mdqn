""" RL primitives.
"""
import torch


class DQNPolicy:
    """ A wrapper over the three main components of the DQN algorithm. This
    basically makes a DQN Agent.
    """

    def __init__(  # pylint: disable=bad-continuation
        self, policy_evaluation, policy_improvement, experience_replay
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
        self.policy_improvement(self.experience_replay.sample())

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


class Episode:
    """ An iterator accepting an environment and a policy, that returns
    experience tuples.
    """

    def __init__(self, env, policy):
        self.env = env
        self.policy = policy
        self.__state, self.__done = self.env.reset(), False
        self.__R = 0
        self.__step_cnt = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.__done:
            raise StopIteration

        pi = self.policy.act(self.__state)
        _state, _action = self.__state.clone(), pi.action
        self.__state, reward, self.__done, _ = self.env.step(pi.action)

        self.__R += reward
        self.__step_cnt += 1
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
