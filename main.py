""" MiniGrid DQN
"""
from copy import deepcopy

import gym
import gym_minigrid  # pylint: disable=unused-import
from torch import optim

import rlog
from liftoff import parse_opts
from wintermute.policy_evaluation import EpsilonGreedyPolicy
from wintermute.policy_improvement import DQNPolicyImprovement
from wintermute.replay import ExperienceReplay

from src.models import MiniGridNet, MiniGridDropnet, BootstrappedEstimator
from src.policies import (
    DropPE,
    BootstrappedDQNPolicyImprovement,
    BootstrappedPE,
)
from src.utils import (
    augment_options,
    config_to_string,
    configure_logger,
    wrap_env,
)
from src.rl_routines import Episode, DQNPolicy


def test(opt, estimator, crt_step):
    """ Test the agent's performance.
    """
    test_log = rlog.getLogger(f"{opt.experiment}.test")
    test_log.info("Test agent after %d training steps.", crt_step)
    test_log.reset()  # required for proper timing

    # construct env and policy
    env = wrap_env(gym.make(opt.game), opt)
    if hasattr(opt.estimator, "ensemble"):
        policy = BootstrappedPE(
            estimator,
            env.action_space.n,
            epsilon={"name": "constant", "start": 0.01},
            vote=True,
        )
    else:
        policy = EpsilonGreedyPolicy(
            estimator,
            env.action_space.n,
            epsilon={"name": "constant", "start": 0.01},
        )

    step_cnt = 0
    while step_cnt < opt.test_steps:
        for transition, pi in Episode(env, policy, with_pi=True):
            _, _, reward, _, done = transition
            test_log.put(reward=reward, done=done, frame_no=1, qval=pi.q_value)
            step_cnt += 1
            if opt.test_render:
                env.render()
    env.close()

    # do some logging
    summary = test_log.summarize()
    test_log.info(
        (
            "[{0:8d}/{ep_cnt:8d}] R/ep={R/ep:6.2f}"
            + "\n             | steps/ep={steps/ep:6.2f}, "
            + "fps={test_fps:8.2f}, maxq={max_q:6.2f}."
        ).format(step_cnt, **summary)
    )
    test_log.trace(step=crt_step, **summary)


def policy_iteration(env, policy, opt):
    """ Policy improvement routine.

    Args:
        env (gym.env): The game we are training on.
        policy (DQNPolicy): A DQN agent.
        opt (Namespace): Configuration values.
    """

    train_log = rlog.getLogger(f"{opt.experiment}.train")

    while policy.steps < opt.train_steps:
        for _state, _action, reward, state, done in Episode(env, policy):

            # push to memory
            policy.push((_state, _action, reward, state, done))

            # learn
            if policy.steps >= opt.start_learning:
                if policy.steps % opt.update_freq == 0:
                    policy.learn()

                if policy.steps % opt.target_update == 0:
                    policy.policy_improvement.update_target_estimator()

            # log
            train_log.put(
                reward=reward, done=done, frame_no=opt.er.batch_size, step_no=1
            )

            if policy.steps % 10_000 == 0:
                summary = train_log.summarize()
                train_log.info(
                    (
                        "[{0:8d}/{ep_cnt:8d}] R/ep={R/ep:6.2f}"
                        + "\n             | steps/ep={steps/ep:6.2f}, "
                        + "fps={learning_fps:8.2f}."
                    ).format(policy.steps, **summary)
                )
                train_log.trace(step=policy.steps, **summary)
                train_log.reset()

            if policy.steps % 50_000 == 0 and policy.steps != 0:
                test(opt, deepcopy(policy.estimator), policy.steps)


def run(opt):
    opt = augment_options(opt)
    configure_logger(opt)
    rlog.info(f"\n{config_to_string(opt)}")

    # configure the Environment and the Policy
    env = wrap_env(gym.make(opt.game), opt)

    estimator = MiniGridNet(
        opt.er.hist_len * 3,
        env.action_space.n,
        hidden_size=opt.estimator.lin_size,
    ).cuda()

    if hasattr(opt.estimator, "ensemble"):
        # Build Bootstrapped Ensembles objects
        estimator = BootstrappedEstimator(estimator, B=opt.estimator.ensemble.B)
        policy_evaluation = BootstrappedPE(
            estimator, env.action_space.n, opt.exploration.__dict__, vote=True
        )
        policy_improvement = BootstrappedDQNPolicyImprovement(
            estimator,
            optim.Adam(estimator.parameters(), lr=opt.lr, eps=1e-4),
            opt.gamma,
            is_double=opt.double,
        )
    elif hasattr(opt.estimator, "dropout"):
        # Build Variational Dropout objects
        estimator = MiniGridDropnet(
            opt.er.hist_len * 3,
            env.action_space.n,
            hidden_size=opt.estimator.lin_size,
            p=opt.estimator.dropout
        ).cuda()
        policy_evaluation = DropPE(
            estimator, env.action_space.n, epsilon=opt.exploration.__dict__,
            thompson=opt.estimator.thompson
        )
        policy_improvement = DQNPolicyImprovement(
            estimator,
            optim.Adam(estimator.parameters(), lr=opt.lr, eps=1e-4),
            opt.gamma,
            is_double=opt.double,
        )
    else:
        policy_evaluation = EpsilonGreedyPolicy(
            estimator, env.action_space.n, epsilon=opt.exploration.__dict__
        )
        policy_improvement = DQNPolicyImprovement(
            estimator,
            optim.Adam(estimator.parameters(), lr=opt.lr, eps=1e-4),
            opt.gamma,
            is_double=opt.double,
        )

    policy = DQNPolicy(
        policy_evaluation,
        policy_improvement,
        ExperienceReplay(**opt.er.__dict__)(),
    )

    # additionally info
    rlog.info(policy)
    rlog.info(estimator)

    # start training
    policy_iteration(env, policy, opt)


def main():
    # read config files using liftoff
    opt = parse_opts()
    run(opt)


if __name__ == "__main__":
    main()
