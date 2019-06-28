""" MiniGrid DQN
"""
from copy import deepcopy

import torch
import gym
import gym_minigrid  # pylint: disable=unused-import
from torch import optim
from numpy import random

import rlog
from liftoff import parse_opts
import wintermute as wt

from src.models import (
    MiniGridNet,
    MiniGridDropnet,
    BootstrappedEstimator,
    MiniGridFF,
)
from src.policies import DropPE, DropPI, BootstrappedPI, BootstrappedPE
from src.utils import config_to_string, configure_logger, wrap_env
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
        policy = wt.EpsilonGreedyPolicy(
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


def augment_options(opt):
    """ Adds fields to `opt`.

        This function and `check_options_are_valid()` are important
        for the setup of the experiment.
    """
    # set the experiment name
    game = f"{''.join(opt.game.split('-')[1:-1])}"
    game = "".join(list(filter(lambda x: x.isupper() or x.isnumeric(), game)))
    algo = "C51" if hasattr(opt.estimator, "categorical") else "DQN"
    if "experiment" not in opt.__dict__:
        opt.experiment = f"{game}-{algo}"
    # sample a number of seeds so that we can limit the no of
    # maze configurations
    if isinstance(opt.seed, str):
        # `opt.seed` is of the form `r10`, `r5`, etc.
        opt.seed = [random.randint(0, 10000) for _ in range(int(opt.seed[1:]))]
    opt.device = torch.device(opt.device)
    # set the degradation schedule of the `beta` importance sampling
    # term
    if hasattr(opt.er, "beta") and opt.er.beta is not None:
        opt.er.optim_steps = (
            opt.train_steps - opt.start_learning
        ) / opt.update_freq
    # set the no of ensemble components in the ER options
    if hasattr(opt.estimator, "ensemble"):
        opt.er.bootstrap_args[0] = opt.estimator.ensemble.B
    return opt


def check_options_are_valid(opt):
    """ Checks if experiment configuration is consistent.
    """
    if hasattr(opt.er, "alpha") and opt.er.alpha is None:
        assert (
            opt.er.priority == "uni"
        ), "Priority can only be uniform if `opt.er.alpha` is None"
    elif hasattr(opt.er, "alpha"):
        assert opt.er.priority in (
            "tde",
            "var",
            "bal",
        ), "Priority cannot be uniform if `opt.er.alpha` has a value."
    if hasattr(opt.estimator, "ensemble"):
        assert (
            opt.er.bootstrap_args[0] == opt.estimator.ensemble.B
        ), "The no of ensemble components cannot differ."


def run(opt):
    torch.set_printoptions(precision=8, sci_mode=False)
    opt = augment_options(opt)
    configure_logger(opt)
    check_options_are_valid(opt)

    rlog.info(f"\n{config_to_string(opt)}")

    # configure the environment
    env = wrap_env(gym.make(opt.game), opt)

    # configure estimator and policy
    if hasattr(opt.estimator, 'categorical'):
        _s = opt.estimator.categorical.support
        support = [_s.min, _s.max, _s.bin_no]
        estimator = MiniGridFF(
            opt.er.hist_len * 3,
            env.action_space.n,
            hidden_size=opt.estimator.lin_size,
            support=support,
        ).cuda()
    elif opt.estimator.ff:
        estimator = MiniGridFF(
            opt.er.hist_len * 3,
            env.action_space.n,
            hidden_size=opt.estimator.lin_size,
        ).cuda()
    else:
        estimator = MiniGridNet(
            opt.er.hist_len * 3,
            env.action_space.n,
            hidden_size=opt.estimator.lin_size,
        ).cuda()

    if hasattr(opt.estimator, "ensemble"):
        # Build Bootstrapped Ensembles objects
        estimator = BootstrappedEstimator(
            estimator, **opt.estimator.ensemble.__dict__
        )
        policy_evaluation = BootstrappedPE(
            estimator, env.action_space.n, opt.exploration.__dict__, vote=True
        )
        if hasattr(opt.estimator, 'categorical'):
            policy_improvement = BootstrappedPI(
                wt.CategoricalPolicyImprovement(
                    estimator,
                    optim.Adam(estimator.parameters(), lr=opt.lr, eps=1e-4),
                    opt.gamma,
                ),
                categorical=True
            )
        else:
            policy_improvement = BootstrappedPI(
                wt.DQNPolicyImprovement(
                    estimator,
                    optim.Adam(estimator.parameters(), lr=opt.lr, eps=1e-4),
                    opt.gamma,
                    is_double=opt.double,
                )
            )
    elif hasattr(opt.estimator, "dropout"):
        # Build Variational Dropout objects
        estimator = MiniGridDropnet(
            opt.er.hist_len * 3,
            env.action_space.n,
            hidden_size=opt.estimator.lin_size,
            p=opt.estimator.dropout,
            mc_samples=opt.estimator.mc_samples,
        ).cuda()
        policy_evaluation = DropPE(
            estimator,
            env.action_space.n,
            epsilon=opt.exploration.__dict__,
            thompson=opt.estimator.thompson,
        )
        policy_improvement = DropPI(
            estimator,
            optim.Adam(estimator.parameters(), lr=opt.lr, eps=1e-4),
            opt.gamma,
            is_double=opt.double,
        )
    elif hasattr(opt.estimator, "categorical"):
        policy_evaluation = wt.EpsilonGreedyPolicy(
            estimator, env.action_space.n, epsilon=opt.exploration.__dict__
        )
        policy_improvement = wt.CategoricalPolicyImprovement(
            estimator,
            optim.Adam(estimator.parameters(), lr=opt.lr, eps=1e-4),
            opt.gamma,
        )
    else:
        policy_evaluation = wt.EpsilonGreedyPolicy(
            estimator, env.action_space.n, epsilon=opt.exploration.__dict__
        )
        policy_improvement = wt.DQNPolicyImprovement(
            estimator,
            optim.Adam(estimator.parameters(), lr=opt.lr, eps=1e-4),
            opt.gamma,
            is_double=opt.double,
        )

    policy = DQNPolicy(
        policy_evaluation,
        policy_improvement,
        wt.ExperienceReplay(**opt.er.__dict__)(),
        priority=opt.er.priority,
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
