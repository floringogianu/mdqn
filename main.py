""" MiniGrid DQN
"""
import random
from copy import deepcopy

import gym
import gym_minigrid  # pylint: disable=unused-import
import torch
from torch import optim

import rlog
from liftoff import parse_opts
from wintermute.policy_evaluation import EpsilonGreedyPolicy
from wintermute.policy_improvement import DQNPolicyImprovement
from wintermute.replay import MemoryEfficientExperienceReplay

from src.models import MiniGridNet
from src.utils import (
    augment_options,
    config_to_string,
    configure_logger,
    wrap_env,
)


def test(opt, estimator, crt_step):
    env = wrap_env(gym.make(opt.game), opt)
    policy_evaluation = EpsilonGreedyPolicy(
        estimator,
        env.action_space.n,
        epsilon={"name": "constant", "start": 0.01},
    )
    test_log = rlog.getLogger(f"{opt.experiment}.test")
    test_log.info("Test agent after %d training steps.", crt_step)
    test_log.reset()  # required for proper timing

    done = True

    for _ in range(1, opt.test_steps + 1):

        if done:
            if opt.seed:
                env.seed(opt.seed)
            elif opt.subset:
                env.seed(random.choice(opt.subset))
            state, done = env.reset(), False

        with torch.no_grad():
            pi = policy_evaluation(state)

        state_, reward, done, _ = env.step(pi.action)
        state = state_.clone()
        env.render()

        test_log.put(reward=reward, done=done, frame_no=1, qval=pi.q_value)

    env.close()

    # do some logging
    summary = test_log.summarize()
    test_log.info(
        (
            "[{0:8d}/{ep_cnt:8d}] R/ep={R/ep:6.2f}"
            + "\n             | steps/ep={steps/ep:6.2f}, "
            + "fps={test_fps:8.2f}, maxq={max_q:6.2f}."
        ).format(crt_step, **summary)
    )
    test_log.trace(step=crt_step, **summary)


def policy_iteration(
    env, policy_evaluation, policy_improvement, experience_replay, opt
):

    train_log = rlog.getLogger(f"{opt.experiment}.train")
    done, ep_cnt = True, 1

    for step_cnt in range(0, opt.train_steps + 1):

        if done:
            if opt.seed:
                env.seed(opt.seed)
            elif opt.subset:
                env.seed(random.choice(opt.subset))
            state, done = env.reset(), False

        with torch.no_grad():
            pi = policy_evaluation(state)
        state_, reward, done, _ = env.step(pi.action)

        experience_replay.push((state, pi.action, reward, state_, done))

        if step_cnt > 10000:
            if step_cnt % opt.update_freq == 0:
                batch = experience_replay.sample()
                policy_improvement(batch)

            if step_cnt % opt.target_update == 0:
                policy_improvement.update_target_estimator()

        state = state_
        # env.render()

        train_log.put(
            reward=reward, done=done, frame_no=opt.batch_size, step_no=1
        )

        if done:
            ep_cnt += 1
            if ep_cnt % 100 == 0:
                summary = train_log.summarize()
                train_log.info(
                    (
                        "[{0:8d}/{ep_cnt:8d}] R/ep={R/ep:6.2f}"
                        + "\n             | steps/ep={steps/ep:6.2f}, "
                        + "fps={learning_fps:8.2f}."
                    ).format(step_cnt, **summary)
                )
                train_log.trace(step=step_cnt, **summary)
                train_log.reset()

        if step_cnt % 50000 == 0 and step_cnt != 0:
            test(opt, deepcopy(policy_evaluation.policy.estimator), step_cnt)


def run(opt):
    opt = augment_options(opt)
    configure_logger(opt)
    rlog.info(config_to_string(opt))

    # start configuring some objects
    env = wrap_env(gym.make(opt.game), opt)

    estimator = MiniGridNet(
        opt.hist_len * 3, env.action_space.n, hidden_size=opt.lin_size
    ).cuda()
    policy_evaluation = EpsilonGreedyPolicy(
        estimator,
        env.action_space.n,
        epsilon={
            "name": "linear",
            "start": 1.0,
            "end": 0.1,
            "steps": opt.epsilon_steps,
        },
    )

    policy_improvement = DQNPolicyImprovement(
        estimator,
        optim.Adam(estimator.parameters(), lr=opt.lr, eps=1e-4),
        opt.gamma,
        # is_double=True,
    )

    experience_replay = MemoryEfficientExperienceReplay(
        capacity=opt.mem_size,
        batch_size=opt.batch_size,
        hist_len=opt.hist_len,
        async_memory=False,
    )

    policy_iteration(
        env, policy_evaluation, policy_improvement, experience_replay, opt
    )


def main():
    # read config files using liftoff
    opt = parse_opts()
    run(opt)


if __name__ == "__main__":
    main()
