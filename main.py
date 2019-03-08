""" MiniGrid DQN
"""
import random
from copy import deepcopy

import torch
from torch import nn
from torch import optim

from wintermute.policy_evaluation import EpsilonGreedyPolicy
from wintermute.policy_improvement import DQNPolicyImprovement
from wintermute.replay import MemoryEfficientExperienceReplay
from wintermute.env_wrappers import FrameStack
from rl_logger import Logger

from liftoff.config import read_config, config_to_string

import gym
from gym import spaces
import gym_minigrid
from gym_minigrid.wrappers import ImgObsWrapper


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


class TorchWrapper(gym.ObservationWrapper):
    """ From numpy to torch. """

    def __init__(self, env, device, verbose=False):
        super().__init__(env)
        self.device = device
        self.max_ratio = int(255 / 9)

        if verbose:
            print("[Torch Wrapper] for returning PyTorch Tensors.")

    def observation(self, obs):
        """ Convert from numpy to torch.
            Also change from (h, w, c*hist) to (batch, hist*c, h, w)
        """
        # (hist, w, h, ch) -> (hit, ch, w, h)
        # obs = torch.from_numpy(obs).permute(0, 3, 1, 2).unsqueeze(0)
        obs = torch.from_numpy(obs).permute(2, 1, 0).unsqueeze(0).unsqueeze(0)
        # scale the symbolic representention from [0,9] to [0, 255]
        obs = obs.mul_(self.max_ratio).byte()
        return obs.to(self.device)


def test(opt, estimator, crt_step):

    env = wrap_env(gym.make(opt.game), opt)
    policy_evaluation = EpsilonGreedyPolicy(
        estimator,
        env.action_space.n,
        epsilon={"name": "constant", "start": 0.01},
    )
    test_log = opt.log.groups["test"]
    test_log.reset()  # required for proper timing
    opt.log.log_info(
        test_log, f"Start evaluation at {crt_step} training steps."
    )

    done = True

    for step_cnt in range(1, opt.test_steps + 1):

        if done:
            if opt.seed:
                env.seed(opt.seed)
            elif opt.subset:
                env.seed(random.choice(opt.subset))
            state, done = env.reset(), False
            ep_rw, ep_steps = 0, 0

        with torch.no_grad():
            pi = policy_evaluation(state)

        state_, reward, done, _ = env.step(pi.action)
        state = state_.clone()
        # env.render()

        ep_rw += reward
        ep_steps += 1
        test_log.update(
            ep_cnt=(1 if done else 0),
            rw_per_ep=(reward, (1 if done else 0)),
            step_per_ep=(1, (1 if done else 0)),
            max_q=pi.q_value,
            test_fps=1,
        )

    env.close()

    # do some logging
    opt.log.log(test_log, crt_step)
    test_log.reset()


def policy_iteration(
    env, policy_evaluation, policy_improvement, experience_replay, opt
):

    train_log = opt.log.groups["train"]
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
            batch = experience_replay.sample()
            policy_improvement(batch)

            if step_cnt % opt.target_update == 0:
                policy_improvement.update_target_estimator()

        state = state_
        # env.render()

        train_log.update(
            ep_cnt=(1 if done else 0),
            rw_per_ep=(reward, (1 if done else 0)),
            step_per_ep=(1, (1 if done else 0)),
            max_q=pi.q_value,
            train_fps=1,
        )

        if done:
            ep_cnt += 1
            if ep_cnt % 25 == 0:
                opt.log.log(train_log, step_cnt)
                train_log.reset()

        if step_cnt % 50000 == 0 and step_cnt != 0:
            test(opt, deepcopy(policy_evaluation.policy.estimator), step_cnt)


def wrap_env(env, opt):
    env = ImgObsWrapper(env)
    env = TorchWrapper(env, device=opt.device)
    return env


def run(opt):

    if opt.subset:
        opt.subset = [random.randint(0, 10000) for _ in range(opt.subset)]
    opt.device = torch.device(opt.device)
    print(config_to_string(opt))

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

    log = Logger(label=opt.experiment, path=opt.out_dir)
    log.add_group(
        tag="train",
        metrics=(
            log.SumMetric("ep_cnt"),
            log.AvgMetric("step_per_ep"),
            log.EpisodicMetric("rw_per_ep", emph=True),
            log.MaxMetric("max_q"),
            log.FPSMetric("train_fps"),
        ),
        console_options=("white", "on_blue", ["bold"]),
    )
    log.add_group(
        tag="test",
        metrics=(
            log.SumMetric("ep_cnt"),
            log.EpisodicMetric("rw_per_ep", emph=True),
            log.AvgMetric("step_per_ep"),
            log.MaxMetric("max_q"),
            log.FPSMetric("test_fps"),
        ),
        console_options=("white", "on_magenta", ["bold"]),
    )
    opt.log = log

    policy_iteration(
        env, policy_evaluation, policy_improvement, experience_replay, opt
    )


def main():
    # read config files using liftoff
    opt = read_config()

    run(opt)


if __name__ == "__main__":
    main()
