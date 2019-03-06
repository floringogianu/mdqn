""" MiniGrid DQN
"""
import random
import time
from copy import deepcopy
from types import SimpleNamespace

import torch
from torch import nn
from torch import optim

from wintermute.policy_evaluation import EpsilonGreedyPolicy
from wintermute.policy_improvement import DQNPolicyImprovement
from wintermute.replay import MemoryEfficientExperienceReplay
from wintermute.env_wrappers import FrameStack

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


def test(opt, estimator):

    env = wrap_env(gym.make(opt.game), opt)
    policy_evaluation = EpsilonGreedyPolicy(
        estimator,
        env.action_space.n,
        epsilon={"name": "constant", "start": 0.01},
    )

    done, ep_cnt, mean_rw, mean_steps = True, 1, [], []

    start = time.time()
    for step_cnt in range(1, opt.test_steps + 1):

        if done:
            if opt.seed:
                env.seed(opt.seed)
            elif opt.subset:
                env.seed(random.choice(opt.subset))
            state, done = env.reset(), False
            ep_rw, ep_steps = 0, 0

        with torch.no_grad():
            action = policy_evaluation(state).action

        state_, reward, done, _ = env.step(action)
        state = state_.clone()
        # env.render()

        ep_rw += reward
        ep_steps += 1

        if done:
            ep_cnt += 1
            mean_rw.append(ep_rw)
            mean_steps.append(ep_steps)

    env.close()

    fps = sum(mean_steps) / (time.time() - start)
    mean_rw_ = torch.tensor(mean_rw).float().mean().item()
    mean_steps_ = torch.tensor(mean_steps).float().mean().item()
    print(
        f"eval on results: avg. rw={mean_rw_:2.2f} in"
        f" {ep_cnt:3d}eps / {step_cnt:6d}steps."
        f"  |  avg. ep_steps={mean_steps_:6.2f}"
        f"  |  fps={fps:3.2f}"
    )


def policy_iteration(
    env, policy_evaluation, policy_improvement, experience_replay, opt
):

    done, ep_cnt, mean_rw, mean_steps = True, 1, [], []

    start = time.time()
    for step_cnt in range(0, opt.train_steps + 1):

        if done:
            if opt.seed:
                env.seed(opt.seed)
            elif opt.subset:
                env.seed(random.choice(opt.subset))
            state, done = env.reset(), False
            ep_rw, ep_steps = 0, 0

        with torch.no_grad():
            action = policy_evaluation(state).action
        state_, reward, done, _ = env.step(action)

        experience_replay.push((state, action, reward, state_, done))

        if step_cnt > 512:
            batch = experience_replay.sample()
            policy_improvement(batch)

            if step_cnt % 1000 == 0:
                policy_improvement.update_target_estimator()

        state = state_
        # env.render()

        ep_rw += reward
        ep_steps += 1

        if done:
            ep_cnt += 1
            mean_rw.append(ep_rw)
            mean_steps.append(ep_steps)
            if ep_cnt % 25 == 0:
                fps = sum(mean_steps) / (time.time() - start)
                mean_rw_ = torch.tensor(mean_rw).float().mean().item()
                mean_steps_ = torch.tensor(mean_steps).float().mean().item()
                print(
                    f"[{ep_cnt:3d}, {step_cnt:6d}]"
                    f"  |  avg. rw={mean_rw_:2.2f}"
                    f"  |  avg. ep_steps={mean_steps_:6.2f}"
                    f"  |  fps={fps:3.2f}"
                )
                mean_rw.clear()
                mean_steps.clear()
                start = time.time()

        if step_cnt % 50000 == 0 and step_cnt != 0:
            test(opt, deepcopy(policy_evaluation.policy.estimator))


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
        opt.hist_len * 3, env.action_space.n, hidden_size=64
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
        0.95,
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
    opt = read_config()

    run(opt)


if __name__ == "__main__":
    main()
