""" MiniGrid DQN
"""
import random
from copy import deepcopy

import gym
import gym_minigrid  # pylint: disable=unused-import
import torch
from gym_minigrid.wrappers import ImgObsWrapper
from torch import nn, optim

import rlog
from liftoff import parse_opts
from src.utils import config_to_string
from wintermute.env_wrappers import FrameStack
from wintermute.policy_evaluation import EpsilonGreedyPolicy
from wintermute.policy_improvement import DQNPolicyImprovement
from wintermute.replay import MemoryEfficientExperienceReplay


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
        # obs = torch.from_numpy(obs).permute(0, 3, 1, 2).unsqueeze(0)
        obs = torch.from_numpy(obs)
        obs = obs.permute(2, 1, 0)

        # [hist_len * channels, w, h] -> [1, hist_len, channels, w, h]
        # we are always using RGB
        obs = obs.view(int(obs.shape[0] / 3), 3, 7, 7).unsqueeze(0)

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


def wrap_env(env, opt):
    env = ImgObsWrapper(env)
    env = FrameStack(env, k=opt.hist_len)
    env = TorchWrapper(env, device=opt.device)
    return env


def augment_options(opt):
    if "experiment" not in opt.__dict__:
        opt.experiment = f"{''.join(opt.game.split('-')[1:-1])}-DQN"
    if opt.subset:
        opt.subset = [random.randint(0, 10000) for _ in range(opt.subset)]
    opt.device = torch.device(opt.device)
    return opt


def configure_logger(opt):
    rlog.init(opt.experiment, path=opt.out_dir)
    rlog.info(
        "Starting experiment using the following settings:\n%s",
        config_to_string(opt),
    )
    train_log = rlog.getLogger(opt.experiment + ".train")
    train_log.addMetrics(
        [
            rlog.AvgMetric("R/ep", metargs=["reward", "done"]),
            rlog.SumMetric("ep_cnt", resetable=False, metargs=["done"]),
            rlog.AvgMetric("steps/ep", metargs=["step_no", "done"]),
            rlog.FPSMetric("learning_fps", metargs=["frame_no"]),
        ]
    )
    test_log = rlog.getLogger(opt.experiment + ".test")
    test_log.addMetrics(
        [
            rlog.AvgMetric("R/ep", metargs=["reward", "done"]),
            rlog.SumMetric("ep_cnt", resetable=False, metargs=["done"]),
            rlog.AvgMetric("steps/ep", metargs=["frame_no", "done"]),
            rlog.FPSMetric("test_fps", metargs=["frame_no"]),
            rlog.MaxMetric("max_q", metargs=["qval"]),
        ]
    )


def run(opt):
    opt = augment_options(opt)
    configure_logger(opt)

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
