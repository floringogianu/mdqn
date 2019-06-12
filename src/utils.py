""" Various utilities.
"""
import random
from argparse import Namespace

import gym
import torch
from gym_minigrid.wrappers import ImgObsWrapper
from termcolor import colored as clr

import rlog
from wintermute.env_wrappers import FrameStack


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


class SeedWrapper(gym.Wrapper):
    def __init__(self, env, seeds):
        """ Resets the env with a random seed.
        
        Args:
            seeds ([int, list<int>, None]): Used to set the seed of the env
            at each reset.
        """
        super().__init__(env)
        self.__seeds = seeds

    def step(self, action):
        return self.env.step(action)

    def reset(self, **kwargs):
        if isinstance(self.__seeds, list):
            self.seed(random.choice(self.__seeds))
        elif isinstance(self.__seeds, int):
            self.seed(self.__seeds)
        return self.env.reset(**kwargs)


def wrap_env(env, opt):
    env = ImgObsWrapper(env)
    env = FrameStack(env, k=opt.er.hist_len)
    env = TorchWrapper(env, device=opt.device)
    env = SeedWrapper(env, opt.seed) if opt.seed is not None else env
    return env


def configure_logger(opt):
    rlog.init(opt.experiment, path=opt.out_dir)
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


def config_to_string(
    cfg: Namespace, indent: int = 0, color: bool = True
) -> str:
    """Creates a multi-line string with the contents of @cfg."""

    text = ""
    for key, value in cfg.__dict__.items():
        ckey = clr(key, "yellow", attrs=["bold"]) if color else key
        text += " " * indent + ckey + ": "
        if isinstance(value, Namespace):
            text += "\n" + config_to_string(value, indent + 2, color=color)
        else:
            cvalue = clr(str(value), "white") if color else str(value)
            text += cvalue + "\n"
    return text
