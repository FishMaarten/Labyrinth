import argparse
import os
import time
from typing import Any, Callable, Dict, List, Set, Tuple

import gym
import matplotlib.pyplot as plt
import numpy as np
import ray
from gym import spaces
from ray.rllib import agents

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dir",
    type=str,
    default=os.path.dirname(os.path.realpath(__file__)),
    help="Location for logs and checkpoints.",
)
parser.add_argument(
    "--iter", type=int, default=20, help="Number of training iterations."
)
parser.add_argument(
    "--workers", type=int, default=0, help="Number of workers set to task."
)
parser.add_argument(
    "--lr", type=float, default=3e-5, help="Alpha parameter, learning rate."
)
parser.add_argument(
    "--checkpoint", type=int, default=None, help="Take checkpoint every n-steps"
)
parser.add_argument(
    "--restore", type=str, default=None, help="File containing path to checkpoint"
)
parser.add_argument(
    "--horizon", type=int, default=None, help="Hard step limit, terminates episode."
)
parser.add_argument(
    "--batch",
    type=int,
    nargs=4,
    default=[512, 4096, 128, 8],
    help="[Fragment, Batch, Resample, Iter]",
)
parser.add_argument(
    "--hidden", type=int, nargs="*", default=[256, 256], help="Hidden layers for NN"
)
parser.add_argument("--curiosity", dest="curiosity", action="store_true")
parser.set_defaults(curiosity=False)


class Labyrinth2D(gym.Env):
    DEFAULT: Dict[str, Any] = {
        "map_arr": plt.imread("./resources/labyrinth_raw.png"),
        "max_step": None,
        "step_cost": 0.1,
        "visit_scale": 0.1,
        "perma_death": True,
        "death_penalty": -1,
        "victory_score": 10,
    }

    def __init__(self, config: Dict[str, Any]):
        super(Labyrinth2D, self).__init__()
        self.config = self.DEFAULT.copy()
        for k, v in config.items():
            self.config[k] = v

        self.max_step: int = self.config["max_step"]
        self.step_cost: int = self.config["step_cost"]
        self.visit_scale: float = self.config["visit_scale"]
        self.perma_death: bool = self.config["perma_death"]
        self.death_penalty: int = self.config["death_penalty"]
        self.victory_score: int = self.config["victory_score"]
        self.raw_img: np.ndarray = self.config["map_arr"]

        self.height, self.width, _ = self.raw_img.shape
        self.num_states = self.height * self.width
        self.yx_states: Set[Tuple[int, int]] = set(
            [(y, x) for y in range(self.height) for x in range(self.width)]
        )

        self.img: np.ndarray = np.einsum("WHC->CWH", self.raw_img)

        self.action_dict: Dict[int, Tuple[int, int]] = {
            0: (0, 1),  # Right
            1: (1, 0),  # Down
            2: (0, -1),  # Left
            3: (-1, 0),  # Up
        }
        world_dict: Dict[str, List[Tuple[int, int]]] = {
            "holes": list(zip(*np.where(self.img[0] != 0))),
            "check": list(zip(*np.where(self.img[1] != 0))),
            "walls": list(zip(*np.where(self.img[2] != 0))),
        }

        self.start: Tuple[int, int] = world_dict["check"][0]
        self.finish: Tuple[int, int] = world_dict["check"][1]

        self.direction: Callable[[Any, Any], Tuple[Any, Any]] = lambda p, a: (
            p[0] + a[0],
            p[1] + a[1],
        )
        self.is_wall: Callable[[Any], bool] = lambda s: s in world_dict["walls"]
        self.is_hole: Callable[[Any], bool] = lambda s: s in world_dict["holes"]
        self.is_legal: Callable[[Any], bool] = lambda s: (
            s[0] >= 0 and s[0] < self.height
        ) and (s[1] >= 0 and s[1] < self.width)

        self.action_space = spaces.Discrete(len(self.action_dict))
        self.observation_space = spaces.Discrete(self.num_states)

        self.time: int = 0
        self.state: int = 0
        self.done: bool = False
        self.visits: Dict[Tuple[int, int], int] = {state: 0 for state in self.yx_states}

    def flatten_s(self, state: Tuple[int, int]) -> int:
        return state[0] * self.width + state[1]

    def expand_s(self, state: int) -> Tuple[int, int]:
        return state // self.width, state % self.width

    def reset(self) -> int:
        self.time = 0
        self.done = False
        self.state = self.flatten_s(self.start)
        self.visits = {state: 0 for state in self.yx_states}
        return self.state

    def step(self, action) -> Tuple[int, float, bool, Any]:
        self.time += 1

        state_yx: Tuple[int, int] = self.expand_s(self.state)
        move: Tuple[int, int] = self.direction(state_yx, self.action_dict[action])
        next_s: Tuple[int, int] = (
            move if self.is_legal(move) and not self.is_wall(move) else state_yx
        )
        self.state = self.flatten_s(next_s)
        self.visits[state_yx] += 1

        if self.max_step and self.time >= self.max_step:
            count = sum(map(lambda f: f != 0, self.visits.values()))
            reward = -1 - (1 - count / self.num_states)
            return self.state, -1, True, {}

        if self.is_hole(next_s):
            self.state = self.flatten_s(self.start)
            reward = self.death_penalty - self.time
            return self.flatten_s(next_s), self.death_penalty, self.perma_death, {}

        if next_s == self.finish:
            return self.state, self.victory_score, True, {}

        reward = (
            self.step_cost * self.visit_scale
            if not self.visits[next_s]
            else -self.step_cost
        )

        return self.state, reward, False, {}


if __name__ == "__main__":
    ray.shutdown()
    ray.init()

    args = parser.parse_args()

    trainer = agents.ppo.PPOTrainer(
        {
            "env": Labyrinth2D,
            "env_config": {
                "max_step": args.horizon,
                "step_cost": 0.1,
                "visit_scale": 0.1,
                "perma_death": True,
                "death_penalty": -1,
                "victory_score": 10,
            },
            "lr": args.lr,
            "framework": "torch",
            "num_workers": args.workers,
            "rollout_fragment_length": args.batch[0],
            "train_batch_size": args.batch[1],
            "sgd_minibatch_size": args.batch[2],
            "num_sgd_iter": args.batch[3],
            "vf_clip_param": 10,
            "exploration_config": {"type": "StochasticSampling"}
            if not args.curiosity
            else {
                "type": "Curiosity",
                "eta": 0.3,
                "lr": 0.001,
                "feature_dim": 288,
                "feature_net_config": {
                    "fcnet_hiddens": [],
                    "fcnet_activation": "relu",
                },
                "inverse_net_hiddens": [256],
                "inverse_net_activation": "relu",
                "forward_net_hiddens": [256],
                "forward_net_activation": "relu",
                "beta": 0.2,
                "sub_exploration": {
                    "type": "StochasticSampling",
                },
            },
        }
    )

    if args.restore:
        trainer.restore(open(os.path.join(args.dir, args.restore), "r").read())

    print("Session started...", time.ctime())

    for i in range(args.iter):
        results = trainer.train()
        print(
            results["training_iteration"],
            {
                "Min:": round(results["episode_reward_min"], 2),
                "Mean:": round(results["episode_reward_mean"], 2),
                "Max:": round(results["episode_reward_max"], 2),
                "Len:": round(results["episode_len_mean"], 2),
            },
        )
        if args.checkpoint and i and i % args.checkpoint == 0:
            checkpoint = trainer.save()
            print(f"Checkpoint saved at: {time.ctime()}\n", checkpoint)
            with open(os.path.join(args.dir, f"labyrinth_2D_{i}.cpt"), "w") as file:
                file.write(checkpoint)

    checkpoint = trainer.save()
    print(f"Checkpoint saved at: {time.ctime()}\n", checkpoint)
    with open("labyrinth_2D.cpt", "w") as file:
        file.write(checkpoint)

    ray.shutdown()
