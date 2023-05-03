import pickle
import random
import warnings

import numpy as np
import torch


class MemoryBuffer:

    def __init__(self, seed: int = 0):
        self.buffer = []
        self.length = 0
        random.seed(seed)

    def add(self, experience):
        self.length += 1
        self.buffer.append(experience)

    def load_expert_data(self, expert_path, num_trajs, seed):
        with open(expert_path, 'rb') as f:
            trajs = pickle.load(f)

        # Sample random `num_trajectories` experts.
        # We hve a separate seed for loading expert data.
        rng = np.random.RandomState(seed)
        perm = np.arange(len(trajs["states"]))
        perm = rng.permutation(perm)

        idx = perm[:num_trajs]
        for k, v in trajs.items():
            trajs[k] = [v[i] for i in idx]

        # We can also consider subsampling the trajectories
        # as in the original code.

        # We transform it for compatibility with the online memory buffer
        for traj_idx in range(num_trajs):
            for step_idx in range(trajs["lengths"][traj_idx]):
                self.add((
                    trajs["states"][traj_idx][step_idx],
                    trajs["next_states"][traj_idx][step_idx],
                    trajs["actions"][traj_idx][step_idx],
                    trajs["rewards"][traj_idx][step_idx],
                    trajs["dones"][traj_idx][step_idx]
                ))

    def get_batch(self, batch_size):
        if batch_size > len(self.buffer):
            warnings.warn(
                f"Requested batch size of {batch_size} is larger than the length of the input data "
                f"({len(self.buffer)}). The function will return the entire dataset.",
                Warning)
            batch_size = len(self.buffer)

        # Select a consecutive batch of data of size batch_size starting from the random start index
        indexes = np.random.choice(np.arange(len(self.buffer)), size=batch_size, replace=False)
        batch = [self.buffer[i] for i in indexes]
        # First convert to np.array for performance, as told by a warning.
        obs_batch = torch.tensor(np.array([t[0] for t in batch]), dtype=torch.float)
        next_obs_batch = torch.tensor(np.array([t[1] for t in batch]), dtype=torch.float)
        # For some environments it may be necessary to unsqueeze an action too.
        action_batch = torch.tensor(np.array([t[2] for t in batch]), dtype=torch.float)
        reward_batch = torch.tensor(np.array([t[3] for t in batch]), dtype=torch.float).unsqueeze(1)
        done_batch = torch.tensor(np.array([t[4] for t in batch]), dtype=torch.float).unsqueeze(1)

        return obs_batch, next_obs_batch, action_batch, reward_batch, done_batch
