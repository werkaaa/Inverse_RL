import pickle
import random
import warnings

import numpy as np


class MemoryBuffer:

    def __init__(self, seed: int = 0):
        self.buffer = []
        self.length = 0
        random.seed(seed)

    def add(self, experience) -> None:
        self.length += 1
        self.buffer.append(experience)

    def load_expert_data(self, expert_path, num_trajectories, seed):
        with open(expert_path, 'rb') as f:
            trajs = pickle.load(f)

        # Sample random `num_trajectories` experts.
        # We hve a separate seed for loading expert data.
        rng = np.random.RandomState(seed)
        perm = np.arange(len(trajs["states"]))
        perm = rng.permutation(perm)

        idx = perm[:num_trajectories]
        for k, v in trajs.items():
            trajs[k] = [v[i] for i in idx]

        # We can also consider subsampling the trajectories
        # as in the original code.

        # We transform it for compatibility with the online memory buffer
        self.length = trajs["lengths"].sum().item()
        for traj_idx in range(num_trajectories):
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

        obs_batch = [t[0] for t in batch]
        next_obs_batch = [t[1] for t in batch]
        action_batch = [t[2] for t in batch]
        reward_batch = [t[3] for t in batch]
        done_batch = [t[4] for t in batch]

        return obs_batch, next_obs_batch, action_batch, reward_batch, done_batch


if __name__ == '__main__':
    mb = MemoryBuffer()
    mb.load_expert_data("/home/weronika/Documents/masters/sem2/AIPMLR/Inverse_RL/experts/HalfCheetah-v2_25.pkl")
