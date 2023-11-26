import random
import numpy as np
import torch

from byol_explore.rl.per.sum_tree import SumTree
from byol_explore.utils.util import get_device, get_n_step_trajectory

class PriortizedReplay:

    def __init__(self, capacity, state_dim, alpha=0.6, beta=0.4, error_threshold=0.01, gamma=0.997):
        self.state_dim = state_dim
        self.gamma = gamma
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.e = error_threshold
        self.max_intrinsic_reward = -1e9
        self.device = get_device()

    @property
    def size(self):
        return self.tree.size

    def _get_priority(self, error):
        return (np.abs(error) + self.e) ** self.alpha

    def add(self, error, sample):
        p = self._get_priority(error)
        self.tree.add(p, sample)

    def sample(self, batch_size, n_step):
        batch = []
        idxs = []
        data_idxs = []
        segment = self.tree.total() / batch_size
        priorities = []

        # self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])
        states = np.zeros((batch_size, self.state_dim), dtype=np.float32)
        actions = np.zeros(batch_size, dtype=np.int64)
        next_states = np.zeros((batch_size, self.state_dim), dtype=np.float32)
        rewards = np.zeros(batch_size, dtype=np.float32)
        dones = np.zeros(batch_size, dtype=np.int8)

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            idx, p, data, data_idx = self.tree.get(s)
            priorities.append(p)
            
            state, action, reward, next_state, done = data
            states[i] = state
            actions[i] = action
            rewards[i] = reward
            next_states[i] = next_state
            dones[i] = done
            idxs.append(idx)
            data_idxs.append(data_idx)

        org_next_states = next_states
        org_dones = dones
        if n_step > 1:
            rewards, next_states, dones = self._get_n_step_trajectory(data_idxs, n_step)
        
        # Anneal beta towards 1.0
        # self.beta += 0.01
        # self.beta = min(self.beta, 1.0)

        sampling_probabilities = priorities / self.tree.total()
        is_weight = np.power(self.tree.size * sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()

        batch = (states, actions, rewards, next_states, org_next_states, dones, org_dones)

        return batch, idxs, is_weight

    def update(self, idx, error):
        p = self._get_priority(error)
        self.tree.update(idx, p)

    def _get_n_step_trajectory(self, ind, n_step):
        # Extract the n_step trajectories from the tree
        rewards = []
        next_states = []
        dones = []
        # print("\nn_step", n_step)
        for idx in ind:
            for i in range(n_step):
                rewards.append(self.tree.data[(idx + i) % self.size][2])
                next_states.append(self.tree.data[(idx + i) % self.size][3])
                # print("NEXT STATE: ", self.tree.data[(idx + i) % self.size][3], "DONE", self.tree.data[(idx + i) % self.size][4])
                dones.append(self.tree.data[(idx + i) % self.size][4])

        # Create new indexes corresponding to the extracted data 
        ind = np.arange(0, n_step * len(ind), n_step)
        # print("UPDATED:", ind)
        # print("next_states", next_states)
        # print("rewards", rewards)
        # print("dones", dones)

        rewards, next_states, dones = get_n_step_trajectory(
            ind, n_step, next_states, rewards, dones, self.size, self.gamma)
        return rewards, next_states, dones

    def save(self, out_file):
        buffer_dict = {
            "tree": self.tree.tree,
            "data": self.tree.data,
            "size": self.tree.size,
            "ptr": self.tree.ptr,
        }
        torch.save(buffer_dict, out_file)

    def load(self, out_file):
        buffer_dict = torch.load(out_file)
        self.tree.tree = buffer_dict["tree"]
        self.tree.data = buffer_dict["data"]
        self.tree.size = buffer_dict["size"]
        self.tree.ptr = buffer_dict["ptr"]

