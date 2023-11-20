import random
import numpy as np
import torch

from byol_explore.rl.per.sum_tree import SumTree
from byol_explore.utils.util import get_device

class PriortizedReplay:

    def __init__(self, capacity, state_dim, alpha=0.6, beta=0.4, error_threshold=0.01):
        self.state_dim = state_dim
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

    def sample(self, batch_size, byol_hindsight):
        batch = []
        idxs = []
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
            (idx, p, data) = self.tree.get(s)
            priorities.append(p)
            
            state, action, reward, next_state, done = data
            states[i] = state
            actions[i] = action
            rewards[i] = reward
            next_states[i] = next_state
            dones[i] = done
            idxs.append(idx)

        sampling_probabilities = priorities / self.tree.total()
        is_weight = np.power(self.tree.size * sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()

        # Compute the intrinsic rewards
        intrinsic_rewards = byol_hindsight.get_intrinsic_reward(
            torch.tensor(states).to(self.device),
            torch.tensor(actions).to(self.device),
            torch.tensor(next_states).to(self.device)).detach().cpu().numpy()

        batch = (states, actions, rewards, intrinsic_rewards, next_states, dones)

        return batch, idxs, is_weight

    def update(self, idx, error):
        p = self._get_priority(error)
        self.tree.update(idx, p)
        
    
    def save(self, out_file):
        buffer_dict = {
            "tree": self.tree.tree,
            "data": self.tree.data,
            "size": self.tree.size,
            "ptr": self.tree.ptr
        }
        torch.save(buffer_dict, out_file)

    def load(self, out_file):
        buffer_dict = torch.load(out_file)
        self.tree.tree = buffer_dict["tree"]
        self.tree.data = buffer_dict["data"]
        self.tree.size = buffer_dict["size"]
        self.tree.ptr = buffer_dict["ptr"]

