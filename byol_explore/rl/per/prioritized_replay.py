import random
import numpy as np
import torch

from byol_explore.rl.per.sum_tree import SumTree
from byol_explore.utils.util import get_device, get_n_step_trajectory, get_k_future_states

class PriortizedReplay:

    def __init__(self, capacity, state_dim, alpha=0.6, beta=0.4, error_threshold=0.01, gamma=0.997, byol_steps=3):
        self.state_dim = state_dim
        self.gamma = gamma
        self.byol_steps = byol_steps
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


        ind, byol_states, byol_actions, _, _, _ = self._create_n_step_data(data_idxs, self.byol_steps + 1)
        byol_states = np.array(byol_states)
        byol_actions = np.array(byol_actions)

        batch = (states, actions, rewards, next_states, org_next_states, dones, org_dones, byol_states, byol_actions)

        return batch, idxs, is_weight

    def update(self, idx, error):
        p = self._get_priority(error)
        self.tree.update(idx, p)

    def _create_n_step_data(self, ind, n_step):
        # Extract the n_step trajectories from the tree
        rewards = []
        states = []
        next_states = []
        actions = []
        dones = []

        for idx in ind:
            states.append([])
            actions.append([])
            for i in range(n_step):
                states[-1].append(self.tree.data[(idx + i) % self.size][0])
                actions[-1].append(self.tree.data[(idx + i) % self.size][1])
                rewards.append(self.tree.data[(idx + i) % self.size][2])
                next_states.append(self.tree.data[(idx + i) % self.size][3])
                # print("NEXT STATE: ", self.tree.data[(idx + i) % self.size][3], "DONE", self.tree.data[(idx + i) % self.size][4])
                dones.append(self.tree.data[(idx + i) % self.size][4])

        # Create new indexes corresponding to the extracted data 
        ind = np.arange(0, n_step * len(ind), n_step)

        return ind, states, actions, rewards, next_states, dones

    def _get_n_step_trajectory(self, ind, n_step):
        ind, _, _, rewards, next_states, dones = self._create_n_step_data(ind, n_step)

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

