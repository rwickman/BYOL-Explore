import torch
import numpy as np
from dataclasses import dataclass
from byol_explore.rl.per.prioritized_replay import PriortizedReplay
import json, os
from byol_explore.utils.util import get_device, get_n_step_trajectory, get_k_future_states
import random

@dataclass
class Trajectory:
    states: list
    actions: list
    rewards: list
    next_states: list
    dones: list
    total_reward: float = 0
    ep_return: float = 0
    total_intrinsic_reward: float = 0

@dataclass
class Experience:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: int

class NGUReplayBuffer:
    def __init__(self, state_dim, max_size):
        self.state = np.zeros((max_size, state_dim), dtype=np.float32)
        self.action = np.zeros(max_size, dtype=np.int64)
        self.next_state = np.zeros((max_size, state_dim), dtype=np.float32)
        self.ptr = 0
        self.size = 0
        self.max_size = max_size
        self.device = get_device()

    def add(self, state, action, next_state):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.LongTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device))


class ReplayBuffer:
    def __init__(self, args, state_dim, max_size):
        self.args = args
        self.states = np.zeros((max_size, state_dim), dtype=np.float32)
        self.actions = np.zeros(max_size, dtype=np.int64)
        self.next_states = np.zeros((max_size, state_dim), dtype=np.float32)
        self.rewards = np.zeros(max_size, dtype=np.float32)
        self.dones = np.zeros(max_size, dtype=np.int8)
    
        self.max_intrinsic_reward = -1e9
        self.ptr = 0
        self.size = 0
        self.max_size = max_size
        self.device = get_device()

    def add(self, state, action, reward, next_state, done):
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.next_states[self.ptr] = next_state
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)


    def sample(self, batch_size, n_step=1):
        ind = np.random.randint(0, self.size, size=batch_size)

        states, actions, next_states = self.states[ind], self.actions[ind], self.next_states[ind]
        byol_states, byol_actions = get_k_future_states(ind, self.states, self.actions, self.args.byol_steps + 1, self.size)
        if n_step == 1:
            return (
                states,
                actions,
                self.rewards[ind],
                next_states,
                next_states,
                self.dones[ind],
                self.dones[ind],
                byol_states,
                byol_actions,
            )
        else:
            org_next_states, org_dones = next_states, self.dones[ind]
            rewards,next_states, dones = self._get_n_step_trajectory(ind, n_step)
            return (
                self.states[ind],
                self.actions[ind],
                rewards,
                next_states,
                org_next_states,
                dones,
                org_dones,
                byol_states,
                byol_actions,
            )

    def save(self, out_file):
        buffer_dict = {
            "states": self.states,
            "actions": self.actions,
            "rewards": self.rewards,
            "next_states": self.next_states,
            "dones": self.dones,
            "ptr": self.ptr,
            "size": self.size,
            "max_intrinsic_reward": self.max_intrinsic_reward
        }

        np.savez(out_file, **buffer_dict)

    def load(self, out_file):
        buffer_dict = np.load(out_file)

        self.states = buffer_dict["states"]
        self.actions = buffer_dict["actions"]
        self.rewards = buffer_dict["rewards"]
        self.next_states = buffer_dict["next_states"]
        self.dones = buffer_dict["dones"]
        self.ptr = buffer_dict["ptr"]
        self.size = buffer_dict["size"]
        self.max_intrinsic_reward = buffer_dict["max_intrinsic_reward"]

    def _get_n_step_trajectory(self, ind, n_step):
        
        rewards, next_states, dones = get_n_step_trajectory(
            ind, n_step, self.next_states, self.rewards, self.dones, self.size, self.args.gamma)
        return rewards, next_states, dones

    # def _get_n_step_trajectory(self, ind, , n_step):
    #     rewards, intrinsic_rewards, next_states, dones = [], [], [], []
        
    #     # Get all the indxs need for the n-step return
    #     all_inds = [
    #         list(range(idx, min(idx + n_step - 1, self.size - 1) + 1))
    #         for idx in ind]
    #     # Flatten them
    #     all_inds = np.array([j for s in all_inds for j in s])

    #     # Get the intrinsic rewards for all current and all n-step indices
    #     one_step_intrinsic_rewards = byol_hindsight.get_intrinsic_reward(
    #         torch.tensor(self.states[all_inds]).to(self.device),
    #         torch.tensor(self.actions[all_inds]).to(self.device),
    #         torch.tensor(self.next_states[all_inds]).to(self.device)).detach().cpu().numpy()

    #     cur_idx = 0
    #     for idx in ind:
            
    #         final_idx = min(idx + n_step - 1, self.size - 1)

    #         reward = self.rewards[final_idx]
    #         intrinsic_reward = one_step_intrinsic_rewards[final_idx - idx + cur_idx]


    #         next_state, done = self.next_states[final_idx], self.dones[final_idx]
            
    #         for i in reversed(range(idx, final_idx)):
    #             reward = self.rewards[i] + reward * self.args.gamma * (1-self.dones[i])
    #             intrinsic_reward = one_step_intrinsic_rewards[i - idx + cur_idx] + intrinsic_reward * self.args.gamma * (1-self.dones[i])
    #             if self.dones[i]:
    #                 next_state, done = self.next_states[i], self.dones[i]
  
    #         cur_idx += final_idx - idx + 1

    #         rewards.append(reward)
    #         intrinsic_rewards.append(intrinsic_reward)
    #         next_states.append(next_state)
    #         dones.append(done)

    #     rewards = np.array(rewards, dtype=np.float32)
    #     intrinsic_rewards = np.array(intrinsic_rewards, dtype=np.float32)
    #     next_states = np.stack(next_states)
    #     dones = np.array(dones, dtype=np.int8)

    #     return rewards, intrinsic_rewards, next_states, dones


class ExpertReplayBuffer(ReplayBuffer):
    """Store experiences from a best performing episode."""
    def __init__(self, args, ep_return, total_reward):
        self.args = args
        self.ep_return = ep_return
        self.total_reward = total_reward
        self.is_expert = False
        self.size = 0
        self.has_saved = False
        self.device = get_device()

    def set_trajectory(self, traj: Trajectory):
        self.states = traj.states
        self.actions = traj.actions
        self.rewards = traj.rewards
        self.next_states = traj.next_states
        self.dones = traj.dones
        self.size = len(self.next_states)
        self.max_intrinsic_reward = traj.total_intrinsic_reward
        self.is_expert = True

    def compare(self, ep_return, total_reward):
        """Compare the expert replay buffer to another episode."""
        if total_reward > self.total_reward:
            return True
        elif total_reward == self.total_reward:
            if ep_return == self.ep_return:
                return random.random() <= 0.001
            else:
                return ep_return > self.ep_return
        else:
            return False

    def save(self, out_file):
        """Save the expert replay buffer."""
        if self.is_expert:
            buffer_dict = {
                "states": self.states,
                "actions": self.actions,
                "rewards": self.rewards,
                "next_states": self.next_states,
                "dones": self.dones,
                "size": self.size,
                "ep_return": self.ep_return,
                "total_reward": self.total_reward,
                "max_intrinsic_reward": self.max_intrinsic_reward,
                "is_expert": self.is_expert
            }

            np.savez(out_file, **buffer_dict)
            self.has_saved = True
    
    def load(self, out_file):
        """Load the expert replay buffer."""
        buffer_dict = np.load(out_file)

        self.states = buffer_dict["states"]
        self.actions = buffer_dict["actions"]
        self.rewards = buffer_dict["rewards"]
        self.next_states = buffer_dict["next_states"]
        self.dones = buffer_dict["dones"]
        self.size = buffer_dict["size"]
        self.ep_return = buffer_dict["ep_return"]
        self.total_reward = buffer_dict["total_reward"]
        self.max_intrinsic_reward = buffer_dict["max_intrinsic_reward"]
        self.is_expert = buffer_dict["is_expert"]
        self.has_saved = True


class ExpertReplayBufferManager:
    def __init__(self, args, state_dim, max_size):
        self.args = args

        # Initialize the normal replay buffer
        self.buffer = PriortizedReplay(
            max_size, state_dim, alpha=self.args.per_alpha, beta=self.args.per_beta, byol_steps=self.args.byol_steps)
        
        # Initialize the expert replay buffers
        self.expert_buffers = [
            ExpertReplayBuffer(args, -1e9, -1e9) for _ in range(self.args.num_experts)
        ]

        self.device = get_device()
        self._reward_stats = {
            "min_total_reward": 1e9,
            "max_total_reward": -1e9,
            "min_total_int_reward": 1e9,
            "max_total_int_reward": -1e9,
        }

    @property
    def expert_size(self):
        return sum(b.size for b in self.expert_buffers)

    @property
    def size(self):
        return self.buffer.size + sum(b.size for b in self.expert_buffers)

    @property
    def reward_bounds(self):
        return self._reward_stats["min_total_reward"], self._reward_stats["max_total_reward"]

    @property
    def intrinsic_reward_bounds(self):
        return self._reward_stats["min_total_int_reward"], self._reward_stats["max_total_int_reward"]

    def _update_reward_stats(self, traj: Trajectory):
        """Update the reward statistics."""
        self._reward_stats["min_total_reward"] = min(self._reward_stats["min_total_reward"], traj.total_reward)
        self._reward_stats["max_total_reward"] = max(self._reward_stats["max_total_reward"], traj.total_reward)
        self._reward_stats["min_total_int_reward"] = min(self._reward_stats["min_total_int_reward"], traj.total_intrinsic_reward)
        self._reward_stats["max_total_int_reward"] = max(self._reward_stats["max_total_int_reward"], traj.total_intrinsic_reward)

    def _add_trajectory(self, traj: Trajectory, errors: list):
        """Add the experiences from the trajectory to the normal replay buffer."""
        for i in range(len(traj.states)):
            self.buffer.add(errors[i],
                [traj.states[i],
                traj.actions[i],
                traj.rewards[i],
                traj.next_states[i],
                traj.dones[i]])

    def add(self, traj: Trajectory, errors: np.ndarray):
        """Add a trajectory."""
        self._update_reward_stats(traj)
        # Check if it outperforms any of the experts
        sorted_idxs = [i[0] for i in sorted(enumerate(self.expert_buffers), key=lambda x: x[1].total_reward)]
        self.buffer.max_intrinsic_reward = max(
            self.buffer.max_intrinsic_reward, traj.total_intrinsic_reward)

        for i in sorted_idxs:
            if self.expert_buffers[i].compare(traj.ep_return, traj.total_reward):
                self.expert_buffers[i] = ExpertReplayBuffer(self.args, traj.ep_return, traj.total_reward)
                self.expert_buffers[i].set_trajectory(traj)

                # Still add to the original replay buffer
                self._add_trajectory(traj, errors)
                return

        # If at this point, then it wasn't better than any of the experts.
        # So, add the experiences to the normal replay buffer
        self._add_trajectory(traj, errors)

    def sample(self, batch_size: int, n_step=1):
        """Sample experiences."""
        # states, actions, rewards, next_states, org_next_states, dones, org_dones byol_states, byol_actions
        batch = [[], [], [], [], [], [], [], [], []]

        batch_is_weight = torch.ones(
            batch_size, dtype=torch.float32, device=self.device)

        # Sample half of the experiences from the experts.
        if self.args.num_experts > 1:
            num_expert_sample = 4
            size_per_expert = self.args.expert_batch_size // num_expert_sample

            # Get the probability of sampling each expert
            expert_rewards = np.array([e.total_reward + e.ep_return for e in self.expert_buffers])
            expert_rewards = expert_rewards - expert_rewards.min() + 1.0
            select_prob = expert_rewards / expert_rewards.sum()
            sample_probs = []
            expert_ind = np.random.choice(self.args.num_experts, p=select_prob, size=num_expert_sample, replace=False)

            # Sample the expert replay buffers
            for idx in range(num_expert_sample):
                i = expert_ind[idx]
                if self.expert_buffers[i].is_expert:
                    expert_batch = self.expert_buffers[i].sample(size_per_expert, n_step)
                    for j in range(len(expert_batch)):
                        batch[j].append(expert_batch[j])
                    sample_probs += [select_prob[i]] *size_per_expert


            # Compute the expert importance weights
            sample_probs = torch.tensor(sample_probs, dtype=torch.float32, device=self.device)
            expert_is_weights = (self.expert_size * sample_probs) ** -self.buffer.beta
            expert_is_weights = expert_is_weights / expert_is_weights.max()
        # Sample the normal replay buffer
        
        idxs, is_weight = [], []
        if self.buffer.size >= self.args.batch_size:
            normal_batch, idxs, is_weight = self.buffer.sample(self.args.batch_size, n_step)
            for i in range(len(normal_batch)):
                batch[i].append(normal_batch[i])

        # Join them and convert them into tensors
        for i in range(len(batch)):
            batch[i] = torch.tensor(np.concatenate(batch[i])).to(self.device)

        batch_is_weight = torch.ones(len(batch[i]), dtype=torch.float32, device=self.device)
        batch_is_weight[:len(expert_is_weights)] = expert_is_weights
        batch_is_weight[-len(is_weight):] = torch.tensor(is_weight, dtype=torch.float32).to(self.device)

        return batch, idxs, batch_is_weight

    def update_priority(self, errors, idxs):
        assert len(idxs) == len(errors)
        for i in range(len(idxs)):
            self.buffer.update(idxs[i], errors[i])

    def save(self, save_dir):
        buffer_out_dir = os.path.join(save_dir, "replay_buffer.pt")
        self.buffer.save(buffer_out_dir)

        for i in range(self.args.num_experts):
            if not self.expert_buffers[i].has_saved:
                expert_out_dir = os.path.join(save_dir, f"expert_replay_buffer_{i}.npz")
                self.expert_buffers[i].save(expert_out_dir)
        
        reward_stats_file = os.path.join(save_dir, "reward_stats.json")
        with open(reward_stats_file, "w") as f:
            json.dump(self._reward_stats, f)


    def load(self, save_dir):
        buffer_out_dir = os.path.join(save_dir, "replay_buffer.pt")
        self.buffer.load(buffer_out_dir)

        for i in range(self.args.num_experts):
            expert_out_dir = os.path.join(save_dir, f"expert_replay_buffer_{i}.npz")
            if os.path.exists(expert_out_dir):
                self.expert_buffers[i].load(expert_out_dir)

        reward_stats_file = os.path.join(save_dir, "reward_stats.json")
        with open(reward_stats_file, "r") as f:
            self._reward_stats = json.load(f)
        