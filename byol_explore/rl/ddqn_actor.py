import torch
import random
import os
import numpy as np
import json
from scipy.stats import betabinom
import math

from byol_explore.networks.q_net import QNet
from byol_explore.rl.replay_buffer import ExpertReplayBufferManager
from byol_explore.rl.byol_hindsight import BYOLHindSight

class DDQNActor:
    def __init__(self, args, state_dim, action_dim):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_net = QNet(
            state_dim,
            action_dim,
            self.args.units,
            self.args.num_hidden,
            gamma=self.args.gamma,
            tau=self.args.tgt_tau,
            continuous=self.args.continuous).to(self.device)
        self.q_net_intrinsic = QNet(
            state_dim,
            action_dim,
            self.args.units,
            self.args.num_hidden,
            gamma=self.args.gamma_intrinsic,
            tau=self.args.tgt_tau,
            continuous=self.args.continuous).to(self.device)

        self.byol_hindsight = BYOLHindSight(
            state_dim,
            action_dim,
            latent_dim=state_dim,
            num_hidden=2,
            num_units=self.args.units,
            emb_dim=state_dim,
            noise_dim=state_dim).to(self.device)

        self.buffer = ExpertReplayBufferManager(
            self.args, state_dim, self.args.memory_cap)

        self.save_file = os.path.join(self.args.save_dir, "models.pt")

        self._train_dict = {
            "episodes" : 0,
            "total_rewards" : [],
            "total_int_rewards": [],
            "loss" : [],
            "intrinsic_loss": [],

        }

    def __call__(self, state, argmax=False):
        with torch.no_grad():
            q_intrinsic_val = self.q_net_intrinsic(state)
            q_val = self.q_net(state)
            out = q_val + self.args.ngu_beta * q_intrinsic_val
            if self.args.print_values:
                print("self.q_net_intrinsic(state)", q_intrinsic_val, q_intrinsic_val.argmax(1).item())
                print("self.q_net(state)", q_val, q_val.argmax(1).item())
                print("out", out, self.epsilon_threshold)
                
            

            action = self._sample_action(out, argmax)
            if self.args.print_values:
                print(f"ACTION BEFORE {action} AFTER INTRINSIC {self._sample_action(q_val, argmax)}", "\n")

            return action

    def _sample_action(self, q_vals: torch.Tensor, argmax=False) -> int:
        """Sample an action from the given Q-values."""
        if not argmax and self.epsilon_threshold >= random.random():
            # Sample a random action
            action = np.random.randint(q_vals.shape[1], size=q_vals.shape[0])
        else:
            with torch.no_grad():
                # Get action with the maximum Q-value
                action = q_vals.argmax(1).detach().cpu().numpy()

        return action

    @property
    def is_train_ready(self):

        return self.buffer.size >= self.args.min_train_exps

    @property
    def epsilon_threshold(self):
        """Return the current epsilon value used for epsilon-greedy exploration."""
        # Adjust for number of expert episodes that have elapsed
        eps = min( (1 - (self._train_dict["episodes"] / self.args.decay_episodes)) * self.args.epsilon, self.args.epsilon) 
        return max(eps, self.args.min_epsilon)

    def add_trajectory(self, traj):
        self._train_dict["episodes"] += 1
        self._train_dict["total_rewards"].append(traj.total_reward)
        self._train_dict["total_int_rewards"].append(traj.total_intrinsic_reward)

        # Compute the TD-errors
        errors = np.zeros(len(traj.states))
        num_batches = math.ceil(len(traj.states) / self.args.batch_size)
        for i in range(num_batches):
            start_idx = i*self.args.batch_size
            end_idx = (i+1)*self.args.batch_size

            q_vals, td_targets = self.q_net.get_val_preds(
                torch.tensor(traj.states[start_idx:end_idx]).to(self.device),
                torch.tensor(traj.actions[start_idx:end_idx]).to(self.device),
                torch.tensor(traj.rewards[start_idx:end_idx]).to(self.device),
                torch.tensor(traj.next_states[start_idx:end_idx]).to(self.device),
                torch.tensor(traj.dones[start_idx:end_idx]).to(self.device),
            )

            q_intrinsic_vals, td_targets_intrinsic = self.q_net_intrinsic.get_val_preds(
                torch.tensor(traj.states[start_idx:end_idx]).to(self.device),
                torch.tensor(traj.actions[start_idx:end_idx]).to(self.device),
                torch.tensor(traj.rewards[start_idx:end_idx]).to(self.device),
                torch.tensor(traj.next_states[start_idx:end_idx]).to(self.device),
                torch.tensor(traj.dones[start_idx:end_idx]).to(self.device),
            )

            td_errors = self.q_net_intrinsic.get_td_errors(q_vals, td_targets)
            td_errors_int = self.q_net_intrinsic.get_td_errors(q_intrinsic_vals, td_targets_intrinsic)
            td_errors = td_errors + self.args.per_intrinsic_priority * td_errors_int


            errors[start_idx:end_idx] = td_errors

        self.buffer.add(traj, errors)

    def save(self):
        self.buffer.save(self.args.save_dir)
        model_dict = {
            "q_net": self.q_net.state_dict(),
            "q_net_intrinsic": self.q_net_intrinsic.state_dict(),
            "hindsight": self.byol_hindsight.state_dict(),
            "l2_norm": self.byol_hindsight.mean_l2norm,
        }

        train_dict_file = os.path.join(self.args.save_dir, "train_dict.json") 
        byol_train_dict_file = os.path.join(self.args.save_dir, "byol_train_dict.json")

        with open(train_dict_file, "w") as f:
            json.dump(self._train_dict, f)
        
        with open(byol_train_dict_file, "w") as f:
            json.dump(self.byol_hindsight.train_dict, f)

        if self._train_dict["episodes"] % self.args.ckpt_iter == 0:
            torch.save(model_dict, os.path.join(self.args.save_dir, f"models_{self._train_dict['episodes']}.pt"))

        torch.save(model_dict, self.save_file)

    def load(self):
        if not self.args.evaluate:
            self.buffer.load(self.args.save_dir)

        model_dict = torch.load(self.save_file)
        self.q_net.load_state_dict(model_dict["q_net"])
        self.q_net_intrinsic.load_state_dict(model_dict["q_net_intrinsic"])
        self.byol_hindsight.load_state_dict(model_dict["hindsight"])
        self.byol_hindsight.mean_l2norm = model_dict["l2_norm"]

        train_dict_file = os.path.join(self.args.save_dir, "train_dict.json")
        byol_train_dict_file = os.path.join(self.args.save_dir, "byol_train_dict.json")
        with open(train_dict_file, "r") as f:
            self._train_dict = json.load(f)

        if not self.args.evaluate:
            with open(byol_train_dict_file, "r") as f:
                self.byol_hindsight.train_dict = json.load(f)

    def train(self):
        """Train the model over the sampled batches of experiences."""
        if self.args.n_steps > 1:
            n_step = betabinom.rvs(self.args.n_steps - 1, self.args.n_step_alpha, self.args.n_step_beta) + 1
        else:
            n_step = self.args.n_steps

        # Sample a batch of experiences
        batch, idxs, is_weight = self.buffer.sample(
            self.args.batch_size, n_step)
        states, actions, rewards, next_states, org_next_states, dones, org_dones = batch

        # Update the BYOL-Hindsight models
        if len(self._train_dict["loss"]) % self.args.byol_delay == 0:
            self.byol_hindsight.update(states, actions, org_next_states)

        # Get the intrinsic reward
        intrinsic_rewards = self.byol_hindsight.get_intrinsic_reward(
            states, actions, org_next_states)
        
        # Update the Q-networks
        loss, td_errors = self.q_net.train(
            states,
            actions,
            rewards,
            next_states,
            dones,
            n_step,
            min_total_reward=self.buffer.reward_bounds[0],
            max_total_reward=self.buffer.reward_bounds[1])

        intrinsic_loss, td_errors_int = self.q_net_intrinsic.train(
            states,
            actions,
            intrinsic_rewards.detach(),
            org_next_states,
            org_dones,
            min_total_reward=self.buffer.intrinsic_reward_bounds[0],
            max_total_reward=self.buffer.intrinsic_reward_bounds[1])
        
        # Update the the PER priorities
        if len(idxs) > 0:
            td_errors = td_errors + self.args.per_intrinsic_priority * td_errors_int
            start_idx = len(td_errors) - len(idxs)
            self.buffer.update_priority(td_errors[start_idx:], idxs)
        

        self._train_dict["loss"].append(loss)
        self._train_dict["intrinsic_loss"].append(intrinsic_loss)
