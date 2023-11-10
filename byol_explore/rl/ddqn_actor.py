import torch
import random
import os
import numpy as np
import json
from scipy.stats import betabinom

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
            if self.args.print_values:
                print("self.q_net_intrinsic(state)", self.q_net_intrinsic(state))
                print("self.q_net(state)", self.q_net(state))
                
            out = self.q_net(state) + self.args.ngu_beta * self.q_net_intrinsic(state)
            action = self._sample_action(out, argmax)
            if self.args.print_values:
                print(f"ACTION BEFORE {action} AFTER INTRINSIC {self._sample_action(self.q_net(state), argmax)}", "\n")

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
        
        self.buffer.add(traj)

    def save(self):
        self.buffer.save(self.args.save_dir)
        model_dict = {
            "q_net": self.q_net.state_dict(),
            "q_net_intrinsic": self.q_net_intrinsic.state_dict(),
            "hindsight": self.byol_hindsight.state_dict()
        }

        train_dict_file = os.path.join(self.args.save_dir, "train_dict.json") 
        with open(train_dict_file, "w") as f:
            json.dump(self._train_dict, f)

        torch.save(model_dict, self.save_file)

    def load(self):
        # if not self.args.evaluate:
        self.buffer.load(self.args.save_dir)

        model_dict = torch.load(self.save_file)
        self.q_net.load_state_dict(model_dict["q_net"])
        self.q_net_intrinsic.load_state_dict(model_dict["q_net_intrinsic"])
        self.byol_hindsight.load_state_dict(model_dict["hindsight"])

        train_dict_file = os.path.join(self.args.save_dir, "train_dict.json") 
        with open(train_dict_file, "r") as f:
            self._train_dict = json.load(f)

    def train(self):
        """Train the model over the sampled batches of experiences."""
        if self.args.n_steps > 1:
            n_step = betabinom.rvs(self.args.n_steps - 1, self.args.n_step_alpha, self.args.n_step_beta) + 1
        else:
            n_step = self.args.n_steps

        # Sample a batch of experiences
        states, actions, rewards, intrinsic_rewards, next_states, dones = self.buffer.sample(
            self.args.batch_size, self.byol_hindsight, n_step)
        
        # Update the BYOL-Hindsight models 
        self.byol_hindsight.update(states, actions, next_states)
        #print("intrinsic_rewards", intrinsic_rewards)
        loss = self.q_net.train(states, actions, rewards, next_states, dones, n_step, max_total_reward=self.buffer.max_total_reward)
        intrinsic_loss = self.q_net_intrinsic.train(
            states,
            actions,
            intrinsic_rewards.detach(),
            next_states,
            dones,
            n_step,
            max_total_reward=self.buffer.max_intrinsic_reward)

        self._train_dict["loss"].append(loss)
        self._train_dict["intrinsic_loss"].append(intrinsic_loss)


    # def train(self):
    #     """Train the model over the sampled batches of experiences."""
    #     if self.args.n_steps > 1:
    #         n_step = betabinom.rvs(self.args.n_steps, self.args.n_step_alpha, self.args.n_step_beta) + 1
    #     else:
    #         n_step = self.args.n_steps

    #     # Sample a batch of experiences
    #     states, actions, rewards, next_states, org_next_states, dones = self.buffer.sample(
    #         self.args.batch_size, n_step)
        
    #     # Update the BYOL-Hindsight models
    #     self.byol_hindsight.update(states, actions, next_states)

    #     # Compute the intrinsic rewards
    #     intrinsic_rewards = self.byol_hindsight.get_intrinsic_reward(
    #         states, actions, org_next_states)

    #     loss = self.q_net.train(states, actions, rewards, next_states, dones, n_step)
    #     intrinsic_loss = self.q_net_intrinsic.train(states, actions, intrinsic_rewards, org_next_states, dones)

    #     self._train_dict["loss"].append(loss)
    #     self._train_dict["intrinsic_loss"].append(intrinsic_loss)
