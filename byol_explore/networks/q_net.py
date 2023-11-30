import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np
import copy
from byol_explore.networks.utils import create_mlp


def weighted_smooth_l1_loss(input, target, weights):
    # type: (Tensor, Tensor, Tensor) -> Tensor
    t = torch.abs(input - target)
    return (weights * torch.where(t < 1, 0.5 * t ** 2, t - 0.5)).mean()


class QNet(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, num_hidden=1, gamma=0.997, tau=0.05, continuous=False):
        super().__init__()
        self.gamma = gamma
        self.tgt_tau = tau
        self.continuous = continuous

        self.net = create_mlp(state_dim, num_hidden, hidden_dim, action_dim)
        
        self.tgt_net = create_mlp(state_dim, num_hidden, hidden_dim, action_dim)
        self.tgt_net.eval()

        self.tgt_net_2 = create_mlp(state_dim, num_hidden, hidden_dim, action_dim)
        self.tgt_net_2.eval()


        self.optimizer = Adam(self.net.parameters(), lr=2e-4)
        self.loss_fn = nn.SmoothL1Loss()

    def _init_net(self, state_dim, action_dim, hidden_dim, num_hidden):
        layers = [nn.Linear(state_dim, hidden_dim), nn.GELU()]
        for _ in range(num_hidden):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),    
            ])
        layers.append(nn.Linear(hidden_dim, action_dim))

        return nn.Sequential(*layers)

    def _update_target(self):
        """Perform soft update of the target policy."""
        for tgt_param, param in zip(self.tgt_net.parameters(), self.net.parameters()):
            tgt_param.data.copy_(
                self.tgt_tau * param.data + (1.0 - self.tgt_tau) * tgt_param.data)
        
        for tgt_param, param in zip(self.tgt_net_2.parameters(), self.net.parameters()):
            tgt_param.data.copy_(
                self.tgt_tau * 0.5 * param.data + (1.0 - self.tgt_tau * 0.5) * tgt_param.data)

    def forward(self, state):
        return self.net(state)
    
    def get_val_preds(self, states, actions, rewards, next_states, dones, n_step=1, min_total_reward=None, max_total_reward=None, ):
        # Get the Q-values for the actions
        q_vals_matrix = self.net(states)
        q_vals = q_vals_matrix.gather(1, actions.unsqueeze(1)).squeeze(1)
    
        # Run policy on next states 75966.9375, 29000
        with torch.no_grad():
            # Get the next actions
            q_next = self.net(next_states)
            next_actions = q_next.argmax(dim=1).detach()
            q_next_target = self.tgt_net(next_states)
            q_next_target_2 = self.tgt_net_2(next_states)

            
            # Compute the td-targets using Double Q-Learning and use TD3 inspired min
            dqn_target_1 = q_next_target.gather(1, next_actions.unsqueeze(1)).squeeze(1).detach()
            dqn_target_2 = q_next_target_2.gather(1, next_actions.unsqueeze(1)).squeeze(1).detach()
            dqn_target = torch.min(dqn_target_1, dqn_target_2)

            # # Small prevention for over-estimation 
            if max_total_reward != None:
                dqn_target = dqn_target.clamp(max=max_total_reward, min=min_total_reward)
            # print("dqn_target", dqn_target)
            if self.continuous:
                td_targets = rewards + self.gamma ** n_step * dqn_target
            else:
                td_targets = rewards + self.gamma ** n_step * dqn_target * (1 - dones)

        return q_vals, td_targets

    def get_td_errors(self, q_vals, td_targets):
        return torch.abs(q_vals-td_targets).detach().cpu().numpy()

    def train(self, states, actions, rewards, next_states, dones, n_step=1, is_weight=None, min_total_reward=None, max_total_reward=None, ):
        """Train the Q-network."""
        q_vals, td_targets = self.get_val_preds(
            states,
            actions,
            rewards,
            next_states,
            dones,
            n_step=n_step,
            min_total_reward=min_total_reward,
            max_total_reward=max_total_reward)

        self.optimizer.zero_grad()
        if is_weight != None:
            loss = weighted_smooth_l1_loss(q_vals, td_targets, is_weight)
        else:
            loss = self.loss_fn(q_vals, td_targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 5.0)
        self.optimizer.step()

        # Update the target Q-Network
        self._update_target()

        td_errors = self.get_td_errors(q_vals, td_targets)

        return loss.item(), td_errors
