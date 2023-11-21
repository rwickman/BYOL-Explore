import torch
import torch.nn as nn

import numpy as np

from torch.optim import Adam

class QNet(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, num_hidden=1, gamma=0.997, tau=0.05, continuous=False):
        super().__init__()
        self.gamma = gamma
        self.tgt_tau = tau
        self.continuous = continuous

        self.net = self._init_net(state_dim, action_dim, hidden_dim, num_hidden)
        self.tgt_net = self._init_net(state_dim, action_dim, hidden_dim, num_hidden)
        self.tgt_net.eval()

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

    def forward(self, state):
        return self.net(state)
    
    def get_val_preds(self, states, actions, rewards, next_states, dones, n_step=1, min_total_reward=None, max_total_reward=None, ):
        # Get the Q-values for the actions
        q_vals_matrix = self.net(states)
        q_vals = q_vals_matrix.gather(1, actions.unsqueeze(1)).squeeze(1)
    
        # Run policy on next states
        with torch.no_grad():
            # Get the next actions
            next_actions = self.net(next_states).argmax(dim=1).detach()
            q_next_target = self.tgt_net(next_states)

            # Compute the td-targets using Double Q-Learning
            dqn_target = q_next_target.gather(1, next_actions.unsqueeze(1)).squeeze(1).detach()
            
            # # Small prevention for over-estimation 
            # if max_total_reward != None:
            #     dqn_target = dqn_target.clamp(max=max_total_reward, min=min_total_reward)
            # print("dqn_target", dqn_target)
            if self.continuous:
                td_targets = rewards + self.gamma ** n_step * dqn_target
            else:
                td_targets = rewards + self.gamma ** n_step * dqn_target * (1 - dones)
        
        return q_vals, td_targets

    def get_td_errors(self, q_vals, td_targets):
        return torch.abs(q_vals-td_targets).detach().cpu().numpy()

    def train(self, states, actions, rewards, next_states, dones, n_step=1, min_total_reward=None, max_total_reward=None, ):
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
        
        # print("q_vals", q_vals)
        self.optimizer.zero_grad()
        loss = self.loss_fn(q_vals, td_targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 5.0)
        self.optimizer.step()

        # Update the target Q-Network
        self._update_target()

        td_errors = self.get_td_errors(q_vals, td_targets)


        return loss.item(), td_errors