import torch
import torch.nn as nn
import os
from torch.optim import Adam
from torch.distributions import Categorical

from byol_explore.utils.util import get_device
from byol_explore.networks.utils import create_mlp

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, num_hidden, gamma, continuous, lr):
        super().__init__()
        self.gamma = gamma
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.num_hidden = num_hidden
        self.lr = lr
        self.device = get_device()
        self.reset()
        self.loss_fn = nn.MSELoss(reduction="none")
        self.actor_loss_fn = nn.CrossEntropyLoss(reduction="none")

    def reset(self):
        self.net = create_mlp(
            self.state_dim, self.num_hidden, self.hidden_dim, self.action_dim).to(self.device)
        self.optimizer = Adam(self.net.parameters(), lr=self.lr)


    # def train(self, states, actions, q_vals, is_weights, n_step=1):
    #     # Normalize the q-values
    #     print("\nBEFORE q_vals", q_vals[:2])
    #     q_vals = q_vals / q_vals.abs().max(1, keepdim=True)[0]
    #     print("AFTER q_vals", q_vals[:2])

    #     is_weights = is_weights.unsqueeze(1).repeat(1, self.action_dim)
    #     action_logits = self.net(states)
    #     dist = Categorical(logits=action_logits)
    #     #action_logits = action_logits - action_logits.logsumexp(dim=-1, keepdim=True)

    #     actor_losses = (q_vals * -dist.logits * is_weights).clamp(min=-2, max=2)
    #     print("ACTOR LOSSES:", actor_losses[:2])
    #     actor_loss = actor_losses.sum()
    #     uniform_tgt = torch.zeros(action_logits.shape, device=self.device)
    #     # uniform_tgt = torch.tile(
    #     #     torch.tensor(1/self.action_dim), action_logits.shape).to(self.device)
    #     entropy_loss = self.loss_fn(action_logits, uniform_tgt)        
    #     loss = actor_loss + 10 * entropy_loss
    #     print("ENTROPY: ", entropy_loss, "actor_loss", actor_loss, "\n")
    #     loss.backward()
    #     torch.nn.utils.clip_grad_norm_(self.net.parameters(), 2.0)
    #     self.optimizer.step()
    #     return loss.item()
 
    def train(self, states, q_vals, is_weights):
        # Get the targets
        tgts = q_vals.argmax(1)
        actor_pred = self.net(states)

        # Compute the entropy loss
        # uniform_tgt = -torch.ones(actor_pred.shape, device=self.device)
        # dist = Categorical(logits=actor_pred)
        # entropy_loss = self.loss_fn(actor_pred, uniform_tgt)
        # loss_vals = entropy_loss[torch.logical_or(dist.probs <= 0.01, dist.probs >= 0.99)]
        # if len(loss_vals) > 0:
        #     entropy_loss = loss_vals.mean()
        # else:
        entropy_loss = 0.0

        # q_vals = q_vals / q_vals.abs().max(1, keepdim=True)[0]
        # dist = Categorical(logits=actor_pred)
        # pg_loss = (q_vals * -dist.logits * is_weights).clamp(min=-2, max=2).sum()

        #actor_loss = actor_losses.sum()
        actor_loss = self.actor_loss_fn(actor_pred, tgts)
        actor_loss = actor_loss * is_weights
        actor_loss = actor_loss.mean()
        print(f"ACTOR LOSS {actor_loss} ENTROPY LOSS {entropy_loss}")
        loss = actor_loss + entropy_loss
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 5.0)
        self.optimizer.step()
        return loss.item()


    def save(self, save_dir):
        model_file = os.path.join(save_dir, "actor.pt")

        model_dict = {
            "actor": self.net.state_dict(),
            "optim": self.optimizer.state_dict()  
        }

        torch.save(model_dict, model_file)

    def load(self, save_dir):
        model_file = os.path.join(save_dir, "actor.pt")
        
        model_dict = torch.load(model_file)

        self.net.load_state_dict(model_dict["actor"])
        self.optimizer.load_state_dict(model_dict["optim"])

    def forward(self, obs):
        return self.net(obs)