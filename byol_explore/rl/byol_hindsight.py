import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F


from byol_explore.networks.byol_hindsight_nets import Embedding, Generator, Critic, WorldModel, OpenLoop
from byol_explore.rl.losses import recon_loss, con_loss
from byol_explore.utils.util import get_device

class MetricNormalizer:
    def __init__(self, alpha=0.99):
        self.mean = 0
        self.square_mean = 1
        self.std = 1
        self.alpha = alpha
        self.c = 0
    
    def update(self, batch):
        sample_mean = batch.mean()
        sample_square = (batch ** 2).mean()
        
        self.mean = self.alpha * self.mean + (1-self.alpha) * sample_mean
        self.square_mean = self.alpha * self.square_mean + (1-self.alpha) * sample_square
        self.std = (max(self.mean - self.square_mean, 0) + 1e-6) ** 0.5
        self.c += 1
        if self.c % 256 == 0:
            print(f"MEAN {self.mean} SQUARE MEAN {self.square_mean} STD {self.std}")
            print(f"SAMPLE MEAN {sample_mean} SAMPLE STD {batch.std()}")

        return torch.clamp((batch - self.mean) / self.std, min=0)


class BYOLHindSight(nn.Module):
    def __init__(self, obs_dim, action_dim, num_hidden, num_units, emb_dim, latent_dim, noise_dim=256, lam=1, alpha=0.99):
        super().__init__()
        # Lambda used for scaling the reconstruction loss
        self.lam = lam

        # Alpha used for updating the mean l2 norm
        self.alpha = alpha
        self.emb_dim = emb_dim

        # Create the networks
        self.embedding = Embedding(obs_dim, emb_dim, num_hidden, num_units)
        self.generator = Generator(
            emb_dim, action_dim, latent_dim, num_hidden, num_units, noise_dim)
        self.critic = Critic(
            emb_dim, action_dim, latent_dim, num_hidden, num_units)
        self.world_model = WorldModel(
            emb_dim, emb_dim, latent_dim, num_hidden, num_units)
        self.open_loop = OpenLoop(emb_dim, action_dim, num_units)

        # Create the normalizers
        self.normalizer = MetricNormalizer(alpha)


        self.train_dict = {
            "contrastive_loss": [],
            "recon_loss": []
        }

        self.optimizer = Adam(self.parameters(), lr=1e-4)
        self.device = get_device()

    def get_intrinsic_reward(self, obs_steps, actions):
        with torch.no_grad():
            belief = self.embedding(obs_steps[:, 0])
    
            intrinsic_reward = torch.zeros(obs_steps.shape[0], device=self.device)
            for i in range(obs_steps.shape[1] - 1):
                # obs = obs_steps[:, i]
                obs_next = obs_steps[:, i+1]
                action = actions[:, i]

                # Compute the open-loop belief
                next_belief = self.open_loop(action, belief)

                # Embed the observations
                obs_emb_next = self.embedding.get_tgt_emb(obs_next)
                obs_emb_next = F.normalize(obs_emb_next).detach()

                # Sample the latent
                latent = self.generator(belief, action, obs_emb_next)

                # Compute the next observation in hindsight
                obs_emb_next_pred = self.world_model(next_belief, latent)
                obs_emb_next_pred = F.normalize(obs_emb_next_pred)

                # Compute the contrastive loss
                critic_reward = con_loss(self.critic, belief, action, latent)

                # Compute the recon loss
                recon_reward = recon_loss(obs_emb_next_pred, obs_emb_next)
                
                intrinsic_reward += self.lam * recon_reward + critic_reward
                belief = next_belief

        return intrinsic_reward / (obs_steps.shape[1] - 1)
    
    def update(self, obs_steps, actions, is_weights):
        self.normalizer.c += 1
        #belief = torch.zeros(obs_steps.shape[0], self.emb_dim).to(self.device)
        con_loss_val = torch.zeros(obs_steps.shape[0]).to(self.device)
        recon_loss_val = torch.zeros(obs_steps.shape[0]).to(self.device)
        belief = self.embedding(obs_steps[:, 0])

        for i in range(obs_steps.shape[1] - 1):
            obs_next = obs_steps[:, i+1]
            action = actions[:, i]

            # Compute the open-loop belief
            next_belief = self.open_loop(action, belief)

            # Embed the observations
            with torch.no_grad():
                obs_emb_next = self.embedding.get_tgt_emb(obs_next)
                obs_emb_next = F.normalize(obs_emb_next).detach()

            # Sample the latent
            latent = self.generator(belief, action, obs_emb_next)

            # Compute the next observation in hindsight
            obs_emb_next_pred = self.world_model(next_belief, latent)
            obs_emb_next_pred = F.normalize(obs_emb_next_pred)

            # Compute the contrastive loss
            con_loss_val += con_loss(self.critic, belief, action, latent) * is_weights

            # Compute the recon loss
            recon_loss_val += recon_loss(obs_emb_next_pred, obs_emb_next) * is_weights
            belief = next_belief

        recon_loss_val = recon_loss_val.mean()  / (obs_steps.shape[1] - 1)
        con_loss_val = con_loss_val.mean() / (obs_steps.shape[1] - 1)
        loss = self.lam * recon_loss_val + con_loss_val

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.embedding.update_target()
            
        self.train_dict["contrastive_loss"].append(con_loss_val.item())
        self.train_dict["recon_loss"].append(recon_loss_val.item())
        return con_loss_val.item(), recon_loss_val.item()
