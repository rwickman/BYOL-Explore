import torch
import torch.nn as nn
from torch.optim import Adam

from byol_explore.networks.byol_hindsight_nets import Embedding, Generator, Critic, WorldModel
from byol_explore.rl.losses import recon_loss, con_loss

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

        # Create the networks
        self.embedding = Embedding(obs_dim, emb_dim, num_hidden, num_units)
        self.generator = Generator(
            emb_dim, action_dim, latent_dim, num_hidden, num_units, noise_dim)
        self.critic = Critic(
            emb_dim, action_dim, latent_dim, num_hidden, num_units)
        self.world_model = WorldModel(
            emb_dim, action_dim, latent_dim, num_hidden, num_units)
        
        # Create the normalizers
        self.normalizer = MetricNormalizer(alpha)
        self.mean_l2norm = torch.ones(emb_dim)


        self.train_dict = {
            "contrastive_loss": [],
            "recon_loss": []
        }

        self.optimizer = Adam(self.parameters(), lr=1e-4)
    
    def _normalize(self, emb):
        """Normalize the embeddings."""
        # Update the mean L2 norm
        self.mean_l2norm = self.alpha * self.mean_l2norm + (1- self.alpha) * torch.norm(emb, dim=0)

        # Normalize the embedding
        norm_emb = emb / self.mean_l2norm
        self.mean_l2norm = self.mean_l2norm.detach()

        return norm_emb



    def get_intrinsic_reward(self, obs, action, obs_next):
        with torch.no_grad():
            # Embed the observations
            obs_emb = self.embedding(obs) / self.mean_l2norm
            obs_emb_next = self.embedding.get_tgt_emb(obs_next) / self.mean_l2norm

            # Sample the latent
            latent = self.generator(obs_emb, action, obs_emb_next)

            # Compute the critic reward
            critic_reward = con_loss(self.critic, obs_emb, action, latent)

            # Compute the next observation in hindsight
            emb_next_pred = self.world_model(obs_emb, action, latent) / self.mean_l2norm

            # Compute the reconstruction error reward
            recon_reward = recon_loss(emb_next_pred, obs_emb_next)
  
            intrinsic_reward = self.lam * recon_reward + critic_reward
            # if (self.normalizer.c + 1) % 256 == 0:
            #     print("BEFORE intrinsic_reward", intrinsic_reward[:5])

            # intrinsic_reward = self.normalizer.update(intrinsic_reward)
            # if self.normalizer.c % 256 == 0:
            #     print("AFTER intrinsic_reward", intrinsic_reward[:5])

            return intrinsic_reward

    
    def update(self, obs, action, obs_next):
        # Embed the observations
        with torch.no_grad():
            obs_emb_next = self.embedding.get_tgt_emb(obs_next)
            obs_emb_next = self._normalize(obs_emb_next).detach()
        obs_emb = self.embedding.get_tgt_emb(obs)
        obs_emb = self._normalize(obs_emb)


        # Sample the latent
        latent = self.generator(obs_emb, action, obs_emb_next)

        # Compute the next observation in hindsight
        obs_emb_next_pred = self.world_model(obs_emb, action, latent)
        # print("BEFORE NORM obs_emb_next_pred", obs_emb_next_pred.mean())
        obs_emb_next_pred = self._normalize(obs_emb_next_pred)
        # print("AFTER NORM obs_emb_next_pred", obs_emb_next_pred.mean())

        # Compute the contrastive loss
        con_loss_val = con_loss(self.critic, obs_emb, action, latent).mean()

        # Compute the recon loss
        recon_loss_val = recon_loss(obs_emb_next_pred, obs_emb_next).mean()

        loss = self.lam * recon_loss_val + con_loss_val

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.embedding.update_target()
        
        self.train_dict["contrastive_loss"].append(con_loss_val.item())
        self.train_dict["recon_loss"].append(recon_loss_val.item())
        return con_loss_val.item(), recon_loss_val.item()

    def to(self, device):
        self.mean_l2norm = self.mean_l2norm.to(device)
        return super().to(device)

# import torch
# obs_dim = 8
# action_dim = 8
# batch_size = 3

# hindsight = BYOLHindSight(obs_dim, action_dim, num_hidden=2, num_units=4, emb_dim=4, noise_dim=32).to("cuda")

# obs = torch.randn(batch_size, obs_dim).to("cuda")
# action = torch.randint(0, action_dim, size=(batch_size, )).to("cuda")
# obs_next = torch.randn(batch_size, obs_dim).to("cuda")
# reward = hindsight.get_intrinsic_reward(obs, action, obs_next)
# print(reward)
# hindsight.update(obs, action, obs_next)
