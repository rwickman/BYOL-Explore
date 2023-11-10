import torch
import torch.nn as nn
from torch.optim import Adam

from byol_explore.networks.byol_hindsight_nets import Embedding, Generator, Critic, WorldModel
from byol_explore.rl.losses import recon_loss, con_loss

# TODO: Perform normalization

class BYOLHindSight(nn.Module):
    def __init__(self, obs_dim, action_dim, num_hidden, num_units, emb_dim, latent_dim, noise_dim=256, lam=1):
        super().__init__()
        # Lambda used for scaling the reconstruction loss
        self.lam = lam
        self.embedding = Embedding(obs_dim, emb_dim, num_hidden, num_units)
        self.generator = Generator(
            emb_dim, action_dim, latent_dim, num_hidden, num_units, noise_dim)
        self.critic = Critic(
            emb_dim, action_dim, latent_dim, num_hidden, num_units)
        self.world_model = WorldModel(
            emb_dim, action_dim, latent_dim, num_hidden, num_units)
        
        self.train_dict = {
            "contrastive_loss": [],
            "recon_loss": []
        }
        # params = [
        #     list(self.embedding.parameters()),
        #     list(self.generator.parameters()),
        #     list(self.critic.parameters()),
        #     list(self.world_model.parameters())
        # ]
        self.optimizer = Adam(self.parameters(), lr=1e-4)
    
    def get_intrinsic_reward(self, obs, action, obs_next):
        with torch.no_grad():
            # Embed the observations
            obs_emb = self.embedding(obs)
            obs_emb_next = self.embedding.get_tgt_emb(obs_next)

            # Sample the latent
            latent = self.generator(obs_emb, action, obs_emb_next)

            # Compute the critic reward
            critic_reward = con_loss(self.critic, obs_emb, action, latent)

            # Compute the next observation in hindsight
            emb_next_pred = self.world_model(obs_emb, action, latent)

            # Compute the reconstruction error reward
            recon_reward = recon_loss(emb_next_pred, obs_emb_next)
  
            intrinsic_reward = self.lam * recon_reward + critic_reward

            return intrinsic_reward
    
    def update(self, obs, action, obs_next):
        # Embed the observations
        with torch.no_grad():
            obs_emb_next = self.embedding.get_tgt_emb(obs_next)
        obs_emb = self.embedding.get_tgt_emb(obs)

        # Sample the latent
        latent = self.generator(obs_emb, action, obs_emb_next)
        


        # Compute the next observation in hindsight
        obs_emb_next_pred = self.world_model(obs_emb, action, latent)

        # Compute the contrastive loss
        con_loss_val = con_loss(self.critic, obs_emb, action, latent).mean()

        # Compute the recon loss
        recon_loss_val = recon_loss(obs_emb_next_pred, obs_emb).mean()

        loss = self.lam * recon_loss_val + con_loss_val
        #print("Contrastive loss: ", con_loss_val.item(), "RECON LOSS", recon_loss_val.item(), "LATENT - NEXT_OBS: ", recon_loss(obs_emb_next, latent).mean())
        

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.embedding.update_target()
        
        self.train_dict["contrastive_loss"].append(con_loss_val.item())
        self.train_dict["recon_loss"].append(recon_loss_val.item())
        return con_loss_val.item(), recon_loss_val.item()

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
