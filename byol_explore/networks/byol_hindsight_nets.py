import torch
import torch.nn as nn
import torch.nn.functional as F

def create_mlp(inp_size, num_hidden, num_units, out_size):
    layers = [nn.Linear(inp_size, num_units), nn.ReLU()]
    for _ in range(num_hidden - 1):
        layers.append(nn.Linear(num_units, num_units))
        layers.append(nn.ReLU())
    
    layers.append(nn.Linear(num_units, out_size))
    
    return nn.Sequential(*layers) 


class Embedding(nn.Module):
    def __init__(self, obs_dim, emb_dim, num_hidden, num_units, tau=0.01):
        super().__init__()
        # Value used to update the target network
        self.tau = tau

        self.net = create_mlp(obs_dim, num_hidden, num_units, emb_dim)
        self.net_tgt = create_mlp(obs_dim, num_hidden, num_units, emb_dim)
    
    def update_target(self):
        """Perform soft update of the target network."""
        for tgt_param, param in zip(self.net_tgt.parameters(), self.net.parameters()):
            tgt_param.data.copy_(
                self.tau * param.data + (1.0 - self.tau) * tgt_param.data)

    def get_tgt_emb(self, obs):
        """Embed the observation using the target."""
        with torch.no_grad():
            return self.net_tgt(obs)

    def forward(self, obs):
        return self.net(obs)


class Generator(nn.Module):
    """Used to sample the hindsight vector.""" 
    def __init__(self, obs_dim, action_dim, latent_dim, num_hidden, num_units, noise_dim=256):
        super().__init__()
        # z_t ~ p(z_t | x_t, a_t, x_t+1, noise)
        self.action_dim = action_dim
        self.noise_dim = noise_dim
        self.net = create_mlp(
            2 * obs_dim + action_dim + noise_dim,
            num_hidden,
            num_units,
            latent_dim)

    
    def forward(self, obs, action, obs_next):
        # Map action to one-hot
        action_one_hot = F.one_hot(action, num_classes=self.action_dim)

        # Sample random noise
        noise = torch.randn(obs.shape[0], self.noise_dim).to("cuda")

        # Predict the latent
        x = torch.concatenate((obs, action_one_hot, obs_next, noise), dim=1)
        x = self.net(x)
        
        return x


class Critic(nn.Module):
    """Used make the hindsight vector independent of the state and action.""" 
    def __init__(self, obs_dim, action_dim, latent_dim, num_hidden, num_units):
        super().__init__()
        # q(x_t, a_t, z_t)
        self.action_dim = action_dim
        self.net = create_mlp(
            obs_dim + action_dim + latent_dim,
            num_hidden,
            num_units,
            1)

    def forward(self, obs, action, latent):
        # print("CRITIC", obs.shape, action.shape, latent.shape)
        # print("action", action)
        # print("latent", latent)
        # Map action to one-hot
        action_one_hot = F.one_hot(action, num_classes=self.action_dim)

        # Predict value for critic
        x = torch.concatenate((obs, action_one_hot, latent), dim=1)
        x = self.net(x)

        return F.sigmoid(x)


class WorldModel(nn.Module):
    """Predicts the next state."""
    def __init__(self, obs_dim, action_dim, latent_dim, num_hidden, num_units):
        super().__init__()
        # x_t+1 = f(x_t, a_t, z_t+1)
        self.action_dim = action_dim

        self.net = create_mlp(
            obs_dim + action_dim + latent_dim,
            num_hidden,
            num_units,
            obs_dim)

    def forward(self, obs, action, latent):
        # Map action to one-hot
        action_one_hot = F.one_hot(action, num_classes=self.action_dim)

        # Predict the next observation
        x = torch.concatenate((obs, action_one_hot, latent), dim=1)
        x = self.net(x)

        return x
