import torch
import torch.nn.functional as F

def recon_loss(x_pred, x_tgt):
    return torch.sum((x_pred - x_tgt) ** 2, dim=1)



def con_loss(critic, obs, actions, latents):
    """Compute the contrastive loss from the critic values."""
    # TODO: FIX THIS TO PERFORM THE CORRECT OPERATION
    # You must run the critic over all latents for each state-action pair
    K = obs.shape[0] 
    latents_tiled = latents.tile((latents.shape[0], 1, 1))

    entry_fn = torch.vmap(critic, in_dims=(None, None, 1), randomness="different")
    # batch_fn = torch.vmap(entry_fn, in_dims=(0, 0, None))

    # print(obs.shape, actions.shape, latents.shape)
    scores = torch.exp(entry_fn(obs, actions, latents_tiled))

    squeezed_scores = scores.squeeze().T

    # Manually compute the scores to verify this is correct
    # real_scores = torch.zeros((obs.shape[0], obs.shape[0]), device="cuda")
    # for i in range(obs.shape[0]):
    #     for j in range(obs.shape[0]):
    #         real_scores[i, j] = critic(obs[i].unsqueeze(0), actions[i].unsqueeze(0), latents[j].unsqueeze(0)).squeeze()
    

    diag_scores = torch.diag(squeezed_scores)
    ratios = diag_scores / (squeezed_scores.sum(dim=0) * (1/ K) + 1e-5)
    # print("diag_scores", diag_scores)
    # print("squeezed_scores.sum(dim=0)", squeezed_scores.sum(dim=0))
    # print("ratios", ratios)
    log_ratios = torch.log(ratios + 1e-4)
    # print("log_ratios", log_ratios)

    return log_ratios