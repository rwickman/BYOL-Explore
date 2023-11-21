import torch
import numpy as np


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_n_step_trajectory(ind, n_step, next_states, rewards, dones, size, gamma):
    n_rewards, n_next_states, n_dones = [], [], []
    cur_idx = 0
    for idx in ind:
        # print("IDX GET N STEP", idx)   
        final_idx = (idx + n_step - 1) % size
        # print("FINAL IDX", final_idx)
        reward = rewards[final_idx]
        next_state, done = next_states[final_idx], dones[final_idx]
        # print("NEXT STATE: ", next_state, done)

        for i in reversed(range(idx, final_idx)):
            i = i % size
            reward = rewards[i] + reward * gamma * (1-dones[i])

            if dones[i]:
                next_state, done = next_states[i], dones[i]            
        #     print("cur_idx", i, "NEXT STATE: ", next_state, done)
        
        # print("FINAL REWARD", reward)
        n_rewards.append(reward)
        n_next_states.append(next_state)
        n_dones.append(done)

    n_rewards = np.array(n_rewards, dtype=np.float32)
    n_next_states = np.stack(n_next_states)
    n_dones = np.array(n_dones, dtype=np.int8)

    return n_rewards, n_next_states, n_dones