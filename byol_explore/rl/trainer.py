import torch
import numpy as np
import time

from byol_explore.rl.replay_buffer import Trajectory

class Trainer:
    def __init__(self, args, agent, env, state_dim):
        self.args = args
        self.agent = agent
        self.env = env
        self.state_dim = state_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.org_beta = self.args.ngu_beta

    def _init_trajectory(self):
        return Trajectory(
                    [],
                    [],
                    [],
                    [],
                    []
                )

    def _setup_trajectories(self):
        self._trajs = []
        for _ in range(self.args.n_envs):
            self._trajs.append(
                self._init_trajectory()
            )

    def _update_trajectories(self, states, actions, rewards, intrinsic_rewards, dones):
        for i in range(len(states)):
            self._trajs[i].states.append(self._last_obs[i].copy())
            self._trajs[i].actions.append(actions[i])
            self._trajs[i].rewards.append(rewards[i])
            self._trajs[i].next_states.append(states[i].copy())
            self._trajs[i].dones.append(dones[i])
            self._trajs[i].total_reward += rewards[i]
            self._trajs[i].total_intrinsic_reward += intrinsic_rewards[i]
            self._trajs[i].ep_return += (0.999 ** (len(self._trajs[i].states) - 1)) * rewards[i]

    def _wrap_up(self, idx):
        """Wrap up one of the episodes."""
        self._trajs[idx].states = np.array(self._trajs[idx].states, dtype=np.float32)
        self._trajs[idx].actions = np.array(self._trajs[idx].actions, dtype=np.int64)
        self._trajs[idx].rewards = np.array(self._trajs[idx].rewards, dtype=np.float32)
        self._trajs[idx].next_states = np.array(self._trajs[idx].next_states, dtype=np.float32)
        self._trajs[idx].dones = np.array(self._trajs[idx].dones, dtype=np.int8)

        self.agent.add_trajectory(self._trajs[idx])
        self._trajs[idx] = self._init_trajectory()

    def run(self):
        start_time = time.time()
        self._setup_trajectories()
        self._last_obs = self.env.reset()
        ep_idx = 0
        self._cur_step = 0
        while ep_idx < self.args.num_episodes:
            with torch.no_grad():
                actions = self.agent(
                    torch.tensor(self._last_obs, dtype=torch.float32).to(self.device))

                states, rewards, dones, info = self.env.step(actions)

                intrinsic_rewards = self.agent.byol_hindsight.get_intrinsic_reward(
                    torch.tensor(self._last_obs, dtype=torch.float32).to(self.device),
                    torch.tensor(actions).to(self.device),
                    torch.tensor(states, dtype=torch.float32).to(self.device)).detach().cpu().numpy()

            
            
            self._update_trajectories(states, actions, rewards, intrinsic_rewards, dones)
            self._last_obs = states.copy()

            # Check if any of the environments terminated
            for i in range(len(dones)):
                if dones[i]:
                    ep_idx += 1

                    print(f"{time.time() - start_time:.2f}: TOTAL REWARD {self._trajs[i].total_reward} INTRINSIC REWARD {self._trajs[i].total_intrinsic_reward.item()} for episode {ep_idx} EPSILON {self.agent.epsilon_threshold}")
                    self._wrap_up(i)
                    if ep_idx % self.args.save_iter == 0:
                        self.agent.save()
                        # self.args.ngu_beta = min(np.random.beta(0.3, 10), self.org_beta)

                    print("self.args.ngu_beta", self.args.ngu_beta)

            if self._cur_step % self.args.dqn_train_iter == 0 and self.agent.is_train_ready:
                self.agent.train()
            
            self._cur_step += 1
