
class Trainer:
    def __init__(self, args, agent, env, state_dim, action_dim):
        self.args = args
        self.agent = agent
        self.state_dim = state_dim
        self.env = env
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.byol_hindsight = BYOLHindSight(
            state_dim,
            action_dim,
            latent_dim=state_dim,
            num_hidden=2,
            num_units=self.args.units,
            emb_dim=state_dim,
            noise_dim=state_dim//2).to(self.device)

        self._cur_step = 0
        self._num_updates = 0

    def _train(self):

        self.agent.train()
        states, actions, rewards, intrinsic_rewards, next_states, dones = self.agent.buffer.sample(
            self.args.batch_size)
        if self._num_updates % 4 == 0:
            con_loss_val, recon_loss_val = self.byol_hindsight.update(states, actions, next_states)
        self._num_updates += 1

    def _add(self, states, actions, rewards, intrinsic_rewards, next_states, dones):
        for i in range(len(states)):
            self.agent.buffer.add(
                states[i],
                actions[i],
                rewards[i],
                intrinsic_rewards[i],
                next_states[i],
                dones[i]
            )

    def run(self):
        last_obs = self.env.reset()
        ep_idx = 0
        total_rewards = torch.zeros(self.args.n_envs)
        total_int_rewards = torch.zeros(self.args.n_envs)


        while ep_idx < self.args.num_episodes:
            with torch.no_grad():
                    
                actions = self.agent(
                    torch.tensor(last_obs, dtype=torch.float32).to(self.device))

                states, rewards, dones, info = self.env.step(actions)

                intrinsic_rewards = self.byol_hindsight.get_intrinsic_reward(
                    torch.tensor(states, dtype=torch.float32).to(self.device),
                    torch.tensor(actions).to(self.device),
                    torch.tensor(last_obs, dtype=torch.float32).to(self.device))

            self._add(states, actions, rewards, intrinsic_rewards, last_obs, dones)
            last_obs = states.copy()
            total_rewards += rewards
            total_int_rewards += intrinsic_rewards.detach().cpu().numpy()
            
            for i in range(len(dones)):
                if dones[i]:
                    print(f"EPISODE {ep_idx} TOTAL REWARDS {total_rewards[i].item()} TOTAL INTRINSIC REWARDS {total_int_rewards[i].item()} EPSILON {self.agent.epsilon_threshold}")
                    self.agent.add_episode_stats(total_rewards[i].item(), total_int_rewards[i].item())
                    total_rewards[i] = 0
                    total_int_rewards[i] = 0
                    ep_idx += 1
                    
                    if ep_idx % self.args.save_iter == 0:
                        self.agent.save()
            
            if self._cur_step % self.args.dqn_train_iter == 0 and self.agent.is_train_ready:
                self._train()
            
            self._cur_step += 1
            