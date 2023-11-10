import gym
import os
import tqdm
from stable_baselines3.common.vec_env import SubprocVecEnv

from byol_explore.args import get_args
from byol_explore.rl.ddqn_actor import DDQNActor
from byol_explore.rl.trainer import Trainer


if __name__ == "__main__":
    args = get_args()
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    # action_space = 4
    # obs_space = 8
    action_space = 3
    obs_space = 2

    agent = DDQNActor(args, obs_space, action_space)
    if args.load:
        agent.load()

    if not args.evaluate:
        # env = SubprocVecEnv(
        #     [lambda:  gym.make("LunarLander-v2") for i in range(args.n_envs)])
        # env = SubprocVecEnv(
        #     [lambda:  gym.make("CartPole-v1") for i in range(args.n_envs)])
        env = SubprocVecEnv(
            [lambda:  gym.make("MountainCar-v0") for i in range(args.n_envs)])

        #env = SubprocVecEnv([lambda:  gym.make("CartPole-v1") for i in range(args.n_envs)])

        #env = gym.make("CartPole-v1")
        
        trainer = Trainer(args, agent, env, obs_space)
        trainer.run()
    else:
        import torch
        done = True
        env = gym.make("MountainCar-v0", render_mode="human")
        total_reward = 0
        total_int_reward = 0
        print(agent.buffer.max_intrinsic_reward)
        for _ in tqdm.tqdm(range(args.max_timesteps * 2)):
            if done:
                state, _ = env.reset()
                done = False
                last_obs = state.copy()
                print(f"TOTAL REWARD {total_reward} TOTAL INT REWARD {total_int_reward}")
                total_reward = 0
                total_int_reward = 0
            else:
                action = agent(
                    torch.tensor(state, dtype=torch.float32).to("cuda").unsqueeze(0),
                    argmax=False).squeeze(0)
               
                
                # obs, _ = model.policy.obs_to_tensor(state)
                # print(model.policy._predict(obs))

                action = int(action)

                # print(action)
                
                state, reward, done, info, terminated = env.step(action)
                total_reward += reward

                states, actions, rewards, intrinsic_rewards, next_states, dones = agent.buffer.sample(
                    args.batch_size, agent.byol_hindsight, n_step=1)
                 
                state_ten = torch.tensor(state, dtype=torch.float32).to("cuda").unsqueeze(0)
                action_ten = torch.tensor(action).to("cuda").unsqueeze(0)  
                next_state_ten = torch.tensor(last_obs, dtype=torch.float32).to("cuda").unsqueeze(0)
                
                states = torch.concatenate([states, state_ten])
                actions = torch.concatenate([actions, action_ten])
                next_states = torch.concatenate([next_states, next_state_ten])

                intrinsic_rewards = agent.byol_hindsight.get_intrinsic_reward(
                    states,
                    actions,
                    next_states
                    )

                total_int_reward += intrinsic_rewards[-1].item()
                total_reward += reward
                last_obs = state.copy()
                if total_reward > 500:
                    done = True

                env.render()
                
