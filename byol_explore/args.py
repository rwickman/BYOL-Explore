import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rom", default="./LoZ.nes", 
        help="Location of the ROM")
    parser.add_argument("--evaluate", action="store_true",
        help="Evaluate the performance")
    parser.add_argument("--load", action="store_true",
        help="Load the models.")
    parser.add_argument("--save_dir", default="",
        help="Location of the saved models")
    parser.add_argument("--ckpt_save_dir", default="logs",
        help="Location of the logs models.")
    parser.add_argument("--save_iter", type=int, default=128,
        help="After how many episode to save the model.")
    parser.add_argument("--ckpt_iter", type=int, default=128,
        help="After how many episode to save a checkpoint of the model.")
    parser.add_argument("--print_values", action="store_true",
        help="Evaluate the performance")
    
    parser.add_argument("--units", type=int, default=256,
        help="Number of hidden units to use in the networks")
    parser.add_argument("--num_hidden", type=int, default=1,
        help="Number of hidden layers to use in the policy networks")
    parser.add_argument("--num_episodes", type=int, default=16384,
        help="Number of episodes to run for each environment.")
    parser.add_argument("--max_timesteps", type=int, default=65536,
        help="Maximum number of timesteps in the environment")
    parser.add_argument("--train_iter", type=int, default=2048, 
        help="How many timesteps to collect in each environment before training the networks.")
    parser.add_argument("--batch_size", type=int, default=128, 
        help="Batch size of the updates.")
    parser.add_argument("--n_envs", type=int, default=16, 
        help="Number of environments to run simultaneously")
    parser.add_argument("--num_experts", type=int, default=8,
        help="Number of expert episodes to preserve.")

    dqn_args = parser.add_argument_group("DQN")
    dqn_args.add_argument("--epsilon", type=float, default=0.99, 
        help="Starting epsilon value used for DQN.")
    dqn_args.add_argument("--gamma", type=float, default=0.997, 
        help="Gamma used to compute return.")
    dqn_args.add_argument("--gamma_intrinsic", type=float, default=0.99, 
        help="Gamma used to compute return.")
    dqn_args.add_argument("--min_epsilon", type=float, default=0.05, 
        help="Minimum epsilon value use for DQN.")
    dqn_args.add_argument("--decay_episodes", type=int, default=8196, 
        help="Number of episodes before epsilon decays to min_epsilon.")
    dqn_args.add_argument("--tgt_tau", type=float, default=0.05,
                    help="The tau value to control the update rate of the target DQN parameters.")
    dqn_args.add_argument("--memory_cap", type=int, default=262144, 
        help="Maximum size of the replay memory.")
    dqn_args.add_argument("--min_train_exps", type=int, default=2048, 
        help="Minimum replay buffer size before training.")
    dqn_args.add_argument("--dqn_train_iter", type=int, default=16, 
        help="How many timesteps elapsed before training the model.")
    dqn_args.add_argument("--continuous", action="store_true",
        help="Assume infinite horizon case.")
    dqn_args.add_argument("--n_steps", type=int, default=10, 
        help="Number of steps to use for n-step return.")
    dqn_args.add_argument("--n_step_alpha", type=float, default=0.6, 
        help="Alpha used for sampling the n-step from beta-binomial distribution.")
    dqn_args.add_argument("--n_step_beta", type=float, default=2.0, 
        help="Beta used for sampling the n-step from beta-binomial distribution.")
    dqn_args.add_argument("--per_alpha", type=float, default=0.6, 
        help="Alpha used for PER.")
    dqn_args.add_argument("--per_beta", type=float, default=0.4, 
        help="Beta used for PER.")
    dqn_args.add_argument("--per_intrinsic_priority", type=float, default=0.1, 
        help="How much to scale the error for intrinsic error for computing the priority.")

    byol_args = parser.add_argument_group("BYOL-Hindsight")
    byol_args.add_argument("--recon_lam", type=float, default=1.0,
        help="Lambda used for scaling the reconstruction error.")

    parser.add_argument("--no_ngu", action="store_true",
        help="Don't use NGU.")
    parser.add_argument("--no_use_intrinsic_reward", action="store_true",
        help="Don't use intrinsic reward.")
    parser.add_argument("--ngu_beta", type=float, default=0.01, 
        help="Factor to use to for the NGU intrinsic reward.")
    parser.add_argument("--ngu_epochs", type=int, default=5, 
        help="Number of epochs to train the NGU networks.")
    parser.add_argument("--max_curiosity_factor", type=float, default=5, 
        help="Maximum curiosity factor for NGU.")
    parser.add_argument("--ngu_memory_capacity", type=int, default=2**17, 
        help="Replay buffer size for NGU trajectories.")
    parser.add_argument("--knn_k", type=int, default=10, 
        help="Number of neighbors to use when calculating the episodic reward for NGU.")
    parser.add_argument("--ngu_ckpt", default="",
        help="NGU checkpoint it load.")
    
    return parser.parse_args()
    
    