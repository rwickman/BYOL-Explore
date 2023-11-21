import matplotlib.pyplot as plt
import argparse
import json
import os
import numpy as np

def plot(save_dir, w):

    def moving_average(x):
        return np.convolve(x, np.ones(w), 'valid') / w

    with open(os.path.join(save_dir, "train_dict.json")) as f:
        train_dict = json.load(f)
    with open(os.path.join(save_dir, "byol_train_dict.json")) as f:
        byol_train_dict = json.load(f)
    

    fig, axs = plt.subplots(4)
    axs[0].plot(moving_average(train_dict["loss"]))
    axs[0].set(ylabel="Q Loss")
    
    axs[1].plot(moving_average(train_dict["intrinsic_loss"]))
    axs[1].set(ylabel="NGU Q Loss")
    
    axs[2].plot(moving_average(train_dict["total_rewards"]))
    axs[2].set(ylabel="Total Rewards")

    axs[3].plot(moving_average(train_dict["total_int_rewards"]))
    axs[3].set(ylabel="Total Intrinsic Rewards")
    plt.show()
    
    fig, axs = plt.subplots(2)
    axs[0].plot(moving_average(byol_train_dict["contrastive_loss"]))
    axs[0].set(ylabel="Contrastive Loss")
    
    axs[1].plot(moving_average(byol_train_dict["recon_loss"]))
    axs[1].set(ylabel="Recon Loss")
    plt.show()
    



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", default="", required=True,
        help="Location of the saved models")
    parser.add_argument("--w", default=10, type=int,
        help="Location of the saved models")
        
    args = parser.parse_args()
    plot(args.save_dir, args.w)