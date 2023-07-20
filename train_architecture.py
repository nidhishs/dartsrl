import argparse
import logging
import os
import time
import sys

import numpy as np
import torch
import utilities as u
import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import CheckpointCallback
from model import SearchNetwork, SEARCH_SPACES
from environment import Scorer, DiscreteDartsRL, ContinuousDartsRL


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="dartsrl-arch-disc", help="Name of the experiment.")
    parser.add_argument("--dataset", type=str, default="cifar10", help="Dataset to use for training.")
    parser.add_argument("--env", type=str, default="disc", help="Environment to use for training (disc/cont).")
    parser.add_argument("--reward", type=str, default="nwot", help="Reward function to use (nwot/loss/acc).")
    parser.add_argument("--grad", action="store_true", default=False, help="Include gradient in obs for continuous env.")
    parser.add_argument("--grad-step", action="store_true", default=False, help="Take gradient step after each action in continuous env.")
    parser.add_argument("--data-path", type=str, default="./data", help="Path to dataset.")
    parser.add_argument("--space", type=str, default="darts", help="darts/s2/s3/s4/nas-bench-201")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for train and valid dataloaders.")
    parser.add_argument("--layers", type=int, default=8, help="Number of layers in the super network.")
    parser.add_argument("--initial-channels", type=int, default=16, help="Initial number of channels after input channels in super network.")
    parser.add_argument("--num-steps", type=int, default=256, help="Number of steps before updating PPO policy.")
    parser.add_argument("--total-steps", type=int, default=2_000_000, help="Total number of steps for PPO training.")
    parser.add_argument("--cutout", action="store_true", default=False, help="Use cutout augmentation.")
    parser.add_argument("--cutout-length", type=int, default=16, help="Cutout length.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to super network weights.")
    parser.add_argument("--save-frequency", type=int, default=250_000, help="Frequency to save agent checkpoints.")
    parser.add_argument("--cpu", action="store_true", default=False, help="Use CPU backend.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--log-dir", type=str, default="./logs", help="Path to save logs.")
    args = parser.parse_args()

    if args.env == "disc" and (args.grad or args.grad_step):
        raise ValueError("Cannot use gradient or gradient-step in discrete environment.")
    if args.env == "cont" and args.reward == Scorer.NWOT:
        raise ValueError("Cannot use NWOT reward in continuous environment.")

    args.job_id = f"{args.name}_{int(time.time())}"
    args.log_dir = os.path.join(args.log_dir, args.job_id)
    args.device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    args.cutout_length = args.cutout_length if args.cutout else None
    os.makedirs(args.log_dir, exist_ok=True)
    del args.cpu, args.cutout

    return args


def get_env(env_type, model, train_loader, valid_loader, reward_fn, device, grad, grad_step):
    if env_type == "disc":
        return DiscreteDartsRL(model, valid_loader, reward_fn, device)
    elif env_type == "cont":
        env = ContinuousDartsRL(model, train_loader, valid_loader, 10, reward_fn, grad, grad_step, device)
        env = gym.wrappers.FrameStack(env, 10)
        return env
    else:
        raise ValueError(f"Invalid environment type {env_type}.")


@u.measure(log_fn=logging.info)
def main(args):

    # ====== Dataset ====== #
    dataset, num_classes, input_channels = u.get_dataset(
        args.dataset, args.data_path, train=True, cutout_length=args.cutout_length
    )
    train_loader, valid_loader = u.get_dataloader(dataset, None, args.batch_size)
    # ===================== #

    # ====== Model ====== #
    model = SearchNetwork(
        c_in=input_channels, c=args.initial_channels, num_classes=num_classes,
        num_layers=args.layers, primitives=SEARCH_SPACES[args.space]
    ).to(args.device)
    if args.checkpoint is not None:
        logging.info(f"Loading super network weights from {args.checkpoint}.")
        model.load_state_dict(torch.load(args.checkpoint, map_location=args.device))
    else:
        logging.info("No checkpoint provided. Using random weights.")
    # =================== #

    # ====== Environment ====== #
    env = get_env(
        args.env, model, train_loader, valid_loader, args.reward, args.device, args.grad,
        args.grad_step
    )
    # ======================== #

    # ====== Agent ====== #
    logger = configure(args.log_dir, ["stdout", "log", "tensorboard"])
    agent = PPO(
        "MlpPolicy", env, n_steps=args.num_steps, seed=args.seed, device=args.device, verbose=1
    )
    agent.set_logger(logger)
    # =================== #

    # ====== Training ====== #
    agent.learn(
        total_timesteps=args.total_steps,
        callback=CheckpointCallback(save_freq=args.save_frequency, save_path=args.log_dir, name_prefix=args.name)
    )
    agent.save(os.path.join(args.log_dir, "final_agent"))
    # ====================== #

    # ====== Evaluation ====== #
    env = get_env(
        args.env, model, train_loader, valid_loader, args.reward, args.device, args.grad,
        args.grad_step
    )
    obs, _ = env.reset()
    done = False
    while not done:
        action, _states = agent.predict(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
    
    logging.info(f"Final Genotype: {model.genotype}")
    logging.info(f"Logs saved at {args.log_dir}")
    # ======================== #
    

if __name__ == "__main__":
    arguments = get_arguments()
    u.setup_logger(arguments.log_dir, arguments.name)
    u.stash_files(arguments.log_dir)
    logging.info(f"python3 {' '.join(sys.argv)}")
    u.log_arguments(arguments, logging.info)

    np.random.seed(arguments.seed)
    torch.manual_seed(arguments.seed)

    main(arguments)
