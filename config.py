'''
This module sets up the argument parser for training a PPO agent in a grid world environment.
It includes configurations for the environment, training hyperparameters, model saving/loading,
and testing/recording options.
'''

import argparse
from pathlib import Path


def create_parser():
    """
    Create and configure the argument parser for PPO training.
    
    Returns:
        argparse.ArgumentParser: Configured argument parser.
    """
    parser = argparse.ArgumentParser(
        description="PPO Agent for Grid World Environments",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="Example: python script.py --mode train --shape 10 10 --num_obstacles 5"
    )

    # Required arguments
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "test", "record"],
        required=True,
        help="Mode to run the agent in"
    )

    # Environment configuration
    env_group = parser.add_argument_group('Environment Configuration')
    env_group.add_argument(
        "--environment_num_channels",
        type=int,
        choices=[1, 3],
        default=3,
        help="Number of channels in the environment observation"
    )
    env_group.add_argument(
        "--shape",
        type=int,
        nargs=2,
        default=[10, 10],
        metavar=('HEIGHT', 'WIDTH'),
        help="Shape of the grid world (height, width)"
    )
    env_group.add_argument(
        "--num_obstacles",
        type=int,
        default=3,
        help="Number of obstacles in the environment"
    )
    env_group.add_argument(
        "--dense_rew",
        action='store_true',
        help="Use dense reward (adds 0.02 for each step closer to goal)"
    )

    # Training hyperparameters
    train_group = parser.add_argument_group('Training Hyperparameters')
    train_group.add_argument(
        "--lr", "--learning_rate",
        type=float,
        default=3e-4,
        help="Learning rate for the optimizer"
    )
    train_group.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="Discount factor"
    )
    train_group.add_argument(
        "--lam",
        type=float,
        default=0.95,
        help="GAE lambda parameter"
    )
    train_group.add_argument(
        "--train_steps",
        type=int,
        default=500000,
        help="Total number of training steps"
    )
    train_group.add_argument(
        "--rollout_size",
        type=int,
        default=16384,
        help="Number of steps to collect before updating"
    )
    train_group.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Size of minibatches for training"
    )
    train_group.add_argument(
        "--num_epochs",
        type=int,
        default=4,
        help="Number of epochs per update"
    )
    train_group.add_argument(
        "--eps_clip",
        type=float,
        default=0.1,
        help="Clipping parameter for PPO"
    )
    train_group.add_argument(
        "--critic_loss_clipping",
        action='store_true',
        help="Enable clipping for critic loss"
    )
    train_group.add_argument(
        "--target_kl",
        type=float,
        default=0.01,
        help="Target KL divergence for early stopping"
    )
    train_group.add_argument(
        "--small_model",
        action='store_true',
        help="Use a smaller neural network model useful for smaller environments"
    )

    # Model and logging
    model_group = parser.add_argument_group('Model and Logging')
    model_group.add_argument(
        "--checkpoint_path",
        type=Path,
        help="Path to model checkpoint for testing/recording"
    )
    model_group.add_argument(
        "--results_dir",
        type=Path,
        default=Path("Autonomous_Projects/Results"),
        help="Directory to save results and plots"
    )
    model_group.add_argument(
        "--save_interval",
        type=int,
        default=10000,
        help="Save model every N steps"
    )

    # Testing/Recording specific
    test_group = parser.add_argument_group('Testing/Recording Options')
    test_group.add_argument(
        "--test_episodes",
        type=int,
        default=10,
        help="Number of episodes to run in test mode"
    )
    test_group.add_argument(
        "--record_episodes",
        type=int,
        default=3,
        help="Number of episodes to record"
    )

    # Advanced options
    # advanced_group = parser.add_argument_group('Advanced Options')
    # advanced_group.add_argument(
    #     "--device",
    #     type=str,
    #     default="auto",
    #     choices=["auto", "cpu", "cuda"],
    #     help="Device to use for training"
    # )
    # advanced_group.add_argument(
    #     "--seed",
    #     type=int,
    #     default=None,
    #     help="Random seed for reproducibility"
    # )
    # advanced_group.add_argument(
    #     "--verbose",
    #     action='store_true',
    #     help="Enable verbose logging"
    # )

    return parser


def parse_arguments():
    """
    Parse and validate command line arguments.

    Returns:
        argparse.Namespace: Parsed and validated arguments
    """
    parser = create_parser()
    args = parser.parse_args()

    return args
