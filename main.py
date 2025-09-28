'''main.py'''
from pathlib import Path
import gymnasium as gym
import custom_envs
from dataclass_config import PPOConfig
from config import parse_arguments
from ppo_grid_world_agent import PPO


def main():
    '''Main function for PPO agent in the grid world environment.'''
    args = parse_arguments()

    config = PPOConfig(**vars(args))

    env_name = f"MovingObstaclesGrid-v{config.environment_num_channels}"
    env = gym.make(
        env_name,
        dense_reward=config.dense_rew,
        num_obstacles=config.num_obstacles,
        nrow=config.shape[0],
        ncol=config.shape[1],
        max_step=500
    )

    ppo = PPO(env, config)

    if config.mode == "train":
        print(f"Starting training with {config.train_steps} steps...")
        print(f"Environment: {env_name}")
        print(f"Grid shape: {config.shape}")
        print(f"Obstacles: {config.num_obstacles}")
        print(f"Dense reward: {config.dense_rew}")

        ppo.train()
        print(f"Training completed! Model saved to: {config.checkpoint_path}")

    elif config.mode == "test":
        env = gym.make(
            env_name,
            dense_reward=config.dense_rew,
            num_obstacles=config.num_obstacles,
            nrow=config.shape[0],
            ncol=config.shape[1],
            max_step=500,
            render_mode="human"
        )
        test_checkpoint = Path(config.checkpoint_path)
        print(f"Testing model: {test_checkpoint}")
        ppo.test(env, episodes=config.test_episodes)

    elif config.mode == "record":
        record_checkpoint = Path(config.checkpoint_path)
        print(f"Recording {config.record_episodes} episodes")
        print(f"Using model: {record_checkpoint}")
        ppo.record(episodes=config.record_episodes)

if __name__ == "__main__":
    main()
