'''ppo_grid_world_agent.py'''
import os
from einops import rearrange
import torch.nn as nn
import torch
import numpy as np
from utils import ActorCritic, ReplayBuffer, ActorCriticSmall
import matplotlib.pyplot as plt
import gymnasium as gym

class PPO:
    """
    Proximal Policy Optimization (PPO) agent for reinforcement learning.
    
    This class implements the PPO algorithm with Generalized Advantage Estimation (GAE)
    for training agents in grid-world environments. It supports both dense and sparse
    reward settings with configurable obstacle layouts.
    
    Attributes:
        env (gym.Env): The training environment instance.
        state_dim (tuple): Dimensions of the observation space.
        action_dim (int): Number of possible actions.
        lr (float): Learning rate for the optimizer.
        gamma (float): Discount factor for future rewards.
        lam (float): Lambda parameter for GAE computation.
        train_steps (int): Total number of training steps.
        eps_clip (float): Clipping parameter for PPO objective.
        num_epochs (int): Number of epochs per update.
        batch_size (int): Size of minibatches for training.
        target_KL (float, optional): Target KL divergence for early stopping.
        rollout_size (int): Number of steps to collect before updating.
        memory (ReplayBuffer): Buffer for storing rollout data.
        policy (ActorCritic): Neural network policy and value function.
        optimizer (torch.optim.Adam): Optimizer for training the policy.
        Loss (nn.MSELoss): Loss function for the critic.
        reward_history (list): History of episode rewards during training.
        critic_loss_clipping (bool): Whether to use value function clipping.

    Args:
        env (gym Environment): Grid world gym environment.
        lr (float): Learning rate for the Adam optimizer.
        gamma (float): Discount factor for computing returns (typically 0.99).
        lam (float): GAE lambda parameter for bias-variance tradeoff (typically 0.95).
        train_steps (int): Total number of environment steps for training.
        rollout_size (int): Number of steps to collect before each policy update.
        minibatch_size (int): Size of minibatches for gradient updates.
        num_epochs (int): Number of optimization epochs per rollout.
        eps_clip (float): Clipping parameter for PPO surrogate objective.
        critic_loss_clipping (bool, optional): Enable value function clipping. 
            Defaults to True.
        target_KL (float, optional): Target KL divergence for early stopping.
            If None, no early stopping is applied. Defaults to None.
    
    """

    def __init__(self, env, config):
        """
        Initialize the PPO agent with specified hyperparameters.
        
        Creates the environment, policy, optimizer and replay buffer
        needed for PPO training.
        """
        self.env = env
        self.config = config
        self.memory = ReplayBuffer(rollout_size=config.rollout_size,
                                  state_dim=self.env.observation_space.shape)
        if config.small_model :
            self.policy = ActorCriticSmall(self.env.action_space.n,
                                           num_channels=config.environment_num_channels
                                           )
        else:
            self.policy = ActorCritic(action_dim=self.env.action_space.n,
                                      num_channels=config.environment_num_channels
                                      )
        self.optimizer = torch.optim.Adam(self.policy.parameters(),
                                        lr=config.lr, eps=1e-5)
        self.loss = nn.MSELoss()
        self.reward_history = []

    def select_action(self, state, timestep):
        """
        Select an action using the current policy and store data in memory.
        
        This function takes the current state, passes it through the policy network
        to get an action, log probability, and state value. All variables are stored
        in the replay buffer for later training.
        
        Args:
            state (np.ndarray): Current environment state.
            timestep (int): Current timestep in the rollout.
            
        Returns:
            int: Selected action as an integer.
        """
        with torch.no_grad():
            state = torch.FloatTensor(state)
            state = rearrange(state, 'c h w -> 1 c h w')  # add batch dimension
            # state = torch.FloatTensor(state).unsqueeze(0)
            action, logprob, state_val = self.policy.act(state)

        # Remove batch dimension to avoid problems when creating batches
        self.memory.states[timestep] = state.squeeze(0)
        self.memory.actions[timestep] = action
        self.memory.logprobs[timestep] = logprob
        self.memory.state_values[timestep] = state_val.squeeze()

        return action.item()

    def compute_gae(self, rewards, values, dones, next_state):
        """
        Compute Generalized Advantage Estimation (GAE) values.
        
        GAE provides a way to estimate advantages that balances bias and variance
        in the advantage estimation, controlled by the lambda parameter.
        
        Args:
            rewards (torch.Tensor): Tensor of rewards from the rollout.
            values (torch.Tensor): Tensor of state values from the rollout.
            dones (torch.Tensor): Tensor indicating terminal states.
            next_state (torch.Tensor): The state after the rollout ends.
            
        Returns:
            torch.Tensor: Computed advantage estimates for each timestep.
        """
        with torch.no_grad():
            advantages = torch.zeros((len(rewards)))
            gae = 0
            for t in reversed(range(len(rewards))):
                if t == len(rewards)-1:
                    next_value = self.policy.get_value(next_state).detach()
                else:
                    next_value = values[t+1]

                mask = 1-dones[t]
                delta = rewards[t] + self.config.gamma * mask * next_value - values[t]
                gae = delta + self.config.gamma * self.config.lam * mask * gae
                advantages[t] = gae

        return advantages

    def update_gae(self, next_state_gae):
        """
        Update the policy using PPO with GAE advantages.
        
        Performs multiple epochs of policy optimization using the collected rollout
        data. Implements PPO's clipped surrogate objective and optional value 
        function clipping.
        
        Args:
            next_state_gae (torch.Tensor): The next state for GAE computation.
        """
        rewards = self.memory.rewards
        dones = self.memory.is_terminals
        old_states = self.memory.states
        old_actions = self.memory.actions
        old_logprobs = self.memory.logprobs
        values = self.memory.state_values
        advantages = self.compute_gae(rewards, values, dones, next_state_gae)
        returns = advantages + values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        indices = np.arange(len(old_states))

        for epoch in range(self.config.num_epochs):
            print(f"Epoch {epoch+1}/{self.config.num_epochs}")
            np.random.shuffle(indices)

            for i in range(0, len(indices), self.config.batch_size):
                batch_indices = indices[i: i + self.config.batch_size]
                states_batch = old_states[batch_indices]
                actions_batch = old_actions[batch_indices]
                logprobs_batch = old_logprobs[batch_indices]
                advantages_batch = advantages[batch_indices]

                logprobs, state_values, dist_entropy = self.policy.evaluate(
                    states_batch, actions_batch.long())
                ratios = torch.exp(logprobs - logprobs_batch)

                with torch.no_grad():
                    approx_kl = ((ratios-1) - (logprobs - logprobs_batch)).mean()

                surrogate_obj = -ratios * advantages_batch
                clipped_surrogate_obj = (-torch.clamp(ratios,
                                       1 - self.config.eps_clip, 1 + self.config.eps_clip) *
                                       advantages_batch)

                if self.config.critic_loss_clipping:
                    value_pred_clipped = (values[batch_indices] + (state_values.squeeze() - values[batch_indices]).clamp(-self.config.eps_clip, self.config.eps_clip))
                    value_losses = (state_values.squeeze() - returns[batch_indices]).pow(2)
                    value_losses_clipped = (value_pred_clipped - returns[batch_indices]).pow(2)
                    critic_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()
                else:
                    critic_loss = self.loss(state_values.squeeze(),
                                          returns[batch_indices])

                loss = (torch.max(surrogate_obj, clipped_surrogate_obj).mean() +
                       0.5 * critic_loss - 0.01 * dist_entropy.mean())

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.optimizer.step()

                if self.config.target_kl is not None:
                    if approx_kl > self.config.target_kl:
                        print('KULBACK LEIBLER GREATER THAN TARGET')
                        print(approx_kl)
                        break

    def save(self):
        """
        Save the current policy parameters to a file.
        
        Args:
            checkpoint_path (str): Path where to save the model parameters.
        """
        os.makedirs(os.path.dirname(self.config.checkpoint_path), exist_ok=True)

        torch.save(self.policy.state_dict(), self.config.checkpoint_path)

    def load(self):
        """
        Load policy parameters from a saved checkpoint.
        
        Args:
            checkpoint_path (str): Path to the saved model parameters.
        """
        self.policy.load_state_dict(torch.load(self.config.checkpoint_path,
                                              weights_only=True))

    def train(self):
        """
        Train the PPO agent in the environment.
        
        Performs the main training loop, collecting rollouts and updating the policy
        using PPO. Implements learning rate annealing and tracks training metrics
        including success rate and average rewards.
        
        Saves training plots and model parameters upon completion.
        """
        num_updates = 0
        state = self.env.reset()[0]
        episode_reward = 0
        episode = 0
        success_rate_list = []
        average_reward = []
        reward_array = np.zeros(10, dtype=np.float32)
        total_updates = int(self.config.train_steps/self.config.rollout_size)

        for i in range(0, total_updates):
            # Collecting rollouts
            print(f"Update {i+1}/{total_updates}")
            success = 0
            ep_in_rol = 0
            for t in range(self.config.rollout_size):
                action = self.select_action(state, t)
                next_state, reward, terminated, truncated, _ = self.env.step(action)

                done = terminated or truncated
                if terminated:
                    success += 1
                    ep_in_rol += 1
                    print('success!')

                if truncated:
                    ep_in_rol += 1
                    print('truncated')

                self.memory.rewards[t] = reward
                self.memory.is_terminals[t] = done
                episode_reward += reward
                state = next_state

                # If done the episode is terminated and so update the stats
                if done:
                    episode += 1
                    self.reward_history.append(episode_reward)
                    print(f"Episode {episode}, Reward: {episode_reward:.3f}")
                    reward_array[episode%100] = episode_reward
                    episode_reward = 0
                    # If episode terminates reset the environment
                    state = self.env.reset()[0]

                # Collect mean values over the updates
                if episode%10 == 0:
                    average_reward.append(reward_array.mean())

            success_rate = success/ep_in_rol
            print(f'Success rate over the last rollout: {success_rate}')
            success_rate_list.append(success_rate)

            # Learning rate annealing
            num_updates += 1
            frac = 1 - (num_updates-1)/total_updates
            new_lr = self.config.lr*frac
            new_lr = max(new_lr, 1e-6)

            self.optimizer.param_groups[0]["lr"] = new_lr
            self.update_gae(torch.FloatTensor(state).unsqueeze(0))

        # Save model parameters at the end of training
        print('saving parameters')
        self.save()
        print('parameters saved')

        # Plot training metrics
        self.plot_train(success_rate_list, average_reward)

    def plot_train(self, success_rate_list, average_reward):
        """
        Plot the training reward history.
        
        Creates and saves a plot showing the reward progression over episodes
        during training.
        """
        os.makedirs(self.config.results_dir, exist_ok=True)

        plt.figure()
        plt.plot(success_rate_list)
        plt.xlabel("updates")
        plt.ylabel("success rate")
        plt.title("PPO Success Rate")
        plt.savefig(self.config.results_dir / "plot_success_rate.png")
        plt.close()

        plt.figure()
        episode_axis = [i*100 for i in np.arange(len(average_reward))]
        plt.plot(episode_axis, average_reward)
        plt.xlabel("episode")
        plt.ylabel("mean")
        plt.title("PPO mean over 100 episodes")
        plt.savefig(self.config.results_dir / "plot_mean_reward.png")
        plt.close()

    def test(self, test_env, episodes=10):
        """
        Test the trained agent in the environment with rendering.
        
        Loads a saved model and runs test episodes with human-readable rendering
        to visually evaluate the agent's performance.
        
        Args:
            checkpoint_path (str): Path to the saved model to test.
            episodes (int, optional): Number of test episodes to run. 
                Defaults to 10.
        """

        self.load()

        for episode in range(episodes):
            state, _ = test_env.reset()
            total_reward = 0
            done, truncated = False, False

            while not (done or truncated):
                action = self.select_action(state, timestep=0)
                state, reward, done, truncated, _ = test_env.step(action)
                total_reward += reward
                test_env.render()

            print(f"Test Episode {episode + 1}: Reward = {total_reward}")

    def record(self, episodes=3):
        """
        Record videos of the agent's performance.
        
        Loads a saved model and records video episodes of the agent interacting
        with the environment for evaluation purposes.
        
        Args:
            checkpoint_path (str): Path to the saved model to record.
            episodes (int, optional): Number of episodes to record. Defaults to 3.
        """
        self.load()
        record_env = gym.make(f"MovingObstaclesGrid-v{self.config.environment_num_channels}", 
                             num_obstacles=self.config.num_obstacles,
                             dense_reward=self.config.dense_rew,
                             nrow=self.config.shape[0],
                             ncol=self.config.shape[1],
                             render_mode='rgb_array')
        record_env = gym.wrappers.RecordVideo(
            record_env, video_folder="videos",
            episode_trigger=lambda x: True,
            name_prefix="MovingObstacles")

        for episode in range(episodes):
            state = record_env.reset()[0]
            total_reward = 0
            done = False
            while not done:
                action = self.select_action(state, timestep=0)
                state, reward, done, _, _ = record_env.step(action)
                total_reward += reward
            print(f"Test Episode {episode + 1}: Reward = {total_reward}")
        record_env.close()
