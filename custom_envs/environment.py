''' environment.py - Custom Gymnasium environment with dynamic obstacle shapes '''
from __future__ import annotations
import random
import time
import numpy as np
from gymnasium import spaces, Env
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors



class DynamicObstacleShapes(Env):
    """
    A gymnasium environment featuring a grid world with dynamic obstacle shapes.
    
    The environment consists of:
    - An agent that moves in a grid world (yellow circle in visualization)
    - A goal position 
    - Dynamic obstacles with various shapes that move randomly each step (red squares)
    
    The agent must navigate to the goal while avoiding collisions with moving obstacles.
    Obstacles can have different shapes (L-shapes, squares, lines) and move in random
    directions each step, making the navigation task challenging and dynamic.
    
    Attributes:
        metadata (dict): Environment metadata including render modes and FPS
        nrow (int): Number of rows in the grid
        ncol (int): Number of columns in the grid  
        max_steps (int): Maximum steps before episode truncation
        num_obstacles (int): Number of dynamic obstacles in the environment
        render_mode (str): Rendering mode ("human" or None)
        dense_reward (bool): Whether to use dense rewards (distance-based) or sparse
        action_space (spaces.Discrete): Action space with 4 discrete actions
        observation_space (spaces.Box): 3-channel observation space (obstacles, agent, goal)
        obstacle_specs (list): Predefined obstacle shapes as relative cell coordinates
    """
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, dense_reward, nrow=10, ncol=10, max_step=500,
                 num_obstacles=2, render_mode=None):
        """
        Initialize the DynamicObstacleShapes environment.
        
        Args:
            dense_reward (bool): 
            If True, provides distance-based rewards for getting closer to goal.
            If False, only provides sparse rewards (goal reached, collision, step penalty).
            
            nrow (int, optional): Number of rows in the grid. Defaults to 10.
            ncol (int, optional): Number of columns in the grid. Defaults to 10.
            max_step (int, optional): Maximum number of steps before truncation. Defaults to 500.
            num_obstacles (int, optional): Number of dynamic obstacles. Defaults to 2.
            render_mode (str, optional): Rendering mode ("human" for visualization). 
                                         Defaults to None.
        """
        super().__init__()
        self.nrow = nrow
        self.ncol = ncol
        self.max_steps = max_step
        self.num_obstacles = num_obstacles
        self.render_mode = render_mode
        self.dense_reward = dense_reward
        # Actions: 0=up, 1=down, 2=left, 3=right
        self.action_space = spaces.Discrete(4)

        # Observation: 3 channels (obstacles, agent, goal)
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(3, nrow, ncol), dtype=np.float32
        )

        # Obstacle shapes
        self.obstacle_specs = [
            {"cells": [(0,0), (0,1), (1,1)]},
            {"cells": [(0,0), (0,1), (1,0), (1,1)]},
            {"cells": [(0,0), (0,1), (0,2)]},
            {"cells": [(0,0), (1,0), (2,0)]}
        ]

        self.agent_pos = None
        self.goal_pos = None
        self.obstacles = []
        self.step_count = 0

        # Initialize rendering attributes
        self._obstacle_patches = []
        self._goal_patch = None
        self._goal_glow = None
        self._goal_inner = None
        self._agent_circle = None
        self._agent_glow = None
        self._fig = None
        self._ax = None
        self._img = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        occupied = set()
        self.obstacles = []

        for _ in range(self.num_obstacles):
            while True:
                shape = random.choice(self.obstacle_specs)
                x = random.randint(0, self.nrow - 1)
                y = random.randint(0, self.ncol - 1)
                abs_cells = {(x + dx, y + dy) for dx, dy in shape["cells"]}
                if (all(0 <= cx < self.nrow and 0 <= cy < self.ncol for cx, cy in abs_cells)
                        and abs_cells.isdisjoint(occupied)):
                    self.obstacles.append({"cells": shape["cells"], "pos": [x, y]})
                    occupied.update(abs_cells)
                    break

        while True:
            ax, ay = random.randint(0, self.nrow - 1), random.randint(0, self.ncol - 1)
            if (ax, ay) not in occupied:
                self.agent_pos = [ax, ay]
                occupied.add((ax, ay))
                break

        while True:
            gx, gy = random.randint(0, self.nrow - 1), random.randint(0, self.ncol - 1)
            if (gx, gy) not in occupied:
                self.goal_pos = [gx, gy]
                occupied.add((gx, gy))
                break

        return self._get_obs(), {}   # obs, info

    def _get_obs(self):

        """Return 3-channel grid: [obstacles, agent, goal]."""
        obs = np.zeros((3, self.nrow, self.ncol), dtype=np.float32)

        for obs_def in self.obstacles:
            ox, oy = obs_def["pos"]
            for dx, dy in obs_def["cells"]:
                obs[0, ox + dx, oy + dy] = 1.0   # obstacles

        obs[1, self.agent_pos[0], self.agent_pos[1]] = 1.0   # agent
        obs[2, self.goal_pos[0], self.goal_pos[1]] = 1.0     # goal

        return obs

    def _move_obstacles(self):
        directions = [(0,0), (-1,0), (1,0), (0,-1), (0,1)]
        occupied = set()

        # Mark current occupied cells
        for obs_def in self.obstacles:
            ox, oy = obs_def["pos"]
            for dx, dy in obs_def["cells"]:
                occupied.add((ox + dx, oy + dy))
        occupied.add(tuple(self.agent_pos))
        occupied.add(tuple(self.goal_pos))

        new_obstacles = []
        for obs_def in self.obstacles:
            shape = obs_def["cells"]
            ox, oy = obs_def["pos"]
            current_cells = {(ox + dx, oy + dy) for dx, dy in shape}
            occupied -= current_cells

            dx, dy = random.choice(directions)
            new_pos = (ox + dx, oy + dy)
            new_cells = {(new_pos[0] + sx, new_pos[1] + sy) for sx, sy in shape}

            if (all(0 <= x < self.nrow and 0 <= y < self.ncol for x,y in new_cells)
                    and new_cells.isdisjoint(occupied)):
                new_obstacles.append({"cells": shape, "pos": [new_pos[0], new_pos[1]]})
                occupied.update(new_cells)
            else:
                new_obstacles.append({"cells": shape, "pos": [ox, oy]})
                occupied.update((ox + sx, oy + sy) for sx, sy in shape)

        self.obstacles = new_obstacles

    def step(self, action):
        self.step_count += 1
        terminated, truncated = False, False
        reward = -0.01   # step penalty
        old_dist = np.linalg.norm(np.array(self.agent_pos) - np.array(self.goal_pos))

        x, y = self.agent_pos
        if action == 0 and x > 0:           # up
            x -= 1
        elif action == 1 and x < self.nrow - 1:  # down
            x += 1
        elif action == 2 and y > 0:         # left
            y -= 1
        elif action == 3 and y < self.ncol - 1:  # right
            y += 1
        else:
            reward -= 0.1   # small penalty for invalid move

        # Check collision at new pos
        collision = False
        for obs_def in self.obstacles:
            ox, oy = obs_def["pos"]
            for dx, dy in obs_def["cells"]:
                if (x, y) == (ox + dx, oy + dy):
                    collision = True
                    reward = -5.0
                    break
            if collision:
                break

        # Update agent
        if not collision:
            self.agent_pos = [x, y]

        # Move obstacles
        self._move_obstacles()

        if self.dense_reward:
            # Add a small Reward for getting closer to goal
            new_dist = np.linalg.norm(np.array([x, y]) - np.array(self.goal_pos))
            reward += 0.2 * (old_dist - new_dist)

        # Collision after obstacles move
        for obs_def in self.obstacles:
            ox, oy = obs_def["pos"]
            for dx, dy in obs_def["cells"]:
                if self.agent_pos == [ox + dx, oy + dy]:
                    reward = -5.0
                    break

        # Check goal
        if self.agent_pos == self.goal_pos:
            reward = 50
            terminated = True

        # Timeout
        if self.step_count >= self.max_steps:
            truncated = True

        return self._get_obs(), reward, terminated, truncated, {}

    def render(self, mode="human"):
        if self.render_mode != "human":
            return

        time.sleep(1 / self.metadata["render_fps"])

        grid = np.zeros((self.nrow, self.ncol), dtype=int)
        ax, ay = self.agent_pos
        cmap = mcolors.ListedColormap(["#0d0d0d"])  # Only background color needed

        if not hasattr(self, "_fig") or self._fig is None:
            self._fig, self._ax = plt.subplots()
            print(f"Created figure: {self._fig is not None}, axes: {self._ax is not None}")
            plt.ion()
            self._img = self._ax.imshow(grid, cmap=cmap, origin="upper", vmin=0, vmax=0)

            # Fixed grid lines - use integer positions for major ticks
            self._ax.set_xticks(np.arange(-0.5, self.ncol, 1), minor=True)
            self._ax.set_yticks(np.arange(-0.5, self.nrow, 1), minor=True)
            self._ax.set_xticks(np.arange(0, self.ncol, 1), minor=False)
            self._ax.set_yticks(np.arange(0, self.nrow, 1), minor=False)

            # Grid styling
            self._ax.grid(which="minor", color="#404040", linestyle='-', linewidth=0.6)
            self._ax.tick_params(which="minor", size=0)
            self._ax.tick_params(which="major", size=0)  # Hide major tick marks
            self._ax.set_facecolor("#0d0d0d")

            # Hide tick labels for cleaner look
            self._ax.set_xticklabels([])
            self._ax.set_yticklabels([])

            # Force square grid cells
            self._ax.set_aspect('equal')

            plt.show()
        else:
            # Safe update - only if _img exists and is not None
            if hasattr(self, "_img") and self._img is not None:
                self._img.set_data(grid)

        # Only proceed with drawing if axes exists
        if self._ax is None:
            print("Warning: _ax is None, skipping rendering")  # Debug
            return

        # Remove old obstacle and goal patches
        if hasattr(self, "_obstacle_patches"):
            for patch in self._obstacle_patches:
                if patch is not None:
                    patch.remove()
        if hasattr(self, "_goal_patch") and self._goal_patch is not None:
            self._goal_patch.remove()
        if hasattr(self, "_goal_glow") and self._goal_glow is not None:
            self._goal_glow.remove()
        if hasattr(self, "_goal_inner") and self._goal_inner is not None:
            self._goal_inner.remove()

        # Draw obstacles as rectangles that fill entire cells
        self._obstacle_patches = []
        for obs_def in self.obstacles:
            ox, oy = obs_def["pos"]
            for dx, dy in obs_def["cells"]:
                x, y = ox + dx, oy + dy
                # Create rectangle that fills the entire cell (from -0.5 to +0.5)
                rect = plt.Rectangle((y - 0.5, x - 0.5), 1.0, 1.0,
                                   facecolor="#ff004d", edgecolor="none")
                self._ax.add_patch(rect)
                self._obstacle_patches.append(rect)

        # Draw goal as retro portal/ring style
        gx, gy = self.goal_pos

        # Outer glow ring (larger, semi-transparent)
        self._goal_glow = plt.Circle((gy, gx), 0.45, facecolor="none",
                                    edgecolor="#00ffcc", linewidth=8, alpha=0.3)
        self._ax.add_patch(self._goal_glow)

        # Main portal ring (thick hollow circle)
        self._goal_patch = plt.Circle((gy, gx), 0.35, facecolor="none",
                                     edgecolor="#00ffcc", linewidth=4)
        self._ax.add_patch(self._goal_patch)

        # Inner sparkle ring (thin bright ring)
        self._goal_inner = plt.Circle((gy, gx), 0.25, facecolor="none",
                                     edgecolor="#ffffff", linewidth=2, alpha=0.8)
        self._ax.add_patch(self._goal_inner)

        # Remove old patches
        if hasattr(self, "_agent_circle") and self._agent_circle is not None:
            self._agent_circle.remove()
        if hasattr(self, "_agent_glow") and self._agent_glow is not None:
            self._agent_glow.remove()

        # Fixed coordinates for proper centering
        # In matplotlib imshow with origin="upper":
        # - x coordinate = column index (ay)
        # - y coordinate = row index (ax)
        circle_x = ay
        circle_y = ax

        # Agent glow - centered in grid cell
        self._agent_glow = plt.Circle((circle_x, circle_y), 0.45, color="#ffcc00", alpha=0.3)
        self._ax.add_patch(self._agent_glow)

        # Agent body - centered in grid cell
        self._agent_circle = plt.Circle((circle_x, circle_y), 0.3, color="#ffcc00")
        self._ax.add_patch(self._agent_circle)

        self._fig.canvas.draw()
        self._fig.canvas.flush_events()

class DynamicObstacleShapesSingleChannel(Env):
    """
    A gymnasium environment featuring a grid world with dynamic obstacle shapes.   
    the environment consists of:
    - An agent that moves in a grid world (yellow circle in visualization)
    - A goal position
    - Dynamic obstacles with various shapes that move randomly each step (red squares)

    The agent must navigate to the goal while avoiding collisions with moving obstacles.
    Obstacles can have different shapes (L-shapes, squares, lines) and move in random
    directions each step, making the navigation task challenging and dynamic.

    Attributes:
        metadata (dict): Environment metadata including render modes and FPS
        nrow (int): Number of rows in the grid
        ncol (int): Number of columns in the grid  
        max_steps (int): Maximum steps before episode truncation
        num_obstacles (int): Number of dynamic obstacles in the environment
        render_mode (str): Rendering mode ("human" or None)
        dense_reward (bool): Whether to use dense rewards (distance-based) or sparse
        action_space (spaces.Discrete): Action space with 4 discrete actions
        observation_space (spaces.Box): 1-channel observation space (obstacles, agent, goal)
        obstacle_specs (list): Predefined obstacle shapes as relative cell coordinates
    
    """
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, dense_reward, nrow=10, ncol=10, max_step=500,
                 num_obstacles=2, render_mode=None):
        super().__init__()
        self.nrow = nrow
        self.ncol = ncol
        self.max_steps = max_step
        self.num_obstacles = num_obstacles
        self.render_mode = render_mode
        self.dense_reward = dense_reward

        # Actions: 0=up, 1=down, 2=left, 3=right
        self.action_space = spaces.Discrete(4)

        # Observation: 1 channel for  (obstacles, agent, goal)
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(1, nrow, ncol), dtype=np.float32
        )

        # Obstacle shapes
        self.obstacle_specs = [
            {"cells": [(0,0), (0,1), (1,1)]},
            {"cells": [(0,0), (0,1), (1,0), (1,1)]},
            {"cells": [(0,0), (0,1), (0,2)]},
            {"cells": [(0,0), (1,0), (2,0)]}
        ]

        self.agent_pos = None
        self.goal_pos = None
        self.obstacles = []
        self.step_count = 0

        # Initialize rendering attributes
        self._obstacle_patches = []
        self._goal_patch = None
        self._goal_glow = None
        self._goal_inner = None
        self._agent_circle = None
        self._agent_glow = None
        self._fig = None
        self._ax = None
        self._img = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        occupied = set()
        self.obstacles = []

        for _ in range(self.num_obstacles):
            while True:
                shape = random.choice(self.obstacle_specs)
                x = random.randint(0, self.nrow - 1)
                y = random.randint(0, self.ncol - 1)
                abs_cells = {(x + dx, y + dy) for dx, dy in shape["cells"]}
                if (all(0 <= cx < self.nrow and 0 <= cy < self.ncol for cx, cy in abs_cells)
                        and abs_cells.isdisjoint(occupied)):
                    self.obstacles.append({"cells": shape["cells"], "pos": [x, y]})
                    occupied.update(abs_cells)
                    break

        while True:
            ax, ay = random.randint(0, self.nrow - 1), random.randint(0, self.ncol - 1)
            if (ax, ay) not in occupied:
                self.agent_pos = [ax, ay]
                occupied.add((ax, ay))
                break

        while True:
            gx, gy = random.randint(0, self.nrow - 1), random.randint(0, self.ncol - 1)
            if (gx, gy) not in occupied:
                self.goal_pos = [gx, gy]
                occupied.add((gx, gy))
                break

        return self._get_obs(), {}   # obs, info

    def _get_obs(self):
        """Return 1-channel grid: [obstacles, agent, goal]."""
        obs = np.zeros((1, self.nrow, self.ncol), dtype=np.float32)

        for obs_def in self.obstacles:
            ox, oy = obs_def["pos"]
            for dx, dy in obs_def["cells"]:
                obs[0, ox + dx, oy + dy] = 100.0   # obstacles

        obs[0, self.agent_pos[0], self.agent_pos[1]] = 255.0   # agent
        obs[0, self.goal_pos[0], self.goal_pos[1]] = 200     # goal

        return obs/255.0

    def _move_obstacles(self):
        directions = [(0,0), (-1,0), (1,0), (0,-1), (0,1)]
        occupied = set()

        # Mark current occupied cells
        for obs_def in self.obstacles:
            ox, oy = obs_def["pos"]
            for dx, dy in obs_def["cells"]:
                occupied.add((ox + dx, oy + dy))
        occupied.add(tuple(self.agent_pos))
        occupied.add(tuple(self.goal_pos))

        new_obstacles = []
        for obs_def in self.obstacles:
            shape = obs_def["cells"]
            ox, oy = obs_def["pos"]
            current_cells = {(ox + dx, oy + dy) for dx, dy in shape}
            occupied -= current_cells

            dx, dy = random.choice(directions)
            new_pos = (ox + dx, oy + dy)
            new_cells = {(new_pos[0] + sx, new_pos[1] + sy) for sx, sy in shape}

            if (all(0 <= x < self.nrow and 0 <= y < self.ncol for x,y in new_cells)
                    and new_cells.isdisjoint(occupied)):
                new_obstacles.append({"cells": shape, "pos": [new_pos[0], new_pos[1]]})
                occupied.update(new_cells)
            else:
                new_obstacles.append({"cells": shape, "pos": [ox, oy]})
                occupied.update((ox + sx, oy + sy) for sx, sy in shape)

        self.obstacles = new_obstacles

    def step(self, action):
        self.step_count += 1
        terminated, truncated = False, False
        reward = -0.01
        old_dist = np.linalg.norm(np.array(self.agent_pos) - np.array(self.goal_pos))


        x, y = self.agent_pos
        if action == 0 and x > 0:
            x -= 1
        elif action == 1 and x < self.nrow - 1:
            x += 1
        elif action == 2 and y > 0:
            y -= 1
        elif action == 3 and y < self.ncol - 1:
            y += 1
        else:
            reward -= 0.1

        # Check collision at new pos
        collision = False
        for obs_def in self.obstacles:
            ox, oy = obs_def["pos"]
            for dx, dy in obs_def["cells"]:
                if (x, y) == (ox + dx, oy + dy):
                    collision = True
                    reward = -5.0
                    break
            if collision:
                break

        # Update agent
        if not collision:
            self.agent_pos = [x, y]

        # Move obstacles
        self._move_obstacles()

        # Add a small Reward for getting closer to goal
        if self.dense_reward:
            new_dist = np.linalg.norm(np.array([x, y]) - np.array(self.goal_pos))
            reward += 0.2 * (old_dist - new_dist)

        # Collision after obstacles move
        for obs_def in self.obstacles:
            ox, oy = obs_def["pos"]
            for dx, dy in obs_def["cells"]:
                if self.agent_pos == [ox + dx, oy + dy]:
                    reward = -5.0
                    break

        # Check goal
        if self.agent_pos == self.goal_pos:
            reward = 50
            terminated = True

        # Timeout
        if self.step_count >= self.max_steps:
            truncated = True

        return self._get_obs(), reward, terminated, truncated, {}

    def render(self, mode="human"):
        if self.render_mode != "human":
            return

        time.sleep(1 / self.metadata["render_fps"])

        grid = np.zeros((self.nrow, self.ncol), dtype=int)
        ax, ay = self.agent_pos
        cmap = mcolors.ListedColormap(["#0d0d0d"])  # Only background color needed

        if not hasattr(self, "_fig") or self._fig is None:
            self._fig, self._ax = plt.subplots()
            print(f"Created figure: {self._fig is not None}, axes: {self._ax is not None}")
            plt.ion()
            self._img = self._ax.imshow(grid, cmap=cmap, origin="upper", vmin=0, vmax=0)

            # Fixed grid lines - use integer positions for major ticks
            self._ax.set_xticks(np.arange(-0.5, self.ncol, 1), minor=True)
            self._ax.set_yticks(np.arange(-0.5, self.nrow, 1), minor=True)
            self._ax.set_xticks(np.arange(0, self.ncol, 1), minor=False)
            self._ax.set_yticks(np.arange(0, self.nrow, 1), minor=False)

            # Grid styling
            self._ax.grid(which="minor", color="#404040", linestyle='-', linewidth=0.6)
            self._ax.tick_params(which="minor", size=0)
            self._ax.tick_params(which="major", size=0)  # Hide major tick marks
            self._ax.set_facecolor("#0d0d0d")

            # Hide tick labels for cleaner look
            self._ax.set_xticklabels([])
            self._ax.set_yticklabels([])

            # Force square grid cells
            self._ax.set_aspect('equal')

            plt.show()
        else:
            # Safe update - only if _img exists and is not None
            if hasattr(self, "_img") and self._img is not None:
                self._img.set_data(grid)

        # Only proceed with drawing if axes exists
        if self._ax is None:
            print("Warning: _ax is None, skipping rendering")  # Debug
            return

        # Remove old obstacle and goal patches
        if hasattr(self, "_obstacle_patches"):
            for patch in self._obstacle_patches:
                if patch is not None:
                    patch.remove()
        if hasattr(self, "_goal_patch") and self._goal_patch is not None:
            self._goal_patch.remove()
        if hasattr(self, "_goal_glow") and self._goal_glow is not None:
            self._goal_glow.remove()
        if hasattr(self, "_goal_inner") and self._goal_inner is not None:
            self._goal_inner.remove()

        # Draw obstacles as rectangles that fill entire cells
        self._obstacle_patches = []
        for obs_def in self.obstacles:
            ox, oy = obs_def["pos"]
            for dx, dy in obs_def["cells"]:
                x, y = ox + dx, oy + dy
                # Create rectangle that fills the entire cell (from -0.5 to +0.5)
                rect = plt.Rectangle((y - 0.5, x - 0.5), 1.0, 1.0,
                                   facecolor="#ff004d", edgecolor="none")
                self._ax.add_patch(rect)
                self._obstacle_patches.append(rect)

        # Draw goal as retro portal/ring style
        gx, gy = self.goal_pos

        # Outer glow ring (larger, semi-transparent)
        self._goal_glow = plt.Circle((gy, gx), 0.45, facecolor="none",
                                    edgecolor="#00ffcc", linewidth=8, alpha=0.3)
        self._ax.add_patch(self._goal_glow)

        # Main portal ring (thick hollow circle)
        self._goal_patch = plt.Circle((gy, gx), 0.35, facecolor="none",
                                     edgecolor="#00ffcc", linewidth=4)
        self._ax.add_patch(self._goal_patch)

        # Inner sparkle ring (thin bright ring)
        self._goal_inner = plt.Circle((gy, gx), 0.25, facecolor="none",
                                     edgecolor="#ffffff", linewidth=2, alpha=0.8)
        self._ax.add_patch(self._goal_inner)

        # Remove old patches
        if hasattr(self, "_agent_circle") and self._agent_circle is not None:
            self._agent_circle.remove()
        if hasattr(self, "_agent_glow") and self._agent_glow is not None:
            self._agent_glow.remove()

        # Fixed coordinates for proper centering
        # In matplotlib imshow with origin="upper":
        # - x coordinate = column index (ay)
        # - y coordinate = row index (ax)
        circle_x = ay
        circle_y = ax

        # Agent glow - centered in grid cell
        self._agent_glow = plt.Circle((circle_x, circle_y), 0.45, color="#ffcc00", alpha=0.3)
        self._ax.add_patch(self._agent_glow)

        # Agent body - centered in grid cell
        self._agent_circle = plt.Circle((circle_x, circle_y), 0.3, color="#ffcc00")
        self._ax.add_patch(self._agent_circle)

        self._fig.canvas.draw()
        self._fig.canvas.flush_events()
