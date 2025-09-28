'''Configuration dataclass for PPO Agent'''
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple


@dataclass
class PPOConfig:
    '''Configuration for PPO Agent'''
    # train, test or record
    mode: str

    # Environment configuration
    environment_num_channels: int = 3
    shape: Tuple[int, int] = (10, 10)
    num_obstacles: int = 3
    dense_rew: bool = False

    # Hyperparametri training
    lr: float = 3e-4
    gamma: float = 0.99
    lam: float = 0.95
    train_steps: int = 500000
    rollout_size: int = 16384
    batch_size: int = 64
    num_epochs: int = 4
    eps_clip: float = 0.1
    critic_loss_clipping: bool = True
    target_kl: Optional[float] = 0.01
    small_model: bool = True

    # Checkpointing e logging
    checkpoint_path: Optional[Path] = None
    results_dir: Path = Path("Autonomous_Projects/Results")
    save_interval: int = 10000

    # Testing e recording
    test_episodes: int = 10
    record_episodes: int = 3
