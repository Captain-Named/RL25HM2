import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch
import numpy as np
from train_PPO import TicTacToeEnv, make_env
from Networks import SimplePolicyNetwork, SimpleValueNetwork
from Config import Config

# Fixtures
@pytest.fixture
def env():
    """Create test environment"""
    return TicTacToeEnv()

@pytest.fixture
def config():
    """Get config object"""
    return Config()

@pytest.fixture
def models(config):
    """Initialize policy and value networks"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    policy_net = SimplePolicyNetwork(config).to(device)
    value_net = SimpleValueNetwork(config).to(device)
    return policy_net, value_net

# Environment Tests
class TestTicTacToeEnv:
    def test_init(self, env):
        """Test environment initialization"""
        # Need to reset first to initialize board
        env.reset()
        assert env.board.shape == (12, 12)

    def test_reset(self, env):
        """Test environment reset"""
        td = env.reset()
        assert "observation" in td
        assert "action_mask" in td
        assert "step_count" in td
        assert "done" in td
        assert td["step_count"].item() == 0
        assert not td["done"].item()

    def test_action_mask(self, env):
        """Test action masking"""
        td = env.reset()
        mask = td["action_mask"]
        assert mask.shape == (80,)
        assert mask.dtype == torch.bool
        # Should have exactly 80 valid positions initially
        assert mask.sum().item() == 80

    def test_move_validation(self, env):
        """Test move validation"""
        td = env.reset()
        action = 0  # First valid position
        # Should not raise error
        try:
            env._step({"action": torch.tensor(action)})
        except Exception as e:
            pytest.fail(f"Valid move raised error: {e}")

        # Test invalid move
        with pytest.raises(Exception):
            env._step({"action": torch.tensor(action)})  # Same position

    def test_win_detection(self, env):
        """Test win detection logic"""
        # Reset first
        env.reset()
        # Create winning pattern
        env.board[4:8, 4] = 1
        winner = env._check_winner()
        assert winner == 1

# Training Component Tests 
class TestTrainingComponents:
    def test_model_output_shapes(self, models, env):
        """Test network output shapes"""
        policy_net, value_net = models
        td = env.reset()
        obs = td["observation"].unsqueeze(0)

        with torch.no_grad():
            # Test policy network
            policy_out = policy_net(obs)
            assert "logits" in policy_out
            assert policy_out["logits"].shape == (1, 80)

            # Test value network  
            value_out = value_net(obs)
            assert value_out["state_value"].shape == (1, 1)

    def test_reward_calculation(self, env):
        """Test pattern reward calculation"""
        # Reset first
        env.reset()
        # Setup 3-in-a-row
        env.board[4:7, 4] = 1
        reward = env.calculate_pattern_reward(1)
        assert reward > 0

    def test_action_conversion(self, env):
        """Test position/index conversion"""
        # Test center position
        i, j = 4, 4  # Center position
        idx = env.position_to_index(i, j)
        assert idx is not None
        i2, j2 = env.index_to_position(idx)
        assert (i, j) == (i2, j2)

# Integration Tests
def test_training_step(env, models):
    """Test full training step"""
    policy_net, value_net = models
    td = env.reset()

    # Forward passes
    with torch.no_grad():
        obs = td["observation"].unsqueeze(0)
        policy_out = policy_net(obs)
        value_out = value_net(obs)

        # Take step
        action = torch.argmax(policy_out["logits"])
        next_td = env._step({"action": action})

        assert "reward" in next_td
        assert "done" in next_td
        assert "observation" in next_td