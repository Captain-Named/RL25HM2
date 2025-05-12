import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch
import numpy as np
from play_the_game import RandomTicTacToeAgent

@pytest.fixture
def agent():
    """Create test agent"""
    return RandomTicTacToeAgent()

class TestRandomTicTacToeAgent:
    def test_initialization(self, agent):
        """Test agent initialization"""
        assert agent.env is not None
        assert agent.policy_net is not None
        assert agent.actor_module is not None
        assert agent.stats == {
            'ppo_wins': 0, 
            'random_wins': 0, 
            'draws': 0, 
            'total_games': 0
        }

    def test_board_reset(self, agent):
        """Test board reset"""
        agent.env._reset()
        board = agent.env.board
        assert np.all(board == 0)
        assert agent.env.current_player == 1

    def test_ppo_move_generation(self, agent):
        """Test PPO agent move generation"""
        agent.env._reset()
        i, j = agent.get_ppo_agent_move()
        assert 0 <= i < 12
        assert 0 <= j < 12
        assert agent.env._is_in_zone(i, j)

    def test_random_move_generation(self, agent):
        """Test random agent move generation"""
        agent.env._reset()
        i, j = agent.get_random_agent_move()
        assert 0 <= i < 12
        assert 0 <= j < 12
        assert agent.env._is_in_zone(i, j)

    def test_game_play(self, agent):
        """Test single game play"""
        winner = agent.play_game(ppo_first=True, visualize=False)
        assert winner in [1, -1, 0]  # Valid winner values
        assert agent.stats['total_games'] == 1
        assert sum([
            agent.stats['ppo_wins'],
            agent.stats['random_wins'],
            agent.stats['draws']
        ]) == 1

    def test_evaluation(self, agent):
        """Test batch evaluation"""
        num_games = 10
        agent.evaluate(num_games=num_games)
        
        # Check statistics
        assert agent.stats['total_games'] == num_games
        assert sum([
            agent.stats['ppo_wins'],
            agent.stats['random_wins'],
            agent.stats['draws']
        ]) == num_games

    def test_valid_moves(self, agent):
        """Test move validation"""
        agent.env._reset()
        
        # Make first move
        i, j = agent.get_ppo_agent_move()
        agent.env.board[i][j] = 1
        
        # Position should no longer be available
        legal_moves = agent.env.get_legal_moves()
        assert (i, j) not in legal_moves

    def test_game_outcome_detection(self, agent):
        """Test win/draw detection"""
        agent.env._reset()
        
        # Create winning line for PPO
        for i in range(4):
            agent.env.board[4+i][4] = 1
            
        winner = agent.env._check_winner()
        assert winner == 1  # PPO wins

    def test_statistics_tracking(self, agent):
        """Test statistics update"""
        initial_stats = agent.stats.copy()
        
        # Play multiple games
        for _ in range(5):
            agent.play_game(ppo_first=True, visualize=False)
            
        assert agent.stats['total_games'] == initial_stats['total_games'] + 5
        assert sum([
            agent.stats['ppo_wins'],
            agent.stats['random_wins'],
            agent.stats['draws']
        ]) == agent.stats['total_games']