import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch
import numpy as np
from Utils import (_calculate_state, _is_in_zone, _is_adjacent_to_player,
                  _calculate_potential, _detect_consecutive_vir, position_to_index)

@pytest.fixture
def empty_board():
    return np.zeros((12, 12))

@pytest.fixture
def sample_input_tensor():
    # Create sample input with batch_size=2
    x = torch.zeros((2, 2, 12, 12))
    # Set some pieces for first batch
    x[0, 0, 4, 4] = 1  # Current player
    x[0, 1, 4, 5] = 1  # Opponent
    return x

class TestZoneValidation:
    def test_is_in_zone(self):
        # Test corners (should be False)  
        assert not _is_in_zone(0, 0)
        assert not _is_in_zone(0, 11)
        assert not _is_in_zone(11, 0)
        assert not _is_in_zone(11, 11)
        
        # Test valid positions
        assert _is_in_zone(4, 4)  # Center
        assert _is_in_zone(0, 5)  # Top
        assert _is_in_zone(5, 0)  # Left
        
        # Test out of bounds
        assert not _is_in_zone(-1, 5)
        assert not _is_in_zone(12, 5)

class TestPositionIndexConversion:
    def test_position_to_index(self):
        # Test valid positions
        assert position_to_index(0, 4) == 0  # Top section start
        assert position_to_index(4, 0) == 16  # Left section start
        assert position_to_index(4, 4) == 32  # Center section start
        assert position_to_index(4, 8) == 48  # Right section start
        assert position_to_index(8, 4) == 64  # Bottom section start
        
        # Test invalid positions
        assert position_to_index(0, 0) is None  # Corner
        assert position_to_index(-1, 4) is None  # Out of bounds

class TestPatternDetection:
    def test_detect_consecutive_horizontal(self, empty_board):
        # Test horizontal pattern
        board = empty_board.copy()
        board[4, 4:7] = 1
        assert _detect_consecutive_vir(1, 3, board) == 1
        assert _detect_consecutive_vir(1, 2, board) == 2
        
    def test_detect_consecutive_vertical(self, empty_board):
        # Test vertical pattern
        board = empty_board.copy()
        board[4:7, 4] = 1
        assert _detect_consecutive_vir(1, 3, board) == 1
        
    def test_detect_consecutive_diagonal(self, empty_board):
        # Test diagonal pattern
        board = empty_board.copy()
        for i in range(3):
            board[4+i, 4+i] = 1
        assert _detect_consecutive_vir(1, 3, board) == 1

class TestPotentialCalculation:
    def test_calculate_potential(self, empty_board):
        board = empty_board.copy()
        # Set up two pieces
        board[4, 4] = board[4, 5] = 1
        
        # Test empty adjacent position
        potential = _calculate_potential(board, 4, 6)
        assert potential > 0  # Should have positive potential
        
        # Test occupied position
        potential = _calculate_potential(board, 4, 4)
        assert potential == 0  # Should have zero potential

class TestStateCalculation:
    def test_calculate_state(self, sample_input_tensor):
        result = _calculate_state(sample_input_tensor)
        assert result.shape == (2, 80)  # Batch size 2, 80 positions
        assert torch.all(result >= 0)  # All scores should be non-negative

    def test_adjacent_to_player(self, empty_board):
        board = empty_board.copy()
        board[4, 4] = 1  # Place piece
        
        # Test adjacent positions
        assert _is_adjacent_to_player(board, 4, 5)
        assert _is_adjacent_to_player(board, 5, 5)
        
        # Test non-adjacent position
        assert not _is_adjacent_to_player(board, 7, 7)