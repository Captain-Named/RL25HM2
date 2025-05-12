import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque, defaultdict
import random
import math
import os
from torch.nn import functional as F
from tqdm import tqdm
# Set GPU device
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Limit GPU memory usage
# PyTorch CUDA settings for memory limitation
if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.25, device=device)  # Limit GPU memory usage to 25%
    print(f"Using device: {device}")

# Hyperparameter Configuration
class Config:
    board_size = 3  # Size of the game board (3x3)
    num_actions = 9  # Number of possible actions (9 positions on 3x3 board)
    success_prob = 1  # Probability of successful move placement
    
    # Model Parameters
    num_res_blocks = 1  # Number of residual blocks in neural network
    num_filters = 64  # Number of filters in convolutional layers
    policy_filters = 32  # Number of filters in policy head
    value_filters = 32  # Number of filters in value head
    
    # Training Parameters  
    batch_size = 64  # Batch size for training iterations
    learning_rate = 0.001  # Learning rate for model updates
    weight_decay = 1e-4  # L2 regularization coefficient
    num_games = 10  # Number of self-play games per iteration
    num_iter = 1000  # Number of training iterations
    
    # MCTS Parameters
    mcts_simulations = 100  # Number of MCTS simulations per move
    c_puct = 1.  # Exploration-exploitation balance parameter
    dirichlet_alpha = 0.3  # Dirichlet noise parameter for exploration
    temp_threshold = 15  # Temperature threshold for action selection
    is_total_N = False  # Whether to use total visit count in UCB formula
    temperature = 0  # Temperature parameter for move probability distribution

# TicTacToe Game Implementation
class TicTacToe:
    def __init__(self, config):
        self.board = np.zeros((3, 3), dtype=int)  # Game board state
        self.current_player = 1  # Current player (1 for X, -1 for O)
        self.winner = 0  # Game winner (0 for no winner)
        self.move_history = []  # History of selected moves
        self.execute_move_history = []  # History of executed moves
        self.config = config

    def dcopy(self):
        """Creates a deep copy of the game state"""
        new_game = TicTacToe(self.config)
        new_game.board = np.copy(self.board)
        new_game.current_player = self.current_player
        new_game.winner = self.winner
        new_game.move_history = self.move_history.copy()
        return new_game

    def get_legal_moves(self):
        """Returns list of available moves on the board"""
        return [(i, j) for i in range(3) for j in range(3) if self.board[i][j] == 0]

    def execute_move(self, move):
        """Executes a move with success probability and adjacent position fallback"""
        i, j = move
        if self.board[i][j] != 0:
            pass
            #print("Invalid move position - programming error")

        # Attempt move with success probability
        if random.random() < self.config.success_prob:
            self.board[i][j] = self.current_player
            execute_move = move
        else:
            # Try adjacent position if primary move fails
            adjacent = []
            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:
                    if di == 0 and dj == 0:
                        continue
                    ni, nj = i + di, j + dj
                    adjacent.append((ni, nj))

            if adjacent:
                ni, nj = random.choice(adjacent)
                if 0 <= ni < 3 and 0 <= nj < 3 and self.board[ni][nj] == 0:
                    self.board[ni][nj] = self.current_player
                    execute_move = (ni, nj)
                else:
                    execute_move = None  # Skip turn if no valid adjacent position
                    return False
            else:
                pass
                #print("No adjacent positions found - programming error")

        self.move_history.append(move)
        self.execute_move_history.append(execute_move)
        self.winner = self._check_winner()
        self.current_player *= -1  # Switch players
        return True
    
    def _check_winner(self):
        # Check rows for winner
        for i in range(3):
            if abs(sum(self.board[i])) == 3 and self.board[i][0] != 0:
                return self.board[i][0]  # Return winner's piece (+1 or -1)
    
        # Check columns for winner
        for j in range(3):
            if abs(sum(self.board[:, j])) == 3 and self.board[0][j] != 0:
                return self.board[0][j]
    
        # Check diagonals for winner
        diag1 = [self.board[i][i] for i in range(3)]  # Main diagonal
        diag2 = [self.board[i][2-i] for i in range(3)]  # Anti-diagonal
    
        if abs(sum(diag1)) == 3 and diag1[0] != 0:
            return diag1[0]
        if abs(sum(diag2)) == 3 and diag2[0] != 0:
            return diag2[0]
    
        # Check for draw - if board is full and no winner
        if len(self.get_legal_moves()) == 0:
            return 0
    
        return 0  # No winner yet
    
    def get_state(self):
        """Returns a copy of the current board state"""
        return np.copy(self.board)
    
    def is_over(self):
        """Checks if game is over (has winner or board is full)"""
        return self.winner != 0 or len(self.get_legal_moves()) == 0
    
    # State encoding for neural network input
    def game_to_state(game):  # game is TicTacToe instance
        """
        Converts game state to 3-channel representation:
        - Channel 0: Current player's pieces
        - Channel 1: Opponent's pieces
        - Channel 2: Current player indicator (all 1's or -1's)
        """
        state = np.zeros((3, 3, 3), dtype=np.float32)
    
        # Encode player positions
        for i in range(3):
            for j in range(3):
                if game.board[i][j] == 1:
                    state[0][i][j] = 1  # Current player's pieces
                elif game.board[i][j] == -1:
                    state[1][i][j] = 1  # Opponent's pieces
    
        # Set current player indicator plane
        state[2] = np.ones((3, 3)) if game.current_player == 1 else np.full((3, 3), -1)
        
        # Add batch dimension and move to device
        return torch.from_numpy(state).unsqueeze(0).to(device)  # Shape: (1, 3, 3, 3)

# ==========================================================================

    def _check_winner(self):
        # Check rows for winner
        for i in range(3):
            if abs(sum(self.board[i])) == 3 and self.board[i][0] != 0:
                return self.board[i][0]  # Return winner's piece (+1 or -1)

        # Check columns for winner
        for j in range(3):
            if abs(sum(self.board[:, j])) == 3 and self.board[0][j] != 0:
                return self.board[0][j]

        # Check diagonals for winner
        diag1 = [self.board[i][i] for i in range(3)]  # Main diagonal
        diag2 = [self.board[i][2-i] for i in range(3)]  # Anti-diagonal

        if abs(sum(diag1)) == 3 and diag1[0] != 0:
            return diag1[0]
        if abs(sum(diag2)) == 3 and diag2[0] != 0:
            return diag2[0]

        # Check for draw - if board is full and no winner
        if len(self.get_legal_moves()) == 0:
            return 0

        return 0  # No winner yet

    def get_state(self):
        """Returns a copy of the current board state"""
        return np.copy(self.board)

    def is_over(self):
        """Checks if game is over (has winner or board is full)"""
        return self.winner != 0 or len(self.get_legal_moves()) == 0

# State encoding for neural network input
def game_to_state(game):  # game is TicTacToe instance
    """
    Converts game state to 3-channel representation:
    - Channel 0: Current player's pieces
    - Channel 1: Opponent's pieces
    - Channel 2: Current player indicator (all 1's or -1's)
    """
    state = np.zeros((3, 3, 3), dtype=np.float32)
    # Encode player positions
    for i in range(3):
        for j in range(3):
            if game.board[i][j] == 1:
                state[0][i][j] = 1  # Current player's pieces
            elif game.board[i][j] == -1:
                state[1][i][j] = 1  # Opponent's pieces
    # Set current player indicator plane
    state[2] = np.ones((3, 3)) if game.current_player == 1 else np.full((3, 3), -1)
    
    # Add batch dimension and move to device
    return torch.from_numpy(state).unsqueeze(0).to(device)  # Shape: (1, 3, 3, 3)
# ==========================================================================
class AlphaZeroNet(nn.Module):
    """Neural network architecture for AlphaZero, implementing both policy and value predictions"""
    
    def __init__(self, config):
        """
        Initialize network architecture
        
        Args:
            config: Configuration object containing network parameters
        """
        super(AlphaZeroNet, self).__init__()
        self.board_size = config.board_size
        self.num_actions = config.num_actions

        # Input convolutional layer
        # 3 channels: current player pieces, opponent pieces, current player indicator
        self.conv1 = nn.Conv2d(3, config.num_filters, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(config.num_filters)

        # Stack of residual blocks for deep feature extraction
        self.res_blocks = nn.Sequential(
            *[ResBlock(config.num_filters) for _ in range(config.num_res_blocks)]
        )

        # Policy head - predicts move probabilities
        self.policy_conv = nn.Conv2d(config.num_filters, config.policy_filters, kernel_size=1)  # 1x1 conv reduces dimensionality
        self.policy_bn = nn.BatchNorm2d(config.policy_filters)
        self.policy_fc = nn.Linear(config.policy_filters * self.board_size * self.board_size, self.num_actions)

        # Value head - predicts game outcome
        self.value_conv = nn.Conv2d(config.num_filters, config.value_filters, kernel_size=1)  # 1x1 conv reduces dimensionality
        self.value_bn = nn.BatchNorm2d(config.value_filters)
        self.value_fc1 = nn.Linear(config.value_filters * self.board_size * self.board_size, 256)
        self.value_fc2 = nn.Linear(256, 1)  # Single scalar output in [-1,1]

    def forward(self, x):
        """
        Forward pass through network
        
        Args:
            x: Input tensor of shape (batch_size, 3, board_size, board_size)
            
        Returns:
            policy: Action probabilities of shape (batch_size, num_actions)
            value: Game outcome prediction of shape (batch_size, 1)
        """
        # Common layers
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.res_blocks(x)

        # Policy head
        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.view(policy.size(0), -1)  # Flatten: (batch_size, channels*height*width)
        policy = F.softmax(self.policy_fc(policy), dim=1)  # Output move probabilities

        # Value head  
        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.view(value.size(0), -1)  # Flatten
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))  # Output in [-1,1] range
        
        return policy, value

class ResBlock(nn.Module):
    """Residual block with two convolutional layers and skip connection"""
    
    def __init__(self, channels):
        """
        Initialize residual block
        
        Args:
            channels: Number of input/output channels
        """
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1) 
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        """Forward pass with skip connection"""
        residual = x  # Store input for skip connection
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual  # Add skip connection 
        return torch.relu(x)
class MCTSNode:
    """
    Monte Carlo Tree Search Node representing either:
    - A game state (root node)
    - A possible action from parent state (non-root nodes)
    """
    def __init__(self, config, parent=None, prior=0):
        # Dictionary of child nodes, only used for root (state) nodes
        # Key: (i,j) move tuple, Value: MCTSNode instance
        self.children = {}  
        
        # Node statistics
        self.Q = 0  # Average action value
        self.N = 0  # Visit count
        self.P = prior  # Prior probability from policy network
        self.W = 0  # Win count for simulating player through this action
        self.Vs = []  # Value estimates (0.5 * (model_value + final_reward))
        
        # Configuration parameters
        self.config = config

    def select_child(self, c_puct):
        """
        Select child node using UCB formula:
        Q + c_puct * P * sqrt(total_N) / (1 + N)
        
        Args:
            c_puct: Exploration constant balancing exploitation vs exploration
        Returns:
            Selected move as (i,j) tuple
        """
        sq_total_N = np.sqrt(sum(child.N for child in self.children.values())) if self.config.is_total_N else 1
        
        return max(self.children.items(),
                  key=lambda item: item[1].Q + c_puct * item[1].P * sq_total_N / (1 + item[1].N))[0]

    def expand(self, moves, probs):
        """
        Expand node by adding child nodes for each valid move
        
        Args:
            moves: List of valid moves as (i,j) tuples
            probs: Prior probabilities for each move from policy network
        """
        for move, prob in zip(moves, probs):
            if move not in self.children:
                self.children[move] = MCTSNode(self.config, prior=prob)

    def get_pi(self):
        """
        Get move probabilities based on visit counts
        
        Returns:
            Normalized probability distribution over valid moves
        """
        visits = np.array([child.N for child in self.children.values()])
        if self.config.temperature == 0:
            return visits / np.sum(visits)
        else:
            # Apply temperature smoothing to visit counts
            visits = visits ** (1/self.config.temperature)
            return visits / np.sum(visits)
 

    def update_child(self, sim_result, choice, value):
        """
        Update statistics of child node after simulation
        
        Args:
            sim_result: Simulation result (-1: loss, 0: draw, 1: win)
            choice: Selected move as (i,j) tuple 
            value: Value prediction from neural network
        """
        child = self.children[choice]
        child.N += 1  # Increment visit count
        
        # Update win count if simulation resulted in win
        if sim_result == 1:
            child.W += 1
    
        # Calculate mixed value (average of model prediction and actual result)
        mixed_value = 0.5 * (value + sim_result)
        child.Vs.append(mixed_value)
    
        # Update Q value (mean of historical mixed values)
        child.Q = np.mean(child.Vs)
    
class MCTS:
    """Monte Carlo Tree Search implementation for AlphaZero"""
    
    def __init__(self, model, config):
        self.model = model  # Neural network model
        self.config = config

    def run(self, game: TicTacToe) -> MCTSNode:
        """
        Run MCTS simulations from current game state
        
        Args:
            game: Current game state
        Returns:
            Root node containing simulation statistics
        """
        # Create root node representing current state
        root = MCTSNode(self.config)

        # Get legal moves and add Dirichlet noise for exploration
        legal_moves = game.get_legal_moves()
        noise = np.random.dirichlet([self.config.dirichlet_alpha] * len(legal_moves)) if self.config.dirichlet_alpha else None

        # Initial expansion of root node
        if not root.children:
            with torch.no_grad():
                state_tensor = game_to_state(game)
                probs, _ = self.model(state_tensor)
                probs = probs.cpu().numpy().flatten()

            # Create move mask and normalize probabilities
            move_indices = [i * 3 + j for (i, j) in legal_moves]
            mask = np.zeros(9)
            mask[move_indices] = 1
            probs = probs * mask
            probs /= np.sum(probs)

            # Apply Dirichlet noise
            if noise is not None:
                probs = 0.75 * probs[move_indices] + 0.25 * noise
            probs /= np.sum(probs)
            root.expand(legal_moves, probs)

        # Run simulations
        for _ in range(self.config.mcts_simulations):
            sim_result, choice, move_value = self.simulate(game.dcopy(), root)
            root.update_child(sim_result, choice, move_value)

        return root

    def simulate(self, game: TicTacToe, node: MCTSNode):
        """
        Run single MCTS simulation
        
        Args:
            game: Copy of current game state
            node: Root node of search tree
        Returns:
            sim_result: Simulation result (-1,0,1)
            selected_move: Move chosen for this simulation
            move_value: Value prediction for chosen move
        """
        current = node
        simulating_player = game.current_player
        selected_move = None
        move_value = 0
        sim_result = None

        # Select action using UCB formula
        move = current.select_child(self.config.c_puct)
        selected_move = move
        game.execute_move(move)

        step_count = 0
        # Play out game using policy network
        while not game.is_over():
            step_count += 1
            with torch.no_grad():
                state_tensor = game_to_state(game)
                probs, value = self.model(state_tensor)
                probs = probs.cpu().numpy().flatten()
            
            # Store value prediction for selected move
            if step_count == 2:
                move_value = value.item()

            # Select moves based on policy network
            legal_moves = game.get_legal_moves()
            move_indices = [i * 3 + j for (i, j) in legal_moves]
            mask = np.zeros(9)
            mask[move_indices] = 1
            probs = probs * mask
            probs /= np.sum(probs)

            move = random.choices(range(9), weights=probs.tolist())[0]
            move = (move // 3, move % 3)
            game.execute_move(move)

        # Determine simulation result
        if simulating_player == game.winner:
            sim_result = 1
        elif game.winner == 0:
            sim_result = 0
        else:
            sim_result = -1

        return sim_result, selected_move, move_value

class Trainer:
    """AlphaZero training implementation with self-play and model updates"""
    
    def __init__(self, config: Config):
        """Initialize trainer with model, optimizer and replay memory"""
        self.config = config
        self.model = AlphaZeroNet(config).to(device)  # Neural network model
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )  # Adam optimizer with L2 regularization
        self.memory = deque(maxlen=10000)  # Replay buffer for training examples

    def self_play(self, num_games=10):
        """
        Generate training data through self-play games
        
        Format of generated training data:
        [
            (state_tensor, pi_tensor, z),  # Player X's first move
            (state_tensor, pi_tensor, z),  # Player O's first move
            ...
        ]
        where:
        - state_tensor: Board state representation
        - pi_tensor: Move probabilities from MCTS
        - z: Final game outcome from current player's perspective
        """
        for _ in range(num_games):
            game = TicTacToe(self.config)  # New game instance
            mcts = MCTS(self.model, self.config)  # MCTS instance for this game
            memory_this_selfplay = []  # Store examples from current game
            
            # Play until game ends
            while not game.is_over():
                # Run MCTS from current state
                root = mcts.run(game)
                
                # Temperature parameter for move selection
                temp = 1 if len(self.memory) < self.config.temp_threshold else 0.1
                
                # Get move probabilities from MCTS visit counts
                pi = root.get_pi()
                
                # Store game state
                state_tensor = game_to_state(game)
                
                # Convert move probabilities to tensor format
                pi_tensor = torch.zeros(9, device=device)
                for (i, j), prob in zip(root.children.keys(), pi):
                    pi_tensor[i * 3 + j] = prob
                
                # Store state, policy and player info
                memory_this_selfplay.append((state_tensor, pi_tensor, game.current_player))
                
                # Select and execute move
                move = random.choices(list(root.children.keys()), weights=pi)[0]
                game.execute_move(move)
            
            # Process final game outcome
            winner = game.winner
            for i in range(len(memory_this_selfplay)):
                state_tensor, pi_tensor, player = memory_this_selfplay[i]
                # Set reward from current player's perspective
                if player == winner:
                    z = 1
                elif winner == 0:
                    z = 0  
                else:
                    z = -1
                memory_this_selfplay[i] = (state_tensor.cpu(), pi_tensor, z)
                
            # Add game examples to replay buffer    
            self.memory.extend(memory_this_selfplay)

    def train(self, epochs=1):
        """
        Train model on collected examples
        
        Returns:
            loss: Total loss
            loss_pi: Policy head loss
            loss_v: Value head loss
        """
        if len(self.memory) < self.config.batch_size:
            return None, None, None

        # Sample training batch
        batch = random.sample(self.memory, self.config.batch_size)
        states, pis, vs = zip(*batch)

        # Prepare tensors
        states = torch.cat(states).to(device)
        target_pis = torch.stack(pis).to(device)
        target_vs = torch.FloatTensor(vs).unsqueeze(1).to(device)

        # Forward pass
        self.optimizer.zero_grad()
        pred_pis, pred_vs = self.model(states)

        # Calculate losses
        loss_pi = -torch.mean(torch.sum(target_pis * torch.log(pred_pis + 1e-10), 1))
        loss_v = torch.mean((target_vs - pred_vs) ** 2)
        loss = loss_pi + loss_v

        # Update model
        loss.backward()
        self.optimizer.step()

        return loss.item(), loss_pi.item(), loss_v.item()

# 主训练循环
if __name__ == "__main__":
    config = Config()
    trainer = Trainer(config)  # 初始化训练器，传入配置参数

    try:
        for iteration in tqdm(range(config.num_iter), desc="Iteration"):  # 进行5次迭代训练
            num_games = config.num_games
            print(f"Iteration {iteration + 1} in {config.num_iter}. We'll conduct {num_games} games./ i.e. self-play {num_games} times.")  # 打印当前迭代次数

            trainer.memory = deque(maxlen=10000) # @ Iter 2,3,···, the model of trainer is updated / we need to clear the memory and generate new data with updated model
            # 自对弈
            trainer.self_play(num_games=num_games)  # 进行5局自对弈，生成训练数据

            # 训练
            loss, loss_pi, loss_v = trainer.train()  # 使用自对弈生成的数据进行模型训练
            if loss:
                print(f"Loss: {loss:.4f}")
                print(f"Loss_p: {loss_pi:.4f}")
                print(f"Loss_v: {loss_v:.4f}")
            else:
                print("Skipping training (not enough data)")  # 打印损失值，如果数据不足则跳过训练

            # ?? 要不要继续trainer.self_play(num_games=5)进行这一层次的迭代？注意self.memory需要清空？
            # 保存模型
            if (iteration + 1) % 5 == 0:  # 每5次迭代保存一次模型
                torch.save(trainer.model.state_dict(), f"tictactoe_model_{iteration+1}.pth")  # 保存模型参数到文件

    except KeyboardInterrupt:  # 捕获键盘中断
        print("Training interrupted")  # 打印中断信息

    finally:
        torch.save(trainer.model.state_dict(), "tictactoe_3plus3.pth")  # 无论是否中断，最终保存模型参数