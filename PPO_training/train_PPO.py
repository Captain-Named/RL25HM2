import torch
import numpy as np
import random
import os
import matplotlib.pyplot as plt
from collections import defaultdict
import time

# TorchRL imports
from tensordict.nn import TensorDictModule, ProbabilisticTensorDictModule, ProbabilisticTensorDictSequential
from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.envs import EnvBase, EnvCreator
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from torchrl.modules import ValueOperator, MaskedCategorical
from torch.distributions import Categorical as cc   # 
from torchrl.envs.utils import TensorDict
from torchrl.data import Composite, Bounded, Categorical
from torchrl.envs.utils import check_env_specs, ExplorationType, set_exploration_type
from tqdm import tqdm
from Utils import *
from Config import Config
from Networks import SimplePolicyNetwork, SimpleValueNetwork

# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# Set GPU device
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


config = Config()
# Create policy and value networks
policy_net = SimplePolicyNetwork(config).to(device)
value_net = SimpleValueNetwork(config).to(device)


class IllegalMoveError(Exception):
    """Custom exception for illegal moves in Tic Tac Toe game."""
    pass

class TicTacToeEnv(EnvBase):
    def __init__(self):
        # 删除 super().__init__ 中的 observation_spec 参数
        super().__init__(device=device)

        # 独立定义观察/动作空间规范
        self.observation_spec = Composite(
            observation=Bounded(
                low=-1,
                high=1,
                shape=(2, 12, 12), 
                dtype=torch.float32,
                device=device
            ),  
            # 添加动作掩码规范
            action_mask=Bounded(
                low=0,
                high=1,
                shape=(80,),
                dtype=torch.bool,
                device=device
            ),
            # 添加步数计数规范
            step_count=Bounded(
                low=1.0,
                high=1000000, #三子棋来说可以设置为100
                shape=(1,),
                dtype=torch.float64,
                device=device
            ),
        )
        self.policy_net = policy_net
        self.action_spec = Categorical(
            n=80,  # 9个可选位置
            shape=(1,), 
            dtype=torch.int64,
            device=device
        )
        
        self.reward_spec = Bounded(
            low=-1.0,
            high=1.0,
            shape=(1,),
            dtype=torch.float32,
            device=device
        )

        # 定义十字形棋盘的有效区域
        self.winning_combinations = self._get_winning_combinations()  # 获取所有可能的获胜组合
 
    def index_to_position(self, index: int) -> tuple:
        """
        Convert index (0-79) to board position (i,j) for cross-shaped board.
        Board consists of 5 4x4 sub-boards:
        - Top (0-15): rows 0-3, cols 4-7
        - Left (16-31): rows 4-7, cols 0-3
        - Center (32-47): rows 4-7, cols 4-7 (excluding corners)
        - Right (48-63): rows 4-7, cols 8-11
        - Bottom (64-79): rows 8-11, cols 4-7

        Args:
            index: Integer from 0 to 79
        Returns:
            tuple: (i,j) board coordinates
        """
        if not 0 <= index < 80:
            return None

        # Top board (0-15)
        if index < 16:
            i = index // 4
            j = 4 + (index % 4)
            return (i, j)

        # Left board (16-31)
        elif index < 32:
            local_idx = index - 16
            i = 4 + (local_idx // 4)
            j = local_idx % 4
            return (i, j)

        # Center board (32-47)
        elif index < 48:
            local_idx = index - 32
            i = 4 + (local_idx // 4)
            j = 4 + (local_idx % 4)
            return (i, j)

        # Right board (48-63)
        elif index < 64:
            local_idx = index - 48
            i = 4 + (local_idx // 4)
            j = 8 + (local_idx % 4)
            return (i, j)

        # Bottom board (64-79)
        else:
            local_idx = index - 64
            i = 8 + (local_idx // 4)
            j = 4 + (local_idx % 4)
            return (i, j)

    def position_to_index(self, i: int, j: int) -> int:
        """
        Convert board position (i,j) to action index (0-79).
        
        Args:
            i: row index (0-11)
            j: column index (0-11)
        Returns:
            int: action index (0-79) or None if invalid position
        """
        # Validate input range
        if not (0 <= i < 12 and 0 <= j < 12):
            return None
            
        # Top board (rows 0-3, cols 4-7)
        if 0 <= i < 4 and 4 <= j < 8:
            return i * 4 + (j - 4)
            
        # Left board (rows 4-7, cols 0-3)
        elif 4 <= i < 8 and 0 <= j < 4:
            return 16 + ((i - 4) * 4 + j)
            
        # Center board (rows 4-7, cols 4-7)
        elif 4 <= i < 8 and 4 <= j < 8:
            return 32 + ((i - 4) * 4 + (j - 4))
            
        # Right board (rows 4-7, cols 8-11)
        elif 4 <= i < 8 and 8 <= j < 12:
            return 48 + ((i - 4) * 4 + (j - 8))
            
        # Bottom board (rows 8-11, cols 4-7)
        elif 8 <= i < 12 and 4 <= j < 8:
            return 64 + ((i - 8) * 4 + (j - 4))
            
        # Invalid position
        return None

    # Example usage:
    # pos = index_to_position(50)  # Convert index 50 to board position
    # valid_positions = get_valid_positions()  # Get all valid positions   

    def _is_in_zone(self, ni, nj) -> bool:
        #在十字形内 就是要既处于棋盘又不处于四个角的位置
        return not((ni<4 and nj<4)or(ni<4 and nj>7)or(ni>7 and nj<4)or(ni>7 and nj>7)) and  (0 <= ni < 12 and 0 <= nj < 12)

    def _reset(self, tensordict=None, **kwargs):
        self.board = np.zeros((12, 12), dtype=int)
        self.current_player = 1  # 1 for X, -1 for O
        self.winner = None
        self.move_history = []
        self.execute_move_history = []
        self.steps = 0  # ..
        self.config = Config()
        
        if random.random() < 0:  # 改为后手    这里在config里加一个
            #这里写的有问题 但不管
            self.policy_net.eval()  # 设置策略网络为评估模式
            with torch.no_grad():
                # 生成随机动作
                action = self.policy_net(self._get_obs().unsqueeze(0).to(device))["logits"].argmax(dim=-1).item()
                i, j = self.index_to_position(action)
                self.board[i][j] = -self.current_player
            self.policy_net.train()  # 恢复策略网络为训练模式

        # Return initial state as TensorDict
        obs = self._get_obs() #已改
        return TensorDict(
            {"observation": obs, "action_mask": self._get_action_mask(), "step_count": torch.tensor([0.0], device=device), "done": torch.tensor(False, device=device),},
            batch_size=[]
        )
    #_step方法在原来的envBase里面是一个抽象方法 这里使用它的同名函数就相当于实现了它
    def _step(self, tensordict):
    
        """Execute one environment step in the game.
        
        The step consists of:
        1. Player Move Phase:
            - Process player's action (0-79)
            - Apply success rate mechanic:
                * success_rate chance to place at chosen position
                * (1-success_rate) chance to place in adjacent position
                * Records both intended and executed moves
        
        2. Win Check Phase:
            - Check if player won (reward +1.0)
            - Check for draw (reward 0.0)
            
        3. Opponent Move Phase (if game continues):
            - Use policy network in eval mode
            - Get opponent observation (inverted board state)
            - Sample action from masked policy logits
            - Execute opponent move
            - Handle sampling failures with random valid moves
            
        4. Final State Check:
            - Check win/loss/draw after opponent move
            - Calculate pattern-based rewards
            - Scale intermediate rewards by 0.1
            
        Args:
            tensordict: TensorDict containing 'action' key with move index
            
        Returns:
            TensorDict containing:
            - observation: Board state [2,12,12]
            - reward: [-1.0 to 1.0]
            - done: Game completion flag
            - action_mask: Valid moves [80]
            - step_count: Current step
            
        Raises:
            IllegalMoveError: If selected position is occupied
        """

        # Get player's action from tensordict and convert to board position
        action = tensordict["action"].item()
        i, j = self.index_to_position(action)  # Convert action index to board coordinates
        
        # Check if move is illegal (position already occupied)
        if self.board[i][j] != 0:
            # Record illegal attempt and get valid moves mask
            illegal_pos = (i, j)
            mask = self._get_action_mask()
            
            # Build detailed error message
            error_msg = f"Attempted move at ({i},{j}) but position is occupied!\nBoard state:\n{self.render_board()}\nValid moves mask: {mask.tolist()}\nAction value: {action}\n"
            raise IllegalMoveError(error_msg)
        
        # Apply success rate mechanic for move execution
        execute_move = None
        if random.random() < self.config.success_rate:
            # Successful placement at intended position
            self.board[i][j] = self.current_player  # Player is always 1
            execute_move = (i, j)
        else:
            # Find valid adjacent positions for failed placement
            adjacent = []
            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:
                    if di == 0 and dj == 0:
                        continue
                    ni, nj = i + di, j + dj
                    adjacent.append((ni, nj))
            
            # Try to place piece in random adjacent position
            if adjacent:
                ni, nj = random.choice(adjacent)
                if self._is_in_zone(ni, nj) and self.board[ni][nj] == 0:
                    self.board[ni][nj] = self.current_player
                    execute_move = (ni, nj)
                else:
                    execute_move = None
        
        # Record move history
        self.move_history.append((i, j))
        self.execute_move_history.append(execute_move)
        
        # Initialize reward
        reward = 0.0
        
        # Check if player won after their move
        self.winner = self._check_winner()
        
        if self.winner == self.current_player:
            reward = 1.0  # Player wins
            done = True
        
        else:  # Game continues - opponent's turn
            done = self.is_over()
            if done:
                reward = 0.0  # Draw game
            
            else:  # Opponent makes move
                # Set policy network to evaluation mode
                self.policy_net.eval()
                with torch.no_grad():
                    # 1. Get opponent observation (inverted board state)
                    opponent_obs = -(self._get_obs().unsqueeze(0)).to(device)
                    
                    # 2. Get policy network predictions
                    opponent_logits = self.policy_net(opponent_obs)["logits"]
                    
                    # 3. Apply action mask to logits
                    mask_for_opponent = self._get_action_mask().unsqueeze(0)
                    opponent_logits = opponent_logits.clone()
                    opponent_logits[~mask_for_opponent] = float('-inf')
                    
                    # 4. Sample action from distribution
                    try:
                        distribution = torch.distributions.Categorical(logits=opponent_logits)
                        action = distribution.sample().item()
                        
                        self.debug =False
                        # Debug logging if enabled
                        if self.debug:
                            probs = torch.softmax(opponent_logits, dim=-1)
                            print(f"Valid actions: {mask_for_opponent.sum().item()}")
                            print(f"Selected action: {action}")
                            print(f"Action probability: {probs[0][action].item():.4f}")
                            
                    except Exception as e:
                        # Fallback to random valid move if sampling fails
                        print(f"Error in action sampling: {e}")
                        valid_actions = torch.where(mask_for_opponent[0])[0]
                        action = valid_actions[torch.randint(len(valid_actions), (1,))].item()
                    
                    # 5. Execute opponent's move
                    i, j = self.index_to_position(action)
                    self.board[i][j] = -self.current_player
                
                # Restore policy network to training mode
                self.policy_net.train()
                
                # Check game state after opponent move
                self.winner = self._check_winner()
                if self.winner == 0:  # Draw
                    done = True
                    reward = 0.0
                elif self.winner == None:  # Game continues
                    done = False
                    # Calculate pattern-based rewards
                    oppo_pattern_reward = self.calculate_pattern_reward(-self.current_player)
                    self_pattern_reward = self.calculate_pattern_reward(self.current_player)
                    reward += self_pattern_reward * 0.1  # Scale intermediate rewards
                elif self.winner == -self.current_player:  # Opponent wins
                    reward = -1.0
                    done = True
        
        # Increment step counter
        self.steps += 1
        
        # Return state as TensorDict
        return TensorDict(
            {"observation": self._get_obs(),
             "reward": torch.tensor(reward, device=device),
             "done": torch.tensor(done, device=device),
             "action_mask": self._get_action_mask(),
             "step_count": torch.tensor([float(self.steps)], dtype=torch.float64, device=device)},
            batch_size=[]
        ) 
    
    def _get_action_mask(self) -> torch.Tensor:
        """Generate boolean mask for valid actions on the board.
        
        Returns:
            torch.Tensor: Boolean mask of shape [80] where:
                - True: Valid move position (empty and in cross shape)
                - False: Invalid move position (occupied or outside cross)
        """
        # Initialize mask tensor with all positions invalid
        mask = torch.zeros(80, dtype=torch.bool, device=self.device)
        
        # Iterate through all board positions
        for i in range(12):
            for j in range(12):
                # Skip positions outside cross-shaped valid zone
                if not self._is_in_zone(i, j):
                    continue
                
                # Mark empty positions as valid moves
                if self.board[i][j] == 0:
                    # Convert 2D position to 1D action index
                    index = self.position_to_index(i, j)
                    mask[index] = True
                    
        return mask
    
    def _get_obs(self) -> torch.Tensor:
        """Get observation of current board state as tensor.
        
        Returns:
            torch.Tensor: Shape (2,12,12) containing:
                Channel 0: Current player positions (1)
                Channel 1: Opponent positions (1)
        """
        state = np.zeros((2, 12, 12), dtype=np.float32)
        
        # Fill channels based on piece positions
        for i in range(12):
            for j in range(12):
                if self.board[i][j] == 1:
                    state[0][i][j] = 1  # Current player
                elif self.board[i][j] == -1:
                    state[1][i][j] = 1  # Opponent
    
        return torch.from_numpy(state).to(device)
    
    def get_legal_moves(self):
        """Get list of valid move positions.
        
        Returns:
            list: Valid (row,col) positions that are empty and in cross shape
        """
        legal_moves = []
        for i in range(12):
            for j in range(12):
                if self._is_in_zone(i, j) and self.board[i][j] == 0:
                    legal_moves.append((i, j))
        return legal_moves
    
    def _get_winning_combinations(self):
        """Generate all possible winning position combinations.
        
        Returns:
            list: Lists of position tuples that form winning lines:
                - Horizontal 4-in-row
                - Vertical 4-in-row
                - Diagonal 5-in-row
        """
        winning_combinations = []
        
        # 1. Horizontal 4-in-row
        # Center horizontal strip
        for i in range(4, 8):
            for j in range(9):
                winning_combinations.append([(i, j+k) for k in range(4)])
        
        # Vertical strip top/bottom
        for i in [0,1,2,3, 8,9,10,11]:
            for j in range(4, 5):
                winning_combinations.append([(i, j+k) for k in range(4)])
        
        # 2. Vertical 4-in-row
        # Center vertical strip
        for j in range(4, 8):
            for i in range(9):
                winning_combinations.append([(i+k, j) for k in range(4)])
        
        # Left/right sides
        for i in range(4, 5):
            for j in [0,1,2,3, 8,9,10,11]:
                winning_combinations.append([(i+k, j) for k in range(4)])
        
        # 3. Diagonal 5-in-row
        # Main diagonal (top-left to bottom-right)
        for i in range(8):
            for j in range(8):
                if self._is_valid_diagonal([(i+k, j+k) for k in range(5)]):
                    winning_combinations.append([(i+k, j+k) for k in range(5)])
        
        # Counter diagonal (top-right to bottom-left)
        for i in range(8):
            for j in range(4, 12):
                if self._is_valid_diagonal([(i+k, j-k) for k in range(5)]):
                    winning_combinations.append([(i+k, j-k) for k in range(5)])
    
        return winning_combinations
    
    def _is_valid_diagonal(self, positions):
        """Check if diagonal line positions are valid.
        
        Args:
            positions: List of (row,col) positions forming diagonal
            
        Returns:
            bool: True if all positions are in cross shape
        """
        for i, j in positions:
            if not self._is_in_zone(i, j):
                return False
        return True
    
    def _check_winner(self):
        """Check if game has a winner.
        
        Returns:
            int: Winner indicator
                1: Current player wins
                -1: Opponent wins
                0: Draw (board full)
                None: Game continues
        """
        winning_combinations = self.winning_combinations
        
        for combo in winning_combinations:
            values = [self.board[i][j] for i, j in combo]
            if len(combo) == 4:  # Horizontal/vertical
                if sum(values) == 4:  # Player 1 wins
                    return 1
                if sum(values) == -4:  # Player -1 wins
                    return -1
            elif len(combo) == 5:  # Diagonal
                if sum(values) == 5:  # Player 1 wins
                    return 1
                if sum(values) == -5:  # Player -1 wins
                    return -1
        
        # Check for draw
        if not self.get_legal_moves():
            return 0
        
        return None  # Game continues
    
    def is_over(self):
        """Check if game is finished.
        
        Returns:
            bool: True if game has winner or is draw
        """
        return self._check_winner() is not None

    def _set_seed(self, seed=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        return [seed]
    
    def render_board(self):
        """Return string representation of current board state.
        
        Returns:
            str: Board visualization where:
                '.': Empty position
                'X': Player 1 piece
                'O': Player -1 piece
        """
        symbols = {0: '.', 1: 'X', -1: 'O'}
        board_str = ""
        for i in range(12):
            board_str += "|"
            for j in range(12):
                board_str += symbols[self.board[i][j]] + "|"
            board_str += "\n"
        return board_str
    
    def _detect_consecutive(self, player, length):
        """Detect number of consecutive pieces for given player.
        
        Args:
            player: Player ID (1 or -1)
            length: Length of consecutive pattern to find
            
        Returns:
            int: Number of patterns found across:
                - Horizontal lines
                - Vertical lines 
                - Main diagonals (top-left to bottom-right)
                - Counter diagonals (top-right to bottom-left)
        """
        count = 0
        
        # Check horizontal patterns
        for i in range(12):
            for j in range(12 - length + 1):
                if all(self.board[i][j+k] == player for k in range(length)):
                    count += 1
        
        # Check vertical patterns
        for j in range(12):
            for i in range(12 - length + 1):
                if all(self.board[i+k][j] == player for k in range(length)):
                    count += 1
        
        # Check main diagonals
        for i in range(12 - length + 1):
            for j in range(12 - length + 1):
                if all(self.board[i+k][j+k] == player for k in range(length)):
                    count += 1
        
        # Check counter diagonals
        for i in range(length-1, 12):
            for j in range(12 - length + 1):
                if all(self.board[i-k][j+k] == player for k in range(length)):
                    count += 1
                    
        return count
    
    def calculate_pattern_reward(self, player):
        """Calculate reward based on consecutive piece patterns.
        
        Args:
            player: Player ID (1 or -1)
            
        Returns:
            float: Pattern-based reward where:
                - 2 consecutive: 0.2 per pattern
                - 3 consecutive: 0.5 per pattern
                - 4 consecutive: 0.8 per pattern
        """
        reward = 0
        # Check patterns of length 2-4
        for length in [2, 3, 4]:
            count = self._detect_consecutive(player, length)
            if length == 4 and count > 0:
                reward += count * 0.8
            elif length == 3 and count > 0:
                reward += count * 0.5
            elif length == 2 and count > 0:
                reward += count * 0.2
        return reward
    
def make_env():
    """Factory function to create TicTacToe environment.
     
        Returns:
        TicTacToeEnv: New environment instance
    """
    env = TicTacToeEnv()
    return env

def train_ppo():
    """Train PPO agent for TicTacToe environment.
    
    Components:
    1. Initialization:
        - Configuration & logging setup
        - Model directories
        - Metrics tracking
        
    2. Network Setup:
        - Policy & value networks
        - Load checkpoints if exists
        - Create TorchRL modules
        
    3. Training Components:
        - Data collector
        - Replay buffer
        - Advantage estimation (GAE)
        - PPO loss module
        - Optimizer & scheduler
        
    4. Training Loop:
        - Collect experiences
        - Calculate advantages
        - Multiple epochs of updates
        - Track metrics & losses
        
    5. Model Saving:
        - Best model based on rewards
        - Periodic checkpoints
        - Final model state
        
    Returns:
        tuple: (PolicyNetwork, ValueNetwork) trained models
    """
    # Print training initialization
    print("\n" + "="*80)
    print(f"【Initialization】 Starting PPO training on {device}...")
    print("="*80)

    # Initialize configuration
    config = Config()

    model_dir = config.model_dir
    os.makedirs(model_dir, exist_ok=True)

    log_file = setup_logging(config.model_dir)
    log_metrics = {
        'epoch_losses': [],
        'policy_losses': [],
        'value_losses': [],
        'entropy_losses': [],
        'rewards': [],
        'lr': []
    }
    # Create environment for inspection
    env_creator = EnvCreator(make_env)
    env = env_creator()
    # check_env_specs(env)
    
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
   
    policy_params = sum(p.numel() for p in policy_net.parameters())
    value_params = sum(p.numel() for p in value_net.parameters())
    print(f"【Model Info】 Policy params: {policy_params:,}, Value params: {value_params:,}")
     
    #load the check_point
    checkpoint_path = os.path.join(model_dir, "ppo_tictactoe_best.pth")
    print(checkpoint_path)
    print(os.path.exists(checkpoint_path))
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        
        # Load model states
        policy_net.load_state_dict(checkpoint['policy'])
        value_net.load_state_dict(checkpoint['value'])
        
        # Load training state
        best_reward = checkpoint.get('reward', float('-inf'))
      

    # Create policy module for TorchRL
    policy_module = TensorDictModule(
        policy_net, 
        in_keys=["observation"],
        out_keys=["logits"],
    )

    # Create probabilistic actor from policy module
    dist_module = ProbabilisticTensorDictModule(
        in_keys={"logits": "logits", "mask": "action_mask"}, #这里的action mask是原来的 
        out_keys=["action"], 
        distribution_class=MaskedCategorical, #被掩码之后的分类的分布 从这个类的对象采样一个action
        return_log_prob=True,
    )

    actor_module = ProbabilisticTensorDictSequential(policy_module, dist_module)

    # Create value module
    value_module = ValueOperator(
        module=value_net,
        in_keys=["observation"],
        out_keys=["state_value"], #torchrl里所有传递的数据的载体都是tensordict
    )

    collector = SyncDataCollector(
        env_creator, #环境 或环境创建器 我们现在是环境创建器 但是也可以传环境； 需要验证这里改回环境
        policy=actor_module,
        frames_per_batch=config.frames_per_batch,   # big_batch / 每次调用enumerate(collector)时，从环境采样这么多steps/frames的数据
        #默认是每次直接接着前面的调用，不会另起一局，可以调整为另起一局
        total_frames=config.total_frames,
        split_trajs=False,
        device=device,
    )
    
    # Create replay buffer - 优化：每批次清空缓冲区
    replay_buffer = ReplayBuffer(
        storage=LazyTensorStorage(max_size=config.frames_per_batch * 2),  # 确保足够大
        sampler=SamplerWithoutReplacement(),
    )
    
    # Create advantage module for GAE
    advantage_module = GAE(
        gamma=config.gamma,
        lmbda=config.gae_lambda,
        value_network=value_module,
        average_gae=True,
    )
    
    # Create PPO loss module
    loss_module = ClipPPOLoss(
        actor_network=actor_module,
        critic_network=value_module,
        clip_epsilon=config.clip_epsilon, 
        # entropy_bonus=False,
        entropy_bonus=bool(config.c2),
        entropy_coef=config.c2,
        critic_coef=config.c1,
        normalize_advantage=True,
        loss_critic_type="smooth_l1",
    )
    
    # Create optimizer
    optimizer = torch.optim.Adam(
        loss_module.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    # Create learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        config.total_frames // config.frames_per_batch, 
        0.0
    )
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    
    # Training loop tracking best performance
    best_reward = float('-inf')
    print("\n" + "="*80)
    print("【Training Start】 Collecting data and training...")
    print("="*80 + "\n")
    
    # 用于存储每个取数据的平均损失值
    draw_losses = [ ]
    
    for i, tensordict_data in enumerate(collector):
        
        replay_buffer.empty()
        advantage_module(tensordict_data)   # assert "advantage" in tensordict.keys()
        data_view = tensordict_data.reshape(-1) 
        replay_buffer.extend(data_view.cpu())
        
        # Train for multiple epochs on collected data
        for j in range(config.update_epochs):  
            epoch_loss_values = []
            
            epoch_losses = []
            policy_losses = []
            value_losses = []
            entropy_losses = []
            
            # Train on mini-batches
            for k in range(config.frames_per_batch // config.sub_batch_size):   # // 整除运算符，返回不大于结果的最大整数
                # Sample a mini-batch
                subdata = replay_buffer.sample(config.sub_batch_size)
                subdata = subdata.to(device)  
                
                # Compute losses
                loss_vals = loss_module(subdata)
                # print(f"{loss_vals}")
                loss_value = (
                    loss_vals["loss_objective"] +
                    loss_vals["loss_critic"] + 
                    loss_vals["loss_entropy"]
                )
            
                optimizer.zero_grad()
                loss_value.backward()
                
                torch.nn.utils.clip_grad_norm_(loss_module.parameters(), 1.0)
                
                optimizer.step()
                # print(f"small_batch loss: {loss_value.item()}")
                epoch_loss_values.append(loss_value.item())
                
                epoch_losses.append(loss_value.item())
                policy_losses.append(loss_vals["loss_objective"].item())
                value_losses.append(loss_vals["loss_critic"].item())
                entropy_losses.append(loss_vals["loss_entropy"].item())

    
            avg_epoch_loss = sum(epoch_loss_values) / len(epoch_loss_values)
            
            log_metrics['epoch_losses'].append(np.mean(epoch_loss_values))
            log_metrics['policy_losses'].append(np.mean(policy_losses))
            log_metrics['value_losses'].append(np.mean(value_losses))
            log_metrics['entropy_losses'].append(np.mean(entropy_losses))
            log_metrics['rewards'].append(tensordict_data["next", "reward"].mean().item()) #这里的reward是当前的下一个状态的reward 感觉没用
            log_metrics['lr'].append(scheduler.get_last_lr()[0])
            
            # Print training progress
            print(f"  Epoch {j+1}/{config.update_epochs}: avg_loss={avg_epoch_loss:.6f}")

            # Record loss for plotting
            draw_losses.append(avg_epoch_loss) 

        avg_reward = tensordict_data["next", "reward"].mean().item()  
        
        # Update learning rate schedule
        scheduler.step()
        
        # Save best model if current reward exceeds best seen
        if avg_reward > best_reward:
            best_reward = avg_reward
            torch.save({
                'policy': policy_net.state_dict(),
                'value': value_net.state_dict(),
                'config': vars(config),
                'metrics': {
                    'reward': avg_reward,
                    'batch': i+1,
                }
            }, os.path.join(model_dir, "ppo_tictactoe_best.pth"))
            print(f"【Save】 New best model saved (reward: {best_reward:.4f})")

        # Save periodic checkpoints for training resumption
        if (i+1) % config.save_frequency == 0:
            torch.save({
                'policy': policy_net.state_dict(),
                'value': value_net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'config': vars(config),
                'metrics': {
                    'reward': avg_reward,
                    'batch': i+1,
                }
            }, os.path.join(model_dir, f"ppo_tictactoe_checkpoint_{i+1}.pth"))
            print(f"【Save】 Checkpoint saved: batch_{i+1}")

    # Save final metrics and models
    log_metrics_function(log_file, log_metrics)
    torch.save({
        'policy': policy_net.state_dict(),
        'value': value_net.state_dict(),
        'config': vars(config),
    }, os.path.join(model_dir, f"ppo_tictactoe_checkpoint_final.pth"))

    # Plot training curves
    plot_training_metrics(log_file)
    plot_smoothed_loss(draw_losses, save_dir=model_dir)
    
    # Plot final loss curve
    plt.figure(figsize=(10, 6))
    plt.plot(draw_losses, label="Loss")
    plt.xlabel("Eopches")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.legend()
    plt.grid()
    plt.savefig("./training_model/training_loss_curve.png")  
    plt.show()
    
    return policy_net, value_net

if __name__ == "__main__":
    policy_net, value_net = train_ppo()