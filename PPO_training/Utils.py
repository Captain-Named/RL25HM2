"""Utility functions for PPO TicTacToe training.

Categories:
1. Core Utilities
2. Training Metrics & Visualization 
3. Board Operations
4. Debug & Testing
"""

import datetime
import torch
import numpy as np
from torch.nn import functional as F      
import json
import matplotlib.pyplot as plt
import os

# 1. Core Utility Functions
#Note that this funciton _calculate_state is used in the network.py
def _calculate_state(x: torch.tensor):
    """Calculate potential scores for each valid position.
    
    Args:
        x: Input tensor of shape [batch_size, 2, 12, 12]
           Channel 0: Current player positions
           Channel 1: Opponent positions
           
    Returns:
        torch.Tensor: Potential scores mask of shape [batch_size, 80]
    """
    batch_size = x.shape[0]
    mask = torch.zeros((x.shape[0],80), dtype=torch.float32, device=x.device)
    
    for b in range(batch_size):
        # Extract board state for current batch
        my_view = x[b, 0].detach().numpy()  
        opponent_view = x[b, 1].detach().numpy()
        
        # Create combined board state (-1: opponent, 0: empty, 1: current player)
        board = np.zeros((12, 12))
        board[my_view == 1] = 1
        board[opponent_view == 1] = -1

        # Calculate potential scores for valid positions
        for i in range(12):
            for j in range(12):
                if board[i][j] == 0 and _is_in_zone(i,j) and _is_adjacent_to_player(board, i,j):
                    my_score = _calculate_potential(board, i,j)
                    index = position_to_index(i, j)
                    mask[b, index] = my_score
    return mask

def _is_in_zone(ni, nj):
    """Check if position is within valid cross-shaped zone.
    
    Args:
        ni, nj: Board coordinates
        
    Returns:
        bool: True if position is valid
    """
    return not((ni<4 and nj<4)or(ni<4 and nj>7)or(ni>7 and nj<4)or(ni>7 and nj>7)) and (0 <= ni < 12 and 0 <= nj < 12)

def _is_adjacent_to_player(board, i, j):
    """Check if position is adjacent to current player's pieces.
    
    Args:
        board: Game board state
        i, j: Position coordinates
        
    Returns:
        bool: True if adjacent to player's piece
    """
    for di in [-1, 0, 1]:
        for dj in [-1, 0, 1]:
            if di == 0 and dj == 0:
                continue
            ni, nj = i + di, j + dj
            if 0 <= ni < 12 and 0 <= nj < 12 and _is_in_zone(ni, nj):
                if board[ni][nj] == 1:
                    return True
    return False

def _calculate_potential(board, i, j):
    """Calculate potential score for placing piece at position.
    
    Args:
        board: Current board state
        i, j: Position coordinates
        
    Returns:
        float: Potential score based on:
            - Two consecutive: +1.0
            - Three consecutive: +3.0
            - Four consecutive: +10.0
    """
    if not _is_in_zone(i, j) or board[i][j] != 0:
        return 0.0
    
    virtual_board = np.copy(board)
    
    # Calculate score before move
    two_len = _detect_consecutive_vir(1, 2, virtual_board)
    three_len = _detect_consecutive_vir(1, 3, virtual_board) 
    four_len = _detect_consecutive_vir(1, 4, virtual_board)
    score_before = 1.0 * two_len + 3.0 * three_len + 10.0 * four_len
    
    # Calculate score after move
    virtual_board[i][j] = 1
    two_len = _detect_consecutive_vir(1, 2, virtual_board)
    three_len = _detect_consecutive_vir(1, 3, virtual_board) 
    four_len = _detect_consecutive_vir(1, 4, virtual_board)
    score_after = 1.0 * two_len + 3.0 * three_len + 10.0 * four_len
    
    return score_after - score_before

def _detect_consecutive_vir(player, length, vir_board):
    """Count consecutive patterns of given length.
    
    Args:
        player: Player ID to check for
        length: Pattern length to find
        vir_board: Board state to analyze
        
    Returns:
        int: Number of patterns found in:
            - Horizontal lines
            - Vertical lines
            - Main diagonals
            - Counter diagonals
    """
    count = 0
    
    # Check horizontal patterns
    for i in range(12):
        for j in range(12 - length + 1):
            if all(vir_board[i][j+k] == player for k in range(length)):
                count += 1
    
    # Check vertical patterns
    for j in range(12):
        for i in range(12 - length + 1):
            if all(vir_board[i+k][j] == player for k in range(length)):
                count += 1
    
    # Check main diagonals
    for i in range(12 - length + 1):
        for j in range(12 - length + 1):
            if all(vir_board[i+k][j+k] == player for k in range(length)):
                count += 1
    
    # Check counter diagonals
    for i in range(length-1, 12):
        for j in range(12 - length + 1):
            if all(vir_board[i-k][j+k] == player for k in range(length)):
                count += 1
                
    return count

def position_to_index(i: int, j: int) -> int:
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


# 2. Training Metrics & Visualization

def plot_training_metrics(log_file):
    """Plot training metrics from log file.
    
    Args:
        log_file: Path to JSON file containing training metrics
        
    Plots:
        1. Training Losses:
            - Total loss
            - Policy loss 
            - Value loss
        2. Training Rewards:
            - Average reward per batch
            
    Saves:
        PNG file with plots in same directory as log file
    """
    # Load metrics from JSON file
    with open(log_file, 'r') as f:
        metrics = json.load(f)
    
    # Create figure with multiple subplots
    plt.figure(figsize=(15, 5))
    
    # Plot various loss curves
    plt.subplot(131)
    plt.plot(metrics['epoch_losses'], label='Total Loss')
    plt.plot(metrics['policy_losses'], label='Policy Loss')
    plt.plot(metrics['value_losses'], label='Value Loss')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.title('Training Losses')
    plt.legend()
    plt.grid(True)
    
    # Plot reward curve
    plt.subplot(132)
    plt.plot(metrics['rewards'], label='Average Reward')
    plt.xlabel('Batch')
    plt.ylabel('Reward')
    plt.title('Training Rewards')
    plt.legend()
    plt.grid(True)
    
    # Adjust layout and save figure
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(log_file), 'training_curves.png'))

def plot_smoothed_loss(losses, window_size=10, save_dir="./training_model", filename="training_loss_curve_avg.png"):
    """绘制平滑和原始损失曲线
    Args:
        losses: 损失值列表
        window_size: 平滑窗口大小
        save_dir: 保存目录
        filename: 保存文件名
    """
    weights = np.ones(window_size) / window_size
    smoothed_losses = np.convolve(losses, weights, mode='valid')
    
    plt.figure(figsize=(12, 6))
    plt.plot(losses, 'lightgray', alpha=0.3, label='原始损失')
    plt.plot(range(window_size-1, len(losses)), 
             smoothed_losses, 
             'b-', 
             linewidth=2, 
             label=f'平滑损失(窗口={window_size})')
    
    plt.xlabel("训练批次")
    plt.ylabel("损失值")
    plt.title("训练损失曲线")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_dir:
        save_path = os.path.join(save_dir, filename)
        plt.savefig(save_path)
    plt.close()


# 3. Debug & Testing Functions 

def print_reset_td(td, env=None, title="env.reset()返回的TensorDict信息", prefix="  "):
    """
    打印环境reset()返回的TensorDict信息，格式化为易读的形式
    
    reset()返回的td形如下：

    TensorDict(
        fields={
            action_mask: Tensor(shape=torch.Size([9]), device=cuda:0, dtype=torch.bool, is_shared=True),
            done: Tensor(shape=torch.Size([1]), device=cuda:0, dtype=torch.bool, is_shared=True),
            observation: Tensor(shape=torch.Size([1, 3, 3]), device=cuda:0, dtype=torch.float32, is_shared=True),      
            step_count: Tensor(shape=torch.Size([1]), device=cuda:0, dtype=torch.float32, is_shared=True),
            terminated: Tensor(shape=torch.Size([1]), device=cuda:0, dtype=torch.bool, is_shared=True)},
        batch_size=torch.Size([]),
        device=None,
        is_shared=False)

    参数:
        td: 要打印的TensorDict
        env: 可选的环境实例，用于打印额外的环境状态
        title: 打印信息的标题
        prefix: 打印时的缩进前缀
    """
    print(f"\n{'='*40}")
    print(f"🔄 {title}")
    print(f"{'='*40}")
    
    # 打印棋盘状态 (observation)
    if "observation" in td:
        print(f"{prefix}👀 观察向量 (observation) / reset过后，棋盘状态:")
        obs = td["observation"].cpu().numpy()
        
        # 提取X和O的位置
        board_display = np.full((3,3), '.')
        board_x = np.argwhere(obs[0] > 0)
        board_o = np.argwhere(obs[0] < 0)
        
        for pos in board_x:
            board_display[tuple(pos)] = 'X'
        for pos in board_o:
            board_display[tuple(pos)] = 'O'
        
        # 打印棋盘
        for row in board_display:
            print(f"{prefix}   |" + "|".join(row) + "|")
        
        current_player = 'X'
        print(f"{prefix}   当前玩家: {current_player}")
    
    # 打印动作掩码
    if "action_mask" in td:
        print(f"\n{prefix}🎯 动作掩码 (action_mask) / reset过后状态的合法动作:")
        mask = td["action_mask"].cpu().bool()
        
        # 格式化为3x3棋盘形式
        mask_array = mask.reshape(3, 3)
        print(f"{prefix}   位置格式: [位置索引:可用性]")
        
        for i in range(3):
            row_str = "   "
            for j in range(3):
                symbol = "✓" if mask_array[i, j] else "✗"
                row_str += f"[{i*3+j}:{symbol}] "
            print(f"{prefix}{row_str}")
    
    # 打印步数计数
    if "step_count" in td:
        step_count = td["step_count"].item()
        print(f"\n{prefix}🔢 步数 (step_count) / reset过后，step置0: {step_count}")
    
    # 打印done状态
    if "done" in td:
        done = td["done"].item()
        print(f"{prefix}🏁 是否结束 (done) / reset过后的状态是否是终止状态: {'是' if done else '否'}")
    
    # 打印terminated状态
    if "terminated" in td:
        terminated = td["terminated"].item()
        print(f"{prefix}⚠️ 是否终止 (terminated) / reset过后的状态是否是终止状态: {'是' if terminated else '否'}")
    
    # 如果提供了环境，打印额外的环境状态
    if env is not None:
        print(f"\n{prefix}🌍 reset过后，env.board的环境状态:")
        if hasattr(env, "render_board"):
            board_str = env.render_board()
            for line in board_str.strip().split('\n'):
                print(f"{prefix}   {line}")
        
        if hasattr(env, "current_player"):
            player_symbol = 'X' if env.current_player == 1 else 'O'
            print(f"{prefix}   当前玩家: {player_symbol}")
        
        if hasattr(env, "winner"):
            winner = env.winner
            winner_str = "无" if winner == 0 else ("X" if winner == 1 else "O")
            print(f"{prefix}   当前赢家: {winner_str}")
    
    print(f"{'='*40}\n")

def print_step_td(td, env=None, title="step(policy_output)后返回的TensorDict信息", prefix="  ", max_tensor_items=30):
    """
    打印step返回的td的详细信息，格式化为易读的形式
    
    step(policy_ouput)返回的td形如下：

    step(policy_ouput): TensorDict(
        fields={
            action: Tensor(shape=torch.Size([1]), device=cuda:0, dtype=torch.int64, is_shared=True),
            action_mask: Tensor(shape=torch.Size([9]), device=cuda:0, dtype=torch.bool, is_shared=True),
            done: Tensor(shape=torch.Size([1]), device=cuda:0, dtype=torch.bool, is_shared=True),
            logits: Tensor(shape=torch.Size([1, 9]), device=cuda:0, dtype=torch.float32, is_shared=True),
            next: TensorDict(
                fields={
                    action_mask: Tensor(shape=torch.Size([9]), device=cuda:0, dtype=torch.bool, is_shared=True),       
                    done: Tensor(shape=torch.Size([1]), device=cuda:0, dtype=torch.bool, is_shared=True),
                    observation: Tensor(shape=torch.Size([3, 3, 3]), device=cuda:0, dtype=torch.float32, is_shared=True),
                    reward: Tensor(shape=torch.Size([1]), device=cuda:0, dtype=torch.float32, is_shared=True),
                    step_count: Tensor(shape=torch.Size([1]), device=cuda:0, dtype=torch.float64, is_shared=True),     
                    terminated: Tensor(shape=torch.Size([1]), device=cuda:0, dtype=torch.bool, is_shared=True)},       
                batch_size=torch.Size([]),
                device=None,
                is_shared=False),
            observation: Tensor(shape=torch.Size([3, 3, 3]), device=cuda:0, dtype=torch.float32, is_shared=True),      
            sample_log_prob: Tensor(shape=torch.Size([1]), device=cuda:0, dtype=torch.float32, is_shared=True),        
            state_value: Tensor(shape=torch.Size([1, 1]), device=cuda:0, dtype=torch.float32, is_shared=True),
            step_count: Tensor(shape=torch.Size([1]), device=cuda:0, dtype=torch.float32, is_shared=True),
            terminated: Tensor(shape=torch.Size([1]), device=cuda:0, dtype=torch.bool, is_shared=True)},
        batch_size=torch.Size([]),
        device=None,
        is_shared=False)

    参数:
        td: 要打印的TensorDict
        env: 可选的环境实例，用于打印棋盘状态
        title: 打印信息的标题
        prefix: 打印时的缩进前缀
        max_tensor_items: 显示tensor的最大元素数量
    """
    def tensor_to_readable(tensor, max_items=max_tensor_items):
        """将张量转换为可读字符串"""
        if tensor.numel() <= max_items:
            return tensor.cpu().tolist()
        else:
            flat = tensor.flatten()
            return "[前{}个: {}, ... 形状: {}]".format(
                max_items,
                flat[:max_items].cpu().tolist(),
                tensor.shape
            )
    
    def print_dict_items(d, level=0, path=""):
        """递归打印字典项"""
        indent = prefix * (level + 1)
        if isinstance(d, dict) or hasattr(d, "keys"):
            for key in d.keys():
                current_path = f"{path}.{key}" if path else key
                if key == "next" and isinstance(d[key], dict):
                    print(f"{indent}🔄 {key}: (next状态)")
                    print_dict_items(d[key], level + 1, current_path)
                elif isinstance(d[key], dict) or hasattr(d[key], "keys"):
                    print(f"{indent}📂 {key}: (嵌套字典)")
                    print_dict_items(d[key], level + 1, current_path)
                elif torch.is_tensor(d[key]):
                    if key == "observation" and d[key].shape[-3:] == (3, 3, 3) and level == 0:
                        print(f"{indent}👀 {key} (step之前的观察向量): ")
                        if d[key].dim() > 3:
                            # 如果是batch形式
                            sample_obs = d[key][0]
                        else:
                            sample_obs = d[key]
                        
                        obs_np = sample_obs.cpu().numpy()
                        # 棋盘表示
                        board_display = np.full((3,3), '.')
                        board_x = np.argwhere(obs_np[0] > 0)
                        board_o = np.argwhere(obs_np[0] < 0)
                        for pos in board_x:
                            board_display[tuple(pos)] = 'X'
                        for pos in board_o:
                            board_display[tuple(pos)] = 'O'
                        print(f"{indent}   观察向量: {tensor_to_readable(sample_obs)}")
                        print(f"{indent}   棋盘状态:")
                        for row in board_display:
                            print(f"{indent}   |" + "|".join(row) + "|")
                        
                        current_player = 'X'
                        print(f"{indent}   当前玩家: {current_player}")
                    elif key == "action_mask":
                        print(f"{indent}🎯 {key} (动作掩码): ")
                        mask = d[key].cpu().bool()
                        if mask.dim() == 1 and mask.shape[0] == 9:
                            # 3x3板
                            mask_array = mask.reshape(3, 3)
                            for i in range(3):
                                row_str = "   "
                                for j in range(3):
                                    symbol = "✓" if mask_array[i, j] else "✗"
                                    row_str += f"[{i*3+j}:{symbol}] "
                                print(f"{indent}{row_str}")
                        else:
                            print(f"{indent}   {tensor_to_readable(mask)}")
                    elif key == "action":
                        action_val = d[key].item()
                        row, col = divmod(action_val, 3)
                        print(f"{indent}🎮 {key} (动作): {action_val} (位置: {row},{col})")
                    elif key == "reward":
                        reward_val = d[key].item()
                        print(f"{indent}🏆 {key} (奖励): {reward_val:.4f}")
                    elif key == "state_value":
                        value = d[key].item()
                        print(f"{indent}💰 {key} (状态价值): {value:.4f}")
                    elif key == "logits":
                        print(f"{indent}🧠 {key} (策略logits):")
                        logits = d[key].squeeze(0).cpu()
                        if logits.shape[0] == 9:
                            # 格式化为3x3的网格
                            logits_array = logits.reshape(3, 3)
                            for i in range(3):
                                row_str = "   "
                                for j in range(3):
                                    row_str += f"[{logits_array[i, j]:.4f}] "
                                print(f"{indent}{row_str}")
                            
                            # 计算softmax概率
                            probs = F.softmax(logits, dim=0)
                            print(f"{indent}   （去掉非法动作前的）概率分布:")
                            probs_array = probs.reshape(3, 3)
                            for i in range(3):
                                row_str = "   "
                                for j in range(3):
                                    row_str += f"[{probs_array[i, j]:.4f}] "
                                print(f"{indent}{row_str}")
                        else:
                            print(f"{indent}   {tensor_to_readable(logits)}")
                    elif key == "sample_log_prob":
                        log_prob = d[key].item()
                        prob = np.exp(log_prob)
                        print(f"{indent}📊 {key} (（去掉非法动作后的）动作对数概率): {log_prob:.4f} (（去掉非法动作后的）概率: {prob:.4%})")
                    elif key == "done" or key == "terminated":
                        print(f"{indent}🏁 {key}: {'是' if d[key].item() else '否'}")
                    elif key == "step_count":
                        print(f"{indent}🔢 {key} (步数): {d[key].item()}")
                    else:
                        print(f"{indent}📊 {key}: {tensor_to_readable(d[key])}")
                else:
                    print(f"{indent}🔧 {key}: {d[key]}")
    
    print(f"\n{'='*40}")
    print(f"🔍 {title}")
    print(f"{'='*40}")
    
    # 打印主体信息
    if isinstance(td, dict) or hasattr(td, "keys"):
        print_dict_items(td)
    else:
        print(f"{prefix}不是TensorDict或字典类型: {type(td)}")
    
    # 如果提供了环境，打印环境状态
    if env is not None:
        print(f"\n{prefix}🌍 执行了step后，env棋盘状态:")
        board_str = env.render_board()
        for line in board_str.strip().split('\n'):
            print(f"{prefix}{line}")
        print(f"{prefix}当前玩家: {'X' if env.current_player == 1 else 'O'}")
    
    print(f"{'='*40}\n")

def print_actor_output(td, env=None, title="actor_module(td)返回的策略输出", prefix="  "):
    """
    打印策略网络(actor_module)输出的TensorDict信息，格式化为易读的形式
    
    actor_module(td)返回的td形如下：
    
    policy_output: TensorDict(
        fields={
            # 这里是根据logits和mask选出的最终的一个action           
            action: Tensor(shape=torch.Size([1]), device=cuda:0, dtype=torch.int64, is_shared=True),
            #这里的action_mask是输入的td的其中原封不动的action_mask
            action_mask: Tensor(shape=torch.Size([9]), device=cuda:0, dtype=torch.bool, is_shared=True),
            #这里的done是原来的done
            done: Tensor(shape=torch.Size([1]), device=cuda:0, dtype=torch.bool, is_shared=True),
            #输入policy网络的输出
            logits: Tensor(shape=torch.Size([1, 9]), device=cuda:0, dtype=torch.float32, is_shared=True),
            observation: Tensor(shape=torch.Size([1, 3, 3]), device=cuda:0, dtype=torch.float32, is_shared=True),      
            sample_log_prob: Tensor(shape=torch.Size([1]), device=cuda:0, dtype=torch.float32, is_shared=True),        
            step_count: Tensor(shape=torch.Size([1]), device=cuda:0, dtype=torch.float32, is_shared=True),
            terminated: Tensor(shape=torch.Size([1]), device=cuda:0, dtype=torch.bool, is_shared=True)},
        batch_size=torch.Size([]),
        device=None,
        is_shared=False)
    
    参数:
        td: 要打印的策略输出TensorDict
        env: 可选的环境实例，用于打印棋盘状态
        title: 打印信息的标题
        prefix: 打印时的缩进前缀
    """
    print(f"\n{'='*40}")
    print(f"🎲 {title}")
    print(f"{'='*40}")
    
    # 1. 策略选择的动作
    if "action" in td:
        action_val = td["action"].item()
        row, col = divmod(action_val, 3)
        print(f"{prefix}🎮 选择的动作: {action_val} (位置: {row},{col})")
    
    # 2. 策略logits和概率分布
    if "logits" in td:
        print(f"\n{prefix}🧠 策略logits值:")
        logits = td["logits"].squeeze(0).cpu()
        
        # 格式化为3x3棋盘
        logits_array = logits.reshape(3, 3)
        for i in range(3):
            row_str = f"{prefix}   "
            for j in range(3):
                row_str += f"[{logits_array[i, j]:.4f}] "
            print(row_str)
        
        # 计算并显示概率分布
        probs = torch.softmax(logits, dim=0)
        print(f"\n{prefix}📊 动作概率分布:")
        prob_array = probs.reshape(3, 3)
        
        # 如果有动作掩码，结合显示
        has_mask = "action_mask" in td
        if has_mask:
            mask = td["action_mask"].cpu().bool()
            mask_array = mask.reshape(3, 3)
        
        for i in range(3):
            row_str = f"{prefix}   "
            for j in range(3):
                pos_idx = i*3 + j
                prob = prob_array[i, j]
                
                if has_mask:
                    symbol = "✓" if mask_array[i, j] else "✗"
                    row_str += f"[{pos_idx}:{prob:.4f}{symbol}] "
                else:
                    row_str += f"[{pos_idx}:{prob:.4f}] "
            print(row_str)
    
    # 3. 动作的对数概率
    if "sample_log_prob" in td:
        log_prob = td["sample_log_prob"].item()
        prob = np.exp(log_prob)
        print(f"\n{prefix}📝 选择动作的对数概率: {log_prob:.4f}")
        print(f"{prefix}   对应概率值: {prob:.4%}")
    
    # 4. 观察向量 (当前棋盘状态)
    if "observation" in td:
        print(f"\n{prefix}👀 当前观察状态 (棋盘):")
        obs = td["observation"].cpu().numpy()
        
        # 提取X和O的位置
        board_display = np.full((3,3), '.')
        board_x = np.argwhere(obs[0] > 0)
        board_o = np.argwhere(obs[0] < 0)
        
        for pos in board_x:
            board_display[tuple(pos)] = 'X'
        for pos in board_o:
            board_display[tuple(pos)] = 'O'
        
        # 打印棋盘
        for row in board_display:
            print(f"{prefix}   |" + "|".join(row) + "|")
        
        
        current_player = 'X'
        print(f"{prefix}   当前玩家: {current_player}")
    
    # 5. 游戏状态信息
    print(f"\n{prefix}🎯 游戏状态信息:")
    
    if "step_count" in td:
        step_count = td["step_count"].item()
        print(f"{prefix}   步数: {step_count}")
    
    if "done" in td:
        done = td["done"].item()
        print(f"{prefix}   是否结束: {'是' if done else '否'}")
    
    if "terminated" in td:
        terminated = td["terminated"].item()
        print(f"{prefix}   是否终止: {'是' if terminated else '否'}")
    
    # 6. 环境状态比对 (如果提供)
    if env is not None:
        print(f"\n{prefix}🌍 环境中的实际状态:")
        if hasattr(env, "render_board"):
            board_str = env.render_board()
            for line in board_str.strip().split('\n'):
                print(f"{prefix}   {line}")
        
        if hasattr(env, "current_player"):
            player_symbol = 'X' if env.current_player == 1 else 'O'
            print(f"{prefix}   当前玩家: {player_symbol}")
    
    # 7. 策略分析 - 最优动作与掩码对比
    if "logits" in td and "action_mask" in td:
        print(f"\n{prefix}🔍 策略分析:")
        mask = td["action_mask"].cpu().bool()
        logits = td["logits"].squeeze(0).cpu()
        probs = torch.softmax(logits, dim=0)
        
        # 找出合法动作中概率最高的
        masked_probs = probs.clone()
        masked_probs[~mask] = -float('inf')
        best_legal_action = torch.argmax(masked_probs).item()
        best_prob = probs[best_legal_action].item()
        
        # 找出整体概率最高的
        overall_best_action = torch.argmax(probs).item()
        
        # 检查是否选择了最优合法动作
        chosen_action = td["action"].item() if "action" in td else None
        best_row, best_col = divmod(best_legal_action, 3)
        
        print(f"{prefix}   最优合法动作: {best_legal_action} (位置: {best_row},{best_col}), 概率: {best_prob:.4f}")
        
        if chosen_action is not None:
            is_optimal = chosen_action == best_legal_action
            print(f"{prefix}   所选动作是否最优: {'是' if is_optimal else '否'}")
            
            if not is_optimal:
                chosen_prob = probs[chosen_action].item()
                print(f"{prefix}   所选动作概率: {chosen_prob:.4f} vs 最优动作概率: {best_prob:.4f}")
        
        if overall_best_action != best_legal_action:
            overall_best_row, overall_best_col = divmod(overall_best_action, 3)
            overall_best_prob = probs[overall_best_action].item()
            print(f"{prefix}   注意: 概率最高的动作 {overall_best_action} (位置: {overall_best_row},{overall_best_col}) 是非法的")
    
    print(f"{'='*40}\n")

def print_value_output(td, env=None, title="value_module(td)返回的价值估计", prefix="  "):
    """
    打印价值网络(value_module)输出的TensorDict信息，格式化为易读的形式
    
    value_module(td)返回的td形如下：
    
    value_output: TensorDict(
        fields={
            action: Tensor(shape=torch.Size([1]), device=cuda:0, dtype=torch.int64, is_shared=True), #从action网络中得到的
            action_mask: Tensor(shape=torch.Size([9]), device=cuda:0, dtype=torch.bool, is_shared=True), #最开始的td里的
            done: Tensor(shape=torch.Size([1]), device=cuda:0, dtype=torch.bool, is_shared=True), #最开始的td里的
            logits: Tensor(shape=torch.Size([1, 9]), device=cuda:0, dtype=torch.float32, is_shared=True), #从action网络中得到的
            observation: Tensor(shape=torch.Size([1, 3, 3]), device=cuda:0, dtype=torch.float32, is_shared=True),    #最开始的td里的  
            sample_log_prob: Tensor(shape=torch.Size([1]), device=cuda:0, dtype=torch.float32, is_shared=True),        
            state_value: Tensor(shape=torch.Size([1, 1]), device=cuda:0, dtype=torch.float32, is_shared=True),
            step_count: Tensor(shape=torch.Size([1]), device=cuda:0, dtype=torch.float32, is_shared=True),
            terminated: Tensor(shape=torch.Size([1]), device=cuda:0, dtype=torch.bool, is_shared=True)},
        batch_size=torch.Size([]),
        device=None,
        is_shared=False)
    
    参数:
        td: 要打印的价值网络输出TensorDict
        env: 可选的环境实例，用于打印棋盘状态
        title: 打印信息的标题
        prefix: 打印时的缩进前缀
    """
    print(f"\n{'='*40}")
    print(f"💰 {title}")
    print(f"{'='*40}")
    
    # 1. 价值估计
    if "state_value" in td:
        value = td["state_value"].item()
        print(f"{prefix}💰 状态价值估计: {value:.4f}")
        
        # 提供价值解释
        if value > 0.8:
            print(f"{prefix}   解释: 非常有利的局面，很可能获胜")
        elif value > 0.5:
            print(f"{prefix}   解释: 相对有利的局面")
        elif value > 0.2:
            print(f"{prefix}   解释: 略微有利的局面")
        elif value > -0.2:
            print(f"{prefix}   解释: 中性局面，胜负难料")
        elif value > -0.5:
            print(f"{prefix}   解释: 略微不利的局面")
        elif value > -0.8:
            print(f"{prefix}   解释: 相对不利的局面")
        else:
            print(f"{prefix}   解释: 非常不利的局面，可能面临失败")
    
    # 2. 观察向量 (当前棋盘状态)
    if "observation" in td:
        print(f"\n{prefix}👀 当前观察状态 (棋盘):")
        obs = td["observation"].cpu().numpy()
        
        # 提取X和O的位置
        board_display = np.full((3,3), '.')
        board_x = np.argwhere(obs[0] > 0)
        board_o = np.argwhere(obs[0] < 0)
        
        for pos in board_x:
            board_display[tuple(pos)] = 'X'
        for pos in board_o:
            board_display[tuple(pos)] = 'O'
        
        # 打印棋盘
        for row in board_display:
            print(f"{prefix}   |" + "|".join(row) + "|")
        
        
        current_player = 'X'
        print(f"{prefix}   当前玩家: {current_player}")
        
        # 棋盘分析
        empty_count = np.sum(board_display == '.')
        x_count = np.sum(board_display == 'X')
        o_count = np.sum(board_display == 'O')
        print(f"{prefix}   棋盘分析: X={x_count}个, O={o_count}个, 空={empty_count}个")
    
    # 3. 游戏状态信息
    print(f"\n{prefix}🎯 游戏状态信息:")
    
    if "step_count" in td:
        step_count = td["step_count"].item()
        print(f"{prefix}   步数: {step_count}")
    
    if "done" in td:
        done = td["done"].item()
        print(f"{prefix}   是否结束: {'是' if done else '否'}")
    
    if "terminated" in td:
        terminated = td["terminated"].item()
        print(f"{prefix}   是否终止: {'是' if terminated else '否'}")
    
    # 4. 动作信息（如果有）
    if "action" in td and "sample_log_prob" in td:
        action_val = td["action"].item()
        row, col = divmod(action_val, 3)
        log_prob = td["sample_log_prob"].item()
        prob = np.exp(log_prob)
        
        print(f"\n{prefix}🎮 动作信息:")
        print(f"{prefix}   选择的动作: {action_val} (位置: {row},{col})")
        print(f"{prefix}   动作对数概率: {log_prob:.4f} (概率: {prob:.4%})")
    
    # 5. 动作掩码（如果有）
    if "action_mask" in td:
        print(f"\n{prefix}🎯 动作掩码:")
        mask = td["action_mask"].cpu().bool()
        mask_array = mask.reshape(3, 3)
        
        for i in range(3):
            row_str = f"{prefix}   "
            for j in range(3):
                pos_idx = i*3 + j
                symbol = "✓" if mask_array[i, j] else "✗"
                row_str += f"[{pos_idx}:{symbol}] "
            print(row_str)
        
        valid_actions = mask.sum().item()
        print(f"{prefix}   可用动作数: {valid_actions}/9")
    
    # 6. 环境状态比对 (如果提供)
    if env is not None:
        print(f"\n{prefix}🌍 环境中的实际状态:")
        if hasattr(env, "render_board"):
            board_str = env.render_board()
            for line in board_str.strip().split('\n'):
                print(f"{prefix}   {line}")
        
        if hasattr(env, "current_player"):
            player_symbol = 'X' if env.current_player == 1 else 'O'
            print(f"{prefix}   当前玩家: {player_symbol}")
            
        if hasattr(env, "winner"):
            winner = env.winner
            winner_str = "无" if winner == 0 else ("X" if winner == 1 else "O")
            print(f"{prefix}   当前赢家: {winner_str}")
    
    print(f"{'='*40}\n")

def test_env(env, actor_module, value_module):
    """Test environment interaction with policy and value networks.
    
    Args:
        env: TicTacToe environment instance
        actor_module: Policy network module
        value_module: Value network module
        
    Flow:
    1. Reset environment and get initial state
    2. Get policy predictions for state
    3. Get value predictions for state  
    4. Step environment with policy action
    5. Repeat 2-4 until episode ends
    """
    print("============================test start=============================")
    
    # Disable gradient computation for testing
    with torch.no_grad():
        # Reset environment and get initial state
        print("====================[[[[[[[env.reset()]]]]]]]===================") 
        td = env.reset()  # Returns TensorDict with obs, mask, step count, done flags
        print_reset_td(td, env)

        # Get policy network predictions
        print("====================[[[[[[actor_module(td)]]]]]]===================")
        policy_output = actor_module(td)  # Get action probabilities
        print_actor_output(policy_output, env, title="actor_module(td)返回的td")

        # Get value network predictions
        print("====================[[[[[[value_module(td)]]]]]]===================")
        value_output = value_module(td)  # Get state value estimate
        print_value_output(value_output, env, title="value_module(td)返回的td")

        # Take first environment step
        print("====================[[[[[[env.step(policy_output)]]]]]]===================")
        td = env.step(policy_output)  # Execute action and get next state
        print(f"~~~~~~~~~~~~~~action_mask: {td['action_mask']}")
        print(f"~~~~~~~~~~~~~~next.action_mask: {td['next']['action_mask']}")
        td = td['next']  # Get next state info
        print_step_td(td, env)

        # Continue episode until done
        step = 0
        while not td["done"].item():
            step += 1
            
            # Get policy prediction for current state
            print(f"====================[[[[[[next{step} -- actor_module(td)]]]]]]===================")
            next_policy_output = actor_module(td)
            print(f"111111111111111 {next_policy_output['action']}")  # Print selected action
            print(f"222222222222222 {next_policy_output['action_mask']}")  # Print valid moves mask

            # Get value prediction for current state
            print(f"====================[[[[[[next{step} -- value_module(td)]]]]]]===================")
            next_value_output = value_module(td)

            # Take environment step
            print(f"====================[[[[[[next{step} -- env.step(next_policy_output)]]]]]]===================")
            td = env.step(next_value_output)  # Execute action
            print(f"~~~~~~~~~~~~~~action_mask: {td['action_mask']}")
            print(f"~~~~~~~~~~~~~~next.action_mask: {td['next']['action_mask']}")
            td = td['next']  # Get next state
    
    print("============================test end=============================")

def inspect_replay_buffer(buffer, num_samples=3):
    """检查并打印回放缓冲区中的数据"""
    print("\n" + "="*60)
    print("【回放缓冲区检查】")
    print(f"缓冲区大小: {len(buffer)}")
    
    if len(buffer) == 0:
        print("缓冲区为空!")
        return
        
    samples = buffer.sample(num_samples)
    print(f"\n抽样查看 {num_samples} 个数据点:")
    
    # 先显示顶层键
    print("\n顶层键:")
    for key in samples.keys():
        if isinstance(samples[key], torch.Tensor):
            tensor = samples[key]
            print(f"  {key}: 形状={tensor.shape}, 类型={tensor.dtype}")
        else:
            print(f"  {key}: 类型={type(samples[key])}")
    
    # 查看第一个样本的详细信息
    print("\n第一个样本的详细信息:")
    sample = buffer.sample(1)
    for key in sample.keys(recursive=True):
        if key.find('next') == 0:  # 只查看next键下的内容
            tensor = sample[key]
            if isinstance(tensor, torch.Tensor):
                if tensor.numel() < 10:  # 如果元素数量少则打印全部
                    print(f"  {key}: {tensor.cpu().tolist()}, 形状={tensor.shape}")
                else:
                    print(f"  {key}: [...], 形状={tensor.shape}, 最大值={tensor.max().item():.4f}, 最小值={tensor.min().item():.4f}")

    print("="*60 + "\n")


def pretty_print_tensordict(td, title="TensorDict 内容详情", max_items=10, max_samples=None):
    """
    以人类可读的方式打印 TensorDict 的内容，支持打印多个样本
    
    参数:
        td: 要打印的 TensorDict
        title: 标题文字
        max_items: 每个张量最多显示的元素数量
        max_samples: 最多显示的样本数量，None表示显示全部
    """
    import numpy as np
    
    def format_tensor(tensor, max_items=max_items):
        """格式化张量为可读字符串"""
        if tensor.numel() <= max_items:
            # 显示全部元素
            content = str(tensor.cpu().tolist())
        else:
            # 显示部分元素和形状
            flat = tensor.flatten()
            content = f"[{', '.join([f'{x:.4f}' if isinstance(x.item(), float) else str(x.item()) for x in flat[:max_items]])}...]"
        
        return f"{content} (形状: {tensor.shape}, 类型: {tensor.dtype})"
    
    def format_board(obs):
        """将观察向量格式化为棋盘表示"""
        if obs.shape[-3:] != (3, 3, 3):
            return "非标准棋盘形状"
            
        # 确保我们有一个单一的观察，而不是批次
        if obs.dim() > 3:
            obs = obs[0]
            
        obs_np = obs.cpu().numpy()
        board = np.full((3, 3), '.')
        
        for i in range(3):
            for j in range(3):
                if obs_np[0, i, j] > 0:
                    board[i, j] = 'X'
                elif obs_np[1, i, j] > 0:
                    board[i, j] = 'O'
        
        player = "X" if obs_np[2, 0, 0] > 0 else "O"
        
        board_str = f"当前玩家: {player}\n"
        for row in board:
            board_str += "|" + "|".join(row) + "|\n"
        return board_str
    
    def format_action_mask(mask):
        """将动作掩码格式化为易读形式"""
        if mask.shape[-1] != 9:
            return format_tensor(mask)
            
        if mask.dim() > 1:
            mask = mask[0]  # 取第一个样本
            
        mask_np = mask.cpu().numpy().reshape(3, 3)
        result = "可用位置 (√=可用, ×=不可用):\n"
        
        for i in range(3):
            row = "|"
            for j in range(3):
                pos = i * 3 + j
                symbol = "√" if mask_np[i, j] else "×"
                row += f"{pos}:{symbol}|"
            result += row + "\n"
            
        return result
    
    def print_tensordict(td, indent=0):
        """递归打印 TensorDict 的内容"""
        prefix = "  " * indent
        
        # 处理空 TensorDict
        if td.is_empty():
            print(f"{prefix}(空 TensorDict)")
            return
            
        for key in td.keys():
            value = td[key]
            
            if key == "observation":
                print(f"{prefix}🎮 {key}:\n{prefix}{format_board(value)}")
            elif key == "action_mask":
                print(f"{prefix}🎯 {key}:\n{prefix}{format_action_mask(value)}")
            elif key == "action":
                # 将动作转换为坐标
                if value.dim() > 1:
                    action_val = value[0].item()  # 取第一个样本
                else:
                    action_val = value.item()
                row, col = divmod(action_val, 3)
                print(f"{prefix}👆 {key}: {action_val} (位置: {row},{col})")
            elif key == "reward":
                if value.dim() > 1:
                    reward_val = value[0].item()  # 取第一个样本
                else:
                    reward_val = value.item()
                print(f"{prefix}🏆 {key}: {reward_val:.4f}")
            elif key == "done" or key == "terminated":
                if value.dim() > 1:
                    done_val = value[0].item()  # 取第一个样本
                else:
                    done_val = value.item()
                print(f"{prefix}🏁 {key}: {'是' if done_val else '否'}")
            elif isinstance(value, dict) or hasattr(value, "keys"):
                # 遇到嵌套的 TensorDict
                print(f"{prefix}📂 {key}:")
                print_tensordict(value, indent + 1)
            elif hasattr(value, "shape"):
                # 其他张量
                print(f"{prefix}📊 {key}: {format_tensor(value)}")
            else:
                # 非张量值
                print(f"{prefix}🔖 {key}: {value}")
    
    print("\n" + "="*60)
    print(f"🔍 {title}")
    print("="*60)
    
    print(f"批次大小: {td.batch_size}")
    print(f"设备: {td.device}\n")
    
    # 如果批次大小非空，打印所有样本的内容
    if td.batch_size and td.batch_size[0] > 0:
        num_samples = td.batch_size[0]
        samples_to_print = num_samples if max_samples is None else min(num_samples, max_samples)
        print(f"【显示 {samples_to_print}/{num_samples} 个样本的信息】\n")
        
        for sample_idx in range(samples_to_print):
            print(f"\n=== 样本 {sample_idx+1}/{samples_to_print} ===")
            
            # 打印样本观察 (棋盘状态)
            if "observation" in td:
                print("棋盘状态:")
                obs = td["observation"][sample_idx]
                obs_np = obs.cpu().numpy()
                board_display = np.full((3,3), '.')
                
                for i in range(3):
                    for j in range(3):
                        if obs_np[0, i, j] > 0:
                            board_display[i, j] = 'X'
                        elif obs_np[1, i, j] > 0:
                            board_display[i, j] = 'O'
                
                for row in board_display:
                    print("|" + "|".join(row) + "|")
                
                player_channel = obs_np[2, 0, 0]
                current_player = 'X' if player_channel > 0 else 'O'
                print(f"当前玩家: {current_player}")
            
            # 打印动作掩码
            if "action_mask" in td:
                print("\n动作掩码:")
                mask = td["action_mask"][sample_idx].reshape(3, 3).cpu().numpy()
                for i in range(3):
                    row = "|"
                    for j in range(3):
                        pos = i * 3 + j
                        symbol = "√" if mask[i, j] else "×"
                        row += f"{pos}:{symbol}|"
                    print(row)
            
            # 打印动作
            if "action" in td:
                action = td["action"][sample_idx].item()
                row, col = divmod(action, 3)
                print(f"\n执行动作: {action} (位置: {row},{col})")
            
            # 打印动作概率
            if "sample_log_prob" in td:
                log_prob = td["sample_log_prob"][sample_idx].item()
                prob = np.exp(log_prob)
                print(f"动作概率: {prob:.4%} (log prob: {log_prob:.4f})")
            
            # 打印奖励
            if "next" in td and "reward" in td["next"]:
                reward = td["next", "reward"][sample_idx].item()
                print(f"获得奖励: {reward:.4f}")
            
            # 打印游戏是否结束
            if "next" in td and "done" in td["next"]:
                done = td["next", "done"][sample_idx].item()
                print(f"游戏结束: {'是' if done else '否'}")
    
    # 打印完整结构
    print("\n完整 TensorDict 结构:")
    print_tensordict(td)
    print("="*60 + "\n")


def setup_logging(model_dir):
    """Set up logging directory and file for training metrics.
    
    Args:
        model_dir: Base directory for model files
        
    Returns:
        str: Path to JSON log file for training metrics
    """
    # Create logs subdirectory
    log_dir = os.path.join(model_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    # Create log file path
    log_file = os.path.join(log_dir, f'training.json')
    return log_file

def log_metrics_function(log_file, metrics):
    """Save training metrics to JSON file.
    
    Args:
        log_file: Path to JSON log file
        metrics: Dictionary containing training metrics:
            - epoch_losses: List of average losses per epoch
            - policy_losses: List of policy losses
            - value_losses: List of value network losses
            - entropy_losses: List of entropy losses
            - rewards: List of average rewards
            - lr: List of learning rates
    """
    with open(log_file, 'w') as f:
        json.dump(metrics, f, indent=4)


