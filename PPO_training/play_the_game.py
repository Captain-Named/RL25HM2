#注意到我们的评估代码没有考虑到下棋后只有50%概率会结束游戏的情况 
#其实可以直接用收集到的数据的胜率来评估智能体的表现 
import torch
import numpy as np
import random
import time
from torchrl.envs.utils import TensorDict
from torchrl.modules import MaskedCategorical
from tensordict.nn import TensorDictModule, ProbabilisticTensorDictModule, ProbabilisticTensorDictSequential
from train_PPO import TicTacToeEnv, SimplePolicyNetwork, device, Config
import os

class RandomTicTacToeAgent:
    def __init__(self, model_path="./training_model/ppo_tictactoe_best.pth"):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.env = TicTacToeEnv()
        self.config = Config()
        self.policy_net = SimplePolicyNetwork(self.config).to(self.device)
        
        print("开始测试随机智能体...")
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            self.policy_net.load_state_dict(checkpoint['policy'])
            print(f"模型已从 {model_path} 加载")
        else:
            print(f"警告: 未找到模型文件 {model_path}，使用初始化模型")
        
        policy_module = TensorDictModule(
            self.policy_net,
            in_keys=["observation"],
            out_keys=["logits"],
        )
        
        dist_module = ProbabilisticTensorDictModule(
            in_keys={"logits": "logits", "mask": "action_mask"},
            out_keys=["action"],
            distribution_class=MaskedCategorical,
            return_log_prob=False,
        )
        
        self.actor_module = ProbabilisticTensorDictSequential(policy_module, dist_module)
        self.actor_module.eval()
        
        self.stats = {'ppo_wins': 0, 'random_wins': 0, 'draws': 0, 'total_games': 0}
        
    def print_board(self):
        """打印当前棋盘状态"""
        symbols = {0: ' . ', 1: ' X ', -1: ' O ', None: ' # '}
        print("\n    " + "".join(f"{j:^3}" for j in range(12)))
        print("   " + "-" * 38)
        
        for i in range(12):
            print(f"{i:2} |", end="")
            for j in range(12):
                if (4 <= i < 8) or (4 <= j < 8):  # 十字形区域
                    print(symbols[self.env.board[i][j]], end="")
                else:
                    print(symbols[None], end="")
            print("|")
        print("   " + "-" * 38)
            
    def get_ppo_agent_move(self):
        """获取PPO智能体的下一步棋"""
        with torch.no_grad():
            obs = self.env._get_obs()
            mask = self.env._get_action_mask()
            
            td = TensorDict({
                "observation": obs.unsqueeze(0),
                "action_mask": mask.unsqueeze(0)
            }, batch_size=[1])
            
            actions = self.actor_module(td)
            action = actions["action"].item()
            
            return self.env.index_to_position(action)
            
    def get_random_agent_move(self):
        """获取随机智能体的下一步棋"""
        legal_moves = self.env.get_legal_moves()
        if legal_moves:
            return random.choice(legal_moves)
        return None
        
    def play_game(self, ppo_first=True, visualize=True):
        """进行一局游戏"""
        self.env._reset()
        current_player = 1 if ppo_first else -1
        
        if visualize:
            print("\n" + "="*50)
            print(f"{'PPO智能体' if ppo_first else '随机智能体'}先手")
            print("="*50)
            print("PPO智能体: X  随机智能体: O")
            self.print_board()
        
        while True:
            if current_player == 1:
                i, j = self.get_ppo_agent_move()
                self.env.board[i][j] = 1
                if visualize:
                    print(f"\nPPO智能体下子在位置: {i},{j}")
            else:
                i, j = self.get_random_agent_move()
                if i is None: #i is none 表示没有合法落子位置
                    break
                self.env.board[i][j] = -1
                if visualize:
                    print(f"\n随机智能体下子在位置: {i},{j}")
            
            if visualize:
                self.print_board()
                time.sleep(0.5)
                
            winner = self.env._check_winner()
            if winner is not None:
                if winner == 1:
                    if visualize:
                        print("\nPPO智能体赢了!")
                    self.stats['ppo_wins'] += 1
                elif winner == -1:
                    if visualize:
                        print("\n随机智能体赢了!")
                    self.stats['random_wins'] += 1
                else: #winner==0
                    if visualize:
                        print("\n平局!")
                    self.stats['draws'] += 1
                break
                
            current_player *= -1
            
        self.stats['total_games'] += 1
        return winner
        
    def evaluate(self, num_games=100):
        """评估PPO智能体与随机智能体的对战表现"""
        self.stats = {'ppo_wins': 0, 'random_wins': 0, 'draws': 0, 'total_games': 0}
        
        print(f"开始评估 {num_games} 局游戏...")
        
        for i in range(num_games):
            if i % 2 == 0:
                self.play_game(ppo_first=True, visualize=False)
            else:
                self.play_game(ppo_first=False, visualize=False)
                
            if (i+1) % 10 == 0:
                print(f"已完成 {i+1}/{num_games} 局")
        
        print("\n" + "="*50)
        print("评估结果:")
        print(f"总游戏数: {self.stats['total_games']}")
        print(f"PPO智能体胜利数: {self.stats['ppo_wins']} ({self.stats['ppo_wins']/self.stats['total_games']*100:.1f}%)")
        print(f"随机智能体胜利数: {self.stats['random_wins']} ({self.stats['random_wins']/self.stats['total_games']*100:.1f}%)")
        print(f"平局数: {self.stats['draws']} ({self.stats['draws']/self.stats['total_games']*100:.1f}%)")
        print("="*50)

def main():
    print("=" * 60)
    print("十字形井字棋游戏 - PPO智能体 vs 随机智能体")
    print("=" * 60)
    
    agent = RandomTicTacToeAgent()
    
    while True:
        print("\n请选择操作:")
        print("[1] 观看单局游戏 (PPO先手)")
        print("[2] 观看单局游戏 (随机智能体先手)")
        print("[3] 批量评估 (100局)")
        print("[4] 批量评估 (1000局)")
        print("[q] 退出")
        
        choice = input("\n请选择: ")
        
        if choice == '1':
            agent.play_game(ppo_first=True, visualize=True)
        elif choice == '2':
            agent.play_game(ppo_first=False, visualize=True)
        elif choice == '3':
            agent.evaluate(num_games=100)
        elif choice == '4':
            agent.evaluate(num_games=1000)
        elif choice.lower() in ['q', 'quit', 'exit']:
            break
        else:
            print("无效的选择，请重试")
    
    print("程序已退出")

if __name__ == "__main__":
    main()