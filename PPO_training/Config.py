class Config:
    model_dir = "training_model"
    # Game rules
    success_rate = 1  # Increased for more stable initial learning
    
    # Board parameters
    board_size = 12
    num_actions = 80
    
    # Model parameters
    num_res_blocks = 4  # Increased for larger board
    num_filters = 128  # Keep same for initial testing
    policy_filters = 32  # Increased for more complex policy
    value_filters = 32  # Increased for better value estimation
    value_num_groups = 8  # Must divide value_filters (32/8=4)
    policy_num_groups = 8  # Must divide policy_filters (32/8=4)
    num_groups = 16  # Must divide num_filters (128/16=8)
    
    # PPO parameters
    learning_rate = 0.0003  # Increased for faster initial learning
    weight_decay = 1e-4
    gamma = 0.99
    gae_lambda = 0.95  # Increased for better advantage estimation
    clip_epsilon = 0.2  # Increased for more exploration
    c1 = 0.2  # Increased value loss weight
    c2 = 0.02  # Slightly increased entropy for exploration
    update_epochs = 3  # Reduced for faster testing
    
    # Training parameters
    frames_per_batch = 256  # Reduced for quicker batches
    total_frames = 4096  # Reduced for initial testing
    sub_batch_size = 8  # Increased for better gradient estimation
    
    #big_batch_nums = total_frames // frames_per_batch 总步长除以每个batch需要的步长得到batch的次数 total_frames是enumerate取得步长数之和 frames——per——batch是一次big batches中得步长数
    #update_epoches是每个big batch需要更新的次数
    #frames_per_batch / sub_batch_size 是 smallbatch的次数
    #训练只需要修改total frames
    
    # Evaluation parameters
    eval_episodes = 10  # Reduced for faster feedback
    eval_frequency = 5  # More frequent evaluation
    
    # Logging parameters
    log_frequency = 1
    verbose = True  # Enable for debugging
    
    # Model saving parameters
    save_frequency = 5
    
    # Opponent parameters
    opponent_strategy = "random" # "network" or "random"