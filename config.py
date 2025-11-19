import argparse


def get_config():
    """
    The configuration parser for common hyperparameters of all environment.
    Returns a parser that includes all necessary parameters including recurrent policy options.
    """
    parser = argparse.ArgumentParser(
        description="on policy RL", formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # 基础参数
    parser.add_argument("--algorithm_name", type=str, default="rmappo", choices=["rmappo", "mappo", "rmappg", "mappg", "trpo"])
    parser.add_argument("--experiment_name", type=str, default="check", help="实验标识符")
    parser.add_argument("--seed", type=int, default=1, help="随机种子")
    parser.add_argument("--cuda", action="store_true", default=True, help="是否使用GPU")
    parser.add_argument("--cuda_deterministic", action="store_true", default=False, help="是否确保随机种子有效")
    parser.add_argument("--n_training_threads", type=int, default=1, help="训练线程数")
    parser.add_argument("--n_rollout_threads", type=int, default=32, help="训练时的并行环境数")
    parser.add_argument("--n_eval_rollout_threads", type=int, default=1, help="评估时的并行环境数")
    parser.add_argument("--n_render_rollout_threads", type=int, default=1, help="渲染时的并行环境数")
    parser.add_argument("--num_env_steps", type=int, default=10000000, help="训练的环境步数")
    parser.add_argument("--user_name", type=str, default="marl", help="用户名")
    parser.add_argument("--use_wandb", action="store_true", default=True, help="是否使用wandb记录数据")

    # 环境参数
    parser.add_argument("--env_name", type=str, default="UAVNetEnv", choices=["UAVNetEnv", "UAVCommAB"], help="环境名称")
    parser.add_argument("--ab_top_k", type=int, default=3, help="UAVCommAB环境的top_k参数")
    parser.add_argument("--agent_num", type=int, default=6, help="智能体数量")
    parser.add_argument("--area_size", type=float, default=500.0, help="区域大小")
    parser.add_argument("--top_k", type=int, default=3, help="top_k参数")
    parser.add_argument("--scenario_name", type=str, default="UAVNet", help="场景名称")
    parser.add_argument("--use_obs_instead_of_state", action="store_true", default=False, help="是否使用观测代替状态")

    # 网络参数
    parser.add_argument("--share_policy", action="store_false", default=True, help="是否共享策略")
    parser.add_argument("--use_centralized_V", action="store_false", default=True, help="是否使用中心化V函数")
    parser.add_argument("--hidden_size", type=int, default=64, help="隐藏层维度")
    parser.add_argument("--layer_N", type=int, default=1, help="网络层数")
    parser.add_argument("--use_ReLU", action="store_false", default=True, help="是否使用ReLU激活函数")

    # 递归策略参数
    parser.add_argument("--use_recurrent_policy", action="store_true", default=False, help="是否使用递归策略")
    parser.add_argument("--use_naive_recurrent_policy", action="store_true", default=False, help="是否使用朴素递归策略")
    parser.add_argument("--recurrent_N", type=int, default=1, help="递归层数")
    parser.add_argument("--data_chunk_length", type=int, default=10, help="用于训练递归策略的数据块长度")

    # optimizer参数
    parser.add_argument("--lr", type=float, default=5e-4, help="learning rate (default: 5e-4)")
    parser.add_argument("--critic_lr", type=float, default=5e-4, help="critic learning rate (default: 5e-4)")
    parser.add_argument("--opti_eps", type=float, default=1e-5, help="RMSprop optimizer epsilon (default: 1e-5)")
    parser.add_argument("--weight_decay", type=float, default=0, help="权重衰减")

    # PPO参数
    parser.add_argument("--ppo_epoch", type=int, default=15, help="PPO轮数")
    parser.add_argument("--use_clipped_value_loss", action="store_false", default=True, help="是否裁剪价值损失")
    parser.add_argument("--clip_param", type=float, default=0.2, help="PPO裁剪参数")
    parser.add_argument("--num_mini_batch", type=int, default=1, help="小批量数量")
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="熵系数")
    parser.add_argument("--value_loss_coef", type=float, default=1, help="价值损失系数")
    parser.add_argument("--use_max_grad_norm", action="store_false", default=True, help="是否使用梯度裁剪")
    parser.add_argument("--max_grad_norm", type=float, default=10.0, help="梯度最大范数")
    parser.add_argument("--use_gae", action="store_false", default=True, help="是否使用GAE优势估计")
    parser.add_argument("--gamma", type=float, default=0.99, help="折扣因子")
    parser.add_argument("--gae_lambda", type=float, default=0.95, help="GAE lambda参数")
    parser.add_argument("--use_proper_time_limits", action="store_true", default=False, help="是否考虑时间限制")
    parser.add_argument("--use_huber_loss", action="store_false", default=True, help="是否使用huber损失")
    parser.add_argument("--use_value_active_masks", action="store_false", default=True, help="是否在价值损失中屏蔽无用数据")
    parser.add_argument("--use_policy_active_masks", action="store_false", default=True, help="是否在策略损失中屏蔽无用数据")
    parser.add_argument("--huber_delta", type=float, default=10.0, help="huber损失系数")

    # 运行参数
    parser.add_argument("--use_linear_lr_decay", action="store_true", default=False, help="是否使用线性学习率衰减")
    parser.add_argument("--episode_length", type=int, default=200, help="每个episode的最大长度")
    parser.add_argument("--save_interval", type=int, default=1, help="模型保存间隔")
    parser.add_argument("--log_interval", type=int, default=5, help="日志打印间隔")

    # 评估参数
    parser.add_argument("--use_eval", action="store_true", default=False, help="是否进行评估")
    parser.add_argument("--eval_interval", type=int, default=25, help="评估间隔")
    parser.add_argument("--eval_episodes", type=int, default=32, help="单次评估的episode数")

    # 渲染参数
    parser.add_argument("--save_gifs", action="store_true", default=False, help="是否保存渲染视频")
    parser.add_argument("--use_render", action="store_true", default=False, help="是否在训练过程中渲染环境")
    parser.add_argument("--render_episodes", type=int, default=5, help="渲染的episode数")
    parser.add_argument("--ifi", type=float, default=0.1, help="渲染视频中每张图像的播放间隔")

    # 预训练参数
    parser.add_argument("--model_dir", type=str, default=None, help="预训练模型路径")

    return parser
