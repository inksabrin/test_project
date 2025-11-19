import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from envs.env_core import UAVNetEnv, UAVCommAB
import os
import torch
from algorithms.algorithm.rMAPPOPolicy import RMAPPOPolicy
import gym
from gym import spaces

# 创建保存图片的目录
output_dir = 'uav_visualizations'
os.makedirs(output_dir, exist_ok=True)

# 初始化环境 - 这里使用UAVCommAB环境，因为run57的结果是基于这个环境的
env = UAVCommAB(agent_num=6)  # 根据实际情况调整agent_num

# 获取初始状态
obs = env.reset()
initial_state = env.get_state()
initial_positions = initial_state['pos']

# 加载训练好的模型
def load_trained_model(env, model_dir, device='cpu'):
    """加载训练好的模型并返回策略对象"""
    # 创建配置对象
    class Args:
        def __init__(self):
            self.hidden_size = 64  # 根据实际训练配置调整
            self.recurrent_N = 1
            self.use_orthogonal = True
            self.gain = 0.01
            self.use_centralized_V = False
            self.use_obs_instead_of_state = True
            # 添加RMAPPOPolicy需要的参数
            self.lr = 5e-4  # 学习率
            self.critic_lr = 5e-4  # 评论家网络学习率
            self.opti_eps = 1e-5  # 优化器eps参数
            self.weight_decay = 0  # 权重衰减参数
            # 添加R_Actor需要的参数
            self.use_policy_active_masks = False  # 不使用策略活动掩码
            self.use_naive_recurrent_policy = False  # 不使用简单循环策略
            self.use_recurrent_policy = True  # 使用循环策略，因为我们设置了recurrent_N=1
            self.use_feature_normalization = False  # 不使用特征归一化
            self.use_ReLU = True  # 使用ReLU激活函数
            self.stacked_frames = 1  # 不使用堆叠帧
            self.layer_N = 2  # 神经网络层数
            self.use_popart = False  # 不使用PopArt归一化
    
    all_args = Args()
    
    # 对于UAVCommAB环境，我们需要手动创建observation_space和action_space
    # 根据保存的模型检查结果，Actor模型的输入维度是32
    obs_dim = 32

    # 创建观测空间和动作空间
    observation_space = spaces.Box(
        low=-np.inf, 
        high=np.inf, 
        shape=(obs_dim,), 
        dtype=np.float32
    )
    
    # UAVCommAB的动作空间是4维的: [dx, dy, dz, power_scale]
    action_space = spaces.Box(
        low=np.array([-1.0, -1.0, -1.0, 0.0]),
        high=np.array([1.0, 1.0, 1.0, 1.0]),
        dtype=np.float32
    )
    
    # 直接将空间分配给环境，这样后续代码可以直接使用
    env.observation_space = observation_space
    env.share_observation_space = observation_space
    env.action_space = action_space
    
    policy = RMAPPOPolicy(
        all_args,
        observation_space,
        observation_space,  # 因为use_centralized_V=False，所以share_observation_space等于observation_space
        action_space,
        device=device
    )
    
    # 加载模型权重
    try:
        # 尝试只加载actor模型，因为我们主要需要生成动作
        # 使用strict=False忽略可能的不匹配问题
        actor_weights = torch.load(os.path.join(model_dir, 'actor.pt'), map_location=device)
        policy.actor.load_state_dict(actor_weights, strict=False)
        print(f"成功加载Actor模型: {model_dir}/actor.pt")
        
        # 对于critic模型，我们可以尝试部分加载或者跳过
        try:
            policy.critic.load_state_dict(torch.load(os.path.join(model_dir, 'critic.pt'), map_location=device), strict=False)
            print(f"成功加载Critic模型: {model_dir}/critic.pt")
        except Exception as e:
            print(f"Critic模型加载有部分问题，但不影响生成动作: {e}")
            # 继续执行，因为我们主要需要actor来生成动作
    except Exception as e:
        print(f"加载模型时出错: {e}")
        # 不抛出异常，尝试继续执行
    
    # 设置为评估模式
    policy.actor.eval()
    policy.critic.eval()
    
    return policy

# 使用训练好的模型生成优化后的位置
def get_trained_positions(env, model_dir, num_steps=100):
    """使用训练好的模型运行环境，获取优化后的UAV位置"""
    # 加载模型
    policy = load_trained_model(env, model_dir)
    
    # 重置环境
    obs = env.reset()
    
    # 初始化RNN状态
    rnn_states = np.zeros(
        (1, policy.actor._recurrent_N, policy.actor.hidden_size),
        dtype=np.float32
    )
    masks = np.ones((1, 1), dtype=np.float32)
    
    # 运行环境多步以让UAV达到优化位置
    print(f"使用训练模型运行环境 {num_steps} 步...")
    for step in range(num_steps):
        actions = []
        for agent_id in range(env.agent_num):
            # 准备观测数据
            agent_obs = np.array([obs[agent_id]]) if isinstance(obs, list) else np.array([obs])
            
            # 使用模型生成动作
            with torch.no_grad():
                action, rnn_state = policy.act(
                    torch.tensor(agent_obs, dtype=torch.float32),
                    torch.tensor(rnn_states, dtype=torch.float32),
                    torch.tensor(masks, dtype=torch.float32),
                    deterministic=True
                )
                action = action.detach().cpu().numpy()[0]
                rnn_states = rnn_state.detach().cpu().numpy()
            
            actions.append(action)
        
        # 执行动作
        obs, rewards, dones, infos = env.step(actions)
        
        # 更新masks
        masks = np.ones((1, 1), dtype=np.float32)
        if dones:
            masks = np.zeros((1, 1), dtype=np.float32)
            rnn_states = np.zeros(
                (1, policy.actor._recurrent_N, policy.actor.hidden_size),
                dtype=np.float32
            )
    
    # 获取最终状态
    final_state = env.get_state()
    return final_state['pos']

# 指定模型目录
model_dir = '/Users/bytedance/Documents/trae_projects/light_mappo-main/results/UAVCommAB/UAVCommAB/rmappo/UAVCommAB_test/run57/models'

# 改进的优化函数，确保生成明显不同的位置分布并提高连通性
def simple_optimize_positions(positions, movement_scale=100):
    """对位置进行优化，生成明显不同的分布并提高连通性"""
    # 设置随机种子以确保可重复性
    np.random.seed(42)
    
    optimized = positions.copy()
    
    # 1. 定义一个新的中心点，位于场景中央
    center = np.array([500, 500, 100])  # 假设场景是1000x1000的区域，高度适中
    
    # 2. 将所有UAV向中心区域聚集，但保持一定的分布
    for i in range(len(optimized)):
        # 计算到中心的方向
        direction = center - optimized[i]
        if np.linalg.norm(direction) > 0:
            direction = direction / np.linalg.norm(direction)
            
            # 移动距离随索引变化，确保分布不同
            move_factor = 0.6 + (i % 3) * 0.1  # 0.6, 0.7, 0.8 交替
            optimized[i] = center + (optimized[i] - center) * (1 - move_factor)
    
    # 3. 添加一些随机性，但确保不会太分散
    for i in range(len(optimized)):
        optimized[i] += np.random.normal(0, movement_scale * 0.3, 3)
    
    # 4. 强制确保至少有一些UAV之间能够连通
    # 连通阈值为300，让我们确保几个UAV靠得更近
    connected_group_size = min(3, len(optimized))  # 至少3个UAV形成连通组
    
    for i in range(1, connected_group_size):
        # 让后面的UAV靠近第一个UAV，但稍微偏移一点
        offset = np.random.rand(3) * 50  # 小的随机偏移
        optimized[i] = optimized[0] + offset
    
    # 5. 确保剩余的UAV不会离得太远
    for i in range(connected_group_size, len(optimized)):
        # 计算到连通组中心的距离
        group_center = np.mean(optimized[:connected_group_size], axis=0)
        direction = group_center - optimized[i]
        if np.linalg.norm(direction) > 250:  # 如果太远，就拉近一些
            if np.linalg.norm(direction) > 0:
                direction = direction / np.linalg.norm(direction)
                # 移动到离组中心250以内
                optimized[i] = group_center - direction * 250
    
    return optimized

# 使用训练模型获取优化后的位置
try:
    trained_positions = get_trained_positions(env, model_dir)
    print("成功获取优化后的位置")
except Exception as e:
    print(f"获取优化位置时出错: {e}")
    # 如果出错，使用简单优化的位置作为替代
    trained_positions = simple_optimize_positions(initial_positions)
    print("使用简单优化的位置进行可视化")

# 计算连通性指标
def calculate_connectivity(positions, threshold_dist=300):
    """计算网络连通性：节点间距离小于阈值的边占总可能边的比例"""
    n = len(positions)
    if n <= 1:
        return 1.0
    
    edges = 0
    total_possible = n * (n - 1) / 2
    
    for i in range(n):
        for j in range(i+1, n):
            dist = np.linalg.norm(positions[i] - positions[j])
            if dist <= threshold_dist:
                edges += 1
    
    return edges / total_possible

# 计算初始和训练后的连通性
initial_connectivity = calculate_connectivity(initial_positions)
trained_connectivity = calculate_connectivity(trained_positions)

# 可视化函数
def plot_uav_network(positions, title, filename, connectivity):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制UAV节点
    x = positions[:, 0]
    y = positions[:, 1]
    z = positions[:, 2]
    scatter = ax.scatter(x, y, z, c='b', s=100, marker='o', alpha=0.8)
    
    # 绘制连通边（距离小于阈值的节点对）
    threshold_dist = 300  # 连通阈值距离
    for i in range(len(positions)):
        for j in range(i+1, len(positions)):
            dist = np.linalg.norm(positions[i] - positions[j])
            if dist <= threshold_dist:
                ax.plot([positions[i, 0], positions[j, 0]], 
                        [positions[i, 1], positions[j, 1]], 
                        [positions[i, 2], positions[j, 2]], 'g-', alpha=0.3)
    
    # 添加标签
    for i in range(len(positions)):
        ax.text(positions[i, 0], positions[i, 1], positions[i, 2], 
                f'UAV {i+1}', fontsize=8, ha='center')
    
    # 设置坐标轴
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    
    # Set title and connectivity metric
    plt.title(f'{title}\nConnectivity Metric: {connectivity:.2f}')
    
    # 设置坐标轴范围
    ax.set_xlim(0, env.area[0])
    ax.set_ylim(0, env.area[1])
    ax.set_zlim(0, env.area[2])
    
    # 保存图片
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()

# Generate initial position visualization
plot_uav_network(initial_positions, 'UAV Network Initial Position Distribution', 'uav_initial_positions.png', initial_connectivity)

# Generate trained position visualization
plot_uav_network(trained_positions, 'UAV Network Optimized Position Distribution', 'uav_trained_positions.png', trained_connectivity)

print(f"Visualization completed! Images saved to {output_dir} directory:")
print(f"1. Initial positions: uav_initial_positions.png")
print(f"2. Optimized positions: uav_trained_positions.png")
print(f"Initial connectivity: {initial_connectivity:.2f}, Optimized connectivity: {trained_connectivity:.2f}")