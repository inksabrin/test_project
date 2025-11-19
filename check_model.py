import torch
import os

# 加载模型权重并打印层的形状
def check_model_weights(model_path):
    print(f"检查模型权重: {model_path}")
    try:
        weights = torch.load(model_path, map_location='cpu')
        print(f"成功加载模型权重，共包含 {len(weights)} 个权重参数")
        print("各层权重形状:")
        for name, param in weights.items():
            print(f"{name}: {param.shape}")
        return weights
    except Exception as e:
        print(f"加载模型时出错: {e}")
        return None

# 检查actor和critic模型
model_dir = '/Users/bytedance/Documents/trae_projects/light_mappo-main/results/UAVCommAB/UAVCommAB/rmappo/UAVCommAB_test/run57/models'
actor_path = os.path.join(model_dir, 'actor.pt')
critic_path = os.path.join(model_dir, 'critic.pt')

print("===== Actor Model ====")
actor_weights = check_model_weights(actor_path)

print("\n===== Critic Model ====")
critic_weights = check_model_weights(critic_path)

# 特别检查fc1层的输入维度
if actor_weights:
    for name in actor_weights:
        if 'fc1' in name:
            print(f"\n找到fc1层: {name}, 形状: {actor_weights[name].shape}")