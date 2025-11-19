"""
Modified from OpenAI Baselines code to work with multi-agent envs and UAVNetEnv
"""

import numpy as np
from gymnasium import spaces

# 自定义空间类，用于处理UAV环境的连续动作和观察空间
class ContinuousSpace:
    def __init__(self, shape, low=-float('inf'), high=float('inf')):
        self.shape = shape
        self.low = low
        self.high = high
        self.dtype = np.float32

# UAV环境包装器
class UAVEnvWrapper:
    def __init__(self, env):
        self.env = env
        self.agent_num = env.agent_num
        
        # 为UAV环境创建模拟的observation_space和action_space
        # 观察空间: 基于UAVNetEnv的_agent_obs方法返回的观测维度
        sample_obs = env._agent_obs(0)
        obs_dim = sample_obs.shape[0]
        self.observation_space = ContinuousSpace(shape=(obs_dim,))
        # 对于集中式值函数，share_observation_space应该包含所有智能体的观测
        # 因此其形状应该是(agent_num * obs_dim,)
        self.share_observation_space = ContinuousSpace(shape=(self.agent_num * obs_dim,))
        
        # 动作空间: 4维连续动作 [dx, dy, dz, power_scale]
        self.action_space = ContinuousSpace(shape=(4,), low=-1.0, high=1.0)
    
    def reset(self, seed=None, options=None):
        # 调用原始环境的reset方法
        result = self.env.reset()
        
        # 处理不同的返回值格式
        if isinstance(result, tuple) and len(result) == 2:
            # gymnasium格式: (obs, info)
            obs, info = result
            return obs, info
        else:
            # 旧格式: 仅返回obs
            # 为了兼容gymnasium，我们需要返回(obs, {})格式
            return result, {}
    
    def step(self, actions):
        # 对动作进行缩放，使其符合UAV环境的期望范围
        scaled_actions = []
        for action in actions:
            # dx, dy, dz可以是任意值，但会被环境中的max_speed限制
            # power_scale确保在[0, 1]范围内
            scaled_action = [action[0], action[1], action[2], (action[3] + 1.0) / 2.0]
            scaled_actions.append(scaled_action)
        
        # 调用原始环境的step方法
        result = self.env.step(scaled_actions)
        
        # 处理不同的返回值格式
        if isinstance(result, tuple) and len(result) == 4:
            # 旧格式: (obs, reward, done, info)
            obs, reward, done, info = result
            # 转换为gymnasium格式: (obs, reward, terminated, truncated, info)
            # 由于原始环境没有区分terminated和truncated，我们将done作为terminated
            return obs, reward, done, False, info
        elif isinstance(result, tuple) and len(result) == 5:
            # 已经是gymnasium格式
            return result
        else:
            # 未知格式，保持原样
            return result
    
    def close(self):
        if hasattr(self.env, 'close'):
            self.env.close()
    
    def render(self, mode="human"):
        if hasattr(self.env, 'render'):
            return self.env.render(mode=mode)
        return None
    
    def seed(self, seed):
        if hasattr(self.env, 'seed'):
            return self.env.seed(seed)
        # UAVNetEnv在初始化时设置seed，这里可以忽略
        return None

# single env
class DummyVecEnv():
    def __init__(self, env_fns):
        # 包装原始环境函数，使其返回UAVEnvWrapper
        wrapped_fns = [lambda fn=fn: UAVEnvWrapper(fn()) for fn in env_fns]
        self.envs = [fn() for fn in wrapped_fns]
        env = self.envs[0]
        self.num_envs = len(env_fns)
        self.observation_space = env.observation_space
        self.share_observation_space = env.share_observation_space
        self.action_space = env.action_space
        self.actions = None

    def step(self, actions):
        """
        Step the environments synchronously.
        This is available for backwards compatibility.
        """
        self.step_async(actions)
        return self.step_wait()

    def step_async(self, actions):
        self.actions = actions

    def step_wait(self):
        results = [env.step(a) for (a, env) in zip(self.actions, self.envs)]
        
        # 处理gymnasium的step返回值 (obs, reward, terminated, truncated, info)
        # 对于仍使用旧API的环境，保持兼容性
        processed_results = []
        for result in results:
            if len(result) == 4:
                # 旧API: (obs, reward, done, info)
                obs, rew, done, info = result
                # 将done转换为terminated和truncated
                # 确保truncated具有与done相同的形状
                if isinstance(done, np.ndarray):
                    truncated = np.zeros_like(done)
                else:
                    truncated = False
                processed_results.append((obs, rew, done, truncated, info))
            else:
                # 新API: (obs, reward, terminated, truncated, info)
                processed_results.append(result)
        
        obs, rews, terminateds, truncateds, infos = map(np.array, zip(*processed_results))
        
        # 确保terminateds和truncateds具有相同的形状
        if terminateds.shape != truncateds.shape:
            # 如果形状不同，将truncateds扩展为与terminateds相同的形状
            if len(truncateds.shape) == 1 and len(terminateds.shape) == 2:
                # 将一维数组扩展为二维
                truncateds = np.repeat(truncateds[:, np.newaxis], terminateds.shape[1], axis=1)
        
        # 合并terminated和truncated为done（向后兼容）
        dones = terminateds | truncateds
        
        for (i, done) in enumerate(dones):
            if 'bool' in done.__class__.__name__:
                if done:
                    # 处理reset返回 (obs, info)
                    reset_result = self.envs[i].reset()
                    if len(reset_result) == 2:
                        obs[i], _ = reset_result
                    else:
                        obs[i] = reset_result
            else:
                if np.all(done):
                    # 处理reset返回 (obs, info)
                    reset_result = self.envs[i].reset()
                    if len(reset_result) == 2:
                        obs[i], _ = reset_result
                    else:
                        obs[i] = reset_result

        self.actions = None
        return obs, rews, dones, infos

    def reset(self):
        obs = []
        for env in self.envs:
            reset_result = env.reset()
            # 处理reset返回 (obs, info)
            if len(reset_result) == 2:
                obs_val, _ = reset_result
            else:
                obs_val = reset_result
            obs.append(obs_val)
        return np.array(obs) # [env_num, agent_num, obs_dim]

    def close(self):
        for env in self.envs:
            env.close()

    def render(self, mode="human"):
        if mode == "rgb_array":
            return np.array([env.render(mode=mode) for env in self.envs])
        elif mode == "human":
            for env in self.envs:
                env.render(mode=mode)
        else:
            raise NotImplementedError