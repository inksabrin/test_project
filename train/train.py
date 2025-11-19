"""
# @Time    : 2021/6/30 10:07 下午
# @Author  : hezhiqiang
# @Email   : tinyzqh@163.com
# @File    : train.py
"""

# !/usr/bin/env python
import os
import sys
import socket
import setproctitle
import numpy as np
from pathlib import Path
import torch

# Get the parent directory of the current file
parent_dir = os.path.abspath(os.path.join(os.getcwd(), "."))

# Append the parent directory to sys.path, otherwise the following import will fail
sys.path.append(parent_dir)

from config import get_config
from envs.env_wrappers import DummyVecEnv

"""Train script for MPEs."""


def make_train_env(all_args):
    def get_env_fn(rank):
        def init_env():
            # 导入环境类
            from envs.env_core import UAVNetEnv, UAVCommAB

            # 根据环境名称选择合适的环境类
            # env = UAVNetEnv(
            #     agent_num=all_args.num_agents,
            #     area_size=(1000.0, 1000.0, 200.0),
            #     dt=1.0,
            #     max_speed=10.0,
            #     top_k=4,
            #     seed=all_args.seed + rank * 1000
            # )
            if all_args.env_name == 'UAVCommAB':
                env = UAVCommAB(
                    agent_num=all_args.agent_num,
                    area_size=getattr(all_args, 'area_size_3d', (float(all_args.area_size), float(all_args.area_size), 200.0)),
                    dt=1.0,
                    max_speed=10.0,
                    top_k=all_args.top_k,
                    seed=all_args.seed + rank * 1000
                )
            else:  # 默认使用UAVNetEnv
                env = UAVNetEnv(
                    agent_num=all_args.agent_num,
                    area_size=(1000.0, 1000.0, 200.0),
                    dt=1.0,
                    max_speed=10.0,
                    top_k=4,
                    seed=all_args.seed + rank * 1000
                )

            return env

        return init_env

    return DummyVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])


def make_eval_env(all_args):
    def get_env_fn(rank):
        def init_env():
            # 导入环境类
            from envs.env_core import UAVNetEnv, UAVCommAB

            # 根据环境名称选择合适的环境类

            # env = UAVNetEnv(
            #     agent_num=all_args.num_agents,
            #     area_size=(1000.0, 1000.0, 200.0),
            #     dt=1.0,
            #     max_speed=10.0,
            #     top_k=4,
            #     seed=all_args.seed + rank * 1000
            # )
            if all_args.env_name == 'UAVCommAB':
                env = UAVCommAB(
                    agent_num=all_args.agent_num,
                    area_size=(all_args.area_size, all_args.area_size, 200.0),
                    dt=1.0,
                    max_speed=10.0,
                    top_k=all_args.top_k,
                    seed=all_args.seed + rank * 1000
                )
            else:  # 默认使用UAVNetEnv
                env = UAVNetEnv(
                    agent_num=all_args.agent_num,
                    area_size=(1000.0, 1000.0, 200.0),
                    dt=1.0,
                    max_speed=10.0,
                    top_k=4,
                    seed=all_args.seed + rank * 1000
                )

            return env

        return init_env

    return DummyVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])


def parse_args(args, parser):
    # 添加config.py中没有定义但train.py需要的参数
    parser.add_argument('--gain', type=float, default=0.01, help="gain for action distribution")
    parser.add_argument('--use_orthogonal', action='store_true', default=True, help="Whether to use orthogonal initialization")
    parser.add_argument('--use_feature_normalization', action='store_true', default=False, help="Whether to use feature normalization")
    parser.add_argument('--stacked_frames', type=int, default=1, help="Number of stacked frames")
    parser.add_argument('--use_popart', action='store_true', default=False, help="Whether to use PopArt for normalized returns")
    parser.add_argument('--use_valuenorm', action='store_true', default=True, help="Whether to use value normalization")
    
    # 使用parse_known_args代替parse_args，这样会忽略未识别的参数
    all_args, unknown_args = parser.parse_known_args(args)
    
    # 手动设置recurrent policy参数，确保它们在all_args中存在
    all_args.use_recurrent_policy = "--use_recurrent_policy" in args
    all_args.use_naive_recurrent_policy = "--use_naive_recurrent_policy" in args
    
    # 确保必要的参数存在
    # 对于UAVCommAB环境，设置特定参数
    if all_args.env_name == 'UAVCommAB':
        all_args.agent_num = 6  # UAVCommAB环境中的UAV数量
        all_args.num_agents = all_args.agent_num  # 确保num_agents与agent_num一致
        # 确保area_size是正确的数值类型
        area_size_value = float(all_args.area_size)
        # 存储为单独的属性，而不是直接修改area_size
        all_args.area_size_3d = (area_size_value, area_size_value, 200.0)
        all_args.top_k = all_args.ab_top_k  # 对UAVCommAB使用ab_top_k
        all_args.scenario_name = "UAVCommAB"  # 更新场景名称以保持一致性
    
    # 确保必要的训练参数存在
    if not hasattr(all_args, 'use_eval'):
        all_args.use_eval = False
    
    if not hasattr(all_args, 'num_agents') and hasattr(all_args, 'agent_num'):
        all_args.num_agents = all_args.agent_num
        
    return all_args


def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)
    
    # 直接在代码中设置use_recurrent_policy和use_naive_recurrent_policy参数
    # 检查命令行参数中是否包含--use_recurrent_policy或--use_naive_recurrent_policy
    all_args.use_recurrent_policy = '--use_recurrent_policy' in args
    all_args.use_naive_recurrent_policy = '--use_naive_recurrent_policy' in args
    
    # 对于rmappo算法，默认启用use_recurrent_policy如果两个参数都未设置
    if all_args.algorithm_name == "rmappo" and not (all_args.use_recurrent_policy or all_args.use_naive_recurrent_policy):
        all_args.use_recurrent_policy = True

    # 验证策略类型与算法的兼容性
    if all_args.algorithm_name == "rmappo":
        assert all_args.use_recurrent_policy or all_args.use_naive_recurrent_policy, "check recurrent policy!"
    elif all_args.algorithm_name == "mappo":
        assert (
            not all_args.use_recurrent_policy and not all_args.use_naive_recurrent_policy
        ), "check recurrent policy!"
    else:
        raise NotImplementedError

    assert (
        all_args.share_policy == True and all_args.scenario_name == "simple_speaker_listener"
    ) == False, "The simple_speaker_listener scenario can not use shared policy. Please check the config.py."

    # cuda
    if all_args.cuda and torch.cuda.is_available():
        print("choose to use gpu...")
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("choose to use cpu...")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    # run dir
    run_dir = (
        Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0] + "/results")
        / all_args.env_name
        / all_args.scenario_name
        / all_args.algorithm_name
        / all_args.experiment_name
    )
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    if not run_dir.exists():
        curr_run = "run1"
    else:
        exst_run_nums = [
            int(str(folder.name).split("run")[1])
            for folder in run_dir.iterdir()
            if str(folder.name).startswith("run")
        ]
        if len(exst_run_nums) == 0:
            curr_run = "run1"
        else:
            curr_run = "run%i" % (max(exst_run_nums) + 1)
    run_dir = run_dir / curr_run
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    setproctitle.setproctitle(
        str(all_args.algorithm_name)
        + "-"
        + str(all_args.env_name)
        + "-"
        + str(all_args.experiment_name)
        + "@"
        + str(all_args.user_name)
    )

    # seed
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    # env init
    envs = make_train_env(all_args)
    eval_envs = make_eval_env(all_args) if all_args.use_eval else None
    
    # 从环境中获取实际的智能体数量（确保与UAVNetEnv设置一致）
    num_agents = all_args.num_agents

    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": num_agents,
        "device": device,
        "run_dir": run_dir,
    }

    # run experiments
    if all_args.share_policy:
        from runner.shared.env_runner import EnvRunner as Runner
    else:
        from runner.separated.env_runner import EnvRunner as Runner

    runner = Runner(config)
    runner.run()

    # post process
    envs.close()
    if all_args.use_eval and eval_envs is not envs:
        eval_envs.close()

    runner.writter.export_scalars_to_json(str(runner.log_dir + "/summary.json"))
    runner.writter.close()


if __name__ == "__main__":
    main(sys.argv[1:])
