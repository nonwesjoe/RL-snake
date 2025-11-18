# RL-Snake（中文）

一个强化学习贪吃蛇项目，环境用 C++（pybind11）实现并在 Python 中调用，包含 PPO、REINFORCE、DQN 三种算法的训练与评估，支持手动游玩可视化。

## 环境与依赖
- Python: 3.10.18
- numpy: 2.2.6
- torch: 2.7.1+cu128
- matplotlib: 3.10.7
- pygame: 2.6.1
- pybind11: 3.0.1

安装依赖（示例）：
- `python -m pip install -U numpy torch matplotlib pygame pybind11`

## 构建 C++ 环境扩展
- 在项目根目录执行：
  - `python setup.py build_ext --inplace`
- 构建后会生成 `rl_snake/snake_env_cpp*.so`，Python 通过 `from rl_snake import SnakeEnv` 使用。

## 项目结构
- `rl_snake/` 环境包
  - `env.py` 环境 Python 封装（使用 C++ 扩展；提供 `reset/step/render`）
  - `snake_env_cpp*.so` C++ 扩展模块（构建产物）
- `cpp/` C++ 源码
  - `snake_env.cpp` 环境逻辑与 pybind11 绑定
- `agents/` 算法实现
  - `reinforce_agent.py`、`ppo_agent.py`、`dqn_agent.py`
- 训练脚本
  - `reinforce_train.py`、`ppo_train.py`、`dqn_train.py`
- 辅助脚本
  - `scripts/manual_play.py` 手动游玩
- 其它
  - `setup.py` 构建与包安装配置
  - `models/` 训练权重输出目录（示例）
  - `imgs/` 训练曲线输出目录（示例）

## 使用方法

### 1. 训练
- PPO：
  - `python ppo_train.py --mode train --episodes 20000 --size 40 --save ./models/ppo_snake_torch.pt --update_freq 2048 --epochs 4 --batch_size 128`
- REINFORCE：
  - `python reinforce_train.py --mode train --episodes 10000 --size 30 --save ./models/reinforce_snake_torch.pt`
- DQN：
  - `python dqn_train.py --mode train --episodes 5000 --size 30 --save ./models/dqn_snake_torch.pt`

参数说明（常用）：
- `--episodes` 训练总轮次；`--size` 地图大小；`--max_steps` 每轮最大步数
- `--save` 模型保存路径；`--seed` 随机种子
- PPO 特有：`--update_freq`、`--epochs`、`--batch_size`

### 2. 断点续训（resume）
- PPO：脚本内置，示例：
  - `python ppo_train.py --mode train --save ./models/ppo_snake_torch.pt --load ./models/ppo_snake_torch.pt --resume 1`
- REINFORCE：
  - `python reinforce_train.py --mode train --save ./models/reinforce_snake_torch.pt --load ./models/reinforce_snake_torch.pt --resume 1`
- DQN：
  - `python dqn_train.py --mode train --save ./models/dqn_snake_torch.pt --load ./models/dqn_snake_torch.pt --resume 1`
- 说明：若加载失败或结构不匹配，会容错打印并继续新训练；保存前会自动创建目录。

### 3. 评估与展示
- PPO：
  - `python ppo_train.py --mode eval --eval 10 --size 40 --render 1 --save ./models/ppo_snake_torch.pt`
- REINFORCE：
  - `python reinforce_train.py --mode eval --eval 3 --size 30 --render 1 --save ./models/reinforce_snake_torch.pt`
- DQN：
  - `python dqn_train.py --mode eval --eval 3 --size 30 --render 1 --save ./models/dqn_snake_torch.pt`
- 脚本会在 `imgs/` 目录输出训练曲线：
  - `imgs/ppo_scores.png`、`imgs/reinforce_scores.png`、`imgs/dqn_scores.png`

### 4. 手动游玩
- 运行：`python scripts/manual_play.py --size 10 --fps 10 --scale 30`
- 操作：方向键控制；`Esc` 退出

## 代码说明
- 环境（C++/pybind11）：
  - 模块 `snake_env_cpp` 实现栈蛇游戏逻辑，提供 `reset()`、`step(action)`、`get_snake()`、`get_head()`、`get_food()`、`get_direction()`、`set_direction(...)`。
  - `reset()` 返回观测向量 `numpy.float32`，长度 36；`step()` 返回 `(obs, reward, done, info)`，与 Gym 接口风格一致。
  - 奖励包含吃食物奖励、距离塑形（曼哈顿距离差），转向惩罚，以及饥饿终止。
- Python 封装与渲染：
  - `rl_snake/env.py` 将 C++ 环境绑定到 Python，并用 `pygame` 绘制蛇、头、食物；接口与训练脚本对接。
- 算法：
  - `agents/ppo_agent.py` 实现策略-价值网络与 GAE、clip 损失
  - `agents/reinforce_agent.py` 实现 REINFORCE 策略梯度（回合式）
  - `agents/dqn_agent.py` 实现 DQN（带 Dropout、Cosine 学习率调度、目标网络同步）

## 常见问题
- 报错 `snake_env_cpp not found`：先执行构建 `python setup.py build_ext --inplace`；或 `pip install -e .` 开发安装。
- 扩展导入路径：
  - 优先包内导入 `from rl_snake import SnakeEnv`；若扩展在根目录，封装有回退导入逻辑。
- 模型加载失败：检查模型结构与当前超参数是否一致，或删除旧模型重新训练。