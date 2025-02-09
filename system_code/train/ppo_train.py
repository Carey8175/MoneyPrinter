import os
import copy
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from datetime import datetime
from sklearn.preprocessing import StandardScaler

from system_code.core.modules.print_system import PrintSystem
from system_code.core.modules.position import HoldingPeriod, HoldingGroup
from stable_baselines3.common.callbacks import BaseCallback
from system_code.core.config import Config


class TrainingLogger(BaseCallback):
    def __init__(self, verbose=1):
        super(TrainingLogger, self).__init__(verbose)

    def _on_step(self) -> bool:
        if self.n_calls % 100 == 0:  # 每 100 个 step 记录一次
            mean_reward = np.mean(self.locals["rewards"])  # 当前平均奖励
            loss = self.model.logger.name_to_value.get("train/loss", None)
            entropy = self.model.logger.name_to_value.get("train/entropy_loss", None)

            print(f"🔹 Step: {self.n_calls}, Mean Reward: {mean_reward:.4f}, Loss: {loss}, Entropy: {entropy}")
        return True  # 继续训练


class HoldingPeriodEnv(gym.Env):
    """
    自定义环境:
      - 输入: 一个 HoldingPeriod 对象 (包含了多根K线行情)
      - 每个 step:
          action=0 => 继续持仓
          action=1 => 平仓(episode结束)
      - episode 在以下情况结束:
          1. 动作为1(平仓)
          2. 到达HoldingPeriod最后一根bar

    """
    def __init__(self, holding_period, fee_rate=0.0005):
        super(HoldingPeriodEnv, self).__init__()

        self.hp = holding_period
        self.fee_rate = fee_rate

        # 将行情数据取出来，后面step逐个取值
        self.data = self.hp.data.reset_index(drop=True)
        self.n_bars = len(self.data)

        # 观察空间
        # 这里先找一下有哪些列可用:
        drop_cols = [
            'inst_id_x', 'inst_id_y', 'ts','taker_sell', 'taker_buy', 'open_interest',
            'elite_long_short_ratio', 'elite_position_long_short_ratio',
            'all_long_short_ratio',
            'bullCondition', 'bearCondition', 'bullCondition_shift1',
            'bearCondition_shift1', 'buySignalRaw', 'sellSignalRaw',
            'buySignal', 'sellSignal', 'longFilter', 'shortFilter'
        ]
        self.data = self.data.drop(columns=drop_cols, errors='ignore')
        self.feature_columns = self.data.columns
        # 标准化处理data
        # scaler = StandardScaler()
        # self.data = pd.DataFrame(scaler.fit_transform(self.data), columns=self.data.columns)


        # gym 需要定义 observation_space，用于描述状态维度、范围等
        # 这里只是一个示例，把所有 feature 做成一个向量
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(len(self.feature_columns),), dtype=np.float32
        )

        # 动作空间: 0=继续持有, 1=平仓
        self.action_space = spaces.Discrete(2)

        # 记录环境在episode中的状态
        self.current_step = 0
        # 持仓期间实时评估，最大浮亏
        self.max_unrealized_loss = 0.0
        # 进场价格: 这里就拿 HoldingPeriod 进场时的 entry_price
        self.entry_price = self.hp.entry_price
        # 杠杆
        self.leverage = self.hp.leverage

    def reset(self, **kwargs):
        """
        在一个episode开始时 (即一个HoldingPeriod开始时), 将状态置0
        :param **kwargs:
        """
        self.current_step = 0
        # 重置最大浮亏 (若要用 hp 内部的已计算值, 可以直接拿 self.hp.max_unrealized_loss)
        self.max_unrealized_loss = 0.0
        info = {}

        return self._get_observation(), info

    def step(self, action):
        """
        执行动作
        """
        done = False
        reward = 0.0

        # 先根据当前bar的信息来更新 max_unrealized_loss
        # unrealized PnL = (当前价格 - entry_price) / entry_price * leverage (多头情况)
        # 如果是空头可改逻辑。也可以拿 HoldingPeriod 内置的 logic.
        row = self.data.iloc[self.current_step]
        current_price = row['close']
        if self.hp.side == 'long':
            unrealized_pnl = self.leverage * ((current_price - self.entry_price) / self.entry_price)
        else:
            unrealized_pnl = self.leverage * ((self.entry_price - current_price) / self.entry_price)

        # 最大浮亏, 其实就是最小的"unrealized_pnl"与0的差值, 或者 negative part
        # 为演示简单, 这里定义:
        if unrealized_pnl < 0:
            # negative part
            if unrealized_pnl < self.max_unrealized_loss:
                self.max_unrealized_loss = unrealized_pnl

        # 判断是否要平仓 or 到达最后一个bar
        if action == 1 or self.current_step == (self.n_bars - 1):
            # done
            done = True

            # 计算最终收益
            final_price = row['close']
            if self.hp.side == 'long':
                final_profit = self.leverage * ((final_price - self.entry_price) / self.entry_price)
            else:
                final_profit = self.leverage * ((self.entry_price - final_price) / self.entry_price)

            # 扣除手续费 (进场+出场)
            # 例如: final_profit_fee = (1 - fee_rate) * (1 + final_profit) * (1 - fee_rate) - 1
            # 这里简单化处理
            final_profit_fee = (1 - self.fee_rate) * (1 + final_profit) * (1 - self.fee_rate) - 1

            # 持仓 bar 数
            holding_bars = self.current_step + 1  # 从0数起，要+1
            if self.max_unrealized_loss == 0:
                # 如果从未浮亏，可以给予更大加成 or 直接视为 final_profit
                drawdown_ratio = final_profit_fee
            else:
                # final_profit_fee / abs(self.max_unrealized_loss)
                drawdown_ratio = final_profit_fee / abs(self.max_unrealized_loss)

            # 将三个要素简单相加
            # 1) final_profit_fee
            # 2) final_profit_fee / holding_bars
            # 3) drawdown_ratio
            reward = self.compute_reward(final_profit_fee, holding_bars, self.max_unrealized_loss)

        # 如果没有结束，还要往后走一步
        self.current_step += 1
        if self.current_step >= self.n_bars:
            self.current_step = self.n_bars - 1  # 防止越界

        # 组装下一个 observation
        obs = self._get_observation()
        info = {}

        return obs, reward, done, False, info

    def _get_observation(self):
        """
        返回当前bar的特征向量
        """
        row = self.data.iloc[self.current_step]
        obs = row[self.feature_columns].values.astype(np.float32)

        return obs

    @staticmethod
    def compute_reward(final_profit_fee, holding_bars, max_unrealized_loss, max_hold_bars=288):
        """
        计算奖励函数，优化奖励值的尺度，确保训练稳定。
        """
        # 1. 归一化收益: [-0.2, 0.2] -> [-1, 1]
        scaled_profit = final_profit_fee * 10

        # 2. 单位时间收益，归一化持仓时间影响
        if final_profit_fee > 0:
            scaled_time_profit = final_profit_fee * 10 / holding_bars
        # 2.1 考虑情况：相同亏损，但是持仓时间更长，应该更差
        else:
            scaled_time_profit = final_profit_fee * 10 / holding_bars * max_hold_bars

        # 3. 收益/最大浮亏比值，避免数值极端
        drawdown_ratio = final_profit_fee / (abs(max_unrealized_loss) + 1e-6)
        drawdown_ratio = np.clip(drawdown_ratio, -2, 2)  # 限制范围

        # 4. 组合奖励，并使用 tanh 限制整体范围
        a = 3 / 5
        b = 1 / 5
        c = 1 / 5
        reward = np.tanh(a * scaled_profit +
                         b * scaled_time_profit +
                         c * drawdown_ratio)

        return reward


class TrainingModule(PrintSystem):
    def __init__(
            self,
            bar: str,
            inst_id: str,
            train_begin: datetime,
            train_end: datetime,
            test_begin: datetime,
            test_end: datetime,
            load_from_local=False,
            total_timesteps=250000
    ):
        super().__init__(bar, inst_id, train_begin, train_end)
        self.load_from_local = load_from_local
        self.total_timesteps = total_timesteps
        self.test_begin = test_begin
        self.test_end = test_end
        self.train_begin = train_begin
        self.train_end = train_end

    @staticmethod
    def train_model_for_group(holding_group, total_timesteps=200000):
        """
        针对一个HoldingGroup，训练一个RL模型
        :param holding_group: HoldingGroup对象
        :param total_timesteps: 训练的总timesteps
        :return: 训练好的模型
        """

        # 1. 为HoldingGroup中的每个HoldingPeriod创建一个环境
        envs = []
        for hp in holding_group.holdings:
            # 创建环境实例
            def _make_env(hp_local):
                # stable-baselines3 需要无参函数，所以封一个lambda
                return lambda: HoldingPeriodEnv(hp_local)

            envs.append(_make_env(hp))

        # 2. 用 DummyVecEnv 并行（或串行）封装这些环境
        # envs 是个list[callable], 需传给 DummyVecEnv
        vec_env = DummyVecEnv(envs)

        # 3. 初始化PPO模型
        # 可以自己选取合适的超参数
        model = PPO(
            policy='MlpPolicy',
            env=vec_env,
            verbose=1,
            n_steps=128,
            learning_rate=1e-4,
            gamma=0.99
        )

        # 4. 开始训练
        model.learn(total_timesteps=total_timesteps, callback=TrainingLogger())

        return model

    def run_reinforcement_learning_system(self, holding_groups):
        """
        假设我们已经通过 open_module.open_position(data) 得到了多个HoldingGroup
        然后对每个HoldingGroup单独训练一个RL模型
        """
        # 1. 比如你已经获得了 holding_groups 列表: List[HoldingGroup]
        # 例如:
        # holding_groups = self.open_module.open_position(data)
        # 这里我们先假设 holding_groups 已经有了

        # 2. 对每个HoldingGroup训练并保存模型
        group_models = []
        for i, group in enumerate(holding_groups):
            # 训练
            model = self.train_model_for_group(group, total_timesteps=self.total_timesteps)
            group_models.append(model)

            # 保存模型
            model.save(os.path.join(Config.MODEL_DIR, f"{self.inst_id}_ppo_model_group_{i}.zip"))

        print("训练完成，已生成所有模型")
        return group_models

    def inference_on_holding_period(self, model, hp: HoldingPeriod):
        """
        使用训练好的 model 对指定的 holding_period 进行决策
        """
        env = HoldingPeriodEnv(hp)
        obs, info = env.reset()
        done = False
        step_count = 0
        while not done:
            # 1) 获取动作
            try:
                action, _states = model.predict(obs, deterministic=True)
            except Exception as e:
                print(f"Error: {e}")
                action = 0

            # 2) 与环境交互
            obs, reward, done, _, info = env.step(action)

            step_count += 1
            if done:
                hp.end = datetime.fromtimestamp(hp.data['ts'].iloc[step_count] / 1000)
                print(f"Episode done at step: {step_count}, reward={reward}")
                break

    def simulate_close_module(self, model, holding_group):
        """
        使用训练好的模型对每个 HoldingGroup 进行决策
        """
        for hp in holding_group.holdings:
            self.inference_on_holding_period(model, hp)

    def run(self):
        self.begin = self.test_begin
        self.end = self.test_end

        data = self.fetch_data()
        test_groups = self.open_module.open_position(data)

        if not self.load_from_local:
            self.begin = self.train_begin
            self.end = self.train_end

            data = self.fetch_data()
            train_groups = self.open_module.open_position(data)

            group_models = self.run_reinforcement_learning_system(train_groups)

            for model, group in zip(group_models, train_groups):
                self.simulate_close_module(model, group)
                group.update()

            df = self.backtest.run(train_groups)
            print(df.to_string())
            self.backtest.plot_all(save=True, inst_id=self.inst_id, to_add_title='train')
        else:
            group_models = []
            for i in range(len(test_groups)):
                model = PPO.load(os.path.join(Config.MODEL_DIR, f"{self.inst_id}_ppo_model_group_{i}.zip"))
                group_models.append(model)

            print("从本地加载模型，开始测试")

        original_test_groups = copy.deepcopy(test_groups)
        for model, group in zip(group_models, test_groups):
            self.simulate_close_module(model, group)
            group.update()

        df = self.backtest.run(test_groups + original_test_groups)
        print(df.to_string())
        self.backtest.plot_all(save=True, inst_id=self.inst_id, to_add_title='test')


if __name__ == '__main__':
    from datetime import timedelta


    time_delta = timedelta(days=365)
    train_begin = datetime(2023, 1, 4)
    train_end = datetime(2024, 2, 4)
    test_begin = train_begin + time_delta
    test_end = train_end + time_delta
    inst_id = 'DOGE-USDT-SWAP'
    bar = '5m'

    tm = TrainingModule(bar, inst_id, train_begin, train_end, test_begin, test_end, load_from_local=False, total_timesteps=5000000)
    tm.run()

