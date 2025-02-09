import os
import joblib
import pandas as pd
import numpy as np
import pandas_ta as ta
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from system_code.core.config import Config
from system_code.core.modules.position import HoldingPeriod, HoldingGroup


class OpenPositionModule:
    def __init__(self):
        self.name = 'basic_open_position_module'

    @staticmethod
    def open_position(data: pd.DataFrame, leverage: float = 1.0) -> list[HoldingGroup]:
        """
        开仓, 实际上就是把dataframe按照买入切分成多个HoldingPeriod对象, 组合成HoldingGroup对象
        可以返回多个HoldingGroup对象, 设计多个HoldingGroup对象的目的是为了支持针对不同的Group进行回测评价
        例如：聚类可以将数据分成n个Group，每个Group进行不同的回测评价
        Args:
            data: 数据
            leverage: 杠杆

        Returns:
            list[HoldingGroup]: 1个或多个HoldingGroup对象
        """
        # 默认间隔100个测试开仓
        holding_group = HoldingGroup()

        for i in range(0, len(data), 1000):
            holding_period = HoldingPeriod(data[i:i + 100], 'long', leverage)
            holding_group.add(holding_period)

        holding_group.update()

        return [holding_group]


class BBRsiMaOpenPositionModule(OpenPositionModule):
    def __init__(self):
        super().__init__()
        self.name = 'bb_rsi_ma_open_module'

    @staticmethod
    def open_position(
            data: pd.DataFrame,
            leverage: float = 1.0,
            lengthBB: int = 20,
            multBB: float = 2.0,
            lengthRSI: int = 14,
            overSold: float = 30,
            overBought: float = 70,
            maPeriod: int = 3,
            useMAFilter: bool = True,
            max_hold_bars: int = 288
    ) -> list[HoldingGroup]:
        """
        开仓, 根据BB+RSI+MA的策略逻辑，把DataFrame中出现买点的片段
        封装成多个HoldingPeriod对象, 并构成一个HoldingGroup返回。

        Args:
            data (pd.DataFrame): 包含相关价格和指标列的数据。
            leverage (float): 杠杆倍数。
            lengthBB (int): Bollinger长度。
            multBB (float): Bollinger倍数。
            lengthRSI (int): RSI长度。
            overSold (float): RSI超卖线。
            overBought (float): RSI超买线。
            maPeriod (int): MA过滤周期。
            useMAFilter (bool): 是否启用MA过滤。

        Returns:
            list[HoldingGroup]: 可能包含一个或多个HoldingGroup对象，
                                本示例仅返回一个HoldingGroup，里面有若干HoldingPeriod。
        """

        # ==================== 1. 若尚未计算布林带、RSI、MA，补充计算 ====================

        data['basis'] = data['close'].rolling(window=lengthBB).mean()
        # 注意：Pandas 的 rolling(std) 需要 ddof=0 才能与 Pine 中 stdev 更接近
        data['dev'] = data['close'].rolling(window=lengthBB).std(ddof=0) * multBB
        data['upper'] = data['basis'] + data['dev']
        data['lower'] = data['basis'] - data['dev']

        # RSI 通常可用 ta 库或自行计算，这里简单示例
        # 如果想更贴近 TradingView，需要用更严谨的公式
        data['rsi'] = data.ta.rsi(lengthRSI)

        # MA 过滤计算
        data['maValue'] = data['close'].rolling(window=maPeriod).mean()
        # ==================== 2. 策略条件与信号 ====================

        # 1. 开多信号
        # bullCondition:  close < lower and rsi < overSold
        data['bullCondition'] = (data['close'] < data['lower']) & (data['rsi'] < overSold)

        # 这里的 [1] 在 pine 中表示上一个bar，这里用 shift(1)
        # buySignalRaw = bullCondition[1] and close > open and rsi[1] <= overSold and rsi > overSold
        data['bullCondition_shift1'] = data['bullCondition'].shift(1, fill_value=False)
        data['buySignalRaw'] = (
                data['bullCondition_shift1'] &
                (data['close'] > data['open']) &
                (data['rsi'].shift(1, fill_value=50) <= overSold) &  # 这里随意给了个 fill_value=50，可按需调整
                (data['rsi'] > overSold)
        )

        # 根据MA过滤
        # longFilter = (close > maValue)
        if useMAFilter:
            data['longFilter'] = data['close'] > data['maValue']
            data['buySignal'] = data['buySignalRaw'] & data['longFilter']
        else:
            data['buySignal'] = data['buySignalRaw']

        # 2. 开空信号
        # bearCondition: close > upper and rsi > overBought
        data['bearCondition'] = (data['close'] > data['upper']) & (data['rsi'] > overBought)

        # sellSignalRaw = bearCondition[1] and close < open and rsi[1] >= overBought and rsi < overBought
        data['bearCondition_shift1'] = data['bearCondition'].shift(1, fill_value=False)
        data['sellSignalRaw'] = (
                data['bearCondition_shift1'] &
                (data['close'] < data['open']) &
                (data['rsi'].shift(1, fill_value=50) >= overBought) &  # 这里随意给了个 fill_value=50，可按需调整
                (data['rsi'] < overBought)
        )

        # 根据MA过滤
        # shortFilter = (close < maValue)
        if useMAFilter:
            data['shortFilter'] = data['close'] < data['maValue']
            data['sellSignal'] = data['sellSignalRaw'] & data['shortFilter']
        else:
            data['sellSignal'] = data['sellSignalRaw']

        # ==================== 3. 封装 HoldingPeriod ====================
        holding_group = HoldingGroup()

        # 遍历所有行，找出 buySignal == True 的行，构造 HoldingPeriod
        buy_signal_indices = data.index[data['buySignal'] == True].tolist()
        sell_signal_indices = data.index[data['sellSignal'] == True].tolist()

        for idx in buy_signal_indices:
            holding_data = data.iloc[idx+1:idx + max_hold_bars]
            holding_period = HoldingPeriod(holding_data, side='long', leverage=leverage, signal_candle_info=data.iloc[[idx]])
            holding_group.add(holding_period)

        for idx in sell_signal_indices:
            holding_data = data.iloc[idx+1:idx + max_hold_bars]
            holding_period = HoldingPeriod(holding_data, side='short', leverage=leverage, signal_candle_info=data.iloc[[idx]])
            holding_group.add(holding_period)

        # 统计、更新持仓组信息
        holding_group.update()

        return [holding_group]


class BBRsiMaOpenClusterPositionModule(OpenPositionModule):
    def __init__(self, n_clusters: int = 2, inst_id: str = 'DOGE-USDT-SWAP'):
        super().__init__()
        self.name = 'bb_rsi_ma_open_module'
        self.model = None
        self.n_clusters = n_clusters
        self.inst_id = inst_id
        self.load_model()

    def load_model(self, name: str = None):
        if name is None:
            name = f'kmeans_{self.inst_id}_{self.n_clusters}.pkl'

        self.model = joblib.load(os.path.join(Config.MODEL_DIR, name))

    def open_position(
            self,
            data: pd.DataFrame,
            leverage: float = 1.0,
            lengthBB: int = 20,
            multBB: float = 2.0,
            lengthRSI: int = 14,
            overSold: float = 30,
            overBought: float = 70,
            maPeriod: int = 3,
            useMAFilter: bool = True,
            max_hold_bars: int = 288,
            n_clusters: int = 2
    ) -> list[HoldingGroup]:
        """
        开仓, 根据BB+RSI+MA的策略逻辑，把DataFrame中出现买点的片段
        封装成多个HoldingPeriod对象, 并构成一个HoldingGroup返回。

        Args:
            data (pd.DataFrame): 包含相关价格和指标列的数据。
            leverage (float): 杠杆倍数。
            lengthBB (int): Bollinger长度。
            multBB (float): Bollinger倍数。
            lengthRSI (int): RSI长度。
            overSold (float): RSI超卖线。
            overBought (float): RSI超买线。
            maPeriod (int): MA过滤周期。
            useMAFilter (bool): 是否启用MA过滤。

        Returns:
            list[HoldingGroup]: 可能包含一个或多个HoldingGroup对象，
                                本示例仅返回一个HoldingGroup，里面有若干HoldingPeriod。
        """

        # ==================== 1. 若尚未计算布林带、RSI、MA，补充计算 ====================

        data['basis'] = data['close'].rolling(window=lengthBB).mean()
        # 注意：Pandas 的 rolling(std) 需要 ddof=0 才能与 Pine 中 stdev 更接近
        data['dev'] = data['close'].rolling(window=lengthBB).std(ddof=0) * multBB
        data['upper'] = data['basis'] + data['dev']
        data['lower'] = data['basis'] - data['dev']

        # RSI 通常可用 ta 库或自行计算，这里简单示例
        # 如果想更贴近 TradingView，需要用更严谨的公式
        data['rsi'] = data.ta.rsi(lengthRSI)

        # MA 过滤计算
        data['maValue'] = data['close'].rolling(window=maPeriod).mean()
        # ==================== 2. 策略条件与信号 ====================

        # 1. 开多信号
        # bullCondition:  close < lower and rsi < overSold
        data['bullCondition'] = (data['close'] < data['lower']) & (data['rsi'] < overSold)

        # 这里的 [1] 在 pine 中表示上一个bar，这里用 shift(1)
        # buySignalRaw = bullCondition[1] and close > open and rsi[1] <= overSold and rsi > overSold
        data['bullCondition_shift1'] = data['bullCondition'].shift(1, fill_value=False)
        data['buySignalRaw'] = (
                data['bullCondition_shift1'] &
                (data['close'] > data['open']) &
                (data['rsi'].shift(1, fill_value=50) <= overSold) &  # 这里随意给了个 fill_value=50，可按需调整
                (data['rsi'] > overSold)
        )

        # 根据MA过滤
        # longFilter = (close > maValue)
        if useMAFilter:
            data['longFilter'] = data['close'] > data['maValue']
            data['buySignal'] = data['buySignalRaw'] & data['longFilter']
        else:
            data['buySignal'] = data['buySignalRaw']

        # 2. 开空信号
        # bearCondition: close > upper and rsi > overBought
        data['bearCondition'] = (data['close'] > data['upper']) & (data['rsi'] > overBought)

        # sellSignalRaw = bearCondition[1] and close < open and rsi[1] >= overBought and rsi < overBought
        data['bearCondition_shift1'] = data['bearCondition'].shift(1, fill_value=False)
        data['sellSignalRaw'] = (
                data['bearCondition_shift1'] &
                (data['close'] < data['open']) &
                (data['rsi'].shift(1, fill_value=50) >= overBought) &  # 这里随意给了个 fill_value=50，可按需调整
                (data['rsi'] < overBought)
        )

        # 根据MA过滤
        # shortFilter = (close < maValue)
        if useMAFilter:
            data['shortFilter'] = data['close'] < data['maValue']
            data['sellSignal'] = data['sellSignalRaw'] & data['shortFilter']
        else:
            data['sellSignal'] = data['sellSignalRaw']

        # ==================== 3. 封装 HoldingPeriod ====================
        all_hp = []

        # 遍历所有行，找出 buySignal == True 的行，构造 HoldingPeriod
        buy_signal_indices = data.index[data['buySignal'] == True].tolist()
        sell_signal_indices = data.index[data['sellSignal'] == True].tolist()

        for idx in buy_signal_indices:
            holding_data = data.iloc[idx+1:idx + max_hold_bars]
            holding_period = HoldingPeriod(holding_data, side='long', leverage=leverage, signal_candle_info=data.iloc[[idx]])
            all_hp.append((holding_period, idx))

        for idx in sell_signal_indices:
            holding_data = data.iloc[idx+1:idx + max_hold_bars]
            holding_period = HoldingPeriod(holding_data, side='short', leverage=leverage, signal_candle_info=data.iloc[[idx]])
            all_hp.append((holding_period, idx))

        # # ==================== 4. 聚类 ====================
        # # 4.1. 提取特征
        drop_cols = [
            'inst_id_x', 'inst_id_y', 'ts', 'open', 'high', 'low', 'close',
            'vol', 'taker_sell', 'taker_buy', 'open_interest',
            'elite_long_short_ratio', 'elite_position_long_short_ratio',
            'all_long_short_ratio',
            'bullCondition', 'bearCondition', 'bullCondition_shift1',
            'bearCondition_shift1', 'buySignalRaw', 'sellSignalRaw',
            'buySignal', 'sellSignal', 'longFilter', 'shortFilter'
        ]
        feature_columns = [col for col in data.columns if col not in drop_cols]

        # 确保所有特征列都在 data 中；若没有 volume，可自行去掉或替换
        available_features = [col for col in feature_columns if col in data.columns]

        # 构建一个列表，用于存放聚类所需的特征行
        X_list = []
        for hp, idx in all_hp:
            # 提取该 signal bar 的特征值
            row = data.loc[idx, available_features]
            # row 可能是 Series，需要先变成 np.array 或者 list
            X_list.append(row.values)

        X = np.array(X_list, dtype=float)

        # 数据清洗（如缺失值填充）与缩放（可视需要选择）
        # 这里简单做个 fillna(0) + 标准化示例
        X = np.nan_to_num(X, nan=0.0)
        # scaler = StandardScaler()
        # X_scaled = scaler.fit_transform(X)

        # 4.2. 聚类
        labels = self.model.predict(X)

        # 4.3. 按照聚类结果，将 HoldingPeriod 分组
        holding_groups = [HoldingGroup() for _ in range(n_clusters)]

        # 遍历聚类标签，把对应的 HoldingPeriod 加入对应的组
        for hp, label in zip(all_hp, labels):
            holding_groups[label].add(hp[0])

        # 统计、更新每个HoldingGroup信息（可选）
        for group in holding_groups:
            group.update()

        # 返回聚类后的多个 HoldingGroup
        return holding_groups
