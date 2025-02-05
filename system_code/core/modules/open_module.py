import pandas as pd
import pandas_ta as ta

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
            holding_data = data.iloc[idx:idx + max_hold_bars]
            holding_period = HoldingPeriod(holding_data, side='long', leverage=leverage)
            holding_group.add(holding_period)

        for idx in sell_signal_indices:
            holding_data = data.iloc[idx:idx + max_hold_bars]
            holding_period = HoldingPeriod(holding_data, side='short', leverage=leverage)
            holding_group.add(holding_period)

        # 统计、更新持仓组信息
        holding_group.update()

        return [holding_group]
