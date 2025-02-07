from datetime import datetime, timedelta
import numpy as np
import torch
import pandas as pd


class HoldingPeriod:
    def __init__(
            self,
            data: pd.DataFrame,
            side: str,
            leverage: float = 1.0,
            fee_rate: float = 0.0005
    ):
        """
        初始化持仓类
        :param begin: 持仓开始时间
        :param position_size: 持仓规模（合约数量）
        :param side: 持仓方向 ("long" or "short")
        :param entry_price: 进场价格
        :param leverage: 杠杆倍数（默认 1）
        """
        self.begin: datetime | None = None
        self.end: datetime | None = None  # 结束时间，未平仓时为 None
        self.data = data  # 交易数据，可用于存储行情
        self.side = side.lower()  # 确保输入是 "long" 或 "short"
        self.entry_price = 0.0
        self.exit_price = None  # 出场价格，未平仓时为 None
        self.leverage = leverage
        self.fee_rate = fee_rate  # 手续费费率（可以在平仓时计算）
        self.more_info = {}     # 用于存储额外信息

        # 计算时初始化
        self.final_profit = None  # 最终收益（平仓后计算）
        self.max_unrealized_profit = 0.0  # 最大未实现收益
        self.max_unrealized_loss = 0.0  # 最大未实现亏损
        self.final_profit_fee = None  # 最终收益（扣除手续费）

        self.update()

    def update(self):
        """
        更新持仓信息，例如更新收益、最大浮动盈亏等。
        :param current_price: 当前市场价格
        :param end: 结束时间（如果提供，则认为是平仓）
        """
        if self.end:
            # 只保留end时间前的数据
            end_ts = int(self.end.timestamp() * 1000)
            self.data = self.data[self.data['ts'] < end_ts]
        else:
            self.end = datetime.fromtimestamp(self.data['ts'].iloc[-1] / 1000)

        self.begin = datetime.fromtimestamp(self.data['ts'].iloc[0] / 1000)
        self.entry_price = self.data['open'].iloc[0]
        self.exit_price = self.data['close'].iloc[-1]

        high = self.data['high'].max()
        low = self.data['low'].min()

        # 收益
        if self.side == "long":
            self.final_profit = self.leverage * ((self.exit_price - self.entry_price) / self.entry_price)

            self.max_unrealized_profit = self.leverage * ((high - self.entry_price) / self.entry_price)
            self.max_unrealized_loss = self.leverage * ((low - self.entry_price) / self.entry_price)
        else:
            self.final_profit = self.leverage * ((self.entry_price - self.exit_price) / self.entry_price)
            self.max_unrealized_profit = self.leverage * ((self.entry_price - low) / self.entry_price)
            self.max_unrealized_loss = self.leverage * ((self.entry_price - high) / self.entry_price)

        self.final_profit_fee = (1 - self.fee_rate) * (1 + self.final_profit) * (1 - self.fee_rate) - 1

    def to_array(self):
        """
        将持仓信息转换为 NumPy 数组
        """
        return self.data.to_numpy()

    def to_torch(self):
        """
        将持仓信息转换为 PyTorch Tensor
        """
        return torch.tensor(self.to_array(), dtype=torch.float32)

    def __repr__(self):
        return (f"HoldingPeriod(begin={self.begin}, end={self.end}, "
                f"side={self.side}, entry_price={self.entry_price}, exit_price={self.exit_price}, "
                f"leverage={self.leverage}, final_profit={self.final_profit}, "
                f"max_unrealized_profit={self.max_unrealized_profit}, max_unrealized_loss={self.max_unrealized_loss})")


class HoldingGroup:
    def __init__(self):
        self.holdings: list[HoldingPeriod] = []
        self.begin: datetime | None = None
        self.end: datetime | None = None

    def add(self, holding: HoldingPeriod):
        self.holdings.append(holding)

    def __getitem__(self, item):
        return self.holdings[item]

    def __len__(self):
        return len(self.holdings)

    def update(self):
        for holding in self.holdings:
            holding.update()

        self.begin = min([holding.begin for holding in self.holdings])
        try:
            self.end = max([holding.end for holding in self.holdings])
        except:
            self.end = max([holding.begin for holding in self.holdings])




