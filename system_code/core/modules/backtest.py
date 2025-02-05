import pandas as pd
from datetime import datetime

from system_code.core.modules.position import HoldingGroup, HoldingPeriod


class Backtest:
    def __init__(self, investment, fee_rate):
        self.investment = investment
        self.fee_rate = fee_rate

    def run(self, holding_groups: list[HoldingGroup]) -> pd.DataFrame:
        """
        回测框架，实际上就是对每一个HoldingGroup对象排序后进行回测，每一个holding_period对象都有实现收益率和杠杆等属性
        回测只需要根据这些属性计算相应指标即可
        Args:
            holding_groups: list of HoldingGroup对象

        Returns:
            pd.DataFrame: 装有回测结果的DataFrame，每一行是一个HoldingGroup对象的回测结果
        """



