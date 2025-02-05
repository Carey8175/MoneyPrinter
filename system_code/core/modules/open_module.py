import pandas as pd

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


