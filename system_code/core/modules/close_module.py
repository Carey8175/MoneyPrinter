import pandas as pd
from datetime import datetime

from system_code.core.modules.position import HoldingGroup


class ClosePositionModule:
    def __init__(self):
        self.name = 'basic_close_position_module'

    @staticmethod
    def close_position(holding_groups: list[HoldingGroup]) -> list[HoldingGroup]:
        """
        平仓, 对HoldingGroup对象进行平仓操作
        实际上就是对HoldingGroup对象中的每个HoldingPeriod对象更新end时间
        Args:
            holding_groups: HoldingGroup对象

        Returns: HoldingGroup对象
        """
        # 默认对每个HoldingPeriod对象的end时间设置为倒数第5个时间
        for g in holding_groups:
            for p in g:
                p.end = datetime.fromtimestamp(p.data['ts'].iloc[-5] / 1000)

            g.update()

        return holding_groups


class FixClosePositionModule(ClosePositionModule):
    def __init__(self):
        super().__init__()
        self.name = 'fix_close_position_module'

    @staticmethod
    def close_position(
            holding_groups: list[HoldingGroup],
            take_profit_rate: float = 0.08,
            stop_loss_rate: float = 0.05,
    ) -> list[HoldingGroup]:
        """
        平仓, 对HoldingGroup对象进行平仓操作
        实际上就是对HoldingGroup对象中的每个HoldingPeriod对象更新end时间
        Args:
            holding_groups: HoldingGroup对象
            take_profit_rate: 止盈比例
            stop_loss_rate: 止损比例

        Returns: HoldingGroup对象
        """
        for g in holding_groups:
            for p in g:
                # 持仓的开仓价格：以开仓那根Bar的 close 作为开仓价
                entry_price = p.data['open'].iloc[0]
                # 根据多空方向计算浮动盈亏率
                if p.side == 'long':
                    # 多头
                    for i in range(1, len(p.data)):
                        current_price = p.data['open'].iloc[i]
                        current_ts = p.data['ts'].iloc[i]
                        # 盈亏比
                        profit_ratio = (current_price - entry_price) / entry_price

                        # 判断止盈止损
                        if profit_ratio >= take_profit_rate or profit_ratio <= -stop_loss_rate:
                            p.end = datetime.fromtimestamp(current_ts / 1000)
                            break
                    else:
                        # 如果循环未break，说明没有触发止盈止损，默认在最后一根Bar平仓
                        p.end = datetime.fromtimestamp(p.data['ts'].iloc[-1] / 1000)

                else:
                    # 空头
                    for i in range(1, len(p.data)):
                        current_price = p.data['open'].iloc[i]
                        current_ts = p.data['ts'].iloc[i]
                        # 空头的盈亏比计算
                        profit_ratio = (entry_price - current_price) / entry_price

                        # 判断止盈止损
                        if profit_ratio >= take_profit_rate or profit_ratio <= stop_loss_rate:
                            p.end = datetime.fromtimestamp(current_ts / 1000)
                            break
                    else:
                        # 如果循环未break，说明没有触发止盈止损，默认在最后一根Bar平仓
                        p.end = datetime.fromtimestamp(p.data['ts'].iloc[-1] / 1000)

            # 更新HoldingGroup信息
            g.update()

        return holding_groups

