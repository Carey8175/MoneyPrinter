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

