from datetime import datetime

from system_code.core.clickhouse import CKClient
from system_code.core.modules.open_module import OpenPositionModule
from system_code.core.modules.close_module import ClosePositionModule
from system_code.core.modules.backtest import Backtest


class PrintSystem:
    def __init__(self, bar: str, inst_id: str, begin: datetime, end: datetime):
        self.begin = begin
        self.end = end
        self.inst_id = inst_id

        self.open_module = OpenPositionModule()
        self.close_module = ClosePositionModule()
        self.backtest = Backtest(bar)

        self.ck_client = CKClient(database=f'mc_{bar.upper()}')

    def __str__(self):
        info = f"""\n
        PrintSystem -> OpenModule: {self.open_module.name}
                   |    
                   --> CloseModule: {self.close_module.name}
        """

        return info

    def fetch_data(self):
        data = self.ck_client.fetch_data(self.inst_id, self.begin, self.end)
        return data