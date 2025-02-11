from datetime import datetime

from system_code.core.clickhouse import CKClient, logger
from system_code.core.modules.open_module import OpenPositionModule, BBRsiMaOpenPositionModule, BBRsiMaOpenClusterPositionModule
from system_code.core.modules.close_module import ClosePositionModule, FixClosePositionModule
from system_code.core.modules.backtest import Backtest


class PrintSystem:
    def __init__(self, bar: str, begin: datetime, end: datetime, inst_id: str = None):
        self.begin = begin
        self.end = end
        self.inst_id = inst_id

        self.open_module = BBRsiMaOpenClusterPositionModule()
        self.close_module = ClosePositionModule()
        self.backtest = Backtest(bar)

        self.ck_client = CKClient(database=f'mc_{bar.upper()}')
        if self.inst_id is None:
            self.inst_id = self.ck_client.get_instruments_info()['inst_id'].tolist()

    def __str__(self):
        info = f"""\n
        =================================================================
        PrintSystem -> [OpenModule] {self.open_module.name} 
                    -> [CloseModule] {self.close_module.name}
        =================================================================            
        """

        return info

    def fetch_data(self, indicators=True):
        results = []

        if not isinstance(self.inst_id, list):
            self.inst_id = [self.inst_id]

        for inst_id in self.inst_id:
            data = self.ck_client.fetch_data(inst_id, self.begin, self.end)
            if indicators:
                indicators = self.ck_client.fetch_data(inst_id, self.begin, self.end, table='indicators')
                # sort
                data = data.sort_values('ts')
                indicators = indicators.sort_values('ts')

                data = data.merge(indicators, on='ts', how='left')

            results.append(data)

        return results

    def run(self):
        logger.info(f"[Running] {str(self)}")

        data = self.fetch_data()
        holding_groups = self.open_module.open_position(data)
        holding_groups = self.close_module.close_position(holding_groups)
        df = self.backtest.run(holding_groups)
        print(df.to_string())
        self.backtest.plot_all(save=True, inst_id=self.inst_id)


if __name__ == '__main__':
    begin = datetime(2023, 2, 4)
    end = datetime(2024, 2, 4)
    inst_id = 'DOGE-USDT-SWAP'
    bar = '5m'

    ps = PrintSystem(bar, inst_id, begin, end)
    ps.run()