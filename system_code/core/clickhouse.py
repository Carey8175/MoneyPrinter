import clickhouse_driver
from datetime import datetime

import pandas as pd

from system_code.core.config import Config, logger


class CKClient:
    def __init__(self, database='mc'):
        self.config = Config()
        self.client = clickhouse_driver.Client(
            host=self.config.database['host'],
            port=self.config.database['port'],
            user=self.config.database['user'],
            password=self.config.database['password'],
            database=database
        )

    def execute(self, query):
        return self.client.execute(query)

    def insert(self, table, data, columns=None):
        if columns is None:
            columns = []

        col_str = "(" + ", ".join(columns) + ")" if columns else ""
        sql = f"INSERT INTO {table} {col_str} VALUES"

        self.client.execute(sql, data)

    def close(self):
        self.client.disconnect()

    def merge_table(self, table):
        sql = f"OPTIMIZE TABLE {table} FINAL;"
        self.execute(sql)

    def has_data_for_date(self, inst_id, date: datetime, day_num=3):
        """
        校验数据库是否已经有指定日期的数据
        :param inst_id: 合约ID
        :param date: 日期
        :param day_num: 一天的数据点数
        :return: True 表示有数据，False 表示无数据
        """
        start_ts = date.timestamp() * 1000
        end_ts = start_ts + 24 * 60 * 60 * 1000
        query = f"""
        SELECT COUNT(*) 
        FROM candles 
        WHERE inst_id = '{inst_id}' 
        AND ts >= {start_ts} 
        AND ts < {end_ts}
        """
        count = self.execute(query)
        count = 0 if not count else count[0][0]
        return count >= day_num

    def query_dataframe(self, query):
        return self.client.query_dataframe(query)

    def get_list_time(self, inst_id) -> int | None:
        sql_list_time = f"SELECT list_time FROM instruments_info WHERE inst_id='{inst_id}'"
        result = self.execute(sql_list_time)

        if not result:
            logger.warning(f"inst_id={inst_id} 不存在于 instruments_info 中")
            return None

        list_time = result[0][0]  # 取出 list_time
        return int(list_time)

    def get_instruments_info(self) -> pd.DataFrame:
        sql = "select * from instruments_info"
        return  self.query_dataframe(sql)

    def fetch_data(self, inst_id, begin: datetime, end: datetime) -> pd.DataFrame:
        begin_ts = int(begin.timestamp() * 1000)
        end_ts = int(end.timestamp() * 1000)
        sql = f"select * from candles where ts >= {begin_ts} and ts < {end_ts}"

        return self.query_dataframe(sql)


if __name__ == '__main__':
    ck = CKClient()
    ck.has_data_for_date('BTC-USDT-SWAP', datetime(2024, 3, 21))
    ck.close()