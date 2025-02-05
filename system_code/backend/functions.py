"""
此处用于定义所有自建函数
"""
from loguru import logger
from datetime import datetime, timedelta

from system_code.core.clickhouse import CKClient
from system_code.core.okx_functions import get_candles_database


def get_instruments_info() -> list:
    """
    获取所有合约信息
    :return:
    """
    ck_client = CKClient(database='mc')
    instruments = ck_client.query_dataframe('select * from instruments_info')
    ck_client.close()

    return instruments.to_dict(orient='records')


def upload_data_by_day(bar: str, inst_id: str, date: datetime, insert=True) -> bool | list:
    """
    上传数据
    :param bar: 周期
    :param inst_id: 合约ID
    :param date: 日期
    :param insert: 是否立即插入数据库
    :return:
    """
    # 定义不同粒度所需的数据条数，用于校验
    bar_num = {
        '1m': 1440,
        '5m': 288,
        '15m': 96,
        '1H': 24,
        '4H': 6,
        '1D': 1
    }

    # 如果传入是字符串，转成 datetime（可根据业务场景灵活处理）
    if isinstance(date, str):
        date = datetime.strptime(date, "%Y-%m-%d")

    # 根据 bar 选择数据库
    if bar == '1m':
        database = 'mc'
    else:
        database = f'mc_{bar.upper()}'

    # 先到 instruments_info 表中获取 list_time，用于判断该 inst_id 上市时间
    ck_client_info = CKClient(database='mc')  # instruments_info 一般在 mc 库下
    list_time = ck_client_info.get_list_time(inst_id)
    if list_time is None:
        logger.warning(f"[WARN] {inst_id} 不存在于 instruments_info 表中，无法上传数据。")
        return False
    ck_client_info.close()

    # 如果合约的 list_time 晚于我们要插入的 date，则说明该合约这一天数据无效
    if list_time > date.timestamp() * 1000:
        logger.warning(f"{inst_id} 的上市时间在 {date} 之后，无需上传。")
        return False

    # 切换到目标数据库，检查当天是否有数据
    ck_client_data = CKClient(database=database)
    has_data = ck_client_data.has_data_for_date(inst_id, date, bar_num[bar])
    if has_data:
        logger.warning(f"{inst_id} 在 {date.date()} 的数据已经存在，跳过上传。")
        return True

    # 数据不存在，则开始获取并插入
    for i in range(3):
        data = get_candles_database(inst_id, date, bar)
        if len(data) == bar_num[bar]:
            # 数据格式转换：CK表结构 ['inst_id', 'ts', 'open', 'high', 'low', 'close', 'vol', ...]
            # 注意：原逻辑是把时间戳放到第一位，然后再加上 inst_id
            data = [[int(d[0])] + d[1:] for d in data]  # 将ts强转int，作为第一列
            data = [[inst_id] + d for d in data]  # 将inst_id作为第一列

            # 插入数据库
            if not insert:
                return data

            ck_client_data.insert(
                'candles',
                data,
                columns=[
                    'inst_id', 'ts', 'open', 'high', 'low', 'close',
                    'vol', 'taker_sell', 'taker_buy', 'open_interest',
                    'elite_long_short_ratio', 'elite_position_long_short_ratio',
                    'all_long_short_ratio'
                ]
            )
            logger.info(f"[OK] 成功插入 {inst_id} 在 {date.date()} 的 {len(data)} 条数据。")
            ck_client_info.close()
            return True
        else:
            print(
                f"[WARN] 第 {i + 1} 次获取 {inst_id} 在 {date.date()} 的数据量不符，期望 {bar_num[bar]} 条，实际 {len(data)} 条。重试中...")

    # 如果 3 次都无法获取到正确条数的数据，抛出异常
    logger.error(f"[ERROR] {inst_id} 在 {date.date()} 获取数据条数始终不正确，上传失败。")
    raise Exception(f"[ERROR] {inst_id} 在 {date.date()} 获取数据条数始终不正确，上传失败。")


def upload_data_by_range(bar: str, inst_id: str, begin: datetime, end: datetime) -> list:
    """
    通过时间范围上传数据
    :param bar: 周期
    :param inst_id: 合约ID
    :param begin: 开始日期
    :param end: 结束日期
    :return: 上传失败的日期列表
    """
    failed_dates = []
    data = []
    ck = CKClient(database=f'mc_{bar.upper()}')

    for day in range((end - begin).days + 1):
        date = begin + timedelta(days=day)
        if chunk := upload_data_by_day(bar, inst_id, date, insert=True):
            pass
            # logger.info(f"[OK] 成功插入 {inst_id} 在 {date.date()} 的数据。当前数据量：{len(data)} 条。")
        else:
            failed_dates.append(date)

        # # 插入数据库
        # if len(data) >= 10000:
        #     ck.insert(
        #         'candles',
        #         data,
        #         columns=[
        #             'inst_id', 'ts', 'open', 'high', 'low', 'close',
        #             'vol', 'taker_sell', 'taker_buy', 'open_interest',
        #             'elite_long_short_ratio', 'elite_position_long_short_ratio',
        #             'all_long_short_ratio'
        #         ]
        #     )
        #     logger.info(f"[OK] 成功插入 {inst_id} 在 {date.date()} 的 {len(data)} 条数据。")
        #
        #     data = []

    # if data:
    #     ck.insert(
    #         'candles',
    #         data,
    #         columns=[
    #             'inst_id', 'ts', 'open', 'high', 'low', 'close',
    #             'vol', 'taker_sell', 'taker_buy', 'open_interest',
    #             'elite_long_short_ratio', 'elite_position_long_short_ratio',
    #             'all_long_short_ratio'
    #         ]
    #     )

        # logger.info(f"[OK] 成功插入 {inst_id} 在 {end.date()} 的 {len(data)} 条数据。")

    return failed_dates


if __name__ == '__main__':
    """
    Test Unit 1
    ========================
    get all instruments info
    """
    # instruments = get_instruments_info()
    # logger.info(f'合约信息: {instruments}')

    """
    Test Unit 2
    ========================
    upload data by range
    """
    bar = '5m'
    inst_id = 'BTC-USDT-SWAP'
    begin = datetime(2024, 2, 1)
    end = datetime(2025, 2, 4)
    upload_data_by_range(bar, inst_id, begin, end)

