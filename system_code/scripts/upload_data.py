"""
上传所有粒度的所有数据
"""
import time
from datetime import datetime

from system_code.core.clickhouse import CKClient
from system_code.backend.functions import upload_data_by_range

ck_client = CKClient(database='mc')
instruments = ck_client.get_instruments_info()
inst_ids = ['BTC-USDT-SWAP', 'DOGE-USDT-SWAP', 'ETH-USDT-SWAP', 'LTC-USDT-SWAP', 'XRP-USDT-SWAP', 'SOL-USDT-SWAP',
            'TRUMP-USDT-SWAP', 'OKB-USDT-SWAP', 'TON-USDT-SWAP', 'BCH-USDT-SWAP']

bars = ['5m', '15m']
bars_num = {
    '5m': 288,
    '15m': 96
}

for bar in bars:
    for inst_id in inst_ids:
        # 校验时期范围是否都有数据
        ck_client = CKClient(database=f'mc_{bar.upper()}')
        if ck_client.has_data_for_date(inst_id, datetime(2024, 2, 4), day_num=bars_num[bar]) and ck_client.has_data_for_date(inst_id, datetime(2025, 2, 4), day_num=bars_num[bar]):
            print(f'{inst_id} {bar} has data, skip')
            continue

        for i in range(3):
            try:
                begin = datetime(2024, 2, 4)
                end = datetime(2025, 2, 4)
                upload_data_by_range(bar, inst_id, begin, end)
                break
            except Exception as e:
                print(f'{inst_id} {begin} {end} upload failed, retry {i + 1} {e}')
                time.sleep(5 * 60)
                if i == 2:
                    raise e