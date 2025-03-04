"""
上传所有粒度的所有数据
"""
import time
from datetime import datetime

from system_code.core.clickhouse import CKClient
from system_code.backend.functions import upload_data_by_range

ck_client = CKClient(database='mc')
instruments = ck_client.get_instruments_info()
# inst_ids = ['BTC-USDT-SWAP', 'DOGE-USDT-SWAP', 'ETH-USDT-SWAP', 'LTC-USDT-SWAP', 'XRP-USDT-SWAP', 'SOL-USDT-SWAP',
#             'TRUMP-USDT-SWAP', 'TON-USDT-SWAP', 'BCH-USDT-SWAP']
inst_ids = instruments['inst_id'].tolist()

begin = datetime(2021, 1, 1)
end = datetime(2025, 1, 1)
bars = ['4H', '1D']
bars_num = {
    '1H': 24,
    '5m': 288,
    '4H': 6,
    '1D': 1
}

for bar in bars:
    for i, inst_id in enumerate(inst_ids):
        # 校验时期范围是否都有数据
        ck_client = CKClient(database=f'mc_{bar.upper()}')
        if ck_client.has_data_for_date(inst_id, begin, day_num=bars_num[bar]) and ck_client.has_data_for_date(inst_id, end, day_num=bars_num[bar]):
            print(f'{inst_id} {bar} has data, skip')
            continue

        for j in range(3):
            try:
                upload_data_by_range(bar, inst_id, begin, end)
                print(f'{i / len(inst_ids)} {inst_id} {begin} {end} upload success')
                break
            except Exception as e:
                print(f'{inst_id} {begin} {end} upload failed, retry {i + 1} {e}')
                time.sleep(5 * 60)
                if i == 2:
                    raise e

ck_client.merge_table('candles')
ck_client.close()