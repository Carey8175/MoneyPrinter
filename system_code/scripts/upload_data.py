"""
上传所有粒度的所有数据
"""
import time
from datetime import datetime

from system_code.core.clickhouse import CKClient
from system_code.backend.functions import upload_data_by_range

ck_client = CKClient(database='mc')
instruments = ck_client.get_instruments_info()
inst_ids = instruments['inst_id'].tolist()
ck_client.close()

bars = ['5m', '15m', '1H', '4H', '1D']
for bar in bars:
    for inst_id in inst_ids:
        for i in range(3):
            try:
                begin = datetime(2024, 2, 4)
                end = datetime(2025, 2, 4)
                upload_data_by_range(bar, inst_id, begin, end)
                break
            except Exception as e:
                print(f'{inst_id} {begin} {end} upload failed, retry {i + 1}')
                time.sleep(5 * 60)
                if i == 2:
                    raise e