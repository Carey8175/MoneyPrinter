import json
import time
import requests
import pandas as pd
from loguru import logger
from httpx import TimeoutException, ConnectError
from okx.Account import AccountAPI
from okx.Trade import TradeAPI
from okx.MarketData import MarketAPI
from okx.TradingData import TradingDataAPI
from okx import MarketData
from datetime import datetime

from system_code.core.config import Key, Order, Config


def get_available_instruments(key: Key, proxy: [str, None] = None, save_path=Config.INST_PATH) -> None:
    """
    get all the available instruments and save to local file
    Args:
        key: Key
        proxy: the proxy to use
        save_path: the path to save the available instruments

    return: None
    """
    client = AccountAPI(key.apikey, key.secretkey, key.passphrase, False, flag=key.flag, debug=False,
                        proxy=proxy)

    try:
        result = client.get_instruments(instType="SWAP")

    except TimeoutException as e:
        logger.error(f"TimeoutException: {e}, check the proxy or network")
        return

    except ConnectError as e:
        logger.error(f"ConnectError: {e}, check the proxy or network")
        return

    if not (instruments := result.get('data', None)):
        logger.error(f"get instruments failed: {result}")

    # filter
    instruments = [inst for inst in instruments if inst['settleCcy'] == 'USDT']
    logger.info(f"get {len(instruments)} instruments")

    with open(save_path, 'w') as f:
        json.dump(instruments, f, indent=4, ensure_ascii=False)


def get_balance(key: Key, proxy: [str, None] = None) -> [float, None]:
    """
    get the balance of the account
    Args:
        key: Key
        proxy: the proxy to use

    return: the available USDT balance
    """
    client = AccountAPI(key.apikey, key.secretkey, key.passphrase, False, flag=key.flag, debug=False,
                        proxy=proxy)

    try:
        result = client.get_account_balance()
        logger.debug(f"get account balance: {result['data']}")
        usdt_balance = 0.0
        for ccy in result['data'][0]['details']:
            if ccy['ccy'] == 'USDT':
                usdt_balance = float(ccy['availBal'])
                break

        return usdt_balance

    except TimeoutException as e:
        logger.error(f"TimeoutException: {e}, check the proxy or network")
        return

    except ConnectError as e:
        logger.error(f"ConnectError: {e}, check the proxy or network")
        return


def get_positions(key: Key, proxy: [str, None] = None) -> [float, None]:
    """
    获取当前持仓
    Args:
        key: Key
        proxy: the proxy to use

    return: 欧意直接返回的字典，注意看欧意的返回
    """
    client = AccountAPI(key.apikey, key.secretkey, key.passphrase, False, flag=key.flag, debug=False,
                        proxy=proxy)

    try:
        result = client.get_positions()
        logger.debug(f"get instrument ticker: {result}")
        return result['data']

    except TimeoutException as e:
        logger.error(f"TimeoutException: {e}, check the proxy or network")
        return

    except ConnectError as e:
        logger.error(f"ConnectError: {e}, check the proxy or network")
        return

    except Exception as e:
        logger.error(f"Exception: {e}")
        return


def set_leverage(key: Key, instId: str, leverage: int, proxy: [str, None] = None) -> bool:
    """
    set the leverage of the instrument
    Args:
        key: Key
        instId: the instrument id
        leverage: the leverage to set
        proxy: the proxy to use

    return: the result of setting leverage
    """
    client = AccountAPI(key.apikey, key.secretkey, key.passphrase, False, flag=key.flag, debug=False,
                        proxy=proxy)

    try:
        result = client.set_leverage(
            instId=instId,
            lever=leverage,
            mgnMode='isolated'
        )
        logger.debug(f"set leverage: {result}")
        return result['code'] == '0'

    except TimeoutException as e:
        logger.error(f"TimeoutException: {e}, check the proxy or network")
        return False

    except ConnectError as e:
        logger.error(f"ConnectError: {e}, check the proxy or network")
        return False


def trade(key: Key, order: Order, proxy: [str, None] = None) -> dict:
    """
    trade the instrument
    Args:
        key: Key
        order: the order to trade
        proxy: the proxy to use

    return: the result of trading
    """
    client = TradeAPI(key.apikey, key.secretkey, key.passphrase, False, flag=key.flag, debug=False,
                      proxy=proxy)

    try:
        result = client.place_order(
            instId=order.instrument_id,
            tdMode=order.td_mode,
            side=order.side,
            ordType=order.ord_type,
            sz=order.sz,
            px=order.px,
            attachAlgoOrds=order.attach_algo_ords
        )

        logger.warning(f"trade: {result}")
        return result

    except TimeoutException as e:
        logger.error(f"TimeoutException: {e}, check the proxy or network")
        return False

    except ConnectError as e:
        logger.error(f"ConnectError: {e}, check the proxy or network")
        return False


def tp_sl(key: Key, order: Order, proxy: [str, None] = None) -> bool:
    """
    set the take profit and stop loss of the instrument
    Args:
        key: Key
        order: the order to set tp and sl
        proxy: the proxy to use

    return: the result of setting tp and sl
    """
    client = TradeAPI(key.apikey, key.secretkey, key.passphrase, False, flag=key.flag, debug=False,
                      proxy=proxy)

    try:
        result = client.place_algo_order(
            instId=order.instrument_id,
            tdMode=order.td_mode,
            side=order.side,
            ordType=order.attach_algo_ords['tpOrdType'],
            closeFraction=order.sz,
            tpTriggerPx=order.attach_algo_ords['tpTriggerPx'],
            tpOrdPx=order.attach_algo_ords['tpOrdPx'],
            slTriggerPx=order.attach_algo_ords['slTriggerPx'],
            slOrdPx=order.attach_algo_ords['slOrdPx'],
            reduceOnly=True
        )

        logger.debug(f"set tp sl: {result}")
        return result['code'] == '0'

    except TimeoutException as e:
        logger.error(f"TimeoutException: {e}, check the proxy or network")
        return False

    except ConnectError as e:
        logger.error(f"ConnectError: {e}, check the proxy or network")
        return False


def get_tickers(key: Key, proxy: [dict, None] = None) -> dict:
    """
    get all instruments' now price

    Args:
        key: Key
        proxy: the proxy to use

    return: {instId: price}
    """
    client = MarketAPI(flag=key.flag, debug=False, proxy=proxy)

    try:
        result = client.get_tickers(instType="SWAP")
        # logger.debug(f"get instrument ticker: {result['data']}")
        tickers = {inst['instId']: float(inst['last']) for inst in result['data'] if
                   'USDT' in inst['instId'] and inst['last']}
        return tickers

    except TimeoutException as e:
        logger.error(f"TimeoutException: {e}, check the proxy or network")
        return {}

    except ConnectError as e:
        logger.error(f"ConnectError: {e}, check the proxy or network")
        return {}


def get_24h_candles(
        key: Key,
        inst_id: str,
        proxy: [dict, None] = None,
        after: [int, None] = None,
        limit: int = 24
) -> [list, None]:
    """
    get the 24h candles of the instrument
    unit: hour
    Args:
        key: Key
        inst_id: the instrument id
        proxy: the proxy to use
        after: the time after (the older candle data)
        limit: the number of candles

    return: the 24h candles
    """
    client = MarketAPI(flag=key.flag, debug=False, proxy=proxy)

    try:
        result = client.get_candlesticks(
            instId=inst_id,
            bar='1H',
            after=after,
            limit=limit
        ) if after else client.get_candlesticks(
            instId=inst_id,
            bar='1H',
            limit=limit
        )
        logger.debug(f"get 24h candles: {result['data']}")
        return result['data']

    except TimeoutException as e:
        logger.error(f"TimeoutException: {e}, check the proxy or network")
        return

    except ConnectError as e:
        logger.error(f"ConnectError: {e}, check the proxy or network")
        return


async def get_1h_candles(
        key: Key,
        inst_id: str,
        proxy: [dict, None] = None,
        after: [int, None] = None,
        limit: int = 60
) -> [list, None]:
    """
    get the 1h candles of the instrument
    unit: hour
    Args:
        key: Key
        inst_id: the instrument id
        proxy: the proxy to use
        after: the time after (the older candle data)
        limit: the number of candles

    return: the 1h candles
    """
    client = MarketAPI(flag=key.flag, debug=False, proxy=proxy)

    try:
        result = client.get_candlesticks(
            instId=inst_id,
            bar='1m',
            after=after,
            limit=limit
        ) if after else client.get_candlesticks(
            instId=inst_id,
            bar='1m',
            limit=limit
        )
        logger.debug(f"get 1h candles: {result['data']}")
        return result['data']

    except TimeoutException as e:
        logger.error(f"TimeoutException: {e}, check the proxy or network")
        return

    except ConnectError as e:
        logger.error(f"ConnectError: {e}, check the proxy or network")
        return


def get_now_time() -> int:
    return int(time.time() * 1000)


def get_candles(
        inst_id: str,
        proxy: [dict, None] = None,
        period: str = '5m',
        after: [int, None] = None,
        limit: int = 100
) -> [dict, None]:
    """
    Args:
        key: Key
        inst_id: the instrument id
        proxy: the proxy to use
        period: the period of the candle
        after: the time after (the older candle data)
        limit: the number of candles

    return: the candles
    """
    client = MarketAPI(flag="0", proxy=proxy, debug=False)

    try:
        result = client.get_history_candlesticks(
            instId=inst_id,
            bar=period,
            after=after,
            limit=limit
        ) if after else client.get_candlesticks(
            instId=inst_id,
            bar=period,
            limit=limit
        )

        res = []
        for data in result['data']:
            res.append([datetime.fromtimestamp(int(data[0]) / 1000), float(data[1]), float(data[2]), float(data[3]), float(data[4]), float(data[5]),
                        float(data[6]), float(data[7])])

        return res

    except TimeoutException as e:
        logger.error(f"TimeoutException: {e}, check the proxy or network")
        return

    except ConnectError as e:
        logger.error(f"ConnectError: {e}, check the proxy or network")
        return

    except Exception as e:
        logger.error(f"Exception: {e}")
        return


def close_position(key: Key, inst_id: str, mgn_mode='isolated', proxy: [dict, None] = None) -> bool:
    """
    close the position of the instrument
    Args:
        key: Key
        inst_id: the instrument id
        mgn_mode: the margin mode, isolated
        proxy: the proxy to use

    return: the result of closing position
    """
    client = TradeAPI(key.apikey, key.secretkey, key.passphrase, False, flag=key.flag, debug=False,
                      proxy=proxy)

    try:
        result = client.close_positions(instId=inst_id, mgnMode=mgn_mode)
        logger.info(f"close position: {result}")
        return result['code'] == '0'

    except TimeoutException as e:
        logger.error(f"TimeoutException: {e}, check the proxy or network")
        return False

    except ConnectError as e:
        logger.error(f"ConnectError: {e}, check the proxy or network")
        return False


def position_history(key: Key, duration=60, proxy: [dict, None] = None) -> [dict, None]:
    """
    get the position history of the instrument
    Args:
        key: Key
        duration: the duration of the position history, min
        proxy: the proxy to use

    return: the position history
    """
    client = AccountAPI(key.apikey, key.secretkey, key.passphrase, False, flag=key.flag, debug=False,
                        proxy=proxy)

    try:
        result = client.get_positions_history(instType="SWAP")
        positions = [pos for pos in result['data'] if int(pos['uTime']) >= get_now_time() - duration * 60 * 1000]
        logger.debug(f"get position history: {positions}")
        return positions

    except TimeoutException as e:
        logger.error(f"TimeoutException: {e}, check the proxy or network")
        return

    except ConnectError as e:
        logger.error(f"ConnectError: {e}, check the proxy or network")
        return


def probability_module(min_price, now_price, max_price, refer_price):
    x = abs(now_price - refer_price) / (max_price - min_price)

    return True

    # if x <= 0.25:
    #     return np.exp(-4 * x) >= random.random()
    # elif x < 0.5:
    #     return 0.5 * (1 - np.tanh(10 * (x - 0.25))) >= random.random()
    # else:
    #     return False


def check_position_performance(key: Key, duration: int, proxy: [dict, None] = None) -> [dict, None]:
    """
    check the position performance of the instrument
    Args:
        key: Key
        duration: the duration of the position performance, min
        proxy: the proxy to use

    return: the position performance
    """
    result = {}

    positions = position_history(key, duration, proxy)
    if not positions:
        return

    for pos in positions:
        inst_id = pos['instId']
        pnl = float(pos['pnl'])
        result[inst_id] = result.get(inst_id, {})
        result[inst_id]['pnl'] = result[inst_id].get('pnl', 0) + pnl
        if pnl > 0:
            result[inst_id]['win'] = result[inst_id].get('win', 0) + pnl
        else:
            result[inst_id]['lose'] = result[inst_id].get('lose', 0) - pnl

        result[inst_id]['win_rate'] = result[inst_id].get('win', 0) / (
                result[inst_id].get('win', 0) + result[inst_id].get('lose', 0))

    result['total'] = {
        'win': sum([v.get('win', 0) for v in result.values()]),
        'lose': sum([v.get('lose', 0) for v in result.values()]),
        'win_rate': sum([v.get('win', 0) for v in result.values()]) / (
                sum([v.get('win', 0) for v in result.values()]) + sum([v.get('lose', 0) for v in result.values()]))
    }

    return result


def check_order_status(key: Key, inst_id: str, order_id: str, proxy: [dict, None] = None) -> [dict, None]:
    """
    check the order status of the instrument
    Args:
        key: Key
        inst_id: the instrument id
        order_id: the order id
        proxy: the proxy to use

    return: the order status
        canceled: 已撤单
        live: 未完结
        partially_filled: 部分成交
        filled: 全部成交
    """
    client = TradeAPI(key.apikey, key.secretkey, key.passphrase, False, flag=key.flag, debug=False,
                        proxy=proxy)

    try:
        result = client.get_order(instId=inst_id, ordId=order_id)
        logger.debug(f"check order status: {result}")
        return result['data'][0]['state']

    except TimeoutException as e:
        logger.error(f"TimeoutException: {e}, check the proxy or network")
        return

    except ConnectError as e:
        logger.error(f"ConnectError: {e}, check the proxy or network")
        return


def cancel_order(key: Key, inst_id: str, order_id: str, proxy: [dict, None] = None) -> bool:
    """
    cancel the order of the instrument
    Args:
        key: Key
        inst_id: the instrument id
        order_id: the order id
        proxy: the proxy to use

    return: the result of canceling order
    """
    client = TradeAPI(key.apikey, key.secretkey, key.passphrase, False, flag=key.flag, debug=False,
                        proxy=proxy)

    try:
        result = client.cancel_order(instId=inst_id, ordId=order_id)
        logger.debug(f"cancel order: {result}")
        return result['code'] == '0'

    except TimeoutException as e:
        logger.error(f"TimeoutException: {e}, check the proxy or network")
        return False

    except ConnectError as e:
        logger.error(f"ConnectError: {e}, check the proxy or network")
        return False


def get_taker_volume(inst_id: str, bar='5m', after: int = None, before: int = None, proxy: dict | None = None) -> list | None:
    """
    获取合约主动买入/卖出情况
    获取合约维度taker主动买入和卖出的交易量。每个粒度最多可获取最近1,440条数据。
    对于时间粒度period=1D，数据时间范围最早至2024年1月1日；对于其他时间粒度period，最早至2024年2月初。
    限速： 5次/2s
    Args:
        before: 筛选的开始时间戳 ts，datetime
        period: 时间粒度
        after: 筛选的结束时间戳 ts，datetime
        inst_id: the instrument id
        proxy: the proxy to use

    return: ts, sell_taker, buy_taker
    """
    flag = "0"
    host = "https://www.okx.com"
    url = f"{host}/api/v5/rubik/stat/taker-volume-contract"
    params = {
        'instId': inst_id,
        'instType': 'SWAP',
        'period': bar,
        'begin': before,
        'end': after,
        'limit': 100
    } if before and after else {
        'instId': inst_id,
        'period': bar,
        'instType': 'SWAP',
        'limit': 100
    }

    try:
        resp = requests.get(url, params=params, proxies=proxy, timeout=5)
        resp = resp.json()
        result = []
        for data in resp['data']:
            result.append([int(data[0]), round(float(data[1]), 2), round(float(data[2]), 2)])
        return result

    except TimeoutException as e:
        logger.error(f"TimeoutException: {e}, check the proxy or network")
        return

    except ConnectError as e:
        logger.error(f"ConnectError: {e}, check the proxy or network")
        return


def get_open_interest_history(inst_id: str, bar='5m' ,after: int = None, before: int = None, proxy: dict | None = None) -> list | None:
    """
    获取合约持仓量历史
    获取交割及永续合约的历史持仓量数据。每个粒度最多可获取最近1,440条数据。
    对于时间粒度bar=1D，数据时间范围最早至2024年1月1日；对于其他时间粒度bar，最早至2024年2月初。
    限速：10次/2s
    限速规则：IP + instrumentID
    Args:
        before: 筛选的开始时间戳 ts，datetime
        bar: 时间粒度
        after: 筛选的结束时间戳 ts，datetime
        inst_id: the instrument id
        proxy: the proxy to use

    return: ts, open_interest
    """
    flag = "0"
    host = "https://www.okx.com"
    url = f"{host}/api/v5/rubik/stat/contracts/open-interest-history"
    params = {
        'instId': inst_id,
        'period': bar,
        'begin': before,
        'end': after,
        'limit': 100
    } if before and after else {
        'instId': inst_id,
        'period': bar,
        'limit': 100
    }

    try:
        resp = requests.get(url, params=params, proxies=proxy, timeout=5)
        resp = resp.json()
        result = []
        for data in resp['data']:
            result.append([int(data[0]), round(float(data[1]), 4)])
        return result

    except TimeoutException as e:
        logger.error(f"TimeoutException: {e}, check the proxy or network")
        return

    except ConnectError as e:
        logger.error(f"ConnectError: {e}, check the proxy or network")
        return


def long_short_account_ratio_contract_top_trader(inst_id: str, bar='5m', after: int = None, before: int = None, proxy: dict | None = None) -> list | None:
    """
    获取精英交易员合约多空持仓人数比
    获取精英交易员交割永续净开多持仓用户数与净开空持仓用户数的比值。精英交易员指持仓价值前5%的用户。每个粒度最多可获取最近1,440条数据。数据时间范围最早至2024年3月22日。

    限速： 5次/2s
    限速规则： IP + instrumentID
    Args:
        before: 筛选的开始时间戳 ts，datetime
        bar: 时间粒度
        after: 筛选的结束时间戳 ts，datetime
        inst_id: the instrument id
        proxy: the proxy to use

    return: ts, long_ratio, short_ratio
    """
    flag = "0"
    host = "https://www.okx.com"
    url = f"{host}/api/v5/rubik/stat/contracts/long-short-account-ratio-contract-top-trader"
    params = {
        'instId': inst_id,
        'period': bar,
        'begin': before,
        'end': after,
        'limit': 100
    } if before and after else {
        'instId': inst_id,
        'period': bar,
        'limit': 100
    }

    try:
        resp = requests.get(url, params=params, proxies=proxy, timeout=5)
        resp = resp.json()
        result = []
        for data in resp['data']:
            result.append([int(data[0]), round(float(data[1]), 5)])
        return result

    except TimeoutException as e:
        logger.error(f"TimeoutException: {e}, check the proxy or network")
        return

    except ConnectError as e:
        logger.error(f"ConnectError: {e}, check the proxy or network")
        return


def long_short_position_ratio_contract_top_trader(inst_id: str, bar='5m', after: int = None, before: int = None, proxy: dict | None = None) -> list | None:
    """
    获取交割永续开多、开空仓位占总持仓的比值。精英交易员指持仓价值前5%的用户。每个粒度最多可获取最近1,440条数据。数据时间范围最早至2024年3月22日。

    限速： 5次/2s
    限速规则： IP + instrumentID
    Args:
        before: 筛选的开始时间戳 ts，datetime
        bar: 时间粒度
        after: 筛选的结束时间戳 ts，datetime
        inst_id: the instrument id
        proxy: the proxy to use

    return: ts, long_ratio, short_ratio
    """
    flag = "0"
    host = "https://www.okx.com"
    url = f"{host}/api/v5/rubik/stat/contracts/long-short-position-ratio-contract-top-trader"
    params = {
        'instId': inst_id,
        'period': bar,
        'begin': before,
        'end': after,
        'limit': 100
    } if before and after else {
        'instId': inst_id,
        'period': bar,
        'limit': 100
    }

    try:
        resp = requests.get(url, params=params, proxies=proxy, timeout=5)
        resp = resp.json()
        result = []
        for data in resp['data']:
            result.append([int(data[0]), round(float(data[1]), 5)])
        return result

    except TimeoutException as e:
        logger.error(f"TimeoutException: {e}, check the proxy or network")
        return

    except ConnectError as e:
        logger.error(f"ConnectError: {e}, check the proxy or network")
        return


def long_short_account_ratio(inst_id: str, bar='5m', after: int = None, before: int = None, proxy: dict | None = None) -> list | None:
    """
    获取交割永续净开多持仓用户数与净开空持仓用户数的比值。每个粒度最多可获取最近1,440条数据。
    对于时间粒度bar=1D，数据时间范围最早至2024年1月1日；对于其他时间粒度bar，最早至2024年2月初。

    限速： 5次/2s
    限速规则： IP + instrumentID
    Args:
        before: 筛选的开始时间戳 ts，datetime
        bar: 时间粒度
        after: 筛选的结束时间戳 ts，datetime
        inst_id: the instrument id
        proxy: the proxy to use

    return: ts, long_ratio, short_ratio
    """
    flag = "0"
    host = "https://www.okx.com"
    url = f"{host}/api/v5/rubik/stat/contracts/long-short-account-ratio-contract"
    params = {
        'instId': inst_id,
        'period': bar,
        'begin': before,
        'end': after,
        'limit': 100
    } if before and after else {
        'instId': inst_id,
        'period': bar,
        'limit': 100
    }

    try:
        resp = requests.get(url, params=params, proxies=proxy, timeout=5)
        resp = resp.json()
        result = []
        for data in resp['data']:
            result.append([int(data[0]), round(float(data[1]), 5)])
        return result

    except TimeoutException as e:
        logger.error(f"TimeoutException: {e}, check the proxy or network")
        return

    except ConnectError as e:
        logger.error(f"ConnectError: {e}, check the proxy or network")
        return


def get_candles_database(inst_id, date: datetime, bar, proxies=None):
    """
    获取指定日期和时间周期的K线数据，并合并其他交易数据。

    Args:
        inst_id: 合约ID
        date: 日期 (datetime 对象)
        bar: K线周期 ('1m', '5m', '15m', '1h', '4h', '1d')
        proxies: 代理设置

    Returns:
        处理后的K线数据（列表格式）
    """
    marketDataAPI = MarketData.MarketAPI(flag="0", debug=False, proxy=proxies)

    begin = int(date.timestamp() * 1000) - 1  # 当天 00:00 UTC 时间戳（毫秒）
    end = begin + 24 * 60 * 60 * 1000  # 当天 23:59:59 UTC 时间戳（毫秒）

    def fetch_paginated_data(api_function, columns):
        """
        通用分页请求函数，适用于所有 API，确保获取完整数据。

        Args:
            api_function: API 函数
            columns: 返回数据的列名

        Returns:
            完整的 DataFrame
        """
        result = []
        current_end = end  # 先获取最新的数据

        while True:
            response = None
            for i in range(3):
                try:
                    response = api_function(inst_id, bar=bar, after=current_end, before=begin)
                    break
                except Exception as e:
                    print(f"Error fetching data: {e}")
                    time.sleep(2 + i * 2)
                    continue

            if response is None:
                raise Exception("Error fetching data")

            if (type(response) is dict) and (not response or 'data' not in response or len(response['data']) == 0):
                break  # 没有更多数据，结束请求

            if len(response) == 0:
                break

            if type(response) is dict:
                batch_data = response['data']
            else:
                batch_data = response

            result.extend(batch_data)
            # print(date.fromtimestamp(int(batch_data[-1][0]) / 1000))
            # print(date.fromtimestamp(int(batch_data[0][0]) / 1000))

            if len(batch_data) < 100:
                break  # 数据不足 100 条，说明已经全部获取完

            current_end = int(batch_data[-1][0]) -1  # 继续获取更早的数据

            if current_end <= begin:
                break

            time.sleep(0.2)  # 避免请求过快

        df = pd.DataFrame(result, columns=columns)
        return df

    # **获取 K 线数据**
    result_df = fetch_paginated_data(
        marketDataAPI.get_history_candlesticks,
        columns=['ts', 'open', 'high', 'low', 'close', 'vol', 'a', 'b', 'c']
    )
    result_df.drop(columns=['a', 'b', 'c'], inplace=True)
    # -- 更改数据类型 --
    result_df['ts'] = result_df['ts'].astype('int64')
    result_df['open'] = result_df['open'].astype('float64')
    result_df['high'] = result_df['high'].astype('float64')
    result_df['low'] = result_df['low'].astype('float64')
    result_df['close'] = result_df['close'].astype('float64')
    result_df['vol'] = result_df['vol'].astype('float64')

    # **获取交易者买卖数据**
    takers_df = fetch_paginated_data(
        get_taker_volume,
        columns=['ts', 'sell_takers', 'buy_takers']
    )
    # -- 更改数据类型 --
    takers_df['ts'] = takers_df['ts'].astype('int64')
    takers_df['sell_takers'] = takers_df['sell_takers'].astype('float64')
    takers_df['buy_takers'] = takers_df['buy_takers'].astype('float64')

    # **获取持仓数据**
    open_interest_df = fetch_paginated_data(
        get_open_interest_history,
        columns=['ts', 'open_interest']
    )
    # -- 更改数据类型 --
    open_interest_df['ts'] = open_interest_df['ts'].astype('int64')
    open_interest_df['open_interest'] = open_interest_df['open_interest'].astype('float64')

    # **获取顶级交易者多空比**
    elite_ratio_df = fetch_paginated_data(
        long_short_account_ratio_contract_top_trader,
        columns=['ts', 'elite_long_short_ratio']
    )
    # -- 更改数据类型 --
    elite_ratio_df['ts'] = elite_ratio_df['ts'].astype('int64')
    elite_ratio_df['elite_long_short_ratio'] = elite_ratio_df['elite_long_short_ratio'].astype('float64')

    # **获取顶级交易者持仓多空比**
    elite_position_ratio_df = fetch_paginated_data(
        long_short_position_ratio_contract_top_trader,
        columns=['ts', 'elite_position_long_short_ratio']
    )
    # -- 更改数据类型 --
    elite_position_ratio_df['ts'] = elite_position_ratio_df['ts'].astype('int64')
    elite_position_ratio_df['elite_position_long_short_ratio'] = elite_position_ratio_df['elite_position_long_short_ratio'].astype('float64')

    # **获取所有账户多空比**
    all_ratio_df = fetch_paginated_data(
        long_short_account_ratio,
        columns=['ts', 'all_long_short_ratio']
    )
    # -- 更改数据类型 --
    all_ratio_df['ts'] = all_ratio_df['ts'].astype('int64')
    all_ratio_df['all_long_short_ratio'] = all_ratio_df['all_long_short_ratio'].astype('float64')

    # ** 所有df 按照ts去重，同时二次筛选，只保留ts在date当天的数据 **
    dfs = [result_df, takers_df, open_interest_df, elite_ratio_df, elite_position_ratio_df, all_ratio_df]
    # 去重
    for df in dfs:
        df.drop_duplicates(subset=['ts'], inplace=True)
    # 二次筛选，只保留date当天的数据
    for df in dfs:
        df = df[df['ts'].apply(lambda x: datetime.fromtimestamp(int(x) / 1000).date() == date.date())]

    # **合并所有数据**
    for i in range(1, len(dfs)):
        dfs[0] = dfs[0].merge(dfs[i], on='ts', how='left')

    df = dfs[0]

    # **去除空值**
    # df.dropna(inplace=True)

    # **按照时间戳排序**
    df.sort_values('ts', inplace=True)
    df['ts'] = df['ts'].astype('int64')

    # **转换为列表**
    return df.values.tolist()


if __name__ == '__main__':
    data = long_short_account_ratio_contract_top_trader('SONIC-USDT-SWAP')
    print(data)