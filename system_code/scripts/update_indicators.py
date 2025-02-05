import numpy as np
import pandas as pd
from loguru import logger
from datetime import datetime

from system_code.core.clickhouse import CKClient


class Indicators:
    def ret(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算 收益率
        """
        # 计算价格变化
        df['ret'] = df['close'].pct_change()
        return df

    def avg_ret(self, df: pd.DataFrame, period_list: list = [3,5,10,20]) -> pd.DataFrame:
        """
        计算 收益率
        """
        # 计算价格变化
        for period in period_list:
            df[f'avg_ret_{period}'] = (df['close']/df['close'].shift(1)-1).rolling(window=period).mean()

        return df


    def rsi(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        计算 RSI 指标
        Args:
            df: 数据框，包含以下列：
                - ts: 时间戳
                - open: 开盘价
                - high: 最高价
                - low: 最低价
                - close: 收盘价
                - vol: 交易量
                - volCcy: 交易币量
                - valCcyQuote: 计价货币的量
            period: 计算 RSI 的周期

        Returns:
            pd.DataFrame: 包含 RSI 指标的数据框
        """
        # 计算价格变化
        df['delta'] = df['close'] - df['close'].shift(1)
        # 计算正数价格变化
        df['gain'] = np.where(df['delta'] > 0, df['delta'], 0)
        # 计算负数价格变化
        df['loss'] = np.where(df['delta'] < 0, -df['delta'], 0)
        # 计算平均增益和平均损失
        df['avg_gain'] = df['gain'].rolling(window=period).mean()
        df['avg_loss'] = df['loss'].rolling(window=period).mean()
        # 计算 RSI
        df['rs'] = df['avg_gain'] / df['avg_loss']
        df[f'rsi_{period}'] = 100 - (100 / (1 + df['rs']))

        # 删除中间变量
        df.drop(columns=['delta', 'gain', 'loss', 'avg_gain', 'avg_loss', 'rs'], inplace=True)

        return df

    def macd(self, df: pd.DataFrame, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> pd.DataFrame:
        """
        计算 MACD 指标
        Args:
            df: 数据框，包含以下列：
                - ts: 时间戳
                - open: 开盘价
                - high: 最高价
                - low: 最低价
                - close: 收盘价
                - vol: 交易量
                - volCcy: 交易币量
                - valCcyQuote: 计价货币的量
            fast_period: 快线周期
            slow_period: 慢线周期
            signal_period: 信号线周期

        Returns:
            pd.DataFrame: 包含 MACD 指标的数据框
        """
        # 计算快线
        df[f'ema_{fast_period}'] = df['close'].ewm(span=fast_period).mean()
        # 计算慢线
        df[f'ema_{slow_period}'] = df['close'].ewm(span=slow_period).mean()
        # 计算 MACD(柱状图)
        df['macd'] = df[f'ema_{fast_period}'] - df[f'ema_{slow_period}']
        # 计算信号线
        df['signal'] = df['macd'].ewm(span=signal_period).mean()
        # 计算 MACD 柱状图
        df['histogram'] = df['macd'] - df['signal']

        # MACD衍生
        close_max2min = df['close'].rolling(window=30).max()-df['close'].rolling(window=30).min()
        histogram_max2min = df['histogram'].rolling(window=30).max()-df['histogram'].rolling(window=30).min()
        period_list = [3,5,10]
        for p in period_list:
            close_degree = (df['close']-df['close'].shift(p))/close_max2min
            histogram_degree = (df['histogram'] - df['histogram'].shift(p)) / histogram_max2min
            df[f'macd_dev_degree_{p}'] = close_degree - histogram_degree

        return df



    def mfi(self, df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """
        计算 MFI 指标
        Args:
            df: 数据框，包含以下列：
                - ts: 时间戳
                - open: 开盘价
                - high: 最高价
                - low: 最低价
                - close: 收盘价
                - vol: 交易量
                - volCcy: 交易币量
                - valCcyQuote: 计价货币的量
            period: 计算 MFI 的周期

        Returns:
            pd.DataFrame: 包含 MFI 指标的数据框
        """
        # 计算典型价格
        # 计算典型价格
        df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3

        # 计算资金流量（价格 * 交易量）
        df['money_flow'] = df['typical_price'] * df['vol']

        # 计算资金流入和流出
        df['positive_flow'] = np.where(df['typical_price'] > df['typical_price'].shift(1), df['money_flow'], 0)
        df['negative_flow'] = np.where(df['typical_price'] < df['typical_price'].shift(1), df['money_flow'], 0)

        # 计算资金流入总和和资金流出总和
        df['positive_flow_sum'] = df['positive_flow'].rolling(window=period).sum()
        df['negative_flow_sum'] = df['negative_flow'].rolling(window=period).sum()

        # 计算资金流向比率
        df['money_flow_ratio'] = df['positive_flow_sum'] / df['negative_flow_sum']

        # 计算 MFI
        df[f'mfi_{period}'] = 100 - (100 / (1 + df['money_flow_ratio']))

        # 删除中间变量
        df.drop(columns=['typical_price', 'money_flow', 'positive_flow', 'negative_flow', 'positive_flow_sum',
                         'negative_flow_sum', 'money_flow_ratio'], inplace=True)

        return df


    def psy(self, df: pd.DataFrame, period_list: list = [3,5,10,20]) -> pd.DataFrame:
        """
        计算 PSY 指标
        Args:
            df: 数据框，包含以下列：
                - ts: 时间戳
                - open: 开盘价
                - high: 最高价
                - low: 最低价
                - close: 收盘价
                - vol: 交易量
                - volCcy: 交易币量
                - valCcyQuote: 计价货币的量
            period: 计算 PSY 的周期

        Returns:
            pd.DataFrame: 包含 PSY 指标的数据框
        """
        tmp_df = pd.DataFrame()

        # 计算价格变化
        tmp_df['delta'] = df['close'] - df['close'].shift(1)
        # 计算正数价格变化
        tmp_df['gain'] = (tmp_df['delta'] > 0).astype(int)
        # 计算 PSY
        for period in period_list:
            df[f'psy_{period}'] = tmp_df['gain'].rolling(window=period).sum() / period * 100

        return df

    def bias(self, df: pd.DataFrame, period_list: list = [3,5,10,20]) -> pd.DataFrame:
        """
        计算 BIAS 指标
        Args:
            df: 数据框，包含以下列：
                - ts: 时间戳
                - open: 开盘价
                - high: 最高价
                - low: 最低价
                - close: 收盘价
                - vol: 交易量
                - volCcy: 交易币量
                - valCcyQuote: 计价货币的量
            period: 计算 BIAS 的周期

        Returns:
            pd.DataFrame: 包含 BIAS 指标的数据框
        """
        tmp_df = pd.DataFrame()

        # 计算均线
        for period in period_list:
            tmp_df[f'ma_{period}'] = df['close'].rolling(window=period).mean()
            # 计算 BIAS
            df[f'bias_{period}'] = (df['close'] - tmp_df[f'ma_{period}']) / tmp_df[f'ma_{period}'] * 100

        return df


    def tp(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算 TP 指标
        Args:
            df: 数据框，包含以下列：
                - ts: 时间戳
                - open: 开盘价
                - high: 最高价
                - low: 最低价
                - close: 收盘价
                - vol: 交易量
                - volCcy: 交易币量
                - valCcyQuote: 计价货币的量

        Returns:
            pd.DataFrame: 包含 TP 指标的数据框
        """
        df['tp'] = (df['high'] + df['low'] + df['close']) / 3

        return df

    def tp_max(self, df: pd.DataFrame, period_list: list = [3,5,10,20]) -> pd.DataFrame:
        """
        计算 TP_MAX 指标
        Args:
            df: 数据框，包含以下列：
                - ts: 时间戳
                - open: 开盘价
                - high: 最高价
                - low: 最低价
                - close: 收盘价
                - vol: 交易量
                - volCcy: 交易币量
                - valCcyQuote: 计价货币的量
            period: 计算 TP_MAX 的周期

        Returns:
            pd.DataFrame: 包含 TP_MAX 指标的数据框
        """
        for period in period_list:
            df[f'tp_max_{period}'] = df['tp'].rolling(window=period).max()
            df[f'tp_max_gap_{period}'] = (df['close']-df[f'tp_max_{period}'])/df[f'tp_max_{period}']

        return df

    def tp_min(self, df: pd.DataFrame, period_list: list = [3,5,10,20]) -> pd.DataFrame:
        """
        计算 TP_MIN 指标
        Args:
            df: 数据框，包含以下列：
                - ts: 时间戳
                - open: 开盘价
                - high: 最高价
                - low: 最低价
                - close: 收盘价
                - vol: 交易量
                - volCcy: 交易币量
                - valCcyQuote: 计价货币的量
            period: 计算 TP_MIN 的周期

        Returns:
            pd.DataFrame: 包含 TP_MIN 指标的数据框
        """
        for period in period_list:
            df[f'tp_min_{period}'] = df['tp'].rolling(window=period).min()
            df[f'tp_min_gap_{period}'] = (df['close'] - df[f'tp_min_{period}']) / df[f'tp_min_{period}']

        return df

    def skewness(self, df: pd.DataFrame, period_list: list = [10,20,30]) -> pd.DataFrame:
        """
        计算 Skewness 指标
        Args:
            df: 数据框，包含以下列：
                - ts: 时间戳
                - open: 开盘价
                - high: 最高价
                - low: 最低价
                - close: 收盘价
                - vol: 交易量
                - volCcy: 交易币量
                - valCcyQuote: 计价货币的量
            period: 计算 Skewness 的周期

        Returns:
            pd.DataFrame: 包含 Skewness 指标的数据框
        """
        for period in period_list:
            df[f'skewness_{period}'] = df['close'].pct_change().rolling(window=period).skew()
        return df

    def avg_low_shadow_length(self, df: pd.DataFrame,  period_list: list = [3,5,10]) -> pd.DataFrame:
        """
        计算 Avg Low Shadow Length 指标
        Args:
            df: 数据框，包含以下列：
                - ts: 时间戳
                - open: 开盘价
                - high: 最高价
                - low: 最低价
                - close: 收盘价
                - vol: 交易量
                - volCcy: 交易币量
                - valCcyQuote: 计价货币的量
            period: 计算 Avg Low Shadow Length 的周期

        Returns:
            pd.DataFrame: 包含 Avg Low Shadow Length 指标的数据框
        """
        df['low_shadow_length'] = abs(df['low'] - df[['open', 'close']].min(axis=1))/df[['open', 'close']].min(axis=1)
        for period in  period_list:
            df[f'avg_low_shadow_length_{period}'] = df['low_shadow_length'].rolling(window=period).mean()

        return df

    def avg_high_shadow_length(self, df: pd.DataFrame,  period_list: list = [3,5,10]) -> pd.DataFrame:
        """
        计算 Avg High Shadow Length 指标
        Args:
            df: 数据框，包含以下列：
                - ts: 时间戳
                - open: 开盘价
                - high: 最高价
                - low: 最低价
                - close: 收盘价
                - vol: 交易量
                - volCcy: 交易币量
                - valCcyQuote: 计价货币的量
            period: 计算 Avg High Shadow Length 的周期

        Returns:
            pd.DataFrame: 包含 Avg High Shadow Length 指标的数据框
        """
        df['high_shadow_length'] = abs(df['high'] - df[['open', 'close']].max(axis=1))/ df[['open', 'close']].max(axis=1)
        for period in period_list:
            df[f'avg_high_shadow_length_{period}'] = df['high_shadow_length'].rolling(window=period).mean()

        return df


    def avg_body_length20(self, df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """
        计算 Avg Candle Length 指标
        平均蜡烛长度
        Args:
            df: 数据框，包含以下列：
                - ts: 时间戳
                - open: 开盘价
                - high: 最高价
                - low: 最低价
                - close: 收盘价
                - vol: 交易量
                - volCcy: 交易币量
                - valCcyQuote: 计价货币的量
            period: 计算 Avg Candle Length 的周期

        Returns:
            pd.DataFrame: 包含 Avg Candle Length 指标的数据框
        """
        df['body_length'] = abs(df['open'] - df['close'])
        df['avg_body_length'] = df['body_length'].rolling(window=period).mean()

        return df

    def body_length_to_avg_ratio(self, df: pd.DataFrame, period=20) -> pd.DataFrame:
        """
        计算 Candle Length To Avg Ratio 指标
        蜡烛长度与平均长度的比率
        Args:
            df: 数据框，包含以下列：
                - ts: 时间戳
                - open: 开盘价
                - high: 最高价
                - low: 最低价
                - close: 收盘价
                - vol: 交易量
                - volCcy: 交易币量
                - valCcyQuote: 计价货币的量

        Returns:
            pd.DataFrame: 包含 Candle Length To Avg Ratio 指标的数据框
        """
        if 'avg_body_length' not in df.columns:
            df['body_length'] = abs(df['open'] - df['close'])
            df['avg_body_length'] = df['body_length'].rolling(window=period).mean()

        df['body_length_to_avg_ratio'] = df['body_length'] / df['avg_body_length']

        return df

    def slope_close(self, df: pd.DataFrame, period_list: list=[5,10,20]) -> pd.DataFrame:
        """
        计算 Slope Close 指标
        收盘价的斜率
        Args:
            df: 数据框，包含以下列：
                - ts: 时间戳
                - open: 开盘价
                - high: 最高价
                - low: 最低价
                - close: 收盘价
                - vol: 交易量
                - volCcy: 交易币量
                - valCcyQuote: 计价货币的量
            period: 计算 Slope Close 的周期

        Returns:
            pd.DataFrame: 包��� Slope Close 指标的数据框
        """
        for period in period_list:
            df[f'slope_close_{period}'] = df['close'].diff(period) / period

        return df

    def trend_continuation_check(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算 Trend Continuation Check 指标
        趋势延续检查
        是否延续之前的趋势，也就是说如果前一根k线图为跌，此K线图HIGH并不高于前一根的收盘价。若为涨则low小于前一根K线的收盘价。
        Args:
            df: 数据框，包含以下列：
                - ts: 时间戳
                - open: 开盘价
                - high: 最高价
                - low: 最低价
                - close: 收盘价
                - vol: 交易量
                - volCcy: 交易币量
                - valCcyQuote: 计价货币的量
            period: 计算 Trend Continuation Check 的周期

        Returns:
            pd.DataFrame: 包含 Trend Continuation Check 指标的数据框
        """
        df['trend_continuation_check'] = np.where(
            df['open'].shift(1) > df['close'].shift(1),  # 前一根K线为涨
            np.where(df['low'] >= df['open'].shift(1), 1, 0),  # 当前K线的low不低于前一根K线的open
            np.where(df['high'] <= df['close'].shift(1), 1, 0)  # 前一根K线为跌，当前K线的high不高于前一根K线的close
        )

        return df
    def momentum(self, df: pd.DataFrame, period_list: list=[3,5,10,20]) -> pd.DataFrame:
        """
        计算 Momentum 指标
        动量指标
        用于测量价格变化的速度和幅度
        Args:
            df: 数据框，包含以下列：
                - ts: 时间戳
                - open: 开盘价
                - high: 最高价
                - low: 最低价
                - close: 收盘价
                - vol: 交易量
                - volCcy: 交易币量
                - valCcyQuote: 计价货币的量
            period: 计算 Momentum 的周期

        Returns:
            pd.DataFrame: 包含 Momentum 指标的数据框
        """
        for period in period_list:
            df[f'momentum_{period}'] = df['close'].pct_change().rolling(window=period).mean()

        return df


    def bollinger_band_expansion_ratio(self, df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """
        计算 Bollinger Band Expansion Ratio 指标
        布林带扩张比率
        ��林带宽度与均线的比率
        Args:
            df: 数据框，包含以下列：
                - ts: 时间戳
                - open: 开盘价
                - high: 最高价
                - low: 最低价
                - close: 收盘价
                - vol: 交易量
                - volCcy: 交易币量
                - valCcyQuote: 计价货币的量
            period: 计算 Bollinger Band Expansion Ratio 的周期

        Returns:
            pd.DataFrame: 包含 Bollinger Band Expansion Ratio 指标的数据框
        """
        # 计算中轨（Middle Band）
        df['middle_band'] = df['close'].rolling(window=period).mean()

        # 计算标准差
        df['std'] = df['close'].rolling(window=period).std()

        # 计算上轨和下轨
        df['upper_band'] = df['middle_band'] + 2 * df['std']
        df['lower_band'] = df['middle_band'] - 2 * df['std']

        # 计算布林带宽度
        df['bollinger_band_width'] = df['upper_band'] - df['lower_band']

        # 计算布林带扩张比率
        df['bollinger_band_expansion_ratio'] = df['bollinger_band_width'] / df['middle_band']

        # 布林带衍生
        df['upper_band_gap'] = (df['close']-df['upper_band'])/df['bollinger_band_width']
        df['lower_band_gap'] = (df['close']-df['lower_band'])/df['bollinger_band_width']


        return df

    def jump_percentage(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算 jump Percentage 指标
        缺口百分比
        今日开盘价与昨日收盘价的差值除以昨日收盘价的绝对值
        Args:
            df: 数据框，包含以下列：
                - ts: 时间戳
                - open: 开盘价
                - high: 最高价
                - low: 最低价
                - close: 收盘价
                - vol: 交易量
                - volCcy: 交易币量
                - valCcyQuote: 计价货币的量
            period: 计算 Gap Percentage 的周期

        Returns:
            pd.DataFrame: 包含 Gap Percentage 指标的数据框
        """
        df['jump_percentage'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)

        return df

    # 波动指数 --------------------------------------------------------------
    def vol_chg(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        成交量变化
        """
        df['vol_chg'] = df['vol'].pct_change()
        return df

    def avg_vol_chg(self, df: pd.DataFrame, period_list: list=[3,5,10,20]) -> pd.DataFrame:
        for period in period_list:
            df[f'avg_vol_chg_{period}'] = df['vol_chg'].rolling(window=period).mean()



    # 交易量指标 --------------------------------------------------------------
    def obv(self, df: pd.DataFrame,period_list: list=[3,5,10,20]) -> pd.DataFrame:
        """
        计算 OBV 指标
        On Balance Volume 能量潮指标
        通过比较当日收盘价与前一日收盘价的大小，来判断成交量是成交量是积极的还是消极的
        如果当日收盘价高于前一日收盘价，则认为当日成交量是积极的，将当日成交量累加到OBV上
        如果当日收盘价低于前一日收盘价，则认为当日成交量是消极的，将当日成交量减去到OBV上
        Args:
            df: 数据框，包含以下列：
                - ts: 时间戳
                - open: 开盘价
                - high: 最高价
                - low: 最低价
                - close: 收盘价
                - vol: 交易量
                - volCcy: 交易币量
                - valCcyQuote: 计价货币的量

        Returns:
            pd.DataFrame: 包含 OBV 指标的数据框
        """
        # 计算 OBV
        df['obv'] = np.where(
            df['close'] > df['close'].shift(1),
            df['vol'],
            -df['vol']
        )

        for period in period_list:
            df[f'obv_{period}'] = df['obv'].rolling(window=period).sum()



        return df

    def vol_skewness(self, df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """
        计算 Vol Skewness 指标
        交易量���度
        20日交易量偏度
        Vol Skewness = (vol - vol.rolling(20).mean()) / vol.rolling(20).std()
        20日平均交易量
        20日交易量偏度
        Args:
            df: 数据框，包含以下列：
                - ts: 时间戳
                - open: 开盘价
                - high: 最高价
                - low: 最低价
                - close: 收盘价
                - vol: 交易量
                - volCcy: 交易币量
                - valCcyQuote: 计价货币的量
            period: 计算 Vol Skewness 的周期

        Returns:
            pd.DataFrame: 包含 Vol Skewness 指标的数据框
        """
        # 计算 Vol Skewness
        df['vol_skewness'] = (df['vol'] - df['vol'].rolling(window=period).mean()) / df['vol'].rolling(window=period).std()

        return df

    def vol_kurtosis(self, df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """
        计算 Vol Kurtosis 指标
        交易量峰度
        20日交易量峰度
        Vol Kurtosis = (vol - vol.rolling(20).mean()) / vol.rolling(20).std()
        20日平均交易量
        20日交易量峰度
        Args:
            df: 数据框，包含以下列：
                - ts: 时间戳
                - open: 开盘价
                - high: 最高价
                - low: 最低价
                - close: 收盘价
                - vol: 交易量
                - volCcy: 交易币量
                - valCcyQuote: 计价货币的量
            period: 计算 Vol Kurtosis 的周期

        Returns:
            pd.DataFrame: 包含 Vol Kurtosis 指标的数据框
        """
        # 计算 Vol Kurtosis
        df['vol_kurtosis'] = (df['vol'] - df['vol'].rolling(window=period).mean()) / df['vol'].rolling(window=period).std()

        return df

    def is_daylight(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算 Is Daylight 指标
        是否为白天
        交易时间为白天（9:00 - 15:00）为1，否则为0
        Args:
            df: 数据框，包含以下列：
                - ts: 时间戳
                - open: 开盘价
                - high: 最高价
                - low: 最低价
                - close: 收盘价
                - vol: 交易量
                - volCcy: 交易币量
                - valCcyQuote: 计价货币的量

        Returns:
            pd.DataFrame: 包含 Is Daylight 指标的数据框
        """
        df['datetime'] = pd.to_datetime(df['ts'], unit='ms')
        # 时区
        df['datetime'] = df['datetime'].dt.tz_localize('UTC').dt.tz_convert('Asia/Shanghai')

        # 交易时间为白天（8:00 - 20:00）为1，否则为0
        df['is_daylight'] = np.where(
            (df['datetime'].dt.hour >= 8) & (df['datetime'].dt.hour < 20),
            1,
            0
        )

        df.drop('datetime', axis=1, inplace=True)

        return df

    def next_trend(self, df: pd.DataFrame, windows: int = 30) -> pd.DataFrame:
        """
        根据过去 `windows` 根 K 线数据，使用线性回归拟合并预测下一根 K 线的趋势值（优化版）。

        Args:
            df (pd.DataFrame): 包含 K 线数据的 DataFrame，至少包括以下列：
                - open: 开盘价
                - high: 最高价
                - low: 最低价
                - close: 收盘价
            windows (int): 使用过去多少根 K 线数据拟合下一根 K 线。

        Returns:
            pd.DataFrame: 添加了 `next_trend` 列的数据框。
        """
        if len(df) < windows:
            raise ValueError("DataFrame 长度不足以进行回归计算")

        # 时间步矩阵 (固定不变)
        X = np.arange(windows).reshape(-1, 1)
        X_mean = X.mean()
        X_centered = X - X_mean
        X_norm = np.sum(X_centered ** 2)

        # 初始化趋势值列表
        trends = []

        # 提取收盘价序列
        close_prices = df['close'].values

        # 预计算窗口内的均值，避免重复计算
        rolling_mean = pd.Series(close_prices).rolling(window=windows).mean().values

        for i in range(windows, len(close_prices)):
            # 提取当前窗口的收盘价
            y = close_prices[i - windows:i]
            y_mean = rolling_mean[i - 1]  # 当前窗口内的均值

            # 中心化 y
            y_centered = y - y_mean

            # 计算斜率和截距
            slope = np.sum(X_centered.flatten() * y_centered) / X_norm
            intercept = y_mean - slope * X_mean

            # 预测下一时间步的值
            next_close = slope * windows + intercept

            # 计算趋势值
            current_close = close_prices[i]
            trend = (next_close - current_close) / current_close

            trends.append(trend)

        # 填充趋势数据到 DataFrame
        df['next_trend'] = [np.nan] * windows + trends  # 前 windows 行没有趋势值，填充 NaN

        return df

    def cci(self, df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """
        计算商品通道指数 (CCI) 并优化滚动绝对偏差计算。
        Args:
            df: 数据框，包含以下列：
                - high: 最高价
                - low: 最低价
                - close: 收盘价
            period: 计算 CCI 的周期

        Returns:
            pd.DataFrame: 包含 CCI 指标的数据框
        """
        # 计算典型价格
        df['tp'] = (df['high'] + df['low'] + df['close']) / 3

        # 计算滚动均值（典型价格的均值）
        rolling_mean = df['tp'].rolling(window=period).mean()

        # 使用 NumPy 向量化计算滚动绝对偏差
        tp_array = df['tp'].values
        rolling_mean_array = rolling_mean.values

        # 计算滚动绝对偏差
        abs_diff = np.abs(tp_array - rolling_mean_array)
        rolling_md = np.convolve(abs_diff, np.ones(period) / period, mode='valid')

        # 填充滚动绝对偏差到原始 DataFrame
        df['md'] = np.nan
        df.loc[period - 1: len(rolling_md) + period - 2, 'md'] = rolling_md

        # 计算 CCI 指标
        df['cci'] = (df['tp'] - rolling_mean) / (0.015 * df['md'])

        # 删除中间变量
        df.drop(columns=['tp', 'md'], inplace=True)

        return df

    def calculate_atr(self, df, period=14):
        """
        计算 ATR(平均真实波动范围)：
        - period: 计算ATR的周期 (默认 14)
        """
        # 计算 True Range (TR)
        df['high_low'] = df['high'] - df['low']
        df['high_close'] = (df['high'] - df['close'].shift()).abs()
        df['low_close'] = (df['low'] - df['close'].shift()).abs()
        df['tr'] = df[['high_low', 'high_close', 'low_close']].max(axis=1)

        # 计算 ATR
        df[f'atr_{period}'] = df['tr'].ewm(alpha=1 / period, adjust=False).mean()

        # 删除中间变量
        df.drop(columns=['high_low', 'high_close', 'low_close', 'tr'], inplace=True)

        return df


class Updator:
    def __init__(self, bar: str, begin: datetime, end: datetime):
        self.indicators = Indicators()
        self.bar = bar
        self.ck_client = CKClient(database=f'mc_{self.bar.upper()}')
        self.begin = begin
        self.end = end

    def fetch_data(self, inst_id) -> pd.DataFrame:
        """
        两个表左join，返回合并后右表没有的数据
        :param inst_id:
        :return:
        """
        return self.ck_client.fetch_data(inst_id=inst_id, begin=self.begin, end=self.end)

    def create_table(self):
        sql = f"""
        CREATE TABLE IF NOT EXISTS indicators  (
            ts UInt64,                        -- 时间戳（唯一标识时间，以毫秒为单位）
            inst_id String,               -- 加密货币名称（如 BTC, ETH）
            datetime Datetime MATERIALIZED toDateTime(ts / 1000, 'Asia/Shanghai') -- 默认列，存储 ts 的日期值 0时区的值
        ) ENGINE = ReplacingMergeTree()
        PARTITION BY toDate(toDateTime(ts / 1000))       -- 按日分区，基于 ts 转换
        ORDER BY (inst_id, ts)                       -- 主键：加密货币和时间戳
        SETTINGS index_granularity = 8192;
        """
        self.ck_client.execute(sql)

    def check_columns(self, df: pd.DataFrame):
        """
        检查df的列在数据库中是否存在，如果不存在则创建新列
        :return:
        """
        columns = self.ck_client.get_columns('indicators')
        for col in df.columns:
            if col not in columns:
                self.ck_client.execute(f"ALTER TABLE indicators ADD COLUMN {col} Float32")

    def add_indicators(self, df: pd.DataFrame):
        """
        为指定交易对数据添加指标
        Args:
            df (pd.DataFrame): 包含 OHLCV 数据的数据框
        """
        indicator_list = [method for method in dir(self.indicators) if
                          callable(getattr(self.indicators, method)) and not method.startswith("_")]

        # 遍历指定指标并计算
        for indicator in indicator_list:
            try:
                # (f"Appling indicator: {indicator}")
                func = getattr(self.indicators, indicator)
                df = func(df)
            except Exception as e:
                print(f"Error applying {indicator}: {e}")
                pass

        # 删除包含 NaN 的行
        # df.dropna(inplace=True)

        logger.info(f"Indicators added.Total columns: {len(df.columns)}")

        return df

    def get_inst_ids(self):
        sql = "SELECT DISTINCT inst_id FROM candles"

        return self.ck_client.execute(sql)

    def start(self):
        inst_ids = self.get_inst_ids()
        inst_ids = [inst_id[0] for inst_id in inst_ids]
        self.create_table()

        for inst_id in inst_ids:
            df = self.fetch_data(inst_id)
            cols = df.columns.tolist()
            cols.remove('ts')
            cols.remove('inst_id')
            df.sort_values('ts', inplace=True)
            df = self.add_indicators(df)

            # drop columns
            df.drop(columns=cols, inplace=True)
            df.dropna(inplace=True)

            # insert data
            self.check_columns(df)
            # self.ck_client.insert('indicators', df.values.tolist(), df.columns.tolist())

            # 假设 df 是你要插入的数据，并且包含一个时间字段 'ts'
            df['datetime'] = pd.to_datetime(df['ts'], unit='ms')

            # 按月份分批插入，避免触及太多分区
            for month, sub_df in df.groupby(df['datetime'].dt.to_period('M')):
                sub_df.drop(columns=['datetime'], inplace=True)
                self.ck_client.insert('indicators', sub_df.values.tolist(), sub_df.columns.tolist())

        self.ck_client.merge_table('indicators')


if __name__ == '__main__':
    begin = datetime(2024, 2, 4)
    end = datetime(2025, 2, 4)
    updator = Updator('5m', begin, end)
    updator.start()
