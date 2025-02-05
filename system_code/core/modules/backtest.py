import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

from system_code.core.modules.position import HoldingGroup, HoldingPeriod
from system_code.core.clickhouse import CKClient, logger


class Backtest:
    def __init__(self, bar, fee_rate=0.0005):
        self.fee_rate = fee_rate
        self.bar = bar
        self.results = None
        self.profits = []
        self.profits_compound = []
        self.profits_fee = []
        self.profits_compound_fee = []

    def run(self, holding_groups: list[HoldingGroup]) -> pd.DataFrame:
        """
        回测框架，实际上就是对每一个HoldingGroup对象排序后进行回测，每一个holding_period对象都有实现收益率和杠杆等属性
        回测只需要根据这些属性计算相应指标即可
        Args:
            holding_groups: list of HoldingGroup对象

        Returns:
            pd.DataFrame: 装有回测结果的DataFrame，每一行是一个HoldingGroup对象的回测结果
        """
        results = []

        for holding_group in holding_groups:
            # 对每个HoldingGroup对象的HoldingPeriod对象进行排序
            holding_group.holdings = sorted(holding_group.holdings, key=lambda x: x.begin)

            # 1. 固定仓位最终收益率
            final_profits = [hp.final_profit for hp in holding_group.holdings]
            self.profits = final_profits
            final_return = np.sum(final_profits)

            # 1.1 复利收益率
            # 去除holding period 重叠的情况，只保留begin时间早的holding period
            # 处理重叠情况，只保留重叠中的第一个仓位
            non_overlapping_holdings = []
            prev_end = None
            for hp in holding_group.holdings:
                if prev_end is None or hp.begin >= prev_end:
                    non_overlapping_holdings.append(hp)
                    prev_end = hp.end

            # plus 计算最大重叠持仓数量
            event = []
            for hp in holding_group.holdings:
                event.append((hp.begin, 1))
                event.append((hp.end, -1))
            event.sort(key=lambda x: x[0])
            max_overlap = np.max(np.cumsum([x[1] for x in event]))
            mean_overlap = np.mean([x[1] for x in event])

            final_profits_compound = [1 + hp.final_profit for hp in non_overlapping_holdings]
            self.profits_compound = final_profits_compound
            final_return_compound = np.prod(final_profits_compound) - 1

            # 1.2.1 带手续费的收益率
            final_profits_fee = [hp.final_profit_fee for hp in holding_group.holdings]
            self.profits_fee = final_profits_fee
            final_return_fee = np.sum(final_profits_fee)

            # 1.2.2 复利带手续费的收益率
            final_profits_compound_fee = [1 + hp.final_profit_fee for hp in non_overlapping_holdings]
            self.profits_compound_fee = final_profits_compound_fee
            final_return_compound_fee = np.prod(final_profits_compound_fee) - 1

            # 2. 平均年，月，日收益
            total_days = (holding_group.end - holding_group.begin).days
            total_months = total_days / 30
            total_years = total_days / 365
            average_daily_return = final_return / total_days
            average_monthly_return = final_return / total_months
            average_yearly_return = final_return / total_years

            # 2.1 复利平均年，月，日收益
            average_daily_return_compound = final_return_compound / total_days
            average_monthly_return_compound = final_return_compound / total_months
            average_yearly_return_compound = final_return_compound / total_years

            # 2.2 复利平均年，月，日收益带手续费
            average_daily_return_fee = final_return_fee / total_days
            average_monthly_return_fee = final_return_fee / total_months
            average_yearly_return_fee = final_return_fee / total_years

            # 2.3 复利平均年，月，日收益带手续费
            average_daily_return_compound_fee = final_return_compound_fee / total_days
            average_monthly_return_compound_fee = final_return_compound_fee / total_months
            average_yearly_return_compound_fee = final_return_compound_fee / total_years

            # 3. 最大回撤
            max_drawdown = 0
            cumulative_returns = np.cumsum(final_profits)
            peak = cumulative_returns[0]
            for r in cumulative_returns:
                if r > peak:
                    peak = r
                drawdown = (peak - r) / peak if peak != 0 else 0
                if drawdown > max_drawdown:
                    max_drawdown = drawdown

            # 3.1 复利最大回撤
            max_drawdown_compound = 0
            cumulative_returns_compound = np.cumprod(final_profits_compound) - 1
            peak = cumulative_returns_compound[0]
            for r in cumulative_returns_compound:
                if r > peak:
                    peak = r
                drawdown = (peak - r) / peak if peak != 0 else 0
                if drawdown > max_drawdown_compound:
                    max_drawdown_compound = drawdown

            # 3.2 复利最大回撤带手续费
            max_drawdown_compound_fee = 0
            cumulative_returns_compound_fee = np.cumprod(final_profits_compound_fee) - 1
            peak = cumulative_returns_compound_fee[0]
            for r in cumulative_returns_compound_fee:
                if r > peak:
                    peak = r
                drawdown = (peak - r) / peak if peak != 0 else 0
                if drawdown > max_drawdown_compound_fee:
                    max_drawdown_compound_fee = drawdown

            # 3.3  固定最大回撤带手续费
            max_drawdown_fee = 0
            cumulative_returns_fee = np.cumsum(final_profits_fee)
            peak = cumulative_returns_fee[0]
            for r in cumulative_returns_fee:
                if r > peak:
                    peak = r
                drawdown = (peak - r) / peak if peak != 0 else 0
                if drawdown > max_drawdown_fee:
                    max_drawdown_fee = drawdown

            # 4. 所有仓位的平均收益率，平均最大浮盈，平均最大浮亏，平均最终收益与平均最大浮亏比
            average_final_profit = np.mean(final_profits)
            average_max_unrealized_profit = np.mean([hp.max_unrealized_profit for hp in holding_group.holdings])
            average_max_unrealized_loss = np.mean([hp.max_unrealized_loss for hp in holding_group.holdings])
            profit_loss_ratio = average_final_profit / average_max_unrealized_loss if average_max_unrealized_loss != 0 else 0

            # 5. 夏普比率 和对应时期的BTC大盘作为对比
            risk_free_rate = 0.02  # 假设无风险利率为2%
            sharpe_ratio = (final_return_fee - risk_free_rate) / np.std(final_profits_fee) if np.std(final_profits) != 0 else 0

            # 5.1 复利夏普比率
            sharpe_ratio_compound = (final_return_compound_fee - risk_free_rate) / np.std(final_profits_compound_fee) if np.std(final_profits_compound) != 0 else 0

            # 5.2.1 针对BTC大盘的夏普比率
            btc_profit = self.get_btc_profit(holding_group.begin, holding_group.end)
            if btc_profit == -1:
                sharpe_ratio_btc = -100
            else:
                sharpe_ratio_btc = (final_return_fee - btc_profit) / np.std(final_profits_fee) if np.std(final_profits) != 0 else 0

            # 5.2.2 复利btc夏普比率
            if btc_profit == -1:
                sharpe_ratio_btc_compound = -100
            else:
                sharpe_ratio_btc_compound = (final_return_compound_fee - btc_profit) / np.std(final_profits_compound_fee) if np.std(final_profits_compound) != 0 else 0

            # 6. 所有仓位的胜率，盈亏比
            winning_trades = sum([1 for hp in holding_group.holdings if hp.final_profit_fee > 0])
            losing_trades = sum([1 for hp in holding_group.holdings if hp.final_profit_fee < 0])
            win_rate = winning_trades / len(holding_group) if len(holding_group) != 0 else 0
            profit_loss_ratio_trades = sum([hp.final_profit for hp in holding_group.holdings if hp.final_profit > 0]) / abs(
                sum([hp.final_profit for hp in holding_group.holdings if hp.final_profit < 0])) if losing_trades != 0 else 0

            # 额外指标
            # 7. 交易次数
            trade_count = len(holding_group)

            # 8. 最长持仓时间
            holding_times = [(hp.end - hp.begin).seconds for hp in holding_group.holdings]
            max_holding_time = max(holding_times) if holding_times else 0
            max_holding_time = max_holding_time / 60 / 60  # 转换为小时

            # 9. 最短持仓时间
            min_holding_time = min(holding_times) if holding_times else 0
            min_holding_time = min_holding_time / 60 / 60  # 转换为小时

            # 10. 平均持仓时间
            average_holding_time = np.mean(holding_times) if holding_times else 0
            average_holding_time = average_holding_time / 60 / 60  # 转换为小时

            # 11. 盈利次数
            winning_count = winning_trades

            # 12. 亏损次数
            losing_count = losing_trades

            # 13. 最大单笔盈利
            max_single_profit = max(final_profits_fee) if final_profits_fee else 0

            # 14. 最大单笔亏损
            max_single_loss = min(final_profits_fee) if final_profits_fee else 0

            # 15. 收益波动率
            volatility = np.std(final_profits_fee)

            # 16. 收益偏度
            skewness = pd.Series(final_profits_fee).skew()

            # 17. 收益峰度
            kurtosis = pd.Series(final_profits_fee).kurtosis()

            # 18. 盈利因子
            profit_factor = sum([hp.final_profit_fee for hp in holding_group if hp.final_profit_fee > 0]) / abs(
                sum([hp.final_profit_fee for hp in holding_group if hp.final_profit_fee < 0])) if losing_trades != 0 else 0

            results.append({
                'final_return': final_return,
                'final_return_compound': final_return_compound,
                'final_return_fee': final_return_fee,
                'max_overlap': max_overlap,
                'mean_overlap': mean_overlap,
                'final_return_compound_fee': final_return_compound_fee,
                'average_daily_return': average_daily_return,
                'average_monthly_return': average_monthly_return,
                'average_yearly_return': average_yearly_return,
                'average_daily_return_compound': average_daily_return_compound,
                'average_monthly_return_compound': average_monthly_return_compound,
                'average_yearly_return_compound': average_yearly_return_compound,
                'average_daily_return_fee': average_daily_return_fee,
                'average_monthly_return_fee': average_monthly_return_fee,
                'average_yearly_return_fee': average_yearly_return_fee,
                'average_daily_return_compound_fee': average_daily_return_compound_fee,
                'average_monthly_return_compound_fee': average_monthly_return_compound_fee,
                'average_yearly_return_compound_fee': average_yearly_return_compound_fee,
                'max_drawdown': max_drawdown,
                'max_drawdown_compound': max_drawdown_compound,
                'max_drawdown_fee': max_drawdown_fee,
                'max_drawdown_compound_fee': max_drawdown_compound_fee,
                'average_final_profit': average_final_profit,
                'average_max_unrealized_profit': average_max_unrealized_profit,
                'average_max_unrealized_loss': average_max_unrealized_loss,
                'profit_loss_ratio': profit_loss_ratio,
                'sharpe_ratio': sharpe_ratio,
                'sharpe_ratio_compound': sharpe_ratio_compound,
                'sharpe_ratio_btc': sharpe_ratio_btc,
                'sharpe_ratio_btc_compound': sharpe_ratio_btc_compound,
                'win_rate': win_rate,
                'profit_loss_ratio_trades': profit_loss_ratio_trades,
                'trade_count': trade_count,
                'max_holding_time': max_holding_time,
                'min_holding_time': min_holding_time,
                'average_holding_time': average_holding_time,
                'winning_count': winning_count,
                'losing_count': losing_count,
                'max_single_profit': max_single_profit,
                'max_single_loss': max_single_loss,
                'volatility': volatility,
                'skewness': skewness,
                'kurtosis': kurtosis,
                'profit_factor': profit_factor
            })

        self.results = pd.DataFrame(results)

        return self.results

    def get_btc_profit(self, begin: datetime, end: datetime) -> float:
        """
        获取BTC大盘的收益率
        Args:
            begin: 开始时间
            end: 结束时间

        Returns:
            float: BTC大盘的收益率
        """
        ck_client = CKClient(database=f'mc_{self.bar.upper()}')
        sql = f"select ts, close from candles where inst_id='BTC-USDT-SWAP' and ts>={int(begin.timestamp() * 1000)} and ts<={int(end.timestamp() * 1000)}"
        btc_prices = ck_client.query_dataframe(sql)

        if btc_prices.empty:
            return -1

        # sort
        btc_prices = btc_prices.sort_values('ts')

        btc_prices = btc_prices['close'].tolist()
        return btc_prices[-1] / btc_prices[0] - 1

    def plot_single(self, index, inst_id, begin, end, save=False):
        """
        绘制回测结果图
        Args:
            index: 回测结果的索引
            inst_id: 合约id
            begin: 开始时间
            end: 结束时间
            save: 是否保存图片

        Return:
            None
        """
        if self.results is None:
            logger.error("Please run the backtest first.")
            return

        # some indicators
        final_return = self.results.loc[index, 'final_return']
        final_return_compound = self.results.loc[index, 'final_return_compound']
        final_return_fee = self.results.loc[index, 'final_return_fee']
        final_return_compound_fee = self.results.loc[index, 'final_return_compound_fee']
        sharpe_ratio = self.results.loc[index, 'sharpe_ratio']
        sharpe_ratio_compound = self.results.loc[index, 'sharpe_ratio_compound']
        sharpe_ratio_btc = self.results.loc[index, 'sharpe_ratio_btc']
        sharpe_ratio_btc_compound = self.results.loc[index, 'sharpe_ratio_btc_compound']

        fig, ax = plt.subplots(3, 1, figsize=(12, 20))
        fig.suptitle(f'Sharp ratio: {sharpe_ratio} | Sharp with Btc: {sharpe_ratio_btc}', fontsize=20)

        # 1. 固定仓位最终收益率: 固定仓位 带手续费和不带手续费
        ax[0].plot(np.cumsum(self.profits), label='Fixed Position', color='blue')
        ax[0].plot(np.cumsum(self.profits_fee), label='Fixed Position with Fee', color='red')
        ax[0].set_title(f'Fixed Position | Final Return: {final_return} | Final Return with Fee: {final_return_fee} | '
                        f'Sharpe: {sharpe_ratio} | Sharpe with Btc: {sharpe_ratio_btc}')
        ax[0].legend()

        # 2. 复利收益率: 复利 带手续费和不带手续费
        ax[1].plot(np.cumprod(self.profits_compound) - 1, label='Compound', color='blue')
        ax[1].plot(np.cumprod(self.profits_compound_fee) - 1, label='Compound with Fee', color='red')
        ax[1].set_title(f'Compound | Final Return: {final_return_compound} | Final Return with Fee: {final_return_compound_fee} | '
                        f'Sharpe: {sharpe_ratio_compound} | Sharpe with Btc: {sharpe_ratio_btc_compound}')
        ax[1].legend()

        # 3. 价格走势
        ck_client = CKClient(database=f'mc_{self.bar.upper()}')
        sql = f"select ts, close from candles where inst_id='{inst_id}' and ts>={int(begin.timestamp() * 1000)} and ts<={int(end.timestamp() * 1000)}"
        prices = ck_client.query_dataframe(sql)
        prices = prices.sort_values('ts')['close'].tolist()

        ax[2].plot(prices, label='Price', color='green')
        ax[2].set_title(f'Price | {inst_id}')
        ax[2].legend()

        if save:
            plt.savefig(f'backtest_{index}.png')

        plt.show()

    def plot_all(self, inst_id, begin, end, save=False):
        """
        绘制所有回测结果图
        Args:
            inst_id: 合约id
            begin: 开始时间
            end: 结束时间
            save: 是否保存图片

        Return:
            None
        """
        if self.results is None:
            logger.error("Please run the backtest first.")
            return

        for i in range(len(self.results)):
            self.plot_single(
                index=i,
                inst_id=inst_id,
                begin=begin,
                end=end,
                save=save
            )





