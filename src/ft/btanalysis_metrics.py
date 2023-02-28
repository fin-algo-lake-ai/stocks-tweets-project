import datetime as dt
import math
from typing import Optional

import numpy as np
import pandas as pd
import re


BAD_SHARPE_VALUE = -20.0  # Sharpe Ratio for definitely bad cases (0 or 1 profit records, etc.)


def calc_sharpe_ratio__default(
        results: pd.DataFrame,
        min_date: dt.datetime,
        max_date: dt.datetime,
        column_for_profit_ratio: str = 'profit_ratio',
        column_for_datetime: str = 'close_date',
        slippage_per_trade_ratio=0.0,  # May be useful for trades-based records, but not for time-based records
        zero_std_epsilon=1e-6       # Epsilon, added to denominator. Useful for ideal equal profits with zero std.
        ) -> float:
    """
    Calculates Sharpe Ratio. For "default" variant values are taken as is, without resampling, as opposed
    to "daily" variant.
    """

    # NEW_2021-11-15: added conditions for no trades case
    if len(results) == 0:
        return BAD_SHARPE_VALUE

    # NEW_2021-11-15: added checks for input dates
    assert max_date >= min_date, f"Inconsistent dates: {max_date} vs {min_date}"
    days_period = (max_date - min_date).days  # NEW 2021-12-06: In fact the number of periods is greater by 1
    if days_period <= 0:
        return BAD_SHARPE_VALUE

    # NEW 2021-11-26: check for real dates in results
    min_date_actual = results[column_for_datetime].min()
    max_date_actual = results[column_for_datetime].max()
    if min_date_actual == max_date_actual:
        # All trade(s) in one time point -> bad.
        return BAD_SHARPE_VALUE

    total_profit = results[column_for_profit_ratio]

    total_profit = total_profit - slippage_per_trade_ratio
    expected_returns_mean = total_profit.sum() / days_period
    up_stdev = np.std(total_profit)  # NEW 2021-11-24: numpy implementation of std gives 0.0 for 1-element array
    assert not np.isnan(up_stdev)  # NEW 2021-11-23

    # NEW 2021-11: calculate ratio even is zero std (could be very smooth strategy)
    if len(total_profit) > 1:
        sharp_ratio = expected_returns_mean / (up_stdev + zero_std_epsilon) * np.sqrt(365)
    else:
        assert len(total_profit) == 1
        sharp_ratio = BAD_SHARPE_VALUE  # We cannot estimate risk for only one point

    return sharp_ratio


def calc_sharpe_ratio__daily(
        results: pd.DataFrame,
        min_date: Optional[dt.datetime] = None,  # None means that min_date will be calculated from the results df
        max_date: Optional[dt.datetime] = None,  # None means that max_date will be calculated from the results df
        column_for_profit_ratio: str = 'profit_ratio',
        column_for_datetime: str = 'close_date',
        slippage_per_trade_ratio=0.0,  # May be useful for trades-based records, but not for time-based records
        zero_std_epsilon=1e-6       # Epsilon, added to denominator. Useful for ideal equal profits with zero std.
        ) -> float:
    """
    Calculates Sharpe Ratio. For "daily" variant values are resampled on a per-day basis. See also "default" variant.
    """

    # NEW_2021-11-15: added conditions for no trades
    if len(results) == 0:
        return BAD_SHARPE_VALUE

    # NEW 2021-11-26: check for real dates in results
    min_date_actual = results[column_for_datetime].min()
    max_date_actual = results[column_for_datetime].max()
    if min_date_actual == max_date_actual:
        # All trade(s) in one time point -> bad.
        return BAD_SHARPE_VALUE

    if min_date is None:
        min_date = min_date_actual
    if max_date is None:
        max_date = max_date_actual

    resample_freq = '1D'
    days_in_year = 365
    annual_risk_free_rate = 0.0
    risk_free_rate = annual_risk_free_rate / days_in_year

    # apply slippage per trade to profit_ratio
    results.loc[:, 'profit_ratio_after_slippage'] = \
        results[column_for_profit_ratio] - slippage_per_trade_ratio

    # create the index within the min_date and end max_date
    # NEW 2021-12: normalize flag: "Normalize start/end dates to midnight before generating date range."
    t_index = pd.date_range(start=min_date, end=max_date, freq=resample_freq,
                            normalize=True)

    sum_daily = (
        results.resample(resample_freq, on='close_date').agg(
            {"profit_ratio_after_slippage": sum}).reindex(t_index).fillna(0)
    )

    total_profit = sum_daily["profit_ratio_after_slippage"] - risk_free_rate
    expected_returns_mean = total_profit.mean()
    up_stdev = total_profit.std()  # NEW 2021-11-24: Pandas implementation of std gives NaN for 1-element array.

    # NEW 2021-11: calculate ratio even is zero std (could be very smooth strategy)
    # if (not np.isnan(up_stdev)) and (up_stdev != 0):
    if not np.isnan(up_stdev):
        sharp_ratio = expected_returns_mean / (up_stdev + zero_std_epsilon) * math.sqrt(days_in_year)
    else:
        # Define high (negative) sharpe ratio to be clear that this is NOT optimal.
        sharp_ratio = BAD_SHARPE_VALUE

    return sharp_ratio


# NEW 2022-06: for pd.to_timedelta both "1m" and "1min" works equally
def convert_timeframe_to_pandas(tf: str):
    assert isinstance(tf, str)
    if re.match(r'[0-9]*m', tf):
        tf += 'in'  # m -> min
    return tf


def check_if_date_is_aligned_with_timeframe(checked_date: dt.datetime, pandas_timeframe: str):
    """
    Example: date=10:41:00, tf=1 min -> OK
    Example: date=10:41:00, tf=5 min -> ERROR
    """
    # Convert input data to dummy pandas series
    dummy_series = pd.Series([0.0, ], index=[checked_date, ])

    # Resample it to get aligned date
    resampled_series = dummy_series.resample(rule=pandas_timeframe).mean()
    aligned_date = resampled_series.index[0]

    # Check if date is equal to aligned date
    return checked_date == aligned_date


class EquityHistory:

    COL_BALANCE = 'balance'

    def __init__(self, min_date: dt.datetime, max_date: dt.datetime,
                 starting_balance: float = 1000.0, timeframe: str = 'D'):

        # Set timeframe (Pandas has some different notations for minutes: "min" instead of "m")
        self.timeframe_pandas = convert_timeframe_to_pandas(timeframe)

        # Check dates
        assert isinstance(min_date, dt.datetime)
        assert isinstance(max_date, dt.datetime)
        assert min_date.tzinfo.utcoffset(min_date).seconds == 0, f"The date {min_date} is not in UTC zone"
        assert max_date.tzinfo.utcoffset(max_date).seconds == 0, f"The date {max_date} is not in UTC zone"
        # Check if min_date is aligned with timeframe
        assert check_if_date_is_aligned_with_timeframe(min_date, self.timeframe_pandas), \
            f"{min_date} is not aligned with pandas timeframe '{self.timeframe_pandas}'"
        assert check_if_date_is_aligned_with_timeframe(max_date, self.timeframe_pandas), \
            f"{max_date} is not aligned with pandas timeframe '{self.timeframe_pandas}'"

        self.history_min_date = min_date
        self.history_max_date = max_date

        # Shift min_date by 1 timeframe earlier (required for checking starting balance)
        dates = pd.date_range(end=min_date, freq=self.timeframe_pandas, periods=2)  # Skip start parameter
        min_date = dates[0]

        # Create _cash_history_df
        dates = pd.date_range(start=min_date, end=max_date, freq=self.timeframe_pandas, normalize=False)
        dates = dates.to_series().asfreq(self.timeframe_pandas).index  # Align to timeframe, to be sure
        self._cash_history_df = pd.DataFrame(index=dates)
        self._cash_history_df[self.COL_BALANCE] = starting_balance

        # Create _equity_history_df
        self._equity_history_df = self._cash_history_df.copy()
        self._equity_history_df[self.COL_BALANCE] = 0.0
        self._trade_count: int = 0

    def append_trade(
            self,
            amount: float,  # NEW 2023-01: in case of short this could be < 0
            open_date: dt.datetime, close_date: dt.datetime,
            open_rate: float, close_rate: float,
            open_fee_ratio: float,      # Example: 0.001 for 0.1% fee
            close_fee_ratio: float,     # Example: 0.001 for 0.1% fee
            ticker_history_df: pd.DataFrame, column_for_date: str = 'date', column_for_rate: str = 'close',
            max_reasonable_fee_ratio: float = 0.01,  # 1%, used for smoke-testing
            verbose: bool = False
            ):

        # Check for time zones
        assert isinstance(open_date, dt.datetime)
        assert isinstance(close_date, dt.datetime)
        assert open_date.tzinfo.utcoffset(open_date).seconds == 0, f"The date {open_date} is not in UTC zone"
        assert close_date.tzinfo.utcoffset(close_date).seconds == 0, f"The date {close_date} is not in UTC zone"

        # Check if trade open/close date is consistent with self.history
        assert open_date >= self.history_min_date
        assert close_date <= self.history_max_date

        # Check if trade open/close date is consistent with ticker_history_df
        assert isinstance(ticker_history_df, pd.DataFrame), f"{type(ticker_history_df)}"
        t_hist_min_date = ticker_history_df[column_for_date].min()
        t_hist_max_date = ticker_history_df[column_for_date].max()
        assert open_date >= t_hist_min_date, f'{open_date}, {t_hist_min_date}'
        assert close_date <= t_hist_max_date, f'{close_date}, {t_hist_max_date}'

        # Smoke check for other params
        assert -np.inf < amount < np.inf
        assert 0.0 < open_rate < np.inf
        assert 0.0 < close_rate < np.inf
        assert 0.0 <= open_fee_ratio <= max_reasonable_fee_ratio
        assert 0.0 <= close_fee_ratio <= max_reasonable_fee_ratio

        # Cut ticker_history_df to [open_date ... close_date] range
        mask = (ticker_history_df[column_for_date] >= open_date) & (ticker_history_df[column_for_date] <= close_date)
        ticker_history_cut_df = ticker_history_df.loc[mask]
        # if verbose:
        #     print(f'{ticker_history_cut_df.head()=}')

        # Create resampled balance df to forward-fill holes (ex: 2017 20:14:00 is missing)
        # Note: normalize flag doc: "Normalize start/end dates to midnight before generating date range."
        # t_index = pd.date_range(start=min_date, end=max_date, freq=self.timeframe_pandas, normalize=False)
        ticker_history_resampled_df = ticker_history_cut_df.resample(
            rule=self.timeframe_pandas, on=column_for_date).agg(
            {column_for_rate: np.mean}).ffill()
        assert ticker_history_resampled_df.index.min() <= open_date  # Ex: 19:35:00 <= 19:38:00 for "5m" tf
        assert ticker_history_resampled_df.index.max() >= close_date
        # {"close": np.mean}).reindex(t_index).ffill()  # .reindex did not change anything

        # Add equity history for this ticker. Open fees influence only cash, so 'amount' is not decreased.
        ticker_history_resampled_df[self.COL_BALANCE] = ticker_history_resampled_df[column_for_rate] * amount

        # Update cash history for trade open event
        trade_open_cost = open_rate * amount  # May be < 0
        fees = abs(trade_open_cost) * open_fee_ratio  # Ex: 100 USD * 0.001 = 0.1 USD
        self._cash_history_df[self.COL_BALANCE].loc[
            self._cash_history_df.index >= open_date] -= (trade_open_cost + fees)

        # Update cash history for trade close event
        trade_close_cost = close_rate * amount  # May be < 0
        fees = abs(trade_close_cost) * close_fee_ratio  # Ex: 200 USD * 0.001 = 0.2 USD
        self._cash_history_df[self.COL_BALANCE].loc[
            self._cash_history_df.index >= close_date] += (trade_close_cost - fees)

        # Update equity history
        mask_eh = (self._equity_history_df.index >= open_date) & (self._equity_history_df.index < close_date)
        mask_th = (ticker_history_resampled_df.index >= open_date) & (ticker_history_resampled_df.index < close_date)
        # Check if "true" blocks are equal
        assert mask_eh.sum() == mask_th.sum(), f"Unmatching sums for masks: {mask_eh.sum()} vs {mask_th.sum()} "
        self._equity_history_df[self.COL_BALANCE].loc[mask_eh] += \
            ticker_history_resampled_df[self.COL_BALANCE].loc[mask_th]

        if verbose:
            print(f"Added trade: {amount=}, {open_rate=},{close_rate=}, {open_date=},{close_date=},"
                  f"  {open_fee_ratio=},{close_fee_ratio=}")

        self._trade_count += 1

    def get_equity_history(self):
        return (self._equity_history_df + self._cash_history_df)[self.COL_BALANCE].copy()

    @property
    def trade_count(self) -> int:
        return self._trade_count
