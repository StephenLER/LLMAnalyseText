# 该文件由 LLM 自动生成，请勿手动修改原始表达式，
# 如需调整，请修改提示词并重新生成。
import numpy as np
import pandas as pd
from alpha_ops import (
    ts_mean, ts_std, ts_max, ts_min, ts_rank,
    corr, cov, rank, zscore, decay_linear
)

def compute_alphas(df: pd.DataFrame) -> pd.DataFrame:
    """
    根据大模型生成的 50 个 alpha 表达式，在 df 上计算并追加对应列。
    df：包含文本因子、价格、技术指标等列的 DataFrame。
    返回带有 alpha1 ~ alpha50 新列的 DataFrame 副本。
    """
    df = df.copy()

    # alpha1
    df['alpha1'] = (df['overall_sentiment'] * df['news_consistency'] / 10.0) * (df['MOM_10'] / (ts_std(df['close'], 20) + 1e-6)) * np.where(df['main_net_inflow_amt_ratio'] > ts_mean(df['main_net_inflow_amt_ratio'], 5), 1.0, 0.5)

    # alpha2
    df['alpha2'] = np.where(df['RSI_14'] > 70, -df['overall_sentiment'] * df['overall_impact'] / 10.0, df['overall_sentiment'] * df['overall_impact'] / 10.0) * (df['close'] - df['SMA_20']) / (df['SMA_20'] + 1e-6)

    # alpha3
    df['alpha3'] = decay_linear(ts_mean(df['fcast_ret'] - 5.0, 3), 5) * corr(df['net_active_buy_amt_ratio'], df['close'], 10) * zscore(df['turn'])

    # alpha4
    df['alpha4'] = (df['BB_Upper'] - df['close']) / (df['BB_Upper'] - df['BB_Lower'] + 1e-6) * np.where(df['overall_sentiment'] < -0.3, 1.0, 0.0) * ts_rank(df['volume'], 10)

    # alpha5
    df['alpha5'] = df['overall_duration'] * df['news_consistency'] * (df['MACD'] - df['MACD_Signal']) * np.sign(df['close'] - df['EMA_10'])

    # alpha6
    df['alpha6'] = -ts_std(df['close'], 5) * df['overall_impact'] * np.where(df['RSI_14'] < 40, 1.0, 0.0)

    # alpha7
    df['alpha7'] = rank(df['overall_sentiment'] * df['main_net_inflow_amt_ratio']) * (df['OBV'] - ts_mean(df['OBV'], 10)) / (ts_std(df['OBV'], 10) + 1e-6)

    # alpha8
    df['alpha8'] = np.where(df['close'] < df['BB_Lower'], df['overall_sentiment'] * df['overall_importance'] / 10.0 * (df['BB_Lower'] - df['close']) / (df['BB_Lower'] + 1e-6), 0)

    # alpha9
    df['alpha9'] = (df['close_net_active_buy_amt_ratio'] - df['open_net_active_buy_amt_ratio']) * df['news_consistency'] * ts_rank(df['MOM_3'], 5)

    # alpha10
    df['alpha10'] = df['fcast_ret'] * (1.0 - abs(df['close'] / df['SMA_5'] - 1.0)) * np.where(df['overall_duration'] > 6, 1.0, 0.5)

    # alpha11
    df['alpha11'] = -df['RSI_14'] * df['overall_sentiment'] * df['dq_amtturnover']

    # alpha12
    df['alpha12'] = corr(df['overall_sentiment'], df['close'], 5) * ts_mean(df['main_net_inflow_amt'], 3) / (ts_mean(df['amt'], 3) + 1e-6) * df['MOM_10']

    # alpha13
    df['alpha13'] = np.where(df['high'] / df['low'] - 1.0 > ts_mean(df['high'] / df['low'] - 1.0, 10), -abs(df['overall_sentiment']) * df['overall_impact'], df['overall_sentiment'] * df['overall_impact'])

    # alpha14
    df['alpha14'] = decay_linear(ts_mean(df['close_main_net_inflow_amt_ratio'], 3), 5) * df['news_consistency'] * (df['EMA_10'] - df['SMA_20']) / (df['SMA_20'] + 1e-6)

    # alpha15
    df['alpha15'] = (df['overall_importance'] - ts_mean(df['overall_importance'], 5)) * (df['volume'] - ts_mean(df['volume'], 5)) / (ts_std(df['volume'], 5) + 1e-6) * np.sign(df['MOM_3'])

    # alpha16
    df['alpha16'] = np.minimum(df['RSI_14'] - 50, 0) * df['overall_sentiment'] * df['net_active_buy_amt_ratio']

    # alpha17
    df['alpha17'] = ts_rank(df['fcast_ret'], 10) * corr(df['OBV'], df['close'], 10) * zscore(df['overall_duration'])

    # alpha18
    df['alpha18'] = -(df['close'] - df['open']) / (df['high'] - df['low'] + 1e-6) * df['overall_sentiment'] * np.where(df['turn'] > ts_mean(df['turn'], 20), 1.0, 0.0)

    # alpha19
    df['alpha19'] = df['overall_impact'] * df['main_net_inflow_amt_ratio'] * (df['MACD'] > df['MACD_Signal']).astype(int)

    # alpha20
    df['alpha20'] = (ts_max(df['close'], 20) - df['close']) / (ts_max(df['close'], 20) - ts_min(df['close'], 20) + 1e-6) * df['overall_sentiment'] * np.where(df['news_consistency'] > 7, 1.0, 0.0)

    # alpha21
    df['alpha21'] = np.where(df['RSI_14'] < 30, df['overall_duration'] * (df['close'] - ts_min(df['close'], 5)) / (ts_min(df['close'], 5) + 1e-6), 0)

    # alpha22
    df['alpha22'] = df['net_active_buy_amt_ratio'] * df['fcast_ret'] * (df['SMA_5'] > df['SMA_20']).astype(int)

    # alpha23
    df['alpha23'] = -df['overall_sentiment'] * ts_std(df['close'], 10) * np.where(df['MOM_10'] < 0, 1.0, 0.0)

    # alpha24
    df['alpha24'] = rank(df['close_main_net_inflow_amt'] / (df['amt'] + 1e-6)) * df['news_consistency'] * ts_rank(df['EMA_10'] - df['SMA_20'], 5)

    # alpha25
    df['alpha25'] = (df['fcast_ret'] - 5.0) * (df['OBV'] - ts_mean(df['OBV'], 5)) / (ts_std(df['OBV'], 5) + 1e-6) * df['overall_importance'] / 10.0

    # alpha26
    df['alpha26'] = np.where(df['close'] > df['BB_Upper'], -df['overall_sentiment'] * df['overall_impact'], df['overall_sentiment'] * df['overall_impact']) * df['turn']

    # alpha27
    df['alpha27'] = decay_linear(ts_mean(df['overall_sentiment'], 3), 5) * corr(df['main_net_inflow_amt_ratio'], df['MOM_3'], 10) * (df['RSI_14'] < 60).astype(int)

    # alpha28
    df['alpha28'] = (df['high'] - df['close']) / (df['high'] - df['low'] + 1e-6) * df['overall_sentiment'] * np.where(df['overall_duration'] > 5, 1.0, 0.0)

    # alpha29
    df['alpha29'] = df['MOM_10'] * df['news_consistency'] * (df['net_active_buy_amt_ratio'] - ts_mean(df['net_active_buy_amt_ratio'], 5))

    # alpha30
    df['alpha30'] = np.maximum(0, df['overall_sentiment'] - 0.2) * ts_mean(df['close_main_net_inflow_amt_ratio'], 3) * (df['MACD'] - df['MACD_Signal'])

    # alpha31
    df['alpha31'] = -ts_std(df['overall_sentiment'], 5) * df['turn'] * np.where(df['MOM_10'] > 0, 1.0, 0.0)

    # alpha32
    df['alpha32'] = corr(df['fcast_ret'], df['close'], 5) * df['main_net_inflow_amt_ratio'] * zscore(df['overall_impact'])

    # alpha33
    df['alpha33'] = np.where(df['RSI_14'] > 50, df['overall_duration'] * (df['close'] - df['SMA_20']) / (df['SMA_20'] + 1e-6), 0)

    # alpha34
    df['alpha34'] = (df['open_net_active_buy_amt_ratio'] + df['close_net_active_buy_amt_ratio']) * df['news_consistency'] * ts_rank(df['volume'], 10)

    # alpha35
    df['alpha35'] = df['overall_importance'] * (df['OBV'] - ts_mean(df['OBV'], 10)) / (ts_std(df['OBV'], 10) + 1e-6) * np.sign(df['MOM_10'])

    # alpha36
    df['alpha36'] = df['overall_sentiment'] * df['MOM_3']

    # alpha37
    df['alpha37'] = (df['close'] - df['SMA_20']) / df['SMA_20']

    # alpha38
    df['alpha38'] = df['RSI_14'] - 50

    # alpha39
    df['alpha39'] = df['main_net_inflow_amt_ratio']

    # alpha40
    df['alpha40'] = df['fcast_ret'] - 5.0

    # alpha41
    df['alpha41'] = -df['overall_sentiment']

    # alpha42
    df['alpha42'] = df['BB_Upper'] - df['close']

    # alpha43
    df['alpha43'] = df['turn'] - ts_mean(df['turn'], 20)

    # alpha44
    df['alpha44'] = df['MACD'] - df['MACD_Signal']

    # alpha45
    df['alpha45'] = df['volume'] / ts_mean(df['volume'], 10)

    # alpha46
    df['alpha46'] = np.where(df['overall_sentiment'] > 0.5, df['MOM_10'], 0)

    # alpha47
    df['alpha47'] = df['net_active_buy_amt'] / df['amt']

    # alpha48
    df['alpha48'] = df['overall_impact'] * df['overall_duration']

    # alpha49
    df['alpha49'] = (df['high'] + df['low']) / 2 - df['close']

    # alpha50
    df['alpha50'] = df['EMA_10'] - df['close']

    return df


if __name__ == '__main__':
    # 示例：你可以在这里手动测试 compute_alphas，
    # 例如读取 CSV 后按概念分组计算，再保存回去。
    # 现阶段按你的要求，这里先留空或写测试代码即可。
    pass