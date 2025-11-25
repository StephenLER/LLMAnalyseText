# 该文件由 LLM 自动生成，请勿手动修改原始表达式，
# 如需调整，请修改提示词并重新生成。
import numpy as np
import pandas as pd


def compute_alphas(df: pd.DataFrame) -> pd.DataFrame:
    """
    根据大模型生成的 50 个 alpha 表达式，在 df 上计算并追加对应列。
    df：包含文本因子、价格、技术指标等列的 DataFrame。
    返回带有 alpha1 ~ alpha50 新列的 DataFrame 副本。
    """
    df = df.copy()

    # alpha1
    df['alpha1'] = df['overall_sentiment'] * df['overall_importance']

    # alpha2
    df['alpha2'] = (df['close'] - df['SMA_20']) / df['SMA_20']

    # alpha3
    df['alpha3'] = df['MOM_10'] * (df['news_consistency'] / 10.0)

    # alpha4
    df['alpha4'] = -(df['RSI_14'] - 50) / 50

    # alpha5
    df['alpha5'] = df['fcast_ret'] - 5.0

    # alpha6
    df['alpha6'] = np.where(df['overall_sentiment'] > 0.2, df['MACD'] - df['MACD_Signal'], 0)

    # alpha7
    df['alpha7'] = df['net_active_buy_amt_ratio'] * df['overall_impact'] / 10.0

    # alpha8
    df['alpha8'] = (df['BB_Upper'] - df['close']) / (df['BB_Upper'] - df['BB_Lower'])

    # alpha9
    df['alpha9'] = df['overall_duration'] * df['EMA_10'] / df['SMA_20']

    # alpha10
    df['alpha10'] = np.sign(df['MOM_3']) * np.abs(df['overall_sentiment']) * df['overall_importance']

    # alpha11
    df['alpha11'] = df['main_net_inflow_amt_ratio'] * (df['fcast_ret'] > 6).astype(int)

    # alpha12
    df['alpha12'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-8)

    # alpha13
    df['alpha13'] = df['OBV'] / df['OBV'].rolling(20).mean()

    # alpha14
    df['alpha14'] = df['overall_sentiment'] * df['turn']

    # alpha15
    df['alpha15'] = np.where(df['RSI_14'] < 30, df['overall_sentiment'], np.where(df['RSI_14'] > 70, -df['overall_sentiment'], 0))

    # alpha16
    df['alpha16'] = (df['close_net_active_buy_amt_ratio'] - df['open_net_active_buy_amt_ratio']) * df['overall_duration']

    # alpha17
    df['alpha17'] = df['MOM_10'] / (df['dq_amtturnover'] + 1e-8)

    # alpha18
    df['alpha18'] = df['close_main_net_inflow_amt'] * df['news_consistency'] / 10.0

    # alpha19
    df['alpha19'] = (df['SMA_5'] - df['SMA_20']) / df['SMA_20']

    # alpha20
    df['alpha20'] = np.maximum(0, df['overall_sentiment']) * df['volume'] / df['volume'].rolling(10).mean()

    # alpha21
    df['alpha21'] = (df['fcast_ret'] - 5.0) * (df['close'] < df['SMA_20']).astype(int)

    # alpha22
    df['alpha22'] = df['overall_impact'] * (df['MACD'] > df['MACD_Signal']).astype(int)

    # alpha23
    df['alpha23'] = (df['high'] - df['close']) / (df['high'] - df['low'] + 1e-8) * (-df['overall_sentiment'])

    # alpha24
    df['alpha24'] = df['net_active_buy_amt'] / (df['amt'] + 1e-8) * df['overall_duration']

    # alpha25
    df['alpha25'] = np.where(df['overall_consistency'] > 6, df['MOM_3'], 0)

    # alpha26
    df['alpha26'] = (df['close'] - df['open']) / (df['high'] - df['low'] + 1e-8) * df['overall_sentiment']

    # alpha27
    df['alpha27'] = df['RSI_14'] * (df['overall_sentiment'] < -0.2).astype(int)

    # alpha28
    df['alpha28'] = df['OBV'].diff(5) / df['OBV'].shift(5)

    # alpha29
    df['alpha29'] = df['overall_importance'] * df['main_net_inflow_amt_ratio']

    # alpha30
    df['alpha30'] = np.minimum(df['overall_sentiment'], 0) * df['turn']

    # alpha31
    df['alpha31'] = (df['EMA_10'] - df['close']) / df['close']

    # alpha32
    df['alpha32'] = df['fcast_ret'] * df['news_consistency'] / 100.0

    # alpha33
    df['alpha33'] = np.where(df['close'] > df['BB_Upper'], -df['overall_sentiment'], np.where(df['close'] < df['BB_Lower'], df['overall_sentiment'], 0))

    # alpha34
    df['alpha34'] = df['volume'] / df['volume'].rolling(5).mean() * df['overall_impact'] / 10.0

    # alpha35
    df['alpha35'] = (df['close_net_active_buy_amt'] - df['open_net_active_buy_amt']) / (df['amt'] + 1e-8)

    # alpha36
    df['alpha36'] = df['MOM_10'] * df['overall_duration'] / 10.0

    # alpha37
    df['alpha37'] = np.sign(df['net_active_buy_amt']) * np.abs(df['MOM_3'])

    # alpha38
    df['alpha38'] = df['dq_amtturnover'] * df['overall_importance']

    # alpha39
    df['alpha39'] = (df['MACD'] - df['MACD_Signal']) * df['news_consistency'] / 10.0

    # alpha40
    df['alpha40'] = np.where(df['overall_sentiment'] > 0.6, df['turn'], 0)

    # alpha41
    df['alpha41'] = (df['close'] - df['low']) / (df['open'] - df['low'] + 1e-8) * (df['open'] > df['low']).astype(int)

    # alpha42
    df['alpha42'] = df['overall_duration'] * (df['RSI_14'] < 40).astype(int)

    # alpha43
    df['alpha43'] = df['main_net_inflow_amt'] / (df['amt'] + 1e-8) * df['fcast_ret']

    # alpha44
    df['alpha44'] = (df['high'] - df['open']) / (df['open'] + 1e-8) * df['overall_sentiment']

    # alpha45
    df['alpha45'] = np.maximum(0, df['MOM_10']) * df['overall_consistency'] / 10.0

    # alpha46
    df['alpha46'] = df['volume'] * df['overall_impact'] / (df['turn'] + 1e-8)

    # alpha47
    df['alpha47'] = (df['close'] - df['SMA_5']) / df['SMA_5'] * df['overall_sentiment']

    # alpha48
    df['alpha48'] = np.where(df['net_active_buy_amt_ratio'] > 0, df['fcast_ret'], 0)

    # alpha49
    df['alpha49'] = df['OBV'] - df['OBV'].shift(10)

    # alpha50
    df['alpha50'] = df['overall_sentiment'] * df['overall_importance'] * df['overall_duration'] / 100.0

    return df


if __name__ == '__main__':
    # 示例：你可以在这里手动测试 compute_alphas，
    # 例如读取 CSV 后按概念分组计算，再保存回去。
    # 现阶段按你的要求，这里先留空或写测试代码即可。
    pass