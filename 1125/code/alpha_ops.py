import numpy as np
import pandas as pd


# ---------- 时间序列算子 ----------

def ts_mean(x: pd.Series, window: int) -> pd.Series:
    """x 在过去 window 天的简单均值"""
    return x.rolling(window, min_periods=window).mean()


def ts_std(x: pd.Series, window: int) -> pd.Series:
    """x 在过去 window 天的标准差"""
    return x.rolling(window, min_periods=window).std()


def ts_max(x: pd.Series, window: int) -> pd.Series:
    """x 在过去 window 天的最大值"""
    return x.rolling(window, min_periods=window).max()


def ts_min(x: pd.Series, window: int) -> pd.Series:
    """x 在过去 window 天的最小值"""
    return x.rolling(window, min_periods=window).min()


def ts_rank(x: pd.Series, window: int) -> pd.Series:
    """
    x 在过去 window 天内的当前值排名（0~1），
    即窗口内 rank / window；窗口长度不够时返回 NaN。
    """
    def _rank_last(a: np.ndarray) -> float:
        # a: 过去 window 天的数据
        # 当前值是 a[-1]，看它在窗口内的排序位置
        if len(a) == 0:
            return np.nan
        # 使用 argsort 排名，效率比直接 rank 高一点
        order = a.argsort()
        # order 里是从小到大的索引位置
        rank_pos = np.where(order == len(a) - 1)[0][0] + 1  # 1-based
        return rank_pos / len(a)

    return x.rolling(window, min_periods=window).apply(_rank_last, raw=True)


# ---------- 相关性与协方差 ----------

def corr(x: pd.Series, y: pd.Series, window: int) -> pd.Series:
    """过去 window 天 x 与 y 的相关系数"""
    return x.rolling(window, min_periods=window).corr(y)


def cov(x: pd.Series, y: pd.Series, window: int) -> pd.Series:
    """过去 window 天 x 与 y 的协方差"""
    return x.rolling(window, min_periods=window).cov(y)


# ---------- 截面排名与标准化 ----------

def rank(x: pd.Series) -> pd.Series:
    """
    截面 rank，0~1。若你是按日期截面调用，可以在外面 groupby 后再用这个函数。
    这里默认就对当前传入的 Series 做 0~1 排名。
    """
    return x.rank(pct=True)


def zscore(x: pd.Series) -> pd.Series:
    """
    截面 z-score 标准化： (x - mean) / std
    注意：如果 std 为 0，则返回 NaN。
    """
    mean = x.mean()
    std = x.std()
    return (x - mean) / std if std not in (0, np.nan) else x * np.nan


# ---------- 衰减加权与平滑 ----------

def decay_linear(x: pd.Series, window: int) -> pd.Series:
    """
    线性衰减加权移动平均：
    权重为 1,2,...,window（越近权重越大），再归一化。
    例如 window=5，权重 = [1,2,3,4,5] / 15。
    """
    if window <= 0:
        raise ValueError("window 必须为正整数")

    weights = np.arange(1, window + 1, dtype=float)
    weights /= weights.sum()

    def _apply_decay(a: np.ndarray) -> float:
        if len(a) < window:
            return np.nan
        return np.dot(a, weights)

    return x.rolling(window, min_periods=window).apply(_apply_decay, raw=True)
