import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ===============================
# 1. 前瞻收益计算：fwd_ret_1d
# ===============================

def add_forward_return(
    df: pd.DataFrame,
    price_col: str = "close",
    group_col: str = "concept_code",
    horizon: int = 1,
    out_col: str = None,
) -> pd.DataFrame:
    """
    在 df 上添加前瞻收益列：fwd_ret_horizon
    默认计算： (P_{t+h} / P_t - 1)

    参数：
        df        : 包含价格和 concept_code 的 DataFrame
        price_col : 价格列名（如 'close'）
        group_col : 分组列名（如 'concept_code'）
        horizon   : 前瞻期（单位：交易日），1 表示 t->t+1 收益
        out_col   : 输出列名，不填则为 f"fwd_ret_{horizon}d"

    返回：
        带有新收益列的 DataFrame（copy 一份）
    """
    if out_col is None:
        out_col = f"fwd_ret_{horizon}d"

    df = df.copy()
    # 按概念 + 日期排序，保证时间顺序
    df = df.sort_values([group_col, "date"])

    # 未来价格：按概念分组向后 shift
    future_price = df.groupby(group_col)[price_col].shift(-horizon)

    # 当前价格
    current_price = df[price_col]

    # 前瞻收益：P_{t+h} / P_t - 1
    df[out_col] = future_price / current_price - 1.0

    return df


# ===============================
# 2. daily IC 计算（截面相关）
# ===============================

def calc_daily_ic(
    df: pd.DataFrame,
    factor_col: str,
    ret_col: str = "fwd_ret_1d",
    method: str = "spearman",
    date_col: str = "date",
) -> pd.Series:
    """
    计算某个因子的 daily IC 序列（按 date 截面相关）。

    参数：
        df        : DataFrame，包含因子值和前瞻收益
        factor_col: 因子列名，如 'alpha1'
        ret_col   : 收益列名，如 'fwd_ret_1d'
        method    : 'spearman'（rank IC）或 'pearson'
        date_col  : 日期列名

    返回：
        一个以 date 为 index 的 Series，每天一个 IC 值
    """

    def _cs_ic(group: pd.DataFrame) -> float:
        x = group[factor_col]
        y = group[ret_col]

        # 去掉 NA
        mask = x.notna() & y.notna()
        x = x[mask]
        y = y[mask]
        # 至少要有 3 个点才算
        if x.size < 3:
            return np.nan

        if method == "spearman":
            xr = x.rank(pct=True)
            yr = y.rank(pct=True)
            return xr.corr(yr)
        elif method == "pearson":
            return x.corr(y)
        else:
            raise ValueError(f"未知 method: {method}")

    daily_ic = df.groupby(date_col).apply(_cs_ic)
    daily_ic.name = factor_col
    return daily_ic


# ===============================
# 3. 多因子 IC & ICIR 汇总
# ===============================

def summarize_factors_ic(
    df: pd.DataFrame,
    factor_cols: list,
    ret_col: str = "fwd_ret_1d",
    method: str = "spearman",
    date_col: str = "date",
):
    """
    对多个因子计算 daily IC 序列，并给出 IC_mean / IC_std / ICIR / N_days。

    返回：
        ic_summary_df  : 每个因子一行的汇总表
        daily_ic_panel : index=date, columns=factors 的 DataFrame
    """
    all_daily_ic = []
    summary_rows = []

    for fac in factor_cols:
        daily_ic = calc_daily_ic(
            df, factor_col=fac, ret_col=ret_col,
            method=method, date_col=date_col
        )
        all_daily_ic.append(daily_ic)

        daily_ic_clean = daily_ic.dropna()
        if daily_ic_clean.empty:
            ic_mean = np.nan
            ic_std = np.nan
            icir = np.nan
            n_days = 0
        else:
            ic_mean = daily_ic_clean.mean()
            ic_std = daily_ic_clean.std(ddof=1)
            icir = ic_mean / ic_std if ic_std not in (0, np.nan) and ic_std != 0 else np.nan
            n_days = daily_ic_clean.shape[0]

        summary_rows.append(
            {
                "factor": fac,
                "IC_mean": ic_mean,
                "IC_std": ic_std,
                "ICIR": icir,
                "N_days": n_days,
            }
        )

    daily_ic_panel = pd.concat(all_daily_ic, axis=1)
    ic_summary_df = pd.DataFrame(summary_rows).sort_values(
        "IC_mean", ascending=False
    )
    return ic_summary_df, daily_ic_panel


# ===============================
# 4. 画图函数
# ===============================

def plot_ic_time_series(
    daily_ic_panel: pd.DataFrame,
    factors: list,
    figsize=(10, 6),
    title_suffix: str = "",
):
    """
    画若干因子的 IC 时间序列。
    """
    import matplotlib.dates as mdates

    plt.figure(figsize=figsize)
    for fac in factors:
        if fac in daily_ic_panel.columns:
            plt.plot(daily_ic_panel.index, daily_ic_panel[fac], label=fac)

    plt.axhline(0, linestyle="--")
    plt.xlabel("Date")
    plt.ylabel("Daily IC")
    if title_suffix:
        plt.title(f"Daily IC Time Series {title_suffix}")
    else:
        plt.title("Daily IC Time Series")
    plt.legend()
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.tight_layout()
    plt.savefig("daily_ic_time_series.png")
    plt.show()


def plot_ic_hist(
    daily_ic_panel: pd.DataFrame,
    factor: str,
    bins: int = 30,
    figsize=(6, 4),
    title_suffix: str = "",
):
    """
    画某个因子的 IC 分布直方图。
    """
    if factor not in daily_ic_panel.columns:
        print(f"{factor} 不在 daily_ic_panel 列中")
        return

    data = daily_ic_panel[factor].dropna()

    plt.figure(figsize=figsize)
    plt.hist(data, bins=bins)
    plt.xlabel("IC")
    plt.ylabel("Frequency")
    if title_suffix:
        plt.title(f"IC Distribution for {factor} {title_suffix}")
    else:
        plt.title(f"IC Distribution for {factor}")
    plt.tight_layout()
    plt.savefig(f"ic_hist_{factor}.png")
    plt.show()


# ===============================
# 5. 主流程
# ===============================

if __name__ == "__main__":
    # === 1) 读取你的数据 ===
    csv_path = "/home/wangyuting/share/quant/wangyuting/liangjian/llm4text/1125/concept_factor_standardized_by_concept.csv"
    df = pd.read_csv(csv_path)

    # 确保 date 是 datetime
    if not np.issubdtype(df["date"].dtype, np.datetime64):
        df["date"] = pd.to_datetime(df["date"])

    # === 2) 计算 1 日前瞻收益 ===
    df = add_forward_return(
        df,
        price_col="close",
        group_col="concept_code",
        horizon=1,
        out_col="fwd_ret_1d",
    )

    # 简单看一下
    print(df[["concept_code", "date", "close", "fwd_ret_1d"]]
          .sort_values(["concept_code", "date"])
          .head(10))

    # === 3) 找出所有 alpha 列 ===
    factor_cols = [c for c in df.columns if c.startswith("alpha")]
    factor_cols = sorted(factor_cols, key=lambda x: int(x.replace("alpha", "")))
    print("找到的因子列：", factor_cols)

    # === 4) 计算 IC & ICIR 汇总 + daily IC 面板 ===
    ic_summary_df, daily_ic_panel = summarize_factors_ic(
        df,
        factor_cols=factor_cols,
        ret_col="fwd_ret_1d",
        method="spearman",   # rank IC
        date_col="date",
    )

    print("===== IC 汇总（按 IC_mean 降序）=====")
    print(ic_summary_df.head(20))

    # 保存结果
    ic_summary_path = "/home/wangyuting/share/quant/wangyuting/liangjian/llm4text/1125/res/ic_summary.csv"
    daily_ic_panel_path = "/home/wangyuting/share/quant/wangyuting/liangjian/llm4text/1125/res/daily_ic_panel.csv"
    ic_summary_df.to_csv(ic_summary_path, index=False)
    daily_ic_panel.to_csv(daily_ic_panel_path)

    print("IC 汇总已保存到：", ic_summary_path)
    print("daily IC 面板已保存到：", daily_ic_panel_path)

    # === 5) 画图：选 IC_mean 最高的前 5 个因子 ===
    top_factors = (
        ic_summary_df.sort_values("IC_mean", ascending=False)["factor"]
        .head(5)
        .tolist()
    )
    print("用于画图的前 5 个因子：", top_factors)

    if top_factors:
        plot_ic_time_series(
            daily_ic_panel,
            factors=top_factors,
            figsize=(10, 6),
            title_suffix="(Top 5 factors by IC_mean)",
        )

        # 画 IC_mean 最大的那个因子的 IC 分布
        plot_ic_hist(
            daily_ic_panel,
            factor=top_factors[0],
            bins=30,
            figsize=(6, 4),
            title_suffix="(Top 1 factor)",
        )
