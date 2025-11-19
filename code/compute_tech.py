import pandas as pd
import pandas_ta as ta  # pip install pandas-ta

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    在原 OHLCV DataFrame 上添加论文中用到的技术指标。

    要求:
        - df 至少包含列: ['Open', 'High', 'Low', 'Close', 'Volume']
        - index 为 DatetimeIndex
    """
    df = df.copy()
    df = df.sort_index()

    # 确保列存在（大写）
    required_cols = ["Open", "High", "Low", "Close", "Volume"]
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(
                f"DataFrame 缺少必须列: {c}，当前列名为: {list(df.columns)}"
            )

    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    # 1) 简单移动均线 SMA
    df["SMA_5"] = ta.sma(close=close, length=5)
    df["SMA_20"] = ta.sma(close=close, length=20)

    # 2) 指数移动均线 EMA
    df["EMA_10"] = ta.ema(close=close, length=10)

    # 3) 动量 Momentum
    df["MOM_3"] = ta.mom(close=close, length=3)
    df["MOM_10"] = ta.mom(close=close, length=10)

    # 4) RSI 相对强弱指数
    df["RSI_14"] = ta.rsi(close=close, length=14)

    # 5) MACD（快线、慢线、信号线）
    macd_df = ta.macd(close=close, fast=12, slow=26, signal=9)
    if macd_df is not None and not macd_df.empty:
        # 通常列名类似: MACD_12_26_9, MACDh_12_26_9, MACDs_12_26_9
        df["MACD"] = macd_df.iloc[:, 0]         # MACD 主线
        df["MACD_Signal"] = macd_df.iloc[:, 2]  # 信号线

    # 6) 布林带 Bollinger Bands
    bb_df = ta.bbands(close=close, length=20, std=2)
    if bb_df is not None and not bb_df.empty:
        # 通常列顺序: [BBL, BBM, BBU, BBB, BBP]
        df["BB_Lower"] = bb_df.iloc[:, 0]
        df["BB_Upper"] = bb_df.iloc[:, 2]

    # 7) OBV 能量潮
    df["OBV"] = ta.obv(close=close, volume=volume)

    return df



input_path = "/home/wangyuting/share/quant/wangyuting/liangjian/llm4text/data_original/884126_2025_ohlcv.csv"
output_path = "/home/wangyuting/share/quant/wangyuting/liangjian/llm4text/code/884126_2025_with_ta.csv"

# 1) 读入数据
df = pd.read_csv(input_path)
print("当前列名列表：", df.columns.tolist())
for c in df.columns:
    print(repr(c))

# 2) 处理日期列：兼容 'date' 或 'Date'
if "date" in df.columns:
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
elif "Date" in df.columns:
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date")
else:
    raise ValueError("找不到日期列，请确认有 'date' 或 'Date' 列。")

# 3) 统一把 OHLCV 列名变成大写
rename_map = {}
for col in ["open", "high", "low", "close", "volume"]:
    if col in df.columns:
        rename_map[col] = col.capitalize() if col != "volume" else "Volume"
df = df.rename(columns=rename_map)

print("重命名后列名：", df.columns.tolist())

# 4) 调用技术指标函数
df_with_ta = add_technical_indicators(df)

# 5) 保存为新的 CSV（保留日期索引为第一列）
df_with_ta.to_csv(output_path, encoding="utf-8-sig")
print(f"已保存到: {output_path}")
