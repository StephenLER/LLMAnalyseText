import os
import pandas as pd
import pandas_ta as ta

# =========================
# 0. 你给的技术指标函数
# =========================
def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    在原 OHLCV DataFrame 上添加技术指标。
    要求:
        - df 至少包含列: ['Open', 'High', 'Low', 'Close', 'Volume']
        - index 为 DatetimeIndex
    """
    df = df.copy()
    df = df.sort_index()

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

    # 1) SMA
    df["SMA_5"] = ta.sma(close=close, length=5)
    df["SMA_20"] = ta.sma(close=close, length=20)

    # 2) EMA
    df["EMA_10"] = ta.ema(close=close, length=10)

    # 3) Momentum
    df["MOM_3"] = ta.mom(close=close, length=3)
    df["MOM_10"] = ta.mom(close=close, length=10)

    # 4) RSI
    df["RSI_14"] = ta.rsi(close=close, length=14)

    # 5) MACD
    macd_df = ta.macd(close=close, fast=12, slow=26, signal=9)
    if macd_df is not None and not macd_df.empty:
        df["MACD"] = macd_df.iloc[:, 0]
        df["MACD_Signal"] = macd_df.iloc[:, 2]

    # 6) Bollinger Bands
    bb_df = ta.bbands(close=close, length=20, std=2)
    if bb_df is not None and not bb_df.empty:
        df["BB_Lower"] = bb_df.iloc[:, 0]
        df["BB_Upper"] = bb_df.iloc[:, 2]

    # 7) OBV
    df["OBV"] = ta.obv(close=close, volume=volume)

    return df


# =========================
# 1. 路径
# =========================
csv_path = "/home/wangyuting/share/quant/wangyuting/liangjian/llm4text/1125/new_concept_factor_pv.csv"
pv_path  = "/home/wangyuting/share/quant/wangyuting/liangjian/量价资金数据_concept_new.pkl"

if not os.path.exists(csv_path):
    raise FileNotFoundError(f"找不到 csv: {csv_path}")
if not os.path.exists(pv_path):
    raise FileNotFoundError(f"找不到量价 pkl: {pv_path}")

# =========================
# 2. 读取已有 csv（文本事件日表）
# =========================
event_df = pd.read_csv(csv_path)
event_df["concept_code"] = event_df["concept_code"].astype(str).str.strip()
event_df["date"] = pd.to_datetime(event_df["date"], errors="coerce")

concept_list = event_df["concept_code"].dropna().unique().tolist()
print(f"[INFO] 文本里出现的概念数: {len(concept_list)}")

# =========================
# 3. 读取量价全历史并过滤概念
# =========================
pv_df = pd.read_pickle(pv_path)
pv_df["code"] = pv_df["code"].astype(str).str.strip()
pv_df["date"] = pd.to_datetime(pv_df["date"], errors="coerce")

pv_df = pv_df[pv_df["code"].isin(concept_list)].copy()
pv_df = pv_df.sort_values(["code", "date"])

print(f"[INFO] 过滤后量价行数: {len(pv_df)}")

# =========================
# 4. 按 code 计算技术指标（在完整量价上算）
# =========================
def calc_one_code(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values("date").set_index("date")

    # 改成 add_technical_indicators 需要的列名
    g = g.rename(columns={
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume",
    })

    g = add_technical_indicators(g)

    # 还原列名（小写）或保留技术指标即可
    g = g.reset_index()
    return g

pv_with_tech = pv_df.groupby("code", group_keys=False).apply(calc_one_code)

tech_cols = [
    "SMA_5", "SMA_20", "EMA_10",
    "MOM_3", "MOM_10", "RSI_14",
    "MACD", "MACD_Signal",
    "BB_Lower", "BB_Upper",
    "OBV"
]

pv_with_tech_keep = pv_with_tech[["code", "date"] + tech_cols].copy()

# =========================
# 5. merge 回事件日表（concept_code + date）
# =========================
out_df = event_df.merge(
    pv_with_tech_keep,
    left_on=["concept_code", "date"],
    right_on=["code", "date"],
    how="left"
)

# 删掉多余 code 列
out_df = out_df.drop(columns=["code"])

missing_tech_rate = out_df[tech_cols].isna().all(axis=1).mean()
print(f"[INFO] 事件日技术指标全缺失比例: {missing_tech_rate:.2%}")

# =========================
# 6. 覆盖写回原 csv（追加列）
# =========================
out_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
print(f"[INFO] 已将技术指标追加写回: {csv_path}")
