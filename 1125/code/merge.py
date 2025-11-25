import os
import glob
import pandas as pd

# =========================
# 1. 路径
# =========================
fac_dir = "/home/wangyuting/share/quant/wangyuting/liangjian/fac_data"
pv_path = "/home/wangyuting/share/quant/wangyuting/liangjian/量价资金数据_concept_new.pkl"

# =========================
# 2. 读取并合并 fac_data 下所有文本因子文件
# =========================
fac_files = sorted(glob.glob(os.path.join(fac_dir, "*.pkl")))
if len(fac_files) == 0:
    raise FileNotFoundError(f"fac_data 文件夹下没有找到 pkl 文件：{fac_dir}")

txt_list = []
for fp in fac_files:
    try:
        df = pd.read_pickle(fp)
        print(f"[INFO] 读取成功: {fp}, shape={df.shape}")
        df["source_file"] = os.path.basename(fp)  # 记录来源文件，方便排查
        txt_list.append(df)
    except Exception as e:
        print(f"[WARN] 读取失败: {fp}, error={e}")

txt_df = pd.concat(txt_list, ignore_index=True)

# 统一字段名、类型
txt_df["concept_code"] = txt_df["concept_code"].astype(str).str.strip()
txt_df["last_trade_date"] = pd.to_datetime(txt_df["last_trade_date"], errors="coerce")
txt_df["next_trade_date"] = pd.to_datetime(txt_df["next_trade_date"], errors="coerce")

# 只保留关心的文本因子列
text_factor_cols = [
    "overall_sentiment",
    "overall_importance",
    "overall_impact",
    "overall_duration",
    "news_consistency",
    "fcast_ret",
]

keep_cols = ["concept_code", "concept_name", "last_trade_date"] + text_factor_cols
txt_df = txt_df[keep_cols]

# 如果同一 concept_code、last_trade_date 有多条文本记录，做聚合（取均值）
txt_df = (
    txt_df
    .groupby(["concept_code", "last_trade_date"], as_index=False)
    .agg(
        {**{c: "mean" for c in text_factor_cols},
         "concept_name": "first"}
    )
)

# =========================
# 3. 读取量价数据
# =========================
pv_df = pd.read_pickle(pv_path)

pv_df["code"] = pv_df["code"].astype(str).str.strip()
pv_df["date"] = pd.to_datetime(pv_df["date"], errors="coerce")

pv_keep_cols = [
    "code", "date",
    "amt", "close", "high", "low", "open", "volume",
    "开盘净主动买入额", "尾盘净主动买入额", "主力净流入额", "开盘主力净流入额", "净主动买入额", "尾盘主力净流入额",
    "开盘净主动买入额_成交额占比", "尾盘净主动买入额_成交额占比", "主力净流入额_成交额占比",
    "开盘主力净流入额_成交额占比", "净主动买入额_成交额占比", "尾盘主力净流入额_成交额占比",
    "dq_amtturnover", "turn",
]
pv_df = pv_df[pv_keep_cols]

pv_df = pv_df.rename(columns={
    "开盘净主动买入额": "open_net_active_buy_amt",
    "尾盘净主动买入额": "close_net_active_buy_amt",
    "主力净流入额": "main_net_inflow_amt",
    "开盘主力净流入额": "open_main_net_inflow_amt",
    "净主动买入额": "net_active_buy_amt",
    "尾盘主力净流入额": "close_main_net_inflow_amt",
    "开盘净主动买入额_成交额占比": "open_net_active_buy_amt_ratio",
    "尾盘净主动买入额_成交额占比": "close_net_active_buy_amt_ratio",
    "主力净流入额_成交额占比": "main_net_inflow_amt_ratio",
    "开盘主力净流入额_成交额占比": "open_main_net_inflow_amt_ratio",
    "净主动买入额_成交额占比": "net_active_buy_amt_ratio",
    "尾盘主力净流入额_成交额占比": "close_main_net_inflow_amt_ratio",
})
# =========================
# 4. concept_code -> code 映射 + last_trade_date -> date 对齐
# =========================
merged = txt_df.merge(
    pv_df,
    left_on=["concept_code", "last_trade_date"],
    right_on=["code", "date"],
    how="left",
)

# 提示一下哪些匹配不上（可选）
missing_rate = merged["close"].isna().mean()
print(f"[INFO] 量价未匹配上的比例: {missing_rate:.2%}")

# =========================
# 5. 生成新 df：你要的列
# =========================
new_df = merged[
    ["concept_code", "concept_name", "last_trade_date"] +
    text_factor_cols +[
        "amt", "close", "high", "low", "open", "volume",
        "open_net_active_buy_amt", "close_net_active_buy_amt",
        "main_net_inflow_amt", "open_main_net_inflow_amt",
        "net_active_buy_amt", "close_main_net_inflow_amt",
        "open_net_active_buy_amt_ratio", "close_net_active_buy_amt_ratio",
        "main_net_inflow_amt_ratio", "open_main_net_inflow_amt_ratio",
        "net_active_buy_amt_ratio", "close_main_net_inflow_amt_ratio",
        "dq_amtturnover", "turn",
    ]
].rename(columns={"last_trade_date": "date"})

# =========================
# 6. 缺失文本因子的处理
#    如果你不想填 0，可以把这段注释掉
# =========================
new_df[text_factor_cols] = new_df[text_factor_cols].fillna(0)

# 量价缺失一般不建议乱填，这里保持 NaN
# new_df[["close","high","low","open","volume"]] = new_df[["close","high","low","open","volume"]].fillna(method="ffill")

print(new_df.head())
print(new_df.info())

# =========================
# 7. 保存
# =========================
out_path = "/home/wangyuting/share/quant/wangyuting/liangjian/llm4text/1125/new_concept_factor_pv.pkl"
new_df.to_pickle(out_path)
print(f"[INFO] 已保存到: {out_path}")

out_path = "/home/wangyuting/share/quant/wangyuting/liangjian/llm4text/1125/new_concept_factor_pv.csv"
new_df.to_csv(out_path, index=False, encoding="utf-8-sig")
print(f"[INFO] 已保存到: {out_path}")
