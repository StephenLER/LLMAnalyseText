import pandas as pd
import numpy as np

csv_path = "/home/wangyuting/share/quant/wangyuting/liangjian/llm4text/1125/new_concept_factor_pv.csv"
df = pd.read_csv(csv_path)

if "date" in df.columns:
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

exclude = []  # 可排除不想标准化的数值列
num_cols = [c for c in num_cols if c not in exclude]

def zscore_group(g):
    for c in num_cols:
        mean = g[c].mean(skipna=True)
        std = g[c].std(skipna=True, ddof=0)
        if std == 0 or np.isnan(std):
            g[c] = 0.0
        else:
            g[c] = (g[c] - mean) / std
    return g

df = df.groupby("concept_code", group_keys=False).apply(zscore_group)

out_path = "/home/wangyuting/share/quant/wangyuting/liangjian/llm4text/1125/concept_factor_standardized_by_concept.csv"
df.to_csv(out_path, index=False, encoding="utf-8-sig")
print(f"[INFO] 分概念标准化后已保存到: {out_path}")