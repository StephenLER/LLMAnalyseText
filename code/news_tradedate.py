import pandas as pd

csv_path = "data/884126_2025_with_ta.csv"        # 行情 csv
news_path = "data_original/884126_2025_news.jsonl"  # 原始新闻 jsonl
out_path = "data/884126_2025_news_by_tradedate.jsonl"  # 输出的聚合 jsonl

# 1. 读取行情数据，拿到所有开盘日
df_price = pd.read_csv(csv_path, parse_dates=['date'])
df_price = df_price.sort_values('date')

# 交易日列表（去重+改名，方便后面 merge）
trade_days = (
    df_price[['date']]
    .drop_duplicates()
    .rename(columns={'date': 'trade_date'})
    .sort_values('trade_date')
)

# 2. 读取新闻数据
df_news = pd.read_json(news_path, lines=True)

# 如果 date 是字符串，先转成 datetime
df_news['date'] = pd.to_datetime(df_news['date'])
df_news = df_news.sort_values('date')

# 3. 对齐到下一个交易日（或当天，如果当天就是交易日）
df_merged = pd.merge_asof(
    df_news,
    trade_days,
    left_on='date',
    right_on='trade_date',
    direction='forward'   # 找到 >= date 的第一个 trade_date
)

# 如果新闻日期晚于最后一个交易日，会出现 trade_date 为 NaN：
df_merged = df_merged.dropna(subset=['trade_date'])

# 4. 按交易日聚合新闻
# 每条新闻只保留你需要的字段，比如 date/title/content
df_grouped = (
    df_merged
    .groupby('trade_date')
    .apply(lambda g: g[['date', 'title', 'content']].to_dict('records'))
    .reset_index(name='news_list')
)

# 交易日转成字符串（方便写 json）
df_grouped['trade_date'] = df_grouped['trade_date'].dt.strftime('%Y-%m-%d')
df_grouped['date_list'] = df_grouped['news_list'].apply(
    lambda lst: sorted({item['date'].strftime('%Y-%m-%d') for item in lst})
)

# 5. 存成新的 jsonl 文件
# 每行结构示例：
# {
#   "trade_date": "2025-01-06",
#   "date_list": ["2025-01-04", "2025-01-05", "2025-01-06"],
#   "news_list": [
#       {"date": "2025-01-04", "title": "...", "content": "..."},
#       ...
#   ]
# }
df_out = df_grouped.copy()
# 把内部的 date 也转字符串
for row in df_out['news_list']:
    for item in row:
        item['date'] = item['date'].strftime('%Y-%m-%d')

df_out.to_json(out_path, orient='records', lines=True, force_ascii=False)

print(f"保存完成：{out_path}")
