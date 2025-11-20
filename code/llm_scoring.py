import os
import json
import pandas as pd
from openai import OpenAI

# ---------- 基础配置 ----------
client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

concept_name = "机器人"

input_path = "data/884126_2025_news_by_tradedate.jsonl"
output_path = "data/884126_2025_robot_scores.jsonl"

# ---------- 工具函数 ----------

def build_news_text(news_list):
    """把某一天的多条新闻拼成一段文本"""
    parts = []
    for i, item in enumerate(news_list, start=1):
        title = item.get("title", "")
        content = item.get("content", "")
        parts.append(f"【新闻{i}】标题：{title}\n内容：{content}")
    return "\n\n".join(parts)

def build_prompt(trade_date, news_list):
    """根据交易日和新闻列表构造 user prompt"""
    news_text = build_news_text(news_list)
    prompt = f"""你是一名量化研究团队中的金融分析助手，现在需要你对“某个概念在某一天的所有新闻整体影响”进行评估。
请严格按照要求输出结构化评分，不要输出多余无关内容。

【目标概念】
{concept_name}

【日期】
{trade_date}

【当日新闻列表】
下面是该日期内，与该概念相关的新闻（已按时间排序，可能有重复或类似内容）：

{news_text}

请你从“整体”角度出发，对这一天的新闻相对于该概念的影响进行评估。不要逐条打分，而是综合所有新闻后给出以下维度的评分：

1. overall_sentiment（整体情绪）：-1 到 1 之间的小数，精确到 0.01。
   - 反映当日新闻对该概念整体是利空、利好还是中性。
2. overall_importance（整体重要性）：0-10 整数。
   - 看这一天的新闻，从重要性和罕见程度来看，是否属于“大事件”。
3. overall_impact（整体影响强度）：0-10 整数。
   - 评估这些新闻对该概念相关资产价格或市场预期的潜在冲击幅度。
4. overall_duration（影响持续性）：0-10 整数。
   - 0 表示大多是一次性、只影响当天；10 表示可能导致数月以上的持续影响。
5. news_consistency（一致性）：0-10 整数。
   - 看这些新闻之间的方向是否一致。10 表示高度一致（几乎都偏同一方向），0 表示高度分化（利好利空混杂、观点矛盾）。

请先用 2-4 句中文简单概括：
- 这一天围绕该概念发生了哪些关键事件或主题？
- 整体基调更偏利好、利空还是中性？是否有明显的市场焦点？

然后按以下 JSON 格式输出结果（严格遵守字段名和类型，不要多加字段）：

{{
  "concept": "{concept_name}",
  "summary": "<用 2-4 句中文概括当日整体情况>",
  "scores": {{
    "overall_sentiment": < -1 到 1 的小数，如 0.35 >,
    "overall_importance": < 0-10 的整数 >,
    "overall_impact": < 0-10 的整数 >,
    "overall_duration": < 0-10 的整数 >,
    "news_consistency": < 0-10 的整数 >
  }}
}}
"""
    return prompt

def call_llm(trade_date, news_list):
    """流式调用大模型：一边打印一边累积结果，最后解析 JSON"""
    prompt = build_prompt(trade_date, news_list)

    print(f"\n==================== {trade_date} 开始请求模型 ====================")
    completion = client.chat.completions.create(
        model="qwen3-max",
        messages=[
            {
                "role": "system",
                "content": "你是一名量化研究团队中的金融分析助手，请严格按照用户要求返回 JSON 格式结果。",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,
        stream=True,   # 开启流式
    )

    full_text = ""
    for chunk in completion:
        delta = chunk.choices[0].delta
        if delta and delta.content:
            piece = delta.content
            full_text += piece
            # 中间过程流式打印出来
            print(piece, end="", flush=True)

    print(f"\n==================== {trade_date} 输出结束，开始解析 JSON ====================")

    text = full_text.strip()
    # 处理可能出现的 ```json 包裹
    if text.startswith("```"):
        text = text.strip("`")
        text = text.replace("json\n", "").replace("JSON\n", "").strip()

    result = json.loads(text)  # 如果 json 不合法会抛异常
    return result

# ---------- 主流程 ----------

df_daily = pd.read_json(input_path, lines=True)

outputs = []

for _, row in df_daily.iterrows():
    trade_date = row["trade_date"]        # 如 "2025-01-06"
    news_list = row["news_list"]          # list[dict]

    try:
        llm_result = call_llm(trade_date, news_list)
    except Exception as e:
        print(f"[ERROR] date={trade_date}, 调用或解析失败：{e}")
        continue

    # 在程序里补上日期
    out_item = {
        "trade_date": trade_date,
        "concept": llm_result.get("concept", concept_name),
        "summary": llm_result.get("summary", ""),
        "scores": llm_result.get("scores", {}),
    }
    outputs.append(out_item)

# 写成新的 jsonl
with open(output_path, "w", encoding="utf-8") as f:
    for item in outputs:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"\n全部完成，已保存评分结果到：{output_path}")
