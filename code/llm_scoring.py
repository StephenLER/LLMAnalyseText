import os
import json
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

# ---------- 基础配置 ----------
client = OpenAI(
    api_key='',
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

concept_name = "机器人"

input_path = "data/884126_2025_news_by_tradedate.jsonl"
# 原始结果（追加写入、可断点续跑）
output_path = "data/884126_2025_robot_scores_raw.jsonl"
# 排序后的最终文件
sorted_output_path = "data/884126_2025_robot_scores_sorted.jsonl"

# 并发线程数
MAX_WORKERS = 1


# ---------- 工具函数 ----------

def load_done_dates(path: str):
    """
    断点续跑：从已有结果 jsonl 中读出已经完成的 trade_date 集合。
    即使有坏行（未写完的 JSON），也会自动忽略。
    """
    done = set()
    if not os.path.exists(path):
        return done

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                # 上次异常中断导致的不完整行，跳过
                continue
            td = obj.get("trade_date")
            if td:
                done.add(td)
    print(f"[RESUME] 已检测到 {len(done)} 个已完成交易日，将自动跳过。")
    return done


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
2. overall_importance（整体重要性）：0-10 整数。
3. overall_impact（整体影响强度）：0-10 整数。
4. overall_duration（影响持续性）：0-10 整数。
5. news_consistency（一致性）：0-10 整数。

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
    """
    单个交易日的工作：
    - 构造 prompt
    - 非流式调用 LLM
    - 返回解析后的 dict
    """
    prompt = build_prompt(trade_date, news_list)

    print(f"[START] {trade_date} 提交模型请求")
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
        stream=False,  # 多线程下不再流式输出
    )

    text = completion.choices[0].message.content.strip()
    print(f"[DONE]  {trade_date} 收到模型响应")

    # 处理可能出现的 ```json 包裹
    if text.startswith("```"):
        text = text.strip("`")
        text = text.replace("json\n", "").replace("JSON\n", "").strip()

    result = json.loads(text)
    return result


def sort_result_file(raw_path: str, sorted_path: str):
    """
    读取原始结果文件（可能是乱序、断点续跑来的），
    按 trade_date 排序后写到 sorted_path。
    """
    records = []
    if not os.path.exists(raw_path):
        print(f"[SORT] 原始结果文件不存在：{raw_path}")
        return

    with open(raw_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if "trade_date" in obj:
                records.append(obj)

    if not records:
        print("[SORT] 没有可排序的记录。")
        return

    # 按 trade_date 排序（假设是 YYYY-MM-DD 字符串，直接字符串排序即可）
    records.sort(key=lambda x: x["trade_date"])

    # 写到新文件（如果你想覆盖原文件，可以最后 rename）
    with open(sorted_path, "w", encoding="utf-8") as f:
        for obj in records:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"[SORT] 已按 trade_date 排序并写入：{sorted_path}")


# ---------- 主流程：断点续跑 + 多线程 + 逐条写入 ----------

def main():
    # 1. 读入按交易日聚合好的新闻
    df_daily = pd.read_json(input_path, lines=True)

    # 2. 断点续跑：先找出已经完成的日期
    done_dates = load_done_dates(output_path)

    # 3. 准备需要处理的任务列表
    tasks = []
    for _, row in df_daily.iterrows():
        trade_date = row["trade_date"]
        if trade_date in done_dates:
            print(f"[SKIP] {trade_date} 已存在结果，跳过。")
            continue
        news_list = row["news_list"]
        tasks.append((trade_date, news_list))

    print(f"[INFO] 需要新处理的交易日数量：{len(tasks)}")
    if not tasks:
        print("[INFO] 没有新的任务需要处理。")
        # 即使没有新任务，仍然可以再跑一遍排序
        sort_result_file(output_path, sorted_output_path)
        return

    # 4. 多线程调用 + 主线程统一写文件
    #    文件用追加模式 "a"，不存在会自动创建
    with open(output_path, "a", encoding="utf-8") as f_out:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_date = {
                executor.submit(call_llm, trade_date, news_list): trade_date
                for trade_date, news_list in tasks
            }

            for future in as_completed(future_to_date):
                trade_date = future_to_date[future]
                try:
                    llm_result = future.result()
                except Exception as e:
                    print(f"[ERROR] date={trade_date}, 调用或解析失败：{e}")
                    continue

                out_item = {
                    "trade_date": trade_date,
                    "concept": llm_result.get("concept", concept_name),
                    "summary": llm_result.get("summary", ""),
                    "scores": llm_result.get("scores", {}),
                }

                # 主线程写入，避免写入冲突
                f_out.write(json.dumps(out_item, ensure_ascii=False) + "\n")
                f_out.flush()
                print(f"[WRITE] 已写入 {trade_date}")

    print(f"\n[INFO] 全部任务完成，结果已追加写入：{output_path}")

    # 5. 最后按 trade_date 排序并生成一个干净的排序结果文件
    sort_result_file(output_path, sorted_output_path)


if __name__ == "__main__":
    main()
