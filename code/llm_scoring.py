import os
import json
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
import tempfile

# ---------- 基础配置 ----------
client = OpenAI(
    api_key='sk-9b4b1fb4b60d4a69b258ebcb2b5a122b',
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

def validate_result(obj):
    """
    检查一条结果是否完全符合预期格式。
    返回 (ok: bool, err_msg: str)
    """
    if not isinstance(obj, dict):
        return False, "顶层不是 JSON 对象"

    expected_top_keys = {"trade_date", "concept", "summary", "scores", "score_reasons"}
    actual_top_keys = set(obj.keys())
    if not actual_top_keys.issuperset(expected_top_keys - {"trade_date"}):
        return False, f"顶层字段缺失，期望至少包含 {expected_top_keys}，实际 {actual_top_keys}"

    # concept
    if not isinstance(obj.get("concept", ""), str):
        return False, "concept 必须是字符串"

    # summary
    summary = obj.get("summary")
    if not isinstance(summary, dict):
        return False, "summary 必须是对象"
    expected_summary_keys = {"正面事件", "负面事件", "综合分析"}
    if set(summary.keys()) != expected_summary_keys:
        return False, f"summary 字段必须且仅包含 {expected_summary_keys}"
    for k in expected_summary_keys:
        if not isinstance(summary[k], str):
            return False, f"summary['{k}'] 必须是字符串"

    # scores
    scores = obj.get("scores")
    if not isinstance(scores, dict):
        return False, "scores 必须是对象"
    expected_score_keys = {
        "overall_sentiment",
        "overall_importance",
        "overall_impact",
        "overall_duration",
        "news_consistency",
    }
    if set(scores.keys()) != expected_score_keys:
        return False, f"scores 字段必须且仅包含 {expected_score_keys}"

    s = scores["overall_sentiment"]
    if not isinstance(s, (int, float)):
        return False, "overall_sentiment 必须是数值"
    # if s < -1.0 or s > 1.0:
    #     return False, "overall_sentiment 必须在 [-1, 1] 区间内"

    for k in ["overall_importance", "overall_impact", "overall_duration", "news_consistency"]:
        v = scores[k]
        if not isinstance(v, int):
            return False, f"{k} 必须是整数"
        # if v < 0 or v > 10:
        #     return False, f"{k} 必须在 [0, 10] 区间内"

    # score_reasons
    reasons = obj.get("score_reasons")
    if not isinstance(reasons, dict):
        return False, "score_reasons 必须是对象"
    if set(reasons.keys()) != expected_score_keys:
        return False, f"score_reasons 字段必须且仅包含 {expected_score_keys}"
    for k in expected_score_keys:
        if not isinstance(reasons[k], str):
            return False, f"score_reasons['{k}'] 必须是字符串"

    return True, ""

def clean_output_file(path: str):
    """
    读一遍 output_path：
    - 能解析且 validate_result 通过的行 → 写入临时文件
    - 其他行 → 视为坏行，丢弃
    最后用临时文件覆盖原文件。

    返回 (total_lines, good_lines, bad_lines)
    """
    if not os.path.exists(path):
        print(f"[CLEAN] 结果文件不存在：{path}")
        return 0, 0, 0

    total = 0
    good = 0
    bad = 0

    # 用临时文件避免中途崩掉导致文件空了
    dir_name = os.path.dirname(path) or "."
    fd, tmp_path = tempfile.mkstemp(dir=dir_name, prefix="clean_", suffix=".jsonl")
    os.close(fd)

    with open(path, "r", encoding="utf-8") as f_in, \
         open(tmp_path, "w", encoding="utf-8") as f_out:

        for line in f_in:
            line = line.strip()
            if not line:
                continue
            total += 1

            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                bad += 1
                print(f"[CLEAN] JSON 解析失败，跳过一行：{e}")
                print(f"[CLEAN] 该行前 200 字符：{line[:200]}")
                continue

            ok, err = validate_result(obj)
            if not ok:
                bad += 1
                print(f"[CLEAN] 校验不通过，跳过 trade_date={obj.get('trade_date')}：{err}")
                continue

            # 合法行写入临时文件
            f_out.write(json.dumps(obj, ensure_ascii=False) + "\n")
            good += 1

    # 用清理后的文件替换原文件
    os.replace(tmp_path, path)
    print(f"[CLEAN] 清理完成：总行数={total}, 合法={good}, 删除坏行={bad}")
    return total, good, bad

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
   - 反映当日新闻对该概念整体是利空、利好还是中性。
   - 区间含义建议如下（不是硬性规则，但请参考使用）：
     - [-1.00, -0.60]：强烈利空（几乎所有重要新闻都明显偏负面）。
     - (-0.60, -0.20]：偏利空（负面新闻占主导，但不至于极端）。
     - (-0.20,  0.20)：基本中性（利好利空相互抵消或影响都很弱）。
     - [ 0.20,  0.60)：偏利好（正面新闻占主导）。
     - [ 0.60,  1.00]：强烈利好（重要新闻普遍强烈正面）。
   - 当证据明显偏向一侧时，请适当使用偏两端的分值，不要习惯性集中在 0 附近。

2. overall_importance（整体重要性）：0-10 的整数。
   - 衡量这一天围绕该概念的新闻，在“事件重要性 + 罕见程度”上的整体水平。
   - 请尽量参考以下五档区间使用（不是硬性规则，但请尽量覆盖全区间）：
     - 0-1：极低 —— 几乎没有与该概念实质相关的内容，或都是极小的边缘信息。
     - 2-3：较低 —— 有一些相关新闻，但多为常规、小幅度更新，对整体格局影响有限。
     - 4-6：中等 —— 存在一定分量的消息，如中等规模订单、企业级策略调整、行业内一般性政策。
     - 7-8：较高 —— 涉及较重大的公司/行业事件，如重要政策、核心企业重大决策、规模较大的订单或合作。
     - 9-10：极高 —— 重大转折类事件，如关键监管政策落地、行业范式变化、对中长期格局有显著影响的消息。
   - 如果你判断当天的新闻确实“不大不小”，才使用 4-6 分，不要默认所有情况都给 5 分。

3. overall_impact（整体影响强度）：0-10 的整数。
   - 衡量这些新闻对该概念相关资产价格或市场预期的“潜在冲击幅度”（不区分方向，只看强弱）。
   - 区间参考：
     - 0-1：几乎没有可观察到的影响，市场大概率不会有明显反应。
     - 2-3：影响较弱，可能只引起短期小幅波动或局部个股反应。
     - 4-6：中等影响，概念整体有一定被关注和定价调整的可能。
     - 7-8：较强影响，预期会引起较明显的资金关注或价格波动。
     - 9-10：非常强的影响，可能引发显著行情或中长期定价重估。
   - 若新闻在重要性不算极端，但集中指向同一方向、且与定价高度相关，也可以给到 7-8 分。

4. overall_duration（影响持续性）：0-10 的整数。
   - 衡量这些新闻对该概念影响的“预期持续时间”（不看强度，只看时间长度）。
   - 区间参考：
     - 0-1：几乎只影响当天或极短期（日内/隔日），典型如短期噪音、谣言、一次性事件。
     - 2-3：短期影响为主，大致 2-3 天内逐渐消化。
     - 4-6：中等持续性，大致可以影响 1-4 周（一段行情或预期周期）。
     - 7-8：较长时间影响，大致可影响数月，如中长期订单、阶段性政策、产业趋势强化等。
     - 9-10：长期甚至结构性影响，可能重塑该概念的中长期逻辑。
   - 若消息本身是结构性、制度性或长期规划，请优先考虑 7 分以上。

5. news_consistency（一致性）：0-10 的整数。
   - 衡量这些新闻在“方向和叙事上的一致程度”（而非好坏本身）。
   - 区间参考：
     - 0-1：高度分化或混乱 —— 利好、利空、中性信息混杂，叙事方向互相矛盾。
     - 2-3：较为分化 —— 虽有一定主方向，但相反观点或干扰信息较多。
     - 4-6：中等一致 —— 大致有主线方向，但仍存在少量相反或噪音信息。
     - 7-8：较高一致 —— 绝大部分重要新闻都指向相同的逻辑或方向。
     - 9-10：极高一致 —— 核心信息高度统一，几乎没有明显相反叙事。
   - 如果你看到的是“单一强主题 + 少量噪音”，可以考虑给到 7 分以上。

在给出数值评分前，请先用中文结构化概括当日整体情况，按照下列三个维度输出内容（可以是简短段落或要点列举）：
- “正面事件”：总结与该概念相关的主要利好或积极信息；
- “负面事件”：总结与该概念相关的主要利空或风险信息；如没有明显负面，可以写“无明显负面事件”；
- “综合分析”：对当日整体情况的综合判断和结论，说明整体基调偏利好/中性/利空，以及是否存在需要重点关注的风险或机会。


此外，请对每一个数值评分给出 1-2 句中文理由，说明：
- 该维度的核心依据是什么（对应了哪些新闻或信息特征）；
- 为什么选择该分值区间而不是明显更高或更低的区间。

然后按以下 JSON 格式输出结果（严格遵守字段名和类型，不要多加字段）：

{{
  "concept": "{concept_name}",
  "summary": {{
    "正面事件": "<用中文概括与该概念相关的主要正面/利好事件，若无可写“无明显正面事件”>",
    "负面事件": "<用中文概括与该概念相关的主要负面/风险事件，若无可写“无明显负面事件”>",
    "综合分析": "<用中文综合分析当日整体情况和基调>"
  }},
  "scores": {{
    "overall_sentiment": < -1 到 1 的小数，如 0.35 >,
    "overall_importance": < 0-10 的整数 >,
    "overall_impact": < 0-10 的整数 >,
    "overall_duration": < 0-10 的整数 >,
    "news_consistency": < 0-10 的整数 >
  }},
  "score_reasons": {{
    "overall_sentiment": "<简要说明整体情绪评分的主要依据和理由>",
    "overall_importance": "<简要说明整体重要性评分的主要依据和理由>",
    "overall_impact": "<简要说明整体影响强度评分的主要依据和理由>",
    "overall_duration": "<简要说明影响持续性评分的主要依据和理由>",
    "news_consistency": "<简要说明新闻一致性评分的主要依据和理由>"
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
        # 即使没有新任务，也可以再跑一遍清理+排序
        total, good, bad = clean_output_file(output_path)
        if bad == 0:
            sort_result_file(output_path, sorted_output_path)
        return

    # 4. 多线程调用 + 主线程统一写文件
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
                    "summary": llm_result.get("summary", {}),
                    "scores": llm_result.get("scores", {}),
                    "score_reasons": llm_result.get("score_reasons", {}),
                }

                f_out.write(json.dumps(out_item, ensure_ascii=False) + "\n")
                f_out.flush()
                print(f"[WRITE] 已写入 {trade_date}")

    print(f"\n[INFO] 全部任务完成，结果已追加写入：{output_path}")

    # 5. 跑完后整体检查一次结果文件，删除坏行
    total, good, bad = clean_output_file(output_path)

    if bad > 0:
        print(f"[WARN] 检测到 {bad} 条不合规记录，已从 {output_path} 删除。")
        print("[WARN] 建议重新运行脚本一次，断点续跑会自动处理缺失的 trade_date。")
        return

    # 6. 没有坏行，直接排序
    sort_result_file(output_path, sorted_output_path)


if __name__ == "__main__":
    main()