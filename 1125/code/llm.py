import json
from openai import OpenAI
from pathlib import Path

# ======== 配置区 ========

# 提示词文件（
PROMPT_PATH = Path("/home/wangyuting/share/quant/wangyuting/liangjian/llm4text/prompt/alpha_prompt.txt")

# 生成的 alpha 因子 Python 文件输出位置
OUTPUT_ALPHA_PY = Path("/home/wangyuting/share/quant/wangyuting/liangjian/llm4text/1125/generated_alphas.py")

# 保存大模型原始回答的位置（方便之后排查/复现）
LLM_RESPONSE_PATH = Path("/home/wangyuting/share/quant/wangyuting/liangjian/llm4text/1125/alpha_qwen3_235b_raw_response.json")

# （后面你会用到的 CSV，现阶段只是预留，不在本脚本里真的读取）
CSV_PATH = Path("/home/wangyuting/share/quant/wangyuting/liangjian/llm4text/1125/concept_factor_standardized_by_concept.csv")

# LLM 客户端配置
# client = OpenAI(
#     base_url="http://0.0.0.0:8002/v1",
#     api_key="EMPTY",
# )
# MODEL_NAME = "my_model2"

# 设置 OpenAI 客户端
client = OpenAI(
    api_key="",  # API Key
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)
MODEL_NAME = "qwen3-235b-a22b-thinking-2507"

def load_prompt(prompt_path: Path) -> str:
    """读取提示词文本。"""
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
    return prompt_path.read_text(encoding="utf-8")


def call_llm_streaming(prompt_text: str) -> str:
    """
    流式调用大模型：
    - 一边从接口读取 delta，一边在终端打印；
    - 同时把完整回答拼成一个字符串返回。
    """
    print("[INFO] Calling LLM with streaming...")
    full_content_chunks = []

    # 流式接口
    stream = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt_text}],
        temperature=0.5,
        stream=True,
    )

    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            print(delta, end="", flush=True)
            full_content_chunks.append(delta)

    print()  # 换行
    full_content = "".join(full_content_chunks)
    print("[INFO] LLM streaming finished.")
    return full_content

import json

def clean_and_parse_json(content: str) -> dict:
    """
    按当前大模型的返回格式解析 JSON：

    格式类似：
    好的，我现在需要帮用户设计50个alpha信号……（一大段中文思考）
    </think>

    ```json
    { "alphas": [...] }
    ```

    逻辑：
    1. 找到 ```json 代码块；
    2. 取其中内容作为 JSON 字符串；
    3. 用 json.loads 解析。
    """
    if not content:
        raise ValueError("模型返回内容为空，无法解析 JSON")

    text = content.strip()

    json_str = None

    # 1) 优先找 ```json ... ``` 代码块
    fence = "```json"
    if fence in text:
        start = text.index(fence) + len(fence)
        end = text.find("```", start)
        if end == -1:
            # 没有找到结束的 ```，那就取到结尾
            json_str = text[start:].strip()
        else:
            json_str = text[start:end].strip()
    else:
        # 2) 退一步：找任意 ``` ... ``` 代码块
        fence = "```"
        if fence in text:
            start = text.index(fence) + len(fence)
            end = text.find("```", start)
            if end == -1:
                json_str = text[start:].strip()
            else:
                json_str = text[start:end].strip()
        else:
            # 3) 再退一步：整串当 JSON（一般用不到）
            json_str = text

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"无法解析模型返回的 JSON：{e}\n"
            f"尝试解析的内容为：\n{json_str[:1000]}..."
        )

    return data


def build_alpha_module_code(alpha_list: list[dict]) -> str:
    """
    根据 JSON 里的 alphas 列表，生成一个 Python 模块字符串。

    生成的模块结构大致如下：

    import numpy as np
    import pandas as pd

    def compute_alphas(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['alpha1'] = <expression1>
        ...
        return df
    """
    lines: list[str] = []

    lines.append("# 该文件由 LLM 自动生成，请勿手动修改原始表达式，")
    lines.append("# 如需调整，请修改提示词并重新生成。")
    lines.append("import numpy as np")
    lines.append("import pandas as pd")
    lines.append("")
    lines.append("")
    lines.append("def compute_alphas(df: pd.DataFrame) -> pd.DataFrame:")
    lines.append("    \"\"\"")
    lines.append("    根据大模型生成的 50 个 alpha 表达式，在 df 上计算并追加对应列。")
    lines.append("    df：包含文本因子、价格、技术指标等列的 DataFrame。")
    lines.append("    返回带有 alpha1 ~ alpha50 新列的 DataFrame 副本。")
    lines.append("    \"\"\"")
    lines.append("    df = df.copy()")
    lines.append("")

    for alpha in alpha_list:
        name = alpha.get("name")
        expr = alpha.get("expression")

        if not name or not expr:
            raise ValueError(f"alpha 对象缺少 name 或 expression: {alpha}")

        # 简单检查 name 格式
        if not name.startswith("alpha"):
            raise ValueError(f"alpha name 格式异常：{name}")

        # 为了可读性，先加注释
        lines.append(f"    # {name}")
        # 直接把表达式拼进去：df['alphaX'] = <expr>
        lines.append(f"    df['{name}'] = {expr}")
        lines.append("")

    lines.append("    return df")
    lines.append("")
    lines.append("")
    lines.append("if __name__ == '__main__':")
    lines.append("    # 示例：你可以在这里手动测试 compute_alphas，")
    lines.append("    # 例如读取 CSV 后按概念分组计算，再保存回去。")
    lines.append("    # 现阶段按你的要求，这里先留空或写测试代码即可。")
    lines.append("    pass")

    return "\n".join(lines)


def main():
    # 1. 读取提示词
    prompt_text = load_prompt(PROMPT_PATH)
    print(f"[INFO] Loaded prompt from: {PROMPT_PATH}")

    # 2. 流式调用大模型
    raw_content = call_llm_streaming(prompt_text)

    # 3. 保存原始回答到文件（方便之后排查和复现实验）
    LLM_RESPONSE_PATH.parent.mkdir(parents=True, exist_ok=True)
    LLM_RESPONSE_PATH.write_text(raw_content, encoding="utf-8")
    print(f"[INFO] LLM raw response saved to: {LLM_RESPONSE_PATH}")

    # 4. 解析 JSON
    print("[INFO] Parsing JSON from LLM response...")
    data = clean_and_parse_json(raw_content)

    if "alphas" not in data or not isinstance(data["alphas"], list):
        raise ValueError("JSON 中未找到 'alphas' 字段或格式不正确")

    alphas = data["alphas"]
    print(f"[INFO] Received {len(alphas)} alphas from LLM")

    # 5. 生成 Python 模块代码字符串
    module_code = build_alpha_module_code(alphas)

    # 6. 写入到输出文件
    OUTPUT_ALPHA_PY.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_ALPHA_PY.write_text(module_code, encoding="utf-8")
    print(f"[INFO] Generated alpha module written to: {OUTPUT_ALPHA_PY}")

    # ==== （现在先不执行，留作参考）====
    # 例如：
    # import pandas as pd
    # from generated_alphas import compute_alphas
    #
    # df = pd.read_csv(CSV_PATH)
    # df['date'] = pd.to_datetime(df['date'])
    # # 如果需要按 concept_code 分组再算，可以这样：
    # df = df.groupby('concept_code', group_keys=False).apply(compute_alphas)
    # df.to_csv(CSV_PATH, index=False)


if __name__ == "__main__":
    main()
