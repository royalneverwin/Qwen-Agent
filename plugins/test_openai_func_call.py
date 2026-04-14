from openai import OpenAI
from datetime import datetime
import json
import os
import random

cfg = {
    'api_key': 'empty',
    # qwen3.5
    # 'base_url': "http://{}/v1".format("[fdbd:dc53:53:500::58]:10872"),
    # qwen3 coder next
    'base_url': "http://{}/v1".format("[fdbd:dccd:cdc1:1302:0:50::]:10022"),
    'model_name': 'qwen3'
}

client = OpenAI(
    api_key=cfg['api_key'],
    base_url=cfg['base_url'],
    default_headers={"Destination-Service": "ad.integrity.qwen3_coder_next_80b_a3b"} if cfg['api_key'] == 'empty' else {},
)

# 模拟用户问题
USER_QUESTION = "北京天气咋样呢？"
# 定义工具列表
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "当你想查询指定城市的天气时非常有用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "城市或县区，比如北京市、杭州市、余杭区等。",
                    }
                },
                "required": ["location"],
            },
        },
    },
]


# 模拟天气查询工具
def get_current_weather(arguments):
    weather_conditions = ["晴天", "多云", "雨天"]
    random_weather = random.choice(weather_conditions)
    location = arguments["location"]
    return f"{location}今天是{random_weather}。"


# 封装模型响应函数
def get_response(messages):
    completion = client.chat.completions.create(
            model="pistis_agentic",
            messages=messages,
            tools=tools,
            max_tokens=8192,
            extra_body={"top_k": 20, "repetition_penalty": 1.0},
            temperature=0.6,
            top_p=0.95,
            stream=False,
            timeout=1200
        )

    return completion

messages = [{"role": "user", "content": USER_QUESTION+"\n\nYou must put your answer inside <answer> </answer> tags, i.e., <answer> answer here </answer>. Please reason step by step. Use Python code to process the image if necessary. You can conduct search to seek the Internet. Format strictly as <think> </think> <code> </code> (if code is neededs) or <think> </think> <tool_call> </tool_call> (if function call is neededs) or <think> <think> <answer> </answer>."}]
response = get_response(messages)
assistant_output = response.choices[0].message
if assistant_output.content is None:
    assistant_output.content = ""
assistant_message = {
    "role": "assistant",
    "content": assistant_output.content,
    "tool_calls": assistant_output.tool_calls,  # 保持原始工具输出
    }
messages.append(assistant_message)
print(assistant_output)
# 如果不需要调用工具，直接输出内容
if assistant_output.tool_calls is None:
    print(f"无需调用天气查询工具，直接回复：{assistant_output.content}")
else:
    # 进入工具调用循环
    while len(assistant_output.tool_calls) > 0:
        tool_call = assistant_output.tool_calls[0]
        tool_call_id = tool_call.id
        func_name = tool_call.function.name
        arguments = json.loads(tool_call.function.arguments)
        print(f"正在调用工具 [{func_name}]，参数：{arguments}")
        # 执行工具
        tool_result = get_current_weather(arguments)
        # 构造工具返回信息
        tool_message = {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": tool_result,  # 保持原始工具输出
        }
        print(f"工具返回：{tool_message['content']}")
        messages.append(tool_message)
        print(messages)
        # 再次调用模型，获取总结后的自然语言回复
        response = get_response(messages)
        assistant_output = response.choices[0].message
        print(f"助手回复：{assistant_output}")
        if assistant_output.content is None:
            assistant_output.content = ""
        messages.append(assistant_output)
    print(f"助手最终回复：{assistant_output.content}")