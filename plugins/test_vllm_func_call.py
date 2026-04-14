#!/usr/bin/env python3
"""Function-calling demo against a locally deployed vLLM Qwen3.5 service."""

import argparse
import json
from typing import Any, Dict, List

from openai import OpenAI


DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1"
DEFAULT_API_KEY = "EMPTY"
DEFAULT_MODEL = "Qwen3.5-9B"
DEFAULT_QUESTION = "上海天气咋样呢？"


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "当你需要查询指定城市天气时非常有用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "城市或区县名称，比如上海市、杭州市、余杭区等。",
                    }
                },
                "required": ["location"],
            },
        },
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Call a local vLLM Qwen3.5 deployment with OpenAI-style tool calling.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="Local OpenAI-compatible vLLM endpoint.")
    parser.add_argument("--api-key", default=DEFAULT_API_KEY, help="API key for the local endpoint.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Served model name exposed by vLLM.")
    parser.add_argument("--question", default=DEFAULT_QUESTION, help="User question to send to the model.")
    parser.add_argument("--max-tokens", type=int, default=8192, help="max_tokens used for the chat request.")
    parser.add_argument("--temperature", type=float, default=0.6, help="Sampling temperature.")
    parser.add_argument("--top-p", type=float, default=0.95, help="Sampling top_p.")
    parser.add_argument("--disable-thinking", action="store_true", help="Disable enable_thinking in chat_template_kwargs.")
    return parser.parse_args()


def build_client(args: argparse.Namespace) -> OpenAI:
    return OpenAI(
        api_key=args.api_key,
        base_url=args.base_url,
    )


def get_current_weather(arguments: Dict[str, Any]) -> str:
    location = arguments["location"]
    weather_map = {
        "上海": "上海今天多云，气温 19 到 26 度。",
        "上海市": "上海市今天多云，气温 19 到 26 度。",
        "北京": "北京今天晴，气温 14 到 25 度。",
        "北京市": "北京市今天晴，气温 14 到 25 度。",
        "杭州": "杭州今天小雨，气温 18 到 24 度。",
        "杭州市": "杭州市今天小雨，气温 18 到 24 度。",
    }
    return weather_map.get(location, f"{location}今天晴转多云，气温 20 到 28 度。")


def dump_assistant_message(message: Any) -> Dict[str, Any]:
    payload = message.model_dump(exclude_none=True)
    if payload.get("content") is None:
        payload["content"] = ""
    return payload


def get_response(client: OpenAI, args: argparse.Namespace, messages: List[Dict[str, Any]]):
    return client.chat.completions.create(
        model=args.model,
        messages=messages,
        tools=TOOLS,
        max_tokens=args.max_tokens,
        extra_body={
            "top_k": 20,
            "repetition_penalty": 1.0,
            "chat_template_kwargs": {
                "enable_thinking": not args.disable_thinking,
            },
        },
        temperature=args.temperature,
        top_p=args.top_p,
        tool_choice="auto",
        stream=False,
        timeout=1200,
    )


def render_tool_calls(tool_calls: List[Any]) -> None:
    for tool_call in tool_calls:
        print(
            "模型发起工具调用："
            f" id={tool_call.id},"
            f" name={tool_call.function.name},"
            f" arguments={tool_call.function.arguments}"
        )


def main() -> int:
    args = parse_args()
    client = build_client(args)

    messages: List[Dict[str, Any]] = [{"role": "user", "content": args.question}]
    completion = get_response(client, args, messages)
    assistant_output = completion.choices[0].message
    messages.append(dump_assistant_message(assistant_output))

    print(f"用户问题：{args.question}")
    print("模型首轮回复：")
    print(assistant_output)

    while assistant_output.tool_calls:
        render_tool_calls(assistant_output.tool_calls)

        for tool_call in assistant_output.tool_calls:
            arguments = json.loads(tool_call.function.arguments)
            tool_result = get_current_weather(arguments)
            print(f"工具返回：{tool_result}")
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": tool_result,
                }
            )

        completion = get_response(client, args, messages)
        assistant_output = completion.choices[0].message
        messages.append(dump_assistant_message(assistant_output))
        print("模型拿到工具结果后的回复：")
        print(assistant_output)

    final_answer = assistant_output.content or ""
    print(f"最终回答：{final_answer}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
