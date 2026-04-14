#!/usr/bin/env python3
"""Verify Qwen3.5 tool calling via an OpenAI-compatible API endpoint."""

import argparse
import json
import sys
from typing import Any, Dict, Iterable, List, Optional

from openai import OpenAI


DEFAULT_BASE_URL = "http://[fdbd:dc53:53:500::58]:10872/v1"
DEFAULT_API_KEY = "empty"
DEFAULT_MODEL = "pistis_agentic"
DEFAULT_TICKET = "tool_call_probe"
DEFAULT_ANSWER = "tool-call-pass-1729"

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_deployment_probe",
            "description": (
                "Return the deployment probe token for tool-calling validation. "
                "Use this tool when the user explicitly asks for the probe token."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "ticket": {
                        "type": "string",
                        "description": "The exact probe ticket from the user.",
                    }
                },
                "required": ["ticket"],
            },
        },
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify Qwen3.5 tool calling through an OpenAI-compatible API.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="OpenAI-compatible API endpoint.")
    parser.add_argument("--api-key", default=DEFAULT_API_KEY, help="API key for the endpoint.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model name passed to chat.completions.create.")
    parser.add_argument(
        "--destination-service",
        default="",
        help="Optional Destination-Service header for internal gateways when api_key=empty.",
    )
    parser.add_argument("--ticket", default=DEFAULT_TICKET, help="Expected probe ticket passed to the tool.")
    parser.add_argument(
        "--expected-answer",
        default=DEFAULT_ANSWER,
        help="Expected answer returned by the tool and echoed by the model.",
    )
    parser.add_argument("--max-tokens", type=int, default=8192, help="max_tokens used for the chat request.")
    parser.add_argument("--temperature", type=float, default=0.6, help="Sampling temperature.")
    parser.add_argument("--top-p", type=float, default=0.95, help="Sampling top_p.")
    parser.add_argument("--timeout", type=int, default=1200, help="Request timeout in seconds.")
    parser.add_argument("--disable-thinking", action="store_true", help="Disable enable_thinking in chat_template_kwargs.")
    return parser.parse_args()


def build_client(args: argparse.Namespace) -> OpenAI:
    default_headers = {}
    if args.api_key == "empty" and args.destination_service:
        default_headers["Destination-Service"] = args.destination_service
    return OpenAI(
        api_key=args.api_key,
        base_url=args.base_url,
        default_headers=default_headers,
    )


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
        timeout=args.timeout,
    )


def dump_assistant_message(message: Any) -> Dict[str, Any]:
    payload = message.model_dump(exclude_none=True)
    if payload.get("content") is None:
        payload["content"] = ""
    return payload


def run_probe_tool(arguments: Dict[str, Any], args: argparse.Namespace) -> str:
    ticket = arguments["ticket"]
    if ticket != args.ticket:
        return json.dumps(
            {
                "ticket": ticket,
                "error": f"unexpected ticket, expected {args.ticket}",
            },
            ensure_ascii=False,
        )
    return json.dumps(
        {
            "ticket": ticket,
            "answer": args.expected_answer,
        },
        ensure_ascii=False,
    )


def normalize_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks = []
        for item in content:
            if isinstance(item, dict) and item.get("text"):
                chunks.append(item["text"])
            else:
                chunks.append(str(item))
        return "\n".join(chunks)
    return str(content)


def render_message(message: Dict[str, Any]) -> Optional[str]:
    tool_calls = message.get("tool_calls")
    if tool_calls:
        rendered = []
        for tool_call in tool_calls:
            function = tool_call.get("function", {})
            rendered.append(
                "[assistant.tool_call] "
                f"{function.get('name')}({function.get('arguments', '')})"
            )
        return "\n".join(rendered)

    role = message.get("role", "unknown")
    if role == "tool":
        return f"[tool] {normalize_content(message.get('content'))}"

    text = normalize_content(message.get("content")).strip()
    if text:
        return f"[{role}] {text}"
    return None


def find_probe_tool_call(messages: Iterable[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    for message in messages:
        for tool_call in message.get("tool_calls") or []:
            function = tool_call.get("function", {})
            if function.get("name") == "get_deployment_probe":
                return function
    return None


def validate_response(messages: List[Dict[str, Any]], args: argparse.Namespace) -> List[str]:
    errors = []

    tool_call = find_probe_tool_call(messages)
    if tool_call is None:
        errors.append("assistant did not issue get_deployment_probe")
    else:
        try:
            arguments = json.loads(tool_call.get("arguments", "{}"))
        except json.JSONDecodeError as exc:
            errors.append(f"tool call arguments are not valid JSON: {exc}")
        else:
            if arguments.get("ticket") != args.ticket:
                errors.append(
                    f"tool call ticket mismatch: got {arguments.get('ticket')!r}, expected {args.ticket!r}"
                )

    tool_results = [message for message in messages if message.get("role") == "tool"]
    if not tool_results:
        errors.append("tool result message is missing")
    else:
        tool_text = normalize_content(tool_results[-1].get("content"))
        if args.expected_answer not in tool_text:
            errors.append("tool result does not contain the expected answer")

    assistant_texts = [
        normalize_content(message.get("content")).strip()
        for message in messages
        if message.get("role") == "assistant" and normalize_content(message.get("content")).strip()
    ]
    if not assistant_texts:
        errors.append("final assistant answer is missing")
    elif args.expected_answer not in assistant_texts[-1]:
        errors.append("final assistant answer does not contain the expected answer")

    return errors


def main() -> int:
    args = parse_args()
    client = build_client(args)

    messages: List[Dict[str, Any]] = [
        {
            "role": "user",
            "content": (
                "Please call get_deployment_probe with ticket "
                f'"{args.ticket}" and then reply with the answer field only.'
            ),
        }
    ]
    transcript: List[Dict[str, Any]] = []

    try:
        completion = get_response(client, args, messages)
        assistant_output = completion.choices[0].message
    except Exception as exc:
        print(f"Probe failed before validation: {exc}", file=sys.stderr)
        return 1

    assistant_message = dump_assistant_message(assistant_output)
    messages.append(assistant_message)
    transcript.append(assistant_message)

    while assistant_output.tool_calls:
        for tool_call in assistant_output.tool_calls:
            arguments = json.loads(tool_call.function.arguments)
            tool_result = run_probe_tool(arguments, args)
            tool_message = {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": tool_result,
            }
            messages.append(tool_message)
            transcript.append(tool_message)

        try:
            completion = get_response(client, args, messages)
            assistant_output = completion.choices[0].message
        except Exception as exc:
            print(f"Probe failed after tool execution: {exc}", file=sys.stderr)
            return 1

        assistant_message = dump_assistant_message(assistant_output)
        messages.append(assistant_message)
        transcript.append(assistant_message)

    print("Conversation transcript:")
    for message in transcript:
        rendered = render_message(message)
        if rendered:
            print(rendered)

    errors = validate_response(transcript, args)
    if errors:
        print("\nProbe result: FAILED", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    print("\nProbe result: PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
