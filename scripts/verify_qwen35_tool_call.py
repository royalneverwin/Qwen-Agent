#!/usr/bin/env python3
"""Tool-calling probe for a Qwen3.5 vLLM deployment.

This follows the same Assistant + OpenAI-compatible raw API pattern used in
examples/assistant_qwen3.5.py, but swaps the MCP tools for a deterministic
local probe tool so the deployment can be validated end-to-end.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from qwen_agent.agents import Assistant
from qwen_agent.tools.base import BaseTool


DEFAULT_MODEL = "Qwen3.5-9B"
DEFAULT_MODEL_SERVER = "http://127.0.0.1:8000/v1"
DEFAULT_API_KEY = "EMPTY"
DEFAULT_TICKET = "tool_call_probe"
DEFAULT_ANSWER = "tool-call-pass-1729"


class DeploymentProbeTool(BaseTool):
    name = "get_deployment_probe"
    description = (
        "Return the deployment probe token for tool-calling validation. "
        "Use this tool when the user explicitly asks for the probe token."
    )
    parameters = {
        "type": "object",
        "properties": {
            "ticket": {
                "type": "string",
                "description": "The exact probe ticket from the user.",
            },
        },
        "required": ["ticket"],
    }

    def __init__(self, expected_ticket: str, answer: str):
        super().__init__()
        self.expected_ticket = expected_ticket
        self.answer = answer

    def call(self, params: Any, **kwargs) -> str:
        payload = self._verify_json_format_args(params)
        ticket = payload["ticket"]
        if ticket != self.expected_ticket:
            return json.dumps(
                {
                    "ticket": ticket,
                    "error": f"unexpected ticket, expected {self.expected_ticket}",
                },
                ensure_ascii=False,
            )
        return json.dumps(
            {
                "ticket": ticket,
                "answer": self.answer,
            },
            ensure_ascii=False,
        )


def build_agent(args: argparse.Namespace) -> Assistant:
    llm_cfg = {
        "model": args.model,
        "model_type": "qwenvl_oai",
        "model_server": args.model_server,
        "api_key": args.api_key,
        "generate_cfg": {
            "use_raw_api": True,
            "extra_body": {
                "chat_template_kwargs": {
                    "enable_thinking": not args.disable_thinking,
                }
            },
        },
    }
    system_message = (
        "You are running a deployment probe. "
        "When the user asks for the probe token, you must call the "
        "`get_deployment_probe` tool with the exact ticket from the user. "
        "After the tool returns, reply with the answer field only."
    )
    return Assistant(
        llm=llm_cfg,
        function_list=[DeploymentProbeTool(args.ticket, args.expected_answer)],
        system_message=system_message,
        name="Qwen3.5 Tool-calling Probe",
        description="Validate that a Qwen3.5 deployment performs native tool calling correctly.",
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
    function_call = message.get("function_call")
    if function_call:
        return (
            "[assistant.tool_call] "
            f"{function_call.get('name')}({function_call.get('arguments', '')})"
        )

    role = message.get("role", "unknown")
    if role == "function":
        name = message.get("name", "")
        return f"[function:{name}] {normalize_content(message.get('content'))}"

    if role == "assistant" and message.get("reasoning_content"):
        reasoning = normalize_content(message.get("reasoning_content")).strip()
        if reasoning:
            return f"[assistant.reasoning] {reasoning}"

    text = normalize_content(message.get("content")).strip()
    if text:
        return f"[{role}] {text}"
    return None


def collect_final_response(bot: Assistant, messages: List[Dict[str, str]], seed: int) -> List[Dict[str, Any]]:
    final_response: Optional[List[Dict[str, Any]]] = None
    for response in bot.run(messages=messages, seed=seed):
        final_response = response
    if final_response is None:
        raise RuntimeError("The model returned no response.")
    return final_response


def find_probe_tool_call(messages: Iterable[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    for message in messages:
        function_call = message.get("function_call")
        if function_call and function_call.get("name") == DeploymentProbeTool.name:
            return function_call
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

    function_results = [
        message
        for message in messages
        if message.get("role") == "function" and message.get("name") == DeploymentProbeTool.name
    ]
    if not function_results:
        errors.append("tool result message is missing")
    else:
        function_text = normalize_content(function_results[-1].get("content"))
        if args.expected_answer not in function_text:
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify native tool calling for a Qwen3.5 vLLM deployment.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Served model name exposed by vLLM.")
    parser.add_argument(
        "--model-server",
        default=DEFAULT_MODEL_SERVER,
        help="OpenAI-compatible vLLM endpoint.",
    )
    parser.add_argument("--api-key", default=DEFAULT_API_KEY, help="API key for the OpenAI-compatible endpoint.")
    parser.add_argument("--ticket", default=DEFAULT_TICKET, help="Expected probe ticket passed to the tool.")
    parser.add_argument(
        "--expected-answer",
        default=DEFAULT_ANSWER,
        help="Expected answer returned by the tool and echoed by the model.",
    )
    parser.add_argument("--seed", type=int, default=1234, help="Random seed forwarded to Qwen-Agent.")
    parser.add_argument(
        "--disable-thinking",
        action="store_true",
        help="Disable the chat_template enable_thinking flag when talking to vLLM.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    bot = build_agent(args)
    messages = [
        {
            "role": "user",
            "content": (
                "Please call get_deployment_probe with ticket "
                f'"{args.ticket}" and then reply with the answer field only.'
            ),
        }
    ]

    try:
        final_response = collect_final_response(bot, messages, seed=args.seed)
    except Exception as exc:
        print(f"Probe failed before validation: {exc}", file=sys.stderr)
        return 1

    print("Conversation transcript:")
    for message in final_response:
        rendered = render_message(message)
        if rendered:
            print(rendered)

    errors = validate_response(final_response, args)
    if errors:
        print("\nProbe result: FAILED", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    print("\nProbe result: PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
