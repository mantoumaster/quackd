"""Qwen as the duck's brain, over Alibaba Cloud Model Studio's OpenAI-compatible endpoint.

The base URL here is Model Studio's international one. Alibaba also issues per-workspace regional
hosts, and a reader on one of those passes `--base-url`, which is why that flag reaches this
provider at all.

`tool_choice` is left at `auto`: Model Studio documents `tools` without documenting which
`tool_choice` values it takes, and `auto` is the value that cannot 400. `parallel_tool_calls` is
not sent for the same reason.
"""

from __future__ import annotations

from quackd.agent.providers.openai import OpenAIProvider


class QwenProvider(OpenAIProvider):
    name = "qwen"
    key_env = "DASHSCOPE_API_KEY"
    extra = "qwen"
    base_url = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
    default_tool_choice = "auto"
    send_parallel_flag = False
