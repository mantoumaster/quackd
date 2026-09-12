"""DeepSeek as the duck's brain, over its OpenAI-compatible endpoint.

`tool_choice="required"` is documented, so quackd asks for its one call outright.
`parallel_tool_calls` is not in DeepSeek's request reference, and an unknown field is a 400 on
some gateways, so it is not sent: the loop already takes the first call when a model sends more.
"""

from __future__ import annotations

from quackd.agent.providers.openai import OpenAIProvider


class DeepSeekProvider(OpenAIProvider):
    name = "deepseek"
    key_env = "DEEPSEEK_API_KEY"
    extra = "deepseek"
    base_url = "https://api.deepseek.com"
    send_parallel_flag = False
