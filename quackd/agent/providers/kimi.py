"""Kimi as the duck's brain, over Moonshot's OpenAI-compatible endpoint.

Moonshot's tool-calling guide shows the tool round trip without documenting which `tool_choice`
values it accepts, so quackd sends `auto`, the one value that cannot be refused, and not
`parallel_tool_calls`. Moonshot says the model may return several calls at once, which the loop
already handles by taking the first.
"""

from __future__ import annotations

from quackd.agent.providers.openai import OpenAIProvider


class KimiProvider(OpenAIProvider):
    name = "kimi"
    key_env = "MOONSHOT_API_KEY"
    extra = "kimi"
    base_url = "https://api.moonshot.ai/v1"
    default_tool_choice = "auto"
    send_parallel_flag = False
