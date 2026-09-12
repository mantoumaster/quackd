/**
 * Bring your own key. The page talks to the model; nothing talks to us.
 *
 * quackd hosts no server for this demo. Your key is read from an input, kept in a variable
 * for the life of the tab, and sent straight to the vendor you picked from your own
 * browser. It is never stored, never logged and never proxied, which also means every
 * request is billed to you and visible to you in your vendor's dashboard. If you would
 * rather it never leave the machine at all, run a local model: Ollama and any
 * OpenAI-compatible server (llama.cpp, vLLM, LM Studio) are here too, and need no key.
 *
 * Each provider does the same job: hand the model a system prompt, the turns so far and a
 * list of tools, and return exactly one tool call. Where a vendor can be told to call a
 * tool it is told to; where it cannot, a missing call ends the run rather than being
 * guessed at.
 *
 * OpenAI is the one vendor here with two APIs that can do that job. Chat Completions is
 * asked first, and a model that refuses function tools there is moved to Responses for the
 * rest of the run, on the strength of what the 400 said. `quackd/agent/providers/openai.py`
 * makes the same move for the CLI, and the two have to keep agreeing.
 */

export const PROVIDERS = {
  anthropic: {
    label: "Anthropic (Claude)",
    keyPlaceholder: "sk-ant-...",
    defaultModel: "claude-sonnet-5",
    models: ["claude-sonnet-5", "claude-opus-5", "claude-haiku-4-5-20251001"],
    keyUrl: "https://console.anthropic.com/settings/keys",
    needsKey: true,
  },
  openai: {
    label: "OpenAI",
    keyPlaceholder: "sk-...",
    defaultModel: "gpt-5",
    models: ["gpt-5", "gpt-5-mini", "gpt-6-astra"],
    keyUrl: "https://platform.openai.com/api-keys",
    needsKey: true,
  },
  gemini: {
    label: "Google (Gemini)",
    keyPlaceholder: "AIza...",
    defaultModel: "gemini-2.5-pro",
    models: ["gemini-2.5-pro", "gemini-2.5-flash"],
    keyUrl: "https://aistudio.google.com/apikey",
    needsKey: true,
  },
  local: {
    label: "Local (Ollama or any OpenAI-compatible server)",
    keyPlaceholder: "not needed",
    defaultModel: "qwen3:8b",
    models: [],
    baseUrl: "http://localhost:11434/v1",
    needsKey: false,
    note:
      "Ollama must be told to accept this page: run it with OLLAMA_ORIGINS=* (or your own " +
      "origin). Any OpenAI-compatible server works the same way; change the base URL.",
  },
};

class ProviderError extends Error {}

function oneCall(name, args) {
  return { name, arguments: args ?? {} };
}

async function readError(response) {
  let detail = "";
  try {
    const body = await response.json();
    detail = body?.error?.message ?? JSON.stringify(body).slice(0, 300);
  } catch {
    detail = (await response.text().catch(() => "")).slice(0, 300);
  }
  return detail;
}

/** Arguments arrive as a JSON string. A model that writes a broken one gets an empty object,
 *  which `checkParams` then refuses by name rather than the tab dying on a SyntaxError. */
function parseArguments(raw) {
  try {
    return JSON.parse(raw || "{}");
  } catch {
    return {};
  }
}

/**
 * Is this the 400 that refuses function tools on Chat Completions and names Responses?
 *
 * Matched on what the API says rather than on a model name, because the list of models that
 * behave this way is not ours to keep and gets longer. Observed on `gpt-6-astra`:
 *
 *     Function tools with reasoning_effort are not supported for gpt-6-astra in
 *     /v1/chat/completions. To use function tools, use /v1/responses or set
 *     reasoning_effort to 'none'.
 *
 * Both halves are required so an unrelated 400 that happens to name one of the two does not
 * move a run onto a different API. The other remedy that message offers is a dead end on the
 * model that produced it: `reasoning_effort: "none"` comes back as unsupported for that model,
 * and the tools are refused at every effort it does support. So the fix is the API.
 *
 * This is `_wants_the_responses_api` in `quackd/agent/providers/openai.py`, in JavaScript.
 */
export function wantsTheResponsesApi(status, detail) {
  if (status !== 400) return false;
  const text = String(detail).toLowerCase();
  return text.includes("function tools") && text.includes("responses");
}

/** Anthropic: one tool call is asked for with tool_choice, and thinking is left off. */
function anthropic({ key, model }) {
  return {
    name: "anthropic",
    model,
    async step({ system, history, observation, tools, signal = null }) {
      const messages = [];
      for (const turn of history) {
        messages.push({ role: "user", content: turn.observation });
        messages.push({
          role: "assistant",
          content: [{ type: "tool_use", id: `t${messages.length}`, name: turn.call.name, input: turn.call.arguments ?? {} }],
        });
        messages.push({
          role: "user",
          content: [{ type: "tool_result", tool_use_id: `t${messages.length - 1}`, content: "ok" }],
        });
      }
      messages.push({ role: "user", content: observation });
      const response = await fetch("https://api.anthropic.com/v1/messages", {
        method: "POST",
        signal,   // an aborted run must not keep a request alive, or keep billing for it
        headers: {
          "content-type": "application/json",
          "x-api-key": key,
          "anthropic-version": "2023-06-01",
          // Anthropic blocks browser calls unless the caller says it meant it. It did.
          "anthropic-dangerous-direct-browser-access": "true",
        },
        body: JSON.stringify({
          model,
          max_tokens: 1024,
          system,
          messages,
          tools: tools.map((t) => ({ name: t.name, description: t.description, input_schema: t.input_schema })),
          tool_choice: { type: "any", disable_parallel_tool_use: true },
        }),
      });
      if (!response.ok) throw new ProviderError(`Anthropic said ${response.status}: ${await readError(response)}`);
      const body = await response.json();
      const call = (body.content ?? []).find((block) => block.type === "tool_use");
      if (!call) throw new ProviderError("Claude answered without calling a tool");
      return oneCall(call.name, call.input);
    },
  };
}

/** OpenAI and every OpenAI-compatible server, including Ollama's.
 *
 * Two APIs behind one interface. Chat Completions is the default, and the only one a local
 * server speaks. A model that refuses function tools there names `/v1/responses` in the 400,
 * and every verb here is a function tool, so that refusal is not a degraded path, it is no
 * path: the run moves to Responses and stays. Staying is the point. Retrying chat each turn
 * would pay a failed call per step against the visitor's own key.
 */
function openaiCompatible({ key, model, baseUrl, label }) {
  const root = baseUrl.replace(/\/$/, "");
  // "chat" or "responses". Held across steps, so a model that has refused chat once is never
  // asked again for the life of this provider, which is the life of the run.
  let api = "chat";

  // A replayed call and its result have to quote the same handle. It is invented here rather
  // than echoed from the vendor, and both renderers key it off the turn index, so the pair
  // cannot drift apart.
  const callId = (index) => `t${index}`;

  function chatBody({ system, history, observation, tools }) {
    const messages = [{ role: "system", content: system }];
    history.forEach((turn, index) => {
      messages.push({ role: "user", content: turn.observation });
      messages.push({
        role: "assistant",
        content: null,
        tool_calls: [{
          id: callId(index),
          type: "function",
          function: { name: turn.call.name, arguments: JSON.stringify(turn.call.arguments ?? {}) },
        }],
      });
      messages.push({ role: "tool", tool_call_id: callId(index), content: "ok" });
    });
    messages.push({ role: "user", content: observation });
    return {
      model,
      messages,
      tool_choice: "required",
      tools: tools.map((t) => ({
        type: "function",
        function: { name: t.name, description: t.description, parameters: t.input_schema },
      })),
    };
  }

  /** The same turn in the shapes Responses uses. It agrees with Chat Completions on almost no
   *  field name: tools go flat, the system prompt becomes `instructions`, and a turn is three
   *  items keyed by `call_id` rather than one message carrying a `tool_calls` array. */
  function responsesBody({ system, history, observation, tools }) {
    const input = [];
    history.forEach((turn, index) => {
      input.push({ role: "user", content: [{ type: "input_text", text: turn.observation }] });
      input.push({
        type: "function_call",
        call_id: callId(index),
        name: turn.call.name,
        arguments: JSON.stringify(turn.call.arguments ?? {}),
      });
      input.push({ type: "function_call_output", call_id: callId(index), output: "ok" });
    });
    input.push({ role: "user", content: [{ type: "input_text", text: observation }] });
    return {
      model,
      instructions: system,
      input,
      tool_choice: "required",
      tools: tools.map((t) => ({
        type: "function",
        name: t.name,
        description: t.description,
        parameters: t.input_schema,
      })),
    };
  }

  function fromChat(body) {
    const call = body.choices?.[0]?.message?.tool_calls?.[0];
    if (!call) throw new ProviderError(`${label} answered without calling a tool`);
    return oneCall(call.function.name, parseArguments(call.function.arguments));
  }

  function fromResponses(body) {
    // `output` is a flat list of items rather than one message: reasoning, then any calls.
    const call = (body.output ?? []).find((item) => item.type === "function_call");
    if (!call) throw new ProviderError(`${label} answered without calling a tool`);
    return oneCall(call.name, parseArguments(call.arguments));
  }

  return {
    name: label,
    model,
    async step({ system, history, observation, tools, signal = null }) {
      const turn = { system, history, observation, tools };
      const headers = { "content-type": "application/json" };
      if (key) headers.authorization = `Bearer ${key}`;
      // At most two passes: the only `continue` moves chat to responses, and the guard that
      // reaches it cannot fire from responses, so a genuine Responses failure is thrown
      // rather than retried forever.
      for (;;) {
        const onResponses = api === "responses";
        const response = await fetch(`${root}/${onResponses ? "responses" : "chat/completions"}`, {
          method: "POST",
          signal,   // an aborted run must not keep a request alive, or keep billing for it
          headers,
          body: JSON.stringify(onResponses ? responsesBody(turn) : chatBody(turn)),
        });
        if (response.ok) {
          const body = await response.json();
          return onResponses ? fromResponses(body) : fromChat(body);
        }
        // Read once. A Response body cannot be consumed twice, and both uses below need it.
        const detail = await readError(response);
        if (!onResponses && wantsTheResponsesApi(response.status, detail)) {
          api = "responses";
          continue;
        }
        throw new ProviderError(`${label} said ${response.status}: ${detail}`);
      }
    },
  };
}

/** Gemini, whose tool schema is JSON Schema without the parts it does not accept. */
function gemini({ key, model }) {
  const clean = (schema) => {
    const copy = { ...schema };
    delete copy.additionalProperties;
    if (copy.properties) {
      copy.properties = Object.fromEntries(
        Object.entries(copy.properties).map(([k, v]) => {
          const p = { ...v };
          delete p.minimum; delete p.maximum; delete p.maxLength;
          return [k, p];
        }));
    }
    return copy;
  };
  return {
    name: "gemini",
    model,
    async step({ system, history, observation, tools, signal = null }) {
      const contents = [];
      for (const turn of history) {
        contents.push({ role: "user", parts: [{ text: turn.observation }] });
        contents.push({ role: "model", parts: [{ functionCall: { name: turn.call.name, args: turn.call.arguments ?? {} } }] });
        contents.push({ role: "user", parts: [{ functionResponse: { name: turn.call.name, response: { result: "ok" } } }] });
      }
      contents.push({ role: "user", parts: [{ text: observation }] });
      const url = `https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent`;
      const response = await fetch(url, {
        method: "POST",
        signal,   // an aborted run must not keep a request alive, or keep billing for it
        headers: { "content-type": "application/json", "x-goog-api-key": key },
        body: JSON.stringify({
          systemInstruction: { parts: [{ text: system }] },
          contents,
          tools: [{ functionDeclarations: tools.map((t) => ({ name: t.name, description: t.description, parameters: clean(t.input_schema) })) }],
          toolConfig: { functionCallingConfig: { mode: "ANY" } },
        }),
      });
      if (!response.ok) throw new ProviderError(`Gemini said ${response.status}: ${await readError(response)}`);
      const body = await response.json();
      const parts = body.candidates?.[0]?.content?.parts ?? [];
      const call = parts.find((p) => p.functionCall)?.functionCall;
      if (!call) throw new ProviderError("Gemini answered without calling a tool");
      return oneCall(call.name, call.args);
    },
  };
}

export function makeProvider({ provider, key, model, baseUrl }) {
  const spec = PROVIDERS[provider];
  if (!spec) throw new ProviderError(`unknown provider ${provider}`);
  if (spec.needsKey && !key) throw new ProviderError(`${spec.label} needs your API key`);
  const chosen = model || spec.defaultModel;
  if (provider === "anthropic") return anthropic({ key, model: chosen });
  if (provider === "gemini") return gemini({ key, model: chosen });
  if (provider === "openai") {
    return openaiCompatible({ key, model: chosen, baseUrl: "https://api.openai.com/v1", label: "OpenAI" });
  }
  return openaiCompatible({
    key: key || "",
    model: chosen,
    baseUrl: localBaseUrl(baseUrl || spec.baseUrl),
    label: "your local server",
  });
}

/**
 * A base URL a key may safely be sent to.
 *
 * This box is free text, and whatever is in the key field goes to it as a bearer token. https
 * is fine anywhere; plain http only to this machine, where nothing leaves it. Anything else is
 * refused by name, so a typo or a paste cannot quietly forward a key to a stranger. The same
 * rule is what a `connect-src` policy could express if the host were knowable in advance.
 */
export function localBaseUrl(raw) {
  let url;
  try {
    url = new URL(raw);
  } catch {
    throw new ProviderError(`"${raw}" is not a URL. Try http://localhost:11434/v1`);
  }
  const loopback = ["localhost", "127.0.0.1", "[::1]", "::1"].includes(url.hostname);
  if (url.protocol === "https:" || (url.protocol === "http:" && loopback)) {
    return raw;
  }
  throw new ProviderError(
    `refusing to send a key to ${url.host} over ${url.protocol.replace(":", "")}. ` +
      "Use https, or http only for a server on this machine."
  );
}
