# LMStudio

Elixir client library for LM Studio's chat completions API with full streaming support.

## Installation

Add `lmstudio` to your list of dependencies in `mix.exs`:

```elixir
def deps do
  [
    {:lmstudio, "~> 0.1.0"}
  ]
end
```

## Configuration

You can configure the client in your `config.exs`:

```elixir
config :lmstudio,
  base_url: "http://localhost:1234",  # Default LM Studio URL
  default_model: "your-model-name",
  default_temperature: 0.7,
  default_max_tokens: 2048
```

## Usage

### Simple Chat

```elixir
{:ok, response} = LMStudio.chat("What is the capital of France?")
content = get_in(response, ["choices", Access.at(0), "message", "content"])
IO.puts(content)
```

### Streaming Response

```elixir
LMStudio.chat(
  "Write a story about a robot",
  stream: true,
  stream_callback: fn
    {:chunk, content} -> IO.write(content)
    {:done, _} -> IO.puts("\n\nDone!")
  end
)
```

### Multi-turn Conversation

```elixir
messages = [
  %{role: "system", content: "You are a helpful assistant."},
  %{role: "user", content: "What's the weather like?"},
  %{role: "assistant", content: "I don't have access to real-time weather data."},
  %{role: "user", content: "What would be a good day for a picnic then?"}
]

{:ok, response} = LMStudio.complete(messages)
```

### With System Prompt

```elixir
{:ok, response} = LMStudio.chat(
  "Explain quantum computing",
  system_prompt: "You are a physics professor. Explain concepts simply."
)
```

### List Available Models

```elixir
{:ok, models} = LMStudio.list_models()
Enum.each(models, fn model ->
  IO.puts("Model: #{model["id"]}")
end)
```

## API Reference

### `LMStudio.complete/2`

Sends a chat completion request to LM Studio.

**Parameters:**
- `messages` - List of message maps with `:role` and `:content` keys
- `opts` - Keyword list of options:
  - `:model` - Model to use (default: from config)
  - `:temperature` - Temperature for sampling (default: from config)
  - `:max_tokens` - Maximum tokens to generate (default: from config)
  - `:stream` - Whether to stream the response (default: false)
  - `:system_prompt` - Convenience option to prepend a system message
  - `:stream_callback` - Function to handle streaming chunks (required if stream: true)

### `LMStudio.chat/2`

Convenience function for single-turn conversations.

**Parameters:**
- `user_content` - The user's message as a string
- `opts` - Same options as `complete/2`

### `LMStudio.list_models/0`

Lists all available models in LM Studio.

## Examples

See the `LMStudio.Examples` module for more detailed examples.

## Testing

```bash
mix test
```

## License

MIT