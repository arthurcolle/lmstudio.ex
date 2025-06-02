# Demo script for LMStudio Elixir client
# Run with: mix run demo.exs

# Make sure LM Studio is running on localhost:1234

IO.puts("LMStudio Elixir Client Demo\n")

# Example 1: Simple chat
IO.puts("1. Simple chat example:")
IO.puts("------------------------")

case LMStudio.chat("What is Elixir?", model: "deepseek-r1-0528-qwen3-8b-mlx", max_tokens: 100) do
  {:ok, response} ->
    content = get_in(response, ["choices", Access.at(0), "message", "content"])
    IO.puts("Response: #{content}")
  {:error, reason} ->
    IO.puts("Error: #{inspect(reason)}")
end

IO.puts("\n")

# Example 2: Streaming chat
IO.puts("2. Streaming example:")
IO.puts("------------------------")
IO.write("Response: ")

case LMStudio.chat(
  "Write a haiku about functional programming",
  model: "deepseek-r1-0528-qwen3-8b-mlx",
  stream: true,
  stream_callback: fn
    {:chunk, content} -> IO.write(content)
    {:done, _} -> IO.puts("\n\nStreaming complete!")
  end
) do
  {:ok, :stream_complete} ->
    :ok
  {:error, reason} ->
    IO.puts("\nError: #{inspect(reason)}")
end

IO.puts("\n")

# Example 3: List models
IO.puts("3. Available models:")
IO.puts("------------------------")

case LMStudio.list_models() do
  {:ok, models} when is_list(models) ->
    Enum.each(models, fn model ->
      IO.puts("  - #{model["id"]}")
    end)
  {:ok, _} ->
    IO.puts("  No models found or unexpected response format")
  {:error, reason} ->
    IO.puts("  Error listing models: #{inspect(reason)}")
end

IO.puts("\n")

# Example 4: Multi-turn conversation
IO.puts("4. Multi-turn conversation:")
IO.puts("------------------------")

messages = [
  %{role: "system", content: "You are a helpful Elixir programming assistant."},
  %{role: "user", content: "What is pattern matching?"},
  %{role: "assistant", content: "Pattern matching in Elixir is a powerful feature that allows you to match values against patterns and bind variables in the process."},
  %{role: "user", content: "Can you give me a simple example?"}
]

case LMStudio.complete(messages, model: "deepseek-r1-0528-qwen3-8b-mlx", max_tokens: 150) do
  {:ok, response} ->
    content = get_in(response, ["choices", Access.at(0), "message", "content"])
    IO.puts("Response: #{content}")
  {:error, reason} ->
    IO.puts("Error: #{inspect(reason)}")
end