#!/usr/bin/env elixir

IO.puts "Testing LM Studio connection..."

# Test listing models
case LMStudio.list_models() do
  {:ok, models} ->
    IO.puts "✓ Connected to LM Studio!"
    IO.puts "Available models:"
    Enum.each(models, fn model ->
      IO.puts "  - #{model["id"]}"
    end)
    
  {:error, reason} ->
    IO.puts "✗ Failed to connect to LM Studio"
    IO.puts "Error: #{inspect(reason)}"
end

# Test a simple completion
IO.puts "\nTesting chat completion..."
case LMStudio.chat("Say 'Hello World'", temperature: 0.1, max_tokens: 10) do
  {:ok, response} ->
    content = get_in(response, ["choices", Access.at(0), "message", "content"])
    IO.puts "✓ Chat response: #{content}"
    
  {:error, reason} ->
    IO.puts "✗ Chat failed"
    IO.puts "Error: #{inspect(reason)}"
end