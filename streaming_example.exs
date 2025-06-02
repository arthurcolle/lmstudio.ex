# Example of streaming tokens to a process that uses IO.puts

defmodule TokenPrinter do
  def start do
    spawn(fn -> loop() end)
  end
  
  defp loop do
    receive do
      {:lmstudio_token, token} ->
        IO.write(token)  # Using IO.write to avoid newlines between tokens
        loop()
        
      {:lmstudio_done, :ok} ->
        IO.puts("\n\nStreaming complete!")
        
      {:lmstudio_error, reason} ->
        IO.puts("\nError: #{inspect(reason)}")
    end
  end
end

# Example 1: Simple token printer
IO.puts("Example 1: Streaming to a simple token printer process")
IO.puts("=" <> String.duplicate("=", 50))

printer_pid = TokenPrinter.start()

LMStudio.stream_to_process(
  [%{role: "user", content: "Write a short poem about Elixir"}],
  printer_pid,
  model: "deepseek-r1-0528-qwen3-8b-mlx"
)

# Give the process time to complete
Process.sleep(5000)

# Example 2: Using IO.puts directly with each token
IO.puts("\n\nExample 2: Using IO.puts for each token (with newlines)")
IO.puts("=" <> String.duplicate("=", 50))

# Helper module to receive all tokens
defmodule TokenReceiver do
  def start do
    spawn(fn -> receive_all() end)
  end
  
  def receive_all do
    receive do
      {:lmstudio_token, token} ->
        IO.puts("Token: #{token}")
        receive_all()
        
      {:lmstudio_done, :ok} ->
        IO.puts("\nAll tokens received!")
        
      {:lmstudio_error, reason} ->
        IO.puts("Error: #{inspect(reason)}")
    end
  end
end

io_puts_pid = TokenReceiver.start()

LMStudio.stream_to_process(
  [%{role: "user", content: "Count from 1 to 5"}],
  io_puts_pid,
  model: "deepseek-r1-0528-qwen3-8b-mlx"
)

Process.sleep(3000)

# Example 3: Custom process that collects tokens
IO.puts("\n\nExample 3: Collecting tokens in a process")
IO.puts("=" <> String.duplicate("=", 50))

defmodule TokenCollector do
  def start do
    spawn(fn ->
      tokens = collect_tokens([])
      IO.puts("Collected response: #{Enum.reverse(tokens) |> Enum.join()}")
    end)
  end
  
  defp collect_tokens(acc) do
    receive do
      {:lmstudio_token, token} ->
        collect_tokens([token | acc])
        
      {:lmstudio_done, :ok} ->
        acc
        
      {:lmstudio_error, reason} ->
        IO.puts("Error collecting tokens: #{inspect(reason)}")
        acc
    end
  end
end

collector_pid = TokenCollector.start()

LMStudio.stream_to_process(
  [%{role: "user", content: "What is 2+2?"}],
  collector_pid,
  model: "deepseek-r1-0528-qwen3-8b-mlx"
)

Process.sleep(3000)