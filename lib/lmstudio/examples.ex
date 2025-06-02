defmodule LMStudio.Examples do
  @moduledoc """
  Examples of using the LMStudio client module.
  """

  def simple_chat_example do
    case LMStudio.chat("What is the capital of France?") do
      {:ok, response} ->
        content = get_in(response, ["choices", Access.at(0), "message", "content"])
        IO.puts("Assistant: #{content}")
      {:error, reason} ->
        IO.puts("Error: #{inspect(reason)}")
    end
  end

  def streaming_example do
    IO.write("Assistant: ")
    
    case LMStudio.chat(
      "Write a haiku about Elixir programming",
      stream: true,
      stream_callback: &handle_stream_chunk/1
    ) do
      {:ok, :stream_complete} ->
        IO.puts("\n\nStream completed!")
      {:error, reason} ->
        IO.puts("\nError: #{inspect(reason)}")
    end
  end

  def conversation_example do
    messages = [
      %{role: "system", content: "You are a helpful assistant who speaks like a pirate."},
      %{role: "user", content: "Hello!"},
      %{role: "assistant", content: "Ahoy there, matey! How can this old sea dog help ye today?"},
      %{role: "user", content: "What's the weather like?"}
    ]

    case LMStudio.complete(messages) do
      {:ok, response} ->
        content = get_in(response, ["choices", Access.at(0), "message", "content"])
        IO.puts("Assistant: #{content}")
      {:error, reason} ->
        IO.puts("Error: #{inspect(reason)}")
    end
  end

  def list_models_example do
    IO.puts("Available models:")
    
    case LMStudio.list_models() do
      {:ok, models} ->
        Enum.each(models, fn model ->
          IO.puts("  - #{model["id"]}")
        end)
      {:error, reason} ->
        IO.puts("Error: #{inspect(reason)}")
    end
  end

  defp handle_stream_chunk({:chunk, content}) do
    IO.write(content)
  end

  defp handle_stream_chunk({:done, _}) do
    :ok
  end
end