defmodule LMStudio do
  @moduledoc """
  Elixir client for LM Studio's chat completions API with streaming support.
  """

  @endpoint "/v1/chat/completions"

  @doc """
  Sends a chat completion request with optional streaming.

  ## Parameters
    - messages: List of message maps with :role and :content keys
    - opts: Keyword list of options
      - :model - Model to use (default: from config)
      - :temperature - Temperature for sampling (default: from config)
      - :max_tokens - Maximum tokens to generate (default: from config)
      - :stream - Whether to stream the response (default: false)
      - :system_prompt - Convenience option to prepend a system message
      - :stream_callback - Function to handle streaming chunks (required if stream: true)

  ## Examples

      # Non-streaming request
      {:ok, response} = LMStudio.complete([
        %{role: "user", content: "Hello, how are you?"}
      ])

      # Streaming request
      LMStudio.complete(
        [%{role: "user", content: "Tell me a story"}],
        stream: true,
        stream_callback: fn chunk -> IO.write(chunk) end
      )

      # With system prompt
      LMStudio.complete(
        [%{role: "user", content: "What is Elixir?"}],
        system_prompt: "You are a helpful programming assistant."
      )
  """
  def complete(messages, opts \\ []) do
    model = Keyword.get(opts, :model, LMStudio.Config.default_model())
    temperature = Keyword.get(opts, :temperature, LMStudio.Config.default_temperature())
    max_tokens = Keyword.get(opts, :max_tokens, LMStudio.Config.default_max_tokens())
    stream = Keyword.get(opts, :stream, false)
    stream_callback = Keyword.get(opts, :stream_callback)
    system_prompt = Keyword.get(opts, :system_prompt)

    # Prepend system message if provided
    messages = if system_prompt do
      [%{role: "system", content: system_prompt} | messages]
    else
      messages
    end

    body = %{
      model: model,
      messages: messages,
      temperature: temperature,
      max_tokens: max_tokens,
      stream: stream
    }

    case stream do
      true ->
        stream_request(body, stream_callback)
      false ->
        regular_request(body)
    end
  end

  @doc """
  Convenience function for single user message.
  """
  def chat(user_content, opts \\ []) do
    complete([%{role: "user", content: user_content}], opts)
  end

  @doc """
  Streams a completion response to a process.
  
  ## Parameters
    - messages: List of message maps with :role and :content keys
    - pid: The process to send tokens to
    - opts: Keyword list of options (same as complete/2 but without stream_callback)
    
  ## Message Format
  The target process will receive messages in the format:
    - `{:lmstudio_token, token}` for each token
    - `{:lmstudio_done, :ok}` when streaming is complete
    - `{:lmstudio_error, reason}` if an error occurs
    
  ## Example
      pid = spawn(fn ->
        receive_tokens()
      end)
      
      LMStudio.stream_to_process(
        [%{role: "user", content: "Hello!"}],
        pid
      )
      
      defp receive_tokens do
        receive do
          {:lmstudio_token, token} ->
            IO.puts(token)
            receive_tokens()
          {:lmstudio_done, :ok} ->
            IO.puts("\\nDone!")
          {:lmstudio_error, reason} ->
            IO.puts("Error: \#{inspect(reason)}")
        end
      end
  """
  def stream_to_process(messages, pid, opts \\ []) when is_pid(pid) do
    callback = fn
      {:chunk, content} -> send(pid, {:lmstudio_token, content})
      {:done, _} -> send(pid, {:lmstudio_done, :ok})
    end
    
    opts = Keyword.put(opts, :stream, true)
    opts = Keyword.put(opts, :stream_callback, callback)
    
    case complete(messages, opts) do
      {:ok, result} -> {:ok, result}
      {:error, reason} -> 
        send(pid, {:lmstudio_error, reason})
        {:error, reason}
    end
  end

  @doc """
  Lists available models from LM Studio.
  """
  def list_models do
    url = LMStudio.Config.base_url() <> "/v1/models"
    
    case http_get(url) do
      {:ok, body} ->
        case Jason.decode(body) do
          {:ok, %{"data" => models}} ->
            {:ok, models}
          {:ok, response} ->
            {:ok, response}
          error ->
            error
        end
      error ->
        error
    end
  end

  # Private functions

  defp regular_request(body) do
    url = LMStudio.Config.base_url() <> @endpoint
    
    case http_post(url, body) do
      {:ok, response_body} ->
        Jason.decode(response_body)
      error ->
        error
    end
  end

  defp stream_request(body, callback) do
    unless is_function(callback, 1) do
      raise ArgumentError, "stream_callback must be a function with arity 1"
    end

    url = LMStudio.Config.base_url() <> @endpoint
    
    # For streaming, we'll use a simple approach with httpc
    # In production, you might want to use a more sophisticated streaming client
    case http_post_stream(url, body, callback) do
      :ok -> {:ok, :stream_complete}
      error -> error
    end
  end

  # HTTP helper functions using Erlang's built-in :httpc

  defp http_get(url) do
    :inets.start()
    :ssl.start()
    
    headers = [{~c"Accept", ~c"application/json"}]
    
    case :httpc.request(:get, {String.to_charlist(url), headers}, [{:timeout, 60_000}], []) do
      {:ok, {{_, 200, _}, _, body}} ->
        {:ok, List.to_string(body)}
      {:ok, {{_, status, _}, _, body}} ->
        {:error, {:http_error, status, List.to_string(body)}}
      {:error, reason} ->
        {:error, {:request_failed, reason}}
    end
  end

  defp http_post(url, body) do
    :inets.start()
    :ssl.start()
    
    headers = [
      {~c"Content-Type", ~c"application/json"},
      {~c"Accept", ~c"application/json"}
    ]
    
    json_body = try do
      Jason.encode!(body)
    rescue
      UndefinedFunctionError ->
        # Fallback for testing without Jason - this will trigger connection error
        "{\"error\": \"JSON encoding not available\"}"
    end
    
    case :httpc.request(
      :post,
      {String.to_charlist(url), headers, ~c"application/json", json_body},
      [{:timeout, 60_000}],
      []
    ) do
      {:ok, {{_, 200, _}, _, response_body}} ->
        {:ok, List.to_string(response_body)}
      {:ok, {{_, status, _}, _, response_body}} ->
        {:error, {:http_error, status, List.to_string(response_body)}}
      {:error, reason} ->
        {:error, {:request_failed, reason}}
    end
  end

  defp http_post_stream(url, body, callback) do
    :inets.start()
    :ssl.start()
    
    headers = [
      {~c"Content-Type", ~c"application/json"},
      {~c"Accept", ~c"text/event-stream"}
    ]
    
    json_body = try do
      Jason.encode!(body)
    rescue
      UndefinedFunctionError ->
        # Fallback for testing without Jason - this will trigger connection error
        "{\"error\": \"JSON encoding not available\"}"
    end
    parent = self()
    
    # Spawn a process to handle the streaming response
    spawn(fn ->
      case :httpc.request(
        :post,
        {String.to_charlist(url), headers, ~c"application/json", json_body},
        [{:timeout, 60_000}],
        [{:sync, false}, {:stream, :self}]
      ) do
        {:ok, request_id} ->
          handle_stream_response(request_id, callback, "", parent)
        {:error, reason} ->
          send(parent, {:stream_error, reason})
      end
    end)
    
    # Wait for completion
    receive do
      :stream_complete -> :ok
      {:stream_error, reason} -> {:error, reason}
    after
      120_000 -> {:error, :timeout}
    end
  end

  defp handle_stream_response(request_id, callback, buffer, parent) do
    receive do
      {:http, {^request_id, :stream_start, _headers}} ->
        handle_stream_response(request_id, callback, buffer, parent)
      
      {:http, {^request_id, :stream, chunk}} ->
        new_buffer = buffer <> chunk
        {events, remaining_buffer} = parse_sse_events(new_buffer)
        
        Enum.each(events, fn event ->
          process_sse_event(event, callback)
        end)
        
        handle_stream_response(request_id, callback, remaining_buffer, parent)
      
      {:http, {^request_id, :stream_end, _headers}} ->
        callback.({:done, nil})
        send(parent, :stream_complete)
      
      {:http, {^request_id, {:error, reason}}} ->
        send(parent, {:stream_error, reason})
    after
      60_000 ->
        send(parent, {:stream_error, :timeout})
    end
  end

  defp parse_sse_events(buffer) do
    # Split by double newline to separate events
    parts = String.split(buffer, "\n\n", trim: true)
    
    case parts do
      [] -> 
        {[], buffer}
      [incomplete] ->
        # Check if this is actually complete (ends with \n\n)
        if String.ends_with?(buffer, "\n\n") do
          {[incomplete], ""}
        else
          {[], buffer}
        end
      _ ->
        # All but the last are complete events
        {complete, [maybe_incomplete]} = Enum.split(parts, -1)
        
        # Check if the last part is complete
        if String.ends_with?(buffer, "\n\n") do
          {complete ++ [maybe_incomplete], ""}
        else
          {complete, maybe_incomplete}
        end
    end
  end

  defp process_sse_event(event, callback) do
    # Parse SSE format
    lines = String.split(event, "\n", trim: true)
    
    Enum.each(lines, fn line ->
      case String.split(line, ": ", parts: 2) do
        ["data", "[DONE]"] ->
          :ok  # Done is handled in the main function
        
        ["data", json_data] ->
          case Jason.decode(json_data) do
            {:ok, data} ->
              # Extract content from the typical streaming format
              case get_in(data, ["choices", Access.at(0), "delta", "content"]) do
                nil -> :ok
                content -> callback.({:chunk, content})
              end
            {:error, _} ->
              :ok
          end
        
        _ ->
          :ok
      end
    end)
  end
end