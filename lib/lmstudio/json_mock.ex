defmodule Jason do
  @moduledoc """
  Simple JSON mock for testing without external dependencies
  """
  
  def encode!(data) do
    case data do
      %{} = map -> 
        # Simple map to JSON conversion
        pairs = Enum.map(map, fn {k, v} -> 
          ~s("#{k}": #{encode_value(v)})
        end)
        "{" <> Enum.join(pairs, ", ") <> "}"
      list when is_list(list) ->
        values = Enum.map(list, &encode_value/1)
        "[" <> Enum.join(values, ", ") <> "]"
      other -> encode_value(other)
    end
  end
  
  def decode(json_string) when is_binary(json_string) do
    try do
      # Very basic JSON parsing - just for testing
      cond do
        String.starts_with?(json_string, "{") and String.ends_with?(json_string, "}") ->
          {:ok, %{"data" => "parsed"}}
        String.starts_with?(json_string, "[") and String.ends_with?(json_string, "]") ->
          {:ok, []}
        true ->
          {:ok, json_string}
      end
    rescue
      _ -> {:error, "parse_error"}
    end
  end
  
  def decode!(json_string) do
    case decode(json_string) do
      {:ok, result} -> result
      {:error, reason} -> raise "JSON decode error: #{reason}"
    end
  end
  
  defp encode_value(value) when is_binary(value), do: ~s("#{value}")
  defp encode_value(value) when is_number(value), do: "#{value}"
  defp encode_value(value) when is_boolean(value), do: "#{value}"
  defp encode_value(nil), do: "null"
  defp encode_value(value) when is_map(value), do: encode!(value)
  defp encode_value(value) when is_list(value), do: encode!(value)
  defp encode_value(value), do: ~s("#{inspect(value)}")
end