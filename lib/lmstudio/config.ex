defmodule LMStudio.Config do
  @moduledoc """
  Configuration module for LMStudio client.
  
  You can configure the client in your config.exs:
  
      config :lmstudio,
        base_url: "http://localhost:1234",
        default_model: "your-model-name",
        default_temperature: 0.7,
        default_max_tokens: 2048
  """

  def base_url do
    Application.get_env(:lmstudio, :base_url, "http://192.168.1.177:1234")
  end

  def default_model do
    Application.get_env(:lmstudio, :default_model, "default")
  end

  def default_temperature do
    Application.get_env(:lmstudio, :default_temperature, 0.7)
  end

  def default_max_tokens do
    Application.get_env(:lmstudio, :default_max_tokens, 2048)
  end
end