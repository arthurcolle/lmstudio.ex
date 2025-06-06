defmodule LMStudio.StablecoinNode.KalshiConfig do
  @moduledoc """
  Configuration module for Kalshi integration.
  
  Handles environment variables, API keys, and configuration management
  for the Kalshi prediction market integration.
  """

  @doc """
  Gets Kalshi API configuration from environment variables or config files.
  
  Required environment variables:
  - KALSHI_EMAIL: Your Kalshi account email
  - KALSHI_PASSWORD: Your Kalshi account password
  - KALSHI_API_KEY: Your Kalshi API key (optional for demo mode)
  
  Optional environment variables:
  - KALSHI_DEMO_MODE: Set to "true" to use demo API (default: true)
  - KALSHI_RATE_LIMIT: API rate limit per minute (default: 100)
  - KALSHI_TIMEOUT: Request timeout in milliseconds (default: 10000)
  """
  def get_config do
    %{
      email: get_env("KALSHI_EMAIL"),
      password: get_env("KALSHI_PASSWORD"),
      api_key: get_env("KALSHI_API_KEY"),
      demo_mode: get_boolean_env("KALSHI_DEMO_MODE", true),
      rate_limit: get_integer_env("KALSHI_RATE_LIMIT", 100),
      timeout: get_integer_env("KALSHI_TIMEOUT", 10_000),
      max_retries: get_integer_env("KALSHI_MAX_RETRIES", 3),
      heartbeat_interval: get_integer_env("KALSHI_HEARTBEAT_INTERVAL", 30_000)
    }
  end

  @doc """
  Validates that all required configuration is present.
  """
  def validate_config(config \\ get_config()) do
    errors = []

    errors = if is_nil(config.email) or config.email == "" do
      ["KALSHI_EMAIL is required" | errors]
    else
      errors
    end

    errors = if is_nil(config.password) or config.password == "" do
      ["KALSHI_PASSWORD is required" | errors]
    else
      errors
    end

    # API key is optional in demo mode
    errors = if not config.demo_mode and (is_nil(config.api_key) or config.api_key == "") do
      ["KALSHI_API_KEY is required for production mode" | errors]
    else
      errors
    end

    case errors do
      [] -> :ok
      errors -> {:error, errors}
    end
  end

  @doc """
  Gets configuration for stablecoin integration parameters.
  """
  def get_stablecoin_config do
    %{
      # Prediction market weights for different risk factors
      market_weights: %{
        btc_price: get_float_env("KALSHI_BTC_WEIGHT", 0.25),
        eth_price: get_float_env("KALSHI_ETH_WEIGHT", 0.15),
        stablecoin_depeg: get_float_env("KALSHI_DEPEG_WEIGHT", 0.30),
        crypto_regulation: get_float_env("KALSHI_REGULATION_WEIGHT", 0.20),
        fed_policy: get_float_env("KALSHI_FED_WEIGHT", 0.10)
      },
      
      # Alert thresholds
      alert_thresholds: %{
        volatility_forecast: get_float_env("KALSHI_VOLATILITY_THRESHOLD", 0.70),
        depeg_risk: get_float_env("KALSHI_DEPEG_THRESHOLD", 0.15),
        regulatory_risk: get_float_env("KALSHI_REGULATORY_THRESHOLD", 0.60),
        sentiment_shift: get_float_env("KALSHI_SENTIMENT_THRESHOLD", 0.30),
        market_stress: get_float_env("KALSHI_STRESS_THRESHOLD", 0.80)
      },
      
      # Update intervals
      update_intervals: %{
        market_analysis: get_integer_env("KALSHI_ANALYSIS_INTERVAL", 30_000),
        deep_analysis: get_integer_env("KALSHI_DEEP_ANALYSIS_INTERVAL", 300_000),
        heartbeat: get_integer_env("KALSHI_HEARTBEAT_INTERVAL", 30_000)
      },
      
      # Market selection criteria
      market_criteria: %{
        min_volume: get_integer_env("KALSHI_MIN_VOLUME", 1000),
        max_markets_per_category: get_integer_env("KALSHI_MAX_MARKETS", 50),
        top_markets_to_subscribe: get_integer_env("KALSHI_TOP_MARKETS", 20)
      }
    }
  end

  @doc """
  Creates a sample .env file with all required configuration.
  """
  def create_sample_env_file(path \\ ".env.kalshi.sample") do
    content = """
    # Kalshi API Configuration
    # Required: Your Kalshi account credentials
    KALSHI_EMAIL=your-email@example.com
    KALSHI_PASSWORD=your-password

    # Optional: API key for production mode
    KALSHI_API_KEY=your-api-key

    # Optional: Use demo mode (true/false)
    KALSHI_DEMO_MODE=true

    # Optional: API rate limits and timeouts
    KALSHI_RATE_LIMIT=100
    KALSHI_TIMEOUT=10000
    KALSHI_MAX_RETRIES=3
    KALSHI_HEARTBEAT_INTERVAL=30000

    # Optional: Stablecoin integration weights
    KALSHI_BTC_WEIGHT=0.25
    KALSHI_ETH_WEIGHT=0.15
    KALSHI_DEPEG_WEIGHT=0.30
    KALSHI_REGULATION_WEIGHT=0.20
    KALSHI_FED_WEIGHT=0.10

    # Optional: Alert thresholds (0.0 to 1.0)
    KALSHI_VOLATILITY_THRESHOLD=0.70
    KALSHI_DEPEG_THRESHOLD=0.15
    KALSHI_REGULATORY_THRESHOLD=0.60
    KALSHI_SENTIMENT_THRESHOLD=0.30
    KALSHI_STRESS_THRESHOLD=0.80

    # Optional: Update intervals (milliseconds)
    KALSHI_ANALYSIS_INTERVAL=30000
    KALSHI_DEEP_ANALYSIS_INTERVAL=300000

    # Optional: Market selection criteria
    KALSHI_MIN_VOLUME=1000
    KALSHI_MAX_MARKETS=50
    KALSHI_TOP_MARKETS=20
    """

    case File.write(path, content) do
      :ok ->
        {:ok, "Sample configuration file created at #{path}"}
      {:error, reason} ->
        {:error, "Failed to create config file: #{reason}"}
    end
  end

  @doc """
  Loads configuration from a .env file.
  """
  def load_env_file(path \\ ".env") do
    if File.exists?(path) do
      path
      |> File.read!()
      |> String.split("\n")
      |> Enum.each(fn line ->
        line = String.trim(line)
        unless String.starts_with?(line, "#") or line == "" do
          case String.split(line, "=", parts: 2) do
            [key, value] ->
              System.put_env(String.trim(key), String.trim(value))
            _ ->
              :ok
          end
        end
      end)
      
      :ok
    else
      {:error, "Environment file not found: #{path}"}
    end
  end

  # Private helper functions
  defp get_env(key, default \\ nil) do
    System.get_env(key) || default
  end

  defp get_boolean_env(key, default) do
    case get_env(key) do
      "true" -> true
      "false" -> false
      _ -> default
    end
  end

  defp get_integer_env(key, default) do
    case get_env(key) do
      nil -> default
      value ->
        case Integer.parse(value) do
          {int, _} -> int
          :error -> default
        end
    end
  end

  defp get_float_env(key, default) do
    case get_env(key) do
      nil -> default
      value ->
        case Float.parse(value) do
          {float, _} -> float
          :error -> default
        end
    end
  end
end