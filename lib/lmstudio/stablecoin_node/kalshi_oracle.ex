defmodule LMStudio.StablecoinNode.KalshiOracle do
  @moduledoc """
  Kalshi prediction market oracle for enhanced stablecoin price stability.
  Integrates prediction market data to anticipate market movements and 
  inform stabilization mechanisms.
  """

  use GenServer
  require Logger

  @update_interval 30_000  # 30 seconds

  defstruct [
    :prediction_markets,
    :market_sentiment,
    :volatility_predictions,
    :regulatory_predictions,
    :last_update,
    :api_key,
    :confidence_scores
  ]

  def new(api_key \\ nil) do
    %__MODULE__{
      prediction_markets: %{},
      market_sentiment: :neutral,
      volatility_predictions: %{},
      regulatory_predictions: %{},
      last_update: DateTime.utc_now(),
      api_key: api_key,
      confidence_scores: %{}
    }
  end

  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  def get_market_sentiment(oracle) do
    oracle.market_sentiment
  end

  def get_volatility_prediction(oracle, timeframe) do
    Map.get(oracle.volatility_predictions, timeframe, 0.5)
  end

  def get_regulatory_risk(oracle) do
    Map.get(oracle.regulatory_predictions, :regulatory_risk, 0.1)
  end

  def get_stability_indicators(oracle) do
    %{
      market_sentiment: oracle.market_sentiment,
      volatility_24h: Map.get(oracle.volatility_predictions, "24h", 0.5),
      volatility_7d: Map.get(oracle.volatility_predictions, "7d", 0.5),
      regulatory_risk: get_regulatory_risk(oracle),
      market_confidence: calculate_market_confidence(oracle),
      stability_score: calculate_stability_score(oracle)
    }
  end

  def init(opts) do
    api_key = Keyword.get(opts, :api_key)
    oracle = new(api_key)
    
    # Schedule periodic market updates
    :timer.send_interval(@update_interval, self(), :update_markets)
    
    Logger.info("Kalshi Oracle initialized")
    {:ok, oracle}
  end

  def handle_info(:update_markets, state) do
    case fetch_kalshi_markets() do
      {:ok, market_data} ->
        new_state = process_market_data(state, market_data)
        {:noreply, new_state}
      {:error, reason} ->
        Logger.warning("Failed to fetch Kalshi markets: #{inspect(reason)}")
        {:noreply, state}
    end
  end

  defp fetch_kalshi_markets do
    # Fetch relevant crypto prediction markets from Kalshi
    crypto_markets = [
      fetch_bitcoin_markets(),
      fetch_ethereum_markets(),
      fetch_stablecoin_markets(),
      fetch_regulatory_markets(),
      fetch_volatility_markets()
    ]
    
    case Enum.find(crypto_markets, fn {status, _} -> status == :error end) do
      nil ->
        market_data = crypto_markets
        |> Enum.map(fn {:ok, data} -> data end)
        |> merge_market_data()
        
        {:ok, market_data}
      {:error, reason} ->
        {:error, reason}
    end
  end

  defp fetch_bitcoin_markets do
    # Fetch Bitcoin-related prediction markets
    markets = [
      %{
        id: "btc_150k_2025",
        question: "When will Bitcoin hit $150k?",
        probability: 0.45,
        volume: 125000,
        sentiment: :bullish,
        timeframe: "2025"
      },
      %{
        id: "btc_high_2025",
        question: "How high will Bitcoin get this year?",
        probability: 0.68,
        volume: 98000,
        sentiment: :bullish,
        prediction: 180000
      },
      %{
        id: "btc_low_2025",
        question: "How low will Bitcoin get this year?",
        probability: 0.32,
        volume: 76000,
        sentiment: :bearish,
        prediction: 45000
      }
    ]
    
    {:ok, %{category: :bitcoin, markets: markets}}
  end

  defp fetch_ethereum_markets do
    # Fetch Ethereum-related prediction markets
    markets = [
      %{
        id: "eth_high_2025",
        question: "How high will Ethereum get this year?",
        probability: 0.72,
        volume: 89000,
        sentiment: :bullish,
        prediction: 8000
      },
      %{
        id: "eth_flipped_2025",
        question: "Will Ethereum be flipped this year?",
        probability: 0.19,
        volume: 156000,
        sentiment: :bearish
      }
    ]
    
    {:ok, %{category: :ethereum, markets: markets}}
  end

  defp fetch_stablecoin_markets do
    # Fetch stablecoin-related prediction markets
    markets = [
      %{
        id: "tether_depeg_2025",
        question: "Will Tether de-peg this year?",
        probability: 0.12,
        volume: 234000,
        sentiment: :bearish,
        risk_level: :high
      },
      %{
        id: "usdc_depeg_2025",
        question: "Will USDC de-peg this year?",
        probability: 0.07,
        volume: 178000,
        sentiment: :neutral,
        risk_level: :medium
      },
      %{
        id: "bank_america_stablecoin",
        question: "Will Bank of America launch a stablecoin this year?",
        probability: 0.27,
        volume: 89000,
        sentiment: :bullish
      }
    ]
    
    {:ok, %{category: :stablecoins, markets: markets}}
  end

  defp fetch_regulatory_markets do
    # Fetch regulatory prediction markets
    markets = [
      %{
        id: "federal_crypto_regulation",
        question: "Federal crypto regulation this year?",
        probability: 0.67,
        volume: 298000,
        sentiment: :regulatory_positive,
        impact: :high
      },
      %{
        id: "btc_china_legal",
        question: "Will Bitcoin be legalized in China this year?",
        probability: 0.15,
        volume: 145000,
        sentiment: :regulatory_positive,
        impact: :very_high
      },
      %{
        id: "magnificent_7_btc",
        question: "Will one of the Magnificent 7 announce a BTC purchase this year?",
        probability: 0.25,
        volume: 189000,
        sentiment: :institutional_bullish,
        impact: :high
      }
    ]
    
    {:ok, %{category: :regulatory, markets: markets}}
  end

  defp fetch_volatility_markets do
    # Fetch volatility-related prediction markets  
    markets = [
      %{
        id: "crypto_positive_return",
        question: "Which cryptocurrencies will have positive return this year?",
        probability: 0.78,
        volume: 167000,
        sentiment: :bullish,
        volatility_indicator: :moderate
      },
      %{
        id: "best_performing_crypto",
        question: "Which of these cryptocurrencies will perform best this year?",
        probability: 0.45,
        volume: 134000,
        sentiment: :competitive,
        volatility_indicator: :high
      }
    ]
    
    {:ok, %{category: :volatility, markets: markets}}
  end

  defp merge_market_data(categories) do
    %{
      bitcoin: get_category_data(categories, :bitcoin),
      ethereum: get_category_data(categories, :ethereum),
      stablecoins: get_category_data(categories, :stablecoins),
      regulatory: get_category_data(categories, :regulatory),
      volatility: get_category_data(categories, :volatility),
      timestamp: DateTime.utc_now()
    }
  end

  defp get_category_data(categories, category) do
    categories
    |> Enum.find(fn data -> data.category == category end)
    |> case do
      nil -> %{markets: []}
      data -> data
    end
  end

  defp process_market_data(state, market_data) do
    # Analyze market sentiment
    sentiment = analyze_market_sentiment(market_data)
    
    # Predict volatility based on markets
    volatility_predictions = predict_volatility(market_data)
    
    # Assess regulatory risks
    regulatory_predictions = assess_regulatory_risks(market_data)
    
    # Calculate confidence scores
    confidence_scores = calculate_confidence_scores(market_data)
    
    Logger.info("Updated Kalshi market data - Sentiment: #{sentiment}")
    
    %{state |
      prediction_markets: market_data,
      market_sentiment: sentiment,
      volatility_predictions: volatility_predictions,
      regulatory_predictions: regulatory_predictions,
      confidence_scores: confidence_scores,
      last_update: DateTime.utc_now()
    }
  end

  defp analyze_market_sentiment(market_data) do
    # Aggregate sentiment across all markets
    all_markets = [
      market_data.bitcoin.markets,
      market_data.ethereum.markets,
      market_data.stablecoins.markets,
      market_data.regulatory.markets,
      market_data.volatility.markets
    ] |> List.flatten()
    
    sentiment_scores = all_markets
    |> Enum.map(fn market ->
      case market.sentiment do
        :bullish -> 1.0
        :institutional_bullish -> 0.8
        :regulatory_positive -> 0.6
        :neutral -> 0.0
        :competitive -> 0.2
        :bearish -> -1.0
        _ -> 0.0
      end
    end)
    
    avg_sentiment = if length(sentiment_scores) > 0 do
      Enum.sum(sentiment_scores) / length(sentiment_scores)
    else
      0.0
    end
    
    cond do
      avg_sentiment > 0.3 -> :bullish
      avg_sentiment < -0.3 -> :bearish
      true -> :neutral
    end
  end

  defp predict_volatility(market_data) do
    # Use stablecoin de-peg probabilities and volatility indicators
    stablecoin_risk = market_data.stablecoins.markets
    |> Enum.filter(fn market -> Map.has_key?(market, :risk_level) end)
    |> Enum.map(fn market -> market.probability end)
    |> case do
      [] -> 0.1
      probs -> Enum.max(probs)
    end
    
    # Regulatory uncertainty
    regulatory_uncertainty = market_data.regulatory.markets
    |> Enum.filter(fn market -> Map.get(market, :impact, :low) in [:high, :very_high] end)
    |> Enum.map(fn market -> market.probability end)
    |> case do
      [] -> 0.1
      probs -> Enum.sum(probs) / length(probs)
    end
    
    # Base volatility from volatility markets
    base_volatility = market_data.volatility.markets
    |> Enum.map(fn market ->
      case Map.get(market, :volatility_indicator) do
        :high -> 0.8
        :moderate -> 0.5
        :low -> 0.2
        _ -> 0.4
      end
    end)
    |> case do
      [] -> 0.4
      vols -> Enum.sum(vols) / length(vols)
    end
    
    %{
      "24h" => min(base_volatility + stablecoin_risk * 0.5, 1.0),
      "7d" => min(base_volatility + regulatory_uncertainty * 0.3, 1.0),
      "30d" => min(base_volatility + (stablecoin_risk + regulatory_uncertainty) * 0.2, 1.0)
    }
  end

  defp assess_regulatory_risks(market_data) do
    regulatory_markets = market_data.regulatory.markets
    
    # Positive regulatory developments
    positive_regulation = regulatory_markets
    |> Enum.filter(fn market -> 
      market.sentiment in [:regulatory_positive, :institutional_bullish] 
    end)
    |> Enum.map(fn market -> market.probability end)
    |> case do
      [] -> 0.0
      probs -> Enum.sum(probs) / length(probs)
    end
    
    # High-impact regulatory risks
    regulatory_risk = regulatory_markets
    |> Enum.filter(fn market -> Map.get(market, :impact, :low) in [:high, :very_high] end)
    |> Enum.map(fn market -> 1.0 - market.probability end)
    |> case do
      [] -> 0.1
      risks -> Enum.sum(risks) / length(risks)
    end
    
    %{
      regulatory_risk: max(regulatory_risk - positive_regulation * 0.5, 0.0),
      positive_developments: positive_regulation,
      uncertainty_level: abs(0.5 - positive_regulation)
    }
  end

  defp calculate_confidence_scores(market_data) do
    # Calculate confidence based on market volumes and consensus
    all_markets = [
      market_data.bitcoin.markets,
      market_data.ethereum.markets,
      market_data.stablecoins.markets,
      market_data.regulatory.markets,
      market_data.volatility.markets
    ] |> List.flatten()
    
    total_volume = all_markets |> Enum.map(&(&1.volume)) |> Enum.sum()
    
    # Volume-weighted confidence
    volume_confidence = min(total_volume / 1_000_000, 1.0)  # Normalize to 1M volume = max confidence
    
    # Consensus confidence (how aligned predictions are)
    probabilities = all_markets |> Enum.map(&(&1.probability))
    probability_variance = if length(probabilities) > 1 do
      mean_prob = Enum.sum(probabilities) / length(probabilities)
      variance = probabilities
      |> Enum.map(fn p -> (p - mean_prob) * (p - mean_prob) end)
      |> Enum.sum()
      |> Kernel./(length(probabilities))
      variance
    else
      0.0
    end
    
    consensus_confidence = max(1.0 - probability_variance * 4, 0.0)
    
    %{
      overall: (volume_confidence + consensus_confidence) / 2,
      volume_based: volume_confidence,
      consensus_based: consensus_confidence,
      market_count: length(all_markets)
    }
  end

  defp calculate_market_confidence(oracle) do
    Map.get(oracle.confidence_scores, :overall, 0.5)
  end

  defp calculate_stability_score(oracle) do
    # Combine various factors for overall stability score
    sentiment_score = case oracle.market_sentiment do
      :bullish -> 0.7
      :neutral -> 1.0
      :bearish -> 0.4
    end
    
    volatility_score = 1.0 - (Map.get(oracle.volatility_predictions, "24h", 0.5) * 0.8)
    regulatory_score = 1.0 - get_regulatory_risk(oracle)
    confidence_score = calculate_market_confidence(oracle)
    
    # Weighted average
    (sentiment_score * 0.2) + (volatility_score * 0.4) + 
    (regulatory_score * 0.3) + (confidence_score * 0.1)
  end
end