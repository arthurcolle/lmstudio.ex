defmodule LMStudio.StablecoinNode.KalshiIntegration do
  @moduledoc """
  High-level Kalshi integration module that processes prediction market data
  for stablecoin stabilization decisions and risk management.
  """

  use GenServer
  require Logger

  alias LMStudio.StablecoinNode.KalshiMarketClient

  @update_interval 30_000  # 30 seconds
  @sentiment_decay_factor 0.95  # How quickly sentiment changes decay

  defstruct [
    :client_pid,
    :market_data,
    :processed_insights,
    :historical_data,
    :sentiment_tracker,
    :volatility_predictor,
    :risk_assessor,
    :last_update,
    :alert_thresholds,
    :active_subscriptions
  ]

  # Market categories and their impact weights on stablecoin stability
  @market_categories %{
    "btc_price" => %{weight: 0.25, impact: :high},
    "eth_price" => %{weight: 0.15, impact: :medium},
    "stablecoin_depeg" => %{weight: 0.30, impact: :critical},
    "crypto_regulation" => %{weight: 0.20, impact: :high},
    "fed_policy" => %{weight: 0.10, impact: :medium}
  }

  # Alert thresholds for various risk indicators
  @default_thresholds %{
    volatility_forecast: 0.70,    # 70% volatility prediction
    depeg_risk: 0.15,            # 15% depeg probability  
    regulatory_risk: 0.60,        # 60% adverse regulation
    sentiment_shift: 0.30,        # 30% sentiment change
    market_stress: 0.80           # 80% market stress indicator
  }

  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  def get_stability_insights(integration \\ __MODULE__) do
    GenServer.call(integration, :get_stability_insights)
  end

  def get_volatility_forecast(integration \\ __MODULE__, timeframe) do
    GenServer.call(integration, {:get_volatility_forecast, timeframe})
  end

  def get_market_sentiment(integration \\ __MODULE__) do
    GenServer.call(integration, :get_market_sentiment)
  end

  def get_risk_assessment(integration \\ __MODULE__) do
    GenServer.call(integration, :get_risk_assessment)
  end

  def get_stabilization_recommendations(integration \\ __MODULE__) do
    GenServer.call(integration, :get_stabilization_recommendations)
  end

  def init(opts) do
    # Start Kalshi client
    client_opts = [
      api_key: Keyword.get(opts, :api_key),
      user_id: Keyword.get(opts, :user_id),
      password: Keyword.get(opts, :password),
      demo_mode: Keyword.get(opts, :demo_mode, true)
    ]

    {:ok, client_pid} = KalshiMarketClient.start_link(client_opts)

    state = %__MODULE__{
      client_pid: client_pid,
      market_data: %{},
      processed_insights: %{},
      historical_data: %{},
      sentiment_tracker: init_sentiment_tracker(),
      volatility_predictor: init_volatility_predictor(),
      risk_assessor: init_risk_assessor(),
      last_update: DateTime.utc_now(),
      alert_thresholds: @default_thresholds,
      active_subscriptions: []
    }

    # Subscribe to Kalshi market updates
    Registry.register(LMStudio.PubSub, "kalshi_updates", [])

    # Schedule periodic updates
    :timer.send_interval(@update_interval, self(), :update_market_analysis)
    :timer.send_interval(300_000, self(), :deep_analysis)  # 5 minute deep analysis

    # Initial market data fetch
    :timer.send_after(1000, self(), :initial_setup)

    Logger.info("Kalshi Integration initialized")
    {:ok, state}
  end

  def handle_call(:get_stability_insights, _from, state) do
    insights = generate_stability_insights(state)
    {:reply, insights, state}
  end

  def handle_call({:get_volatility_forecast, timeframe}, _from, state) do
    forecast = get_volatility_forecast_for_timeframe(state, timeframe)
    {:reply, forecast, state}
  end

  def handle_call(:get_market_sentiment, _from, state) do
    sentiment = calculate_current_sentiment(state)
    {:reply, sentiment, state}
  end

  def handle_call(:get_risk_assessment, _from, state) do
    risk_assessment = perform_comprehensive_risk_assessment(state)
    {:reply, risk_assessment, state}
  end

  def handle_call(:get_stabilization_recommendations, _from, state) do
    recommendations = generate_stabilization_recommendations(state)
    {:reply, recommendations, state}
  end

  def handle_info(:initial_setup, state) do
    case setup_market_subscriptions(state) do
      {:ok, new_state} ->
        {:noreply, new_state}
      {:error, reason} ->
        Logger.error("Failed to setup market subscriptions: #{inspect(reason)}")
        # Retry after delay
        :timer.send_after(5000, self(), :initial_setup)
        {:noreply, state}
    end
  end

  def handle_info(:update_market_analysis, state) do
    new_state = update_market_analysis(state)
    {:noreply, new_state}
  end

  def handle_info(:deep_analysis, state) do
    new_state = perform_deep_market_analysis(state)
    
    # Check for alerts
    alerts = check_alert_conditions(new_state)
    if length(alerts) > 0 do
      broadcast_alerts(alerts)
    end
    
    {:noreply, new_state}
  end

  def handle_info({:kalshi_market_update, market_data}, state) do
    new_state = process_market_update(state, market_data)
    {:noreply, new_state}
  end

  # Private functions
  defp setup_market_subscriptions(state) do
    # Authenticate first
    auth_result = KalshiMarketClient.authenticate(state.client_pid, %{
      email: System.get_env("KALSHI_EMAIL"),
      password: System.get_env("KALSHI_PASSWORD")
    })

    case auth_result do
      {:ok, :authenticated} ->
        # Get crypto-related markets
        case KalshiMarketClient.get_crypto_markets(state.client_pid) do
          {:ok, markets} ->
            # Subscribe to key markets
            subscriptions = subscribe_to_key_markets(state, markets)
            
            new_state = %{state |
              market_data: index_markets_by_category(markets),
              active_subscriptions: subscriptions
            }
            
            Logger.info("Subscribed to #{length(subscriptions)} key markets")
            {:ok, new_state}
            
          {:error, reason} ->
            {:error, reason}
        end
        
      {:error, reason} ->
        {:error, reason}
    end
  end

  defp subscribe_to_key_markets(state, markets) do
    key_markets = identify_key_markets(markets)
    
    Enum.map(key_markets, fn market ->
      case KalshiMarketClient.subscribe_to_market(state.client_pid, market.ticker) do
        :ok ->
          Logger.info("Subscribed to #{market.ticker}: #{market.title}")
          market.ticker
        {:error, reason} ->
          Logger.warning("Failed to subscribe to #{market.ticker}: #{inspect(reason)}")
          nil
      end
    end)
    |> Enum.filter(&(&1 != nil))
  end

  defp identify_key_markets(markets) do
    # Identify markets most relevant to stablecoin stability
    markets
    |> Enum.filter(&is_high_priority_market?/1)
    |> Enum.sort_by(&calculate_market_priority/1, :desc)
    |> Enum.take(20)  # Subscribe to top 20 most relevant markets
  end

  defp is_high_priority_market?(market) do
    title = String.downcase(market.title)
    
    high_priority_keywords = [
      "bitcoin", "btc", "150", "100k", "crash", "moon",
      "tether", "usdt", "usdc", "stablecoin", "depeg", "peg",
      "federal", "regulation", "sec", "treasury", "ban",
      "ethereum", "eth", "flip", "defi"
    ]
    
    Enum.any?(high_priority_keywords, fn keyword ->
      String.contains?(title, keyword)
    end) and market.volume > 1000  # Minimum volume threshold
  end

  defp calculate_market_priority(market) do
    title = String.downcase(market.title)
    
    # Base score from volume and open interest
    base_score = (market.volume + market.open_interest) / 1000
    
    # Bonus for stablecoin-related markets
    stablecoin_bonus = if String.contains?(title, "stablecoin") or 
                          String.contains?(title, "usdt") or 
                          String.contains?(title, "usdc") or
                          String.contains?(title, "depeg") do
      100
    else
      0
    end
    
    # Bonus for regulatory markets
    regulatory_bonus = if String.contains?(title, "regulation") or
                          String.contains?(title, "federal") or
                          String.contains?(title, "sec") do
      75
    else
      0
    end
    
    # Bonus for major crypto price markets
    crypto_bonus = if String.contains?(title, "bitcoin") or
                      String.contains?(title, "ethereum") do
      50
    else
      0
    end
    
    base_score + stablecoin_bonus + regulatory_bonus + crypto_bonus
  end

  defp index_markets_by_category(markets) do
    markets
    |> Enum.group_by(&categorize_market/1)
  end

  defp categorize_market(market) do
    title = String.downcase(market.title)
    
    cond do
      String.contains?(title, "bitcoin") or String.contains?(title, "btc") ->
        "btc_price"
      String.contains?(title, "ethereum") or String.contains?(title, "eth") ->
        "eth_price"
      String.contains?(title, "stablecoin") or String.contains?(title, "depeg") or 
      String.contains?(title, "usdt") or String.contains?(title, "usdc") ->
        "stablecoin_depeg"
      String.contains?(title, "regulation") or String.contains?(title, "federal") or
      String.contains?(title, "sec") or String.contains?(title, "treasury") ->
        "crypto_regulation"
      String.contains?(title, "fed") or String.contains?(title, "interest") or
      String.contains?(title, "inflation") ->
        "fed_policy"
      true ->
        "other"
    end
  end

  defp update_market_analysis(state) do
    # Process current market data to generate insights
    insights = %{
      volatility_forecast: calculate_volatility_forecast(state),
      sentiment_analysis: update_sentiment_analysis(state),
      risk_indicators: calculate_risk_indicators(state),
      market_stress: assess_market_stress(state),
      timestamp: DateTime.utc_now()
    }
    
    # Update historical data
    new_historical = add_to_historical_data(state.historical_data, insights)
    
    %{state |
      processed_insights: insights,
      historical_data: new_historical,
      last_update: DateTime.utc_now()
    }
  end

  defp calculate_volatility_forecast(state) do
    # Analyze price-related markets for volatility indicators
    btc_markets = Map.get(state.market_data, "btc_price", [])
    eth_markets = Map.get(state.market_data, "eth_price", [])
    
    # Extract volatility signals from market probabilities
    btc_volatility = extract_volatility_signals(btc_markets, "btc")
    eth_volatility = extract_volatility_signals(eth_markets, "eth")
    
    # Weighted average
    %{
      "1h" => (btc_volatility.short_term * 0.6) + (eth_volatility.short_term * 0.4),
      "24h" => (btc_volatility.medium_term * 0.6) + (eth_volatility.medium_term * 0.4),
      "7d" => (btc_volatility.long_term * 0.6) + (eth_volatility.long_term * 0.4),
      overall: (btc_volatility.overall * 0.6) + (eth_volatility.overall * 0.4)
    }
  end

  defp extract_volatility_signals(markets, _asset) do
    if length(markets) == 0 do
      %{short_term: 0.3, medium_term: 0.3, long_term: 0.3, overall: 0.3}
    else
      # Analyze market titles and probabilities for volatility indicators
      high_volatility_indicators = Enum.count(markets, fn market ->
        title = String.downcase(market.title)
        (String.contains?(title, "crash") and market.probability > 0.2) or
        (String.contains?(title, "moon") and market.probability > 0.3) or
        (String.contains?(title, "150") and market.probability > 0.4) or
        (String.contains?(title, "100") and market.probability > 0.5)
      end)
      
      # Base volatility from market uncertainty (probabilities near 0.5)
      uncertainty_score = markets
      |> Enum.map(fn market -> 
        # Markets with probabilities near 50% indicate high uncertainty
        1.0 - abs(market.probability - 0.5) * 2
      end)
      |> Enum.sum()
      |> Kernel./(length(markets))
      
      volatility_multiplier = 1.0 + (high_volatility_indicators * 0.2)
      base_volatility = uncertainty_score * volatility_multiplier
      
      %{
        short_term: min(base_volatility * 1.2, 1.0),
        medium_term: min(base_volatility, 1.0),
        long_term: min(base_volatility * 0.8, 1.0),
        overall: min(base_volatility, 1.0)
      }
    end
  end

  defp update_sentiment_analysis(state) do
    # Analyze sentiment from all market categories
    category_sentiments = @market_categories
    |> Enum.map(fn {category, config} ->
      markets = Map.get(state.market_data, category, [])
      sentiment = calculate_category_sentiment(markets, category)
      {category, sentiment * config.weight}
    end)
    |> Map.new()
    
    # Calculate overall sentiment
    overall_sentiment = category_sentiments
    |> Map.values()
    |> Enum.sum()
    
    # Update sentiment tracker with exponential smoothing
    previous_sentiment = Map.get(state.sentiment_tracker, :overall, 0.5)
    smoothed_sentiment = previous_sentiment * @sentiment_decay_factor + 
                        overall_sentiment * (1 - @sentiment_decay_factor)
    
    %{
      overall: smoothed_sentiment,
      by_category: category_sentiments,
      trend: calculate_sentiment_trend(state.historical_data),
      confidence: calculate_sentiment_confidence(state.market_data)
    }
  end

  defp calculate_category_sentiment(markets, category) do
    if length(markets) == 0 do
      0.5  # Neutral
    else
      case category do
        "stablecoin_depeg" ->
          # For depeg markets, high probability is negative sentiment
          avg_depeg_risk = markets
          |> Enum.map(&(&1.probability))
          |> Enum.sum()
          |> Kernel./(length(markets))
          1.0 - avg_depeg_risk  # Invert: low depeg risk = positive sentiment
          
        "crypto_regulation" ->
          # Analyze whether regulation is positive or negative
          positive_regulation = Enum.count(markets, fn market ->
            title = String.downcase(market.title)
            (String.contains?(title, "approve") or String.contains?(title, "legal")) and
            market.probability > 0.5
          end)
          
          negative_regulation = Enum.count(markets, fn market ->
            title = String.downcase(market.title)
            (String.contains?(title, "ban") or String.contains?(title, "crackdown")) and
            market.probability > 0.5
          end)
          
          if positive_regulation > negative_regulation do
            0.7  # Positive regulatory environment
          else
            0.3  # Negative regulatory environment
          end
          
        _ ->
          # For price markets, analyze bullish vs bearish indicators
          bullish_signals = Enum.count(markets, fn market ->
            title = String.downcase(market.title)
            (String.contains?(title, "moon") or String.contains?(title, "150") or
             String.contains?(title, "100")) and market.probability > 0.3
          end)
          
          bearish_signals = Enum.count(markets, fn market ->
            title = String.downcase(market.title)
            String.contains?(title, "crash") and market.probability > 0.3
          end)
          
          if bullish_signals > bearish_signals do
            0.7
          else
            0.4
          end
      end
    end
  end

  defp calculate_risk_indicators(state) do
    %{
      depeg_risk: calculate_depeg_risk(state),
      regulatory_risk: calculate_regulatory_risk(state),
      market_correlation_risk: calculate_correlation_risk(state),
      liquidity_risk: calculate_liquidity_risk(state),
      systemic_risk: calculate_systemic_risk(state)
    }
  end

  defp calculate_depeg_risk(state) do
    depeg_markets = Map.get(state.market_data, "stablecoin_depeg", [])
    
    if length(depeg_markets) == 0 do
      0.1  # Default low risk
    else
      # Average probability of depeg events
      depeg_markets
      |> Enum.map(&(&1.probability))
      |> Enum.sum()
      |> Kernel./(length(depeg_markets))
    end
  end

  defp calculate_regulatory_risk(state) do
    reg_markets = Map.get(state.market_data, "crypto_regulation", [])
    
    if length(reg_markets) == 0 do
      0.2  # Default moderate risk
    else
      # Count negative regulatory outcomes
      negative_outcomes = Enum.count(reg_markets, fn market ->
        title = String.downcase(market.title)
        (String.contains?(title, "ban") or String.contains?(title, "crackdown") or
         String.contains?(title, "restrict")) and market.probability > 0.3
      end)
      
      min(negative_outcomes * 0.2, 1.0)
    end
  end

  defp calculate_correlation_risk(state) do
    # Risk from high correlation between crypto assets
    btc_volatility = get_in(state.processed_insights, [:volatility_forecast, "24h"]) || 0.3
    eth_volatility = get_in(state.processed_insights, [:volatility_forecast, "24h"]) || 0.3
    
    # High volatility in major cryptos increases correlation risk
    (btc_volatility + eth_volatility) / 2
  end

  defp calculate_liquidity_risk(state) do
    # Assess liquidity risk from market volumes
    all_markets = state.market_data
    |> Map.values()
    |> List.flatten()
    
    if length(all_markets) == 0 do
      0.3
    else
      avg_volume = all_markets
      |> Enum.map(&(&1.volume))
      |> Enum.sum()
      |> Kernel./(length(all_markets))
      
      # Low volume indicates high liquidity risk
      max(1.0 - (avg_volume / 10000), 0.1)
    end
  end

  defp calculate_systemic_risk(state) do
    # Combine all risk factors for systemic risk assessment
    depeg_risk = calculate_depeg_risk(state)
    regulatory_risk = calculate_regulatory_risk(state)
    correlation_risk = calculate_correlation_risk(state)
    liquidity_risk = calculate_liquidity_risk(state)
    
    # Weighted combination
    (depeg_risk * 0.4) + (regulatory_risk * 0.3) + 
    (correlation_risk * 0.2) + (liquidity_risk * 0.1)
  end

  defp assess_market_stress(state) do
    volatility = get_in(state.processed_insights, [:volatility_forecast, :overall]) || 0.3
    sentiment = get_in(state.processed_insights, [:sentiment_analysis, :overall]) || 0.5
    systemic_risk = get_in(state.processed_insights, [:risk_indicators, :systemic_risk]) || 0.3
    
    # Market stress increases with volatility and risk, decreases with positive sentiment
    stress_level = (volatility * 0.4) + (systemic_risk * 0.4) + ((1.0 - sentiment) * 0.2)
    
    %{
      level: stress_level,
      status: classify_stress_level(stress_level),
      components: %{
        volatility: volatility,
        sentiment: sentiment,
        systemic_risk: systemic_risk
      }
    }
  end

  defp classify_stress_level(level) when level < 0.3, do: :low
  defp classify_stress_level(level) when level < 0.5, do: :moderate
  defp classify_stress_level(level) when level < 0.7, do: :high
  defp classify_stress_level(_level), do: :critical

  defp perform_deep_market_analysis(state) do
    # More comprehensive analysis performed every 5 minutes
    enhanced_insights = %{
      cross_market_correlations: analyze_cross_market_correlations(state),
      trend_analysis: perform_trend_analysis(state),
      anomaly_detection: detect_market_anomalies(state),
      predictive_models: run_predictive_models(state)
    }
    
    updated_insights = Map.merge(state.processed_insights, enhanced_insights)
    
    %{state | processed_insights: updated_insights}
  end

  defp analyze_cross_market_correlations(state) do
    # Analyze correlations between different market categories
    categories = Map.keys(@market_categories)
    
    correlations = for cat1 <- categories, cat2 <- categories, cat1 != cat2 do
      correlation = calculate_category_correlation(state, cat1, cat2)
      {{cat1, cat2}, correlation}
    end
    |> Map.new()
    
    %{
      correlations: correlations,
      max_correlation: correlations |> Map.values() |> Enum.max(fn -> 0 end),
      systemic_correlation: calculate_systemic_correlation(correlations)
    }
  end

  defp calculate_category_correlation(state, cat1, cat2) do
    # Simplified correlation based on sentiment alignment
    sentiment1 = get_category_sentiment(state, cat1)
    sentiment2 = get_category_sentiment(state, cat2)
    
    # High correlation if sentiments move together
    1.0 - abs(sentiment1 - sentiment2)
  end

  defp get_category_sentiment(state, category) do
    get_in(state.processed_insights, [:sentiment_analysis, :by_category, category]) || 0.5
  end

  defp calculate_systemic_correlation(correlations) do
    if map_size(correlations) == 0 do
      0.0
    else
      correlations
      |> Map.values()
      |> Enum.sum()
      |> Kernel./(map_size(correlations))
    end
  end

  defp perform_trend_analysis(state) do
    # Analyze trends in historical data
    %{
      sentiment_trend: calculate_sentiment_trend(state.historical_data),
      volatility_trend: calculate_volatility_trend(state.historical_data),
      risk_trend: calculate_risk_trend(state.historical_data)
    }
  end

  defp calculate_sentiment_trend(_historical_data) do
    # Placeholder for trend calculation
    0.1  # Slight positive trend
  end

  defp calculate_volatility_trend(_historical_data) do
    # Placeholder for volatility trend
    -0.05  # Slight decreasing volatility
  end

  defp calculate_risk_trend(_historical_data) do
    # Placeholder for risk trend
    0.0  # Stable risk
  end

  defp detect_market_anomalies(state) do
    # Detect unusual patterns in market data
    %{
      sentiment_anomalies: detect_sentiment_anomalies(state),
      volume_anomalies: detect_volume_anomalies(state),
      probability_anomalies: detect_probability_anomalies(state)
    }
  end

  defp detect_sentiment_anomalies(_state) do
    # Placeholder for anomaly detection
    []
  end

  defp detect_volume_anomalies(_state) do
    # Placeholder for volume anomaly detection
    []
  end

  defp detect_probability_anomalies(_state) do
    # Placeholder for probability anomaly detection
    []
  end

  defp run_predictive_models(state) do
    # Run predictive models for future market conditions
    %{
      next_hour_forecast: forecast_next_hour(state),
      next_day_forecast: forecast_next_day(state),
      confidence_intervals: calculate_confidence_intervals(state)
    }
  end

  defp forecast_next_hour(state) do
    # Simplified next hour forecast
    current_volatility = get_in(state.processed_insights, [:volatility_forecast, "1h"]) || 0.3
    %{volatility: current_volatility * 1.1, confidence: 0.6}
  end

  defp forecast_next_day(state) do
    # Simplified next day forecast
    current_volatility = get_in(state.processed_insights, [:volatility_forecast, "24h"]) || 0.3
    %{volatility: current_volatility * 0.9, confidence: 0.4}
  end

  defp calculate_confidence_intervals(_state) do
    # Placeholder for confidence interval calculations
    %{
      volatility: %{lower: 0.2, upper: 0.8},
      sentiment: %{lower: 0.3, upper: 0.7}
    }
  end

  defp check_alert_conditions(state) do
    alerts = []
    
    # Check volatility alert
    volatility = get_in(state.processed_insights, [:volatility_forecast, :overall]) || 0
    alerts = if volatility > state.alert_thresholds.volatility_forecast do
      [create_alert(:high_volatility, volatility, state.alert_thresholds.volatility_forecast) | alerts]
    else
      alerts
    end
    
    # Check depeg risk alert
    depeg_risk = get_in(state.processed_insights, [:risk_indicators, :depeg_risk]) || 0
    alerts = if depeg_risk > state.alert_thresholds.depeg_risk do
      [create_alert(:depeg_risk, depeg_risk, state.alert_thresholds.depeg_risk) | alerts]
    else
      alerts
    end
    
    # Check regulatory risk alert
    reg_risk = get_in(state.processed_insights, [:risk_indicators, :regulatory_risk]) || 0
    alerts = if reg_risk > state.alert_thresholds.regulatory_risk do
      [create_alert(:regulatory_risk, reg_risk, state.alert_thresholds.regulatory_risk) | alerts]
    else
      alerts
    end
    
    alerts
  end

  defp create_alert(type, current_value, threshold) do
    %{
      type: type,
      severity: calculate_alert_severity(current_value, threshold),
      current_value: current_value,
      threshold: threshold,
      timestamp: DateTime.utc_now(),
      message: generate_alert_message(type, current_value, threshold)
    }
  end

  defp calculate_alert_severity(value, threshold) do
    ratio = value / threshold
    cond do
      ratio > 2.0 -> :critical
      ratio > 1.5 -> :high
      ratio > 1.2 -> :medium
      true -> :low
    end
  end

  defp generate_alert_message(type, value, threshold) do
    case type do
      :high_volatility ->
        "High volatility detected: #{Float.round(value * 100, 1)}% (threshold: #{Float.round(threshold * 100, 1)}%)"
      :depeg_risk ->
        "Elevated depeg risk: #{Float.round(value * 100, 1)}% (threshold: #{Float.round(threshold * 100, 1)}%)"
      :regulatory_risk ->
        "Regulatory risk alert: #{Float.round(value * 100, 1)}% (threshold: #{Float.round(threshold * 100, 1)}%)"
    end
  end

  defp broadcast_alerts(alerts) do
    Enum.each(alerts, fn alert ->
      Logger.warning("Kalshi Alert: #{alert.message}")
      # Could broadcast to other systems here
    end)
  end

  defp generate_stability_insights(state) do
    %{
      overall_stability_score: calculate_overall_stability_score(state),
      volatility_forecast: state.processed_insights.volatility_forecast || %{},
      sentiment_analysis: state.processed_insights.sentiment_analysis || %{},
      risk_indicators: state.processed_insights.risk_indicators || %{},
      market_stress: state.processed_insights.market_stress || %{},
      recommendations: generate_stabilization_recommendations(state),
      last_updated: state.last_update
    }
  end

  defp calculate_overall_stability_score(state) do
    # Combine various factors into overall stability score
    volatility = get_in(state.processed_insights, [:volatility_forecast, :overall]) || 0.3
    sentiment = get_in(state.processed_insights, [:sentiment_analysis, :overall]) || 0.5
    depeg_risk = get_in(state.processed_insights, [:risk_indicators, :depeg_risk]) || 0.1
    
    # High sentiment and low volatility/risk = high stability
    stability = (sentiment * 0.4) + ((1.0 - volatility) * 0.4) + ((1.0 - depeg_risk) * 0.2)
    max(min(stability, 1.0), 0.0)
  end

  defp generate_stabilization_recommendations(state) do
    recommendations = []
    
    volatility = get_in(state.processed_insights, [:volatility_forecast, :overall]) || 0.3
    sentiment = get_in(state.processed_insights, [:sentiment_analysis, :overall]) || 0.5
    depeg_risk = get_in(state.processed_insights, [:risk_indicators, :depeg_risk]) || 0.1
    
    # High volatility recommendations
    recommendations = if volatility > 0.6 do
      [%{
        type: :monetary_policy,
        action: :tighten,
        urgency: :high,
        details: "Increase interest rates by 15-25% to reduce volatility exposure"
      } | recommendations]
    else
      recommendations
    end
    
    # High depeg risk recommendations
    recommendations = if depeg_risk > 0.2 do
      [%{
        type: :collateral_management,
        action: :increase_reserves,
        urgency: :medium,
        details: "Increase stability fund reserves by 10-20% as precaution"
      } | recommendations]
    else
      recommendations
    end
    
    # Negative sentiment recommendations
    recommendations = if sentiment < 0.4 do
      [%{
        type: :market_operations,
        action: :defensive_positioning,
        urgency: :medium,
        details: "Prepare for potential market stress with conservative policies"
      } | recommendations]
    else
      recommendations
    end
    
    recommendations
  end

  defp get_volatility_forecast_for_timeframe(state, timeframe) do
    get_in(state.processed_insights, [:volatility_forecast, timeframe]) || 0.3
  end

  defp calculate_current_sentiment(state) do
    get_in(state.processed_insights, [:sentiment_analysis, :overall]) || 0.5
  end

  defp perform_comprehensive_risk_assessment(state) do
    state.processed_insights.risk_indicators || %{}
  end

  defp process_market_update(state, market_data) do
    # Update market data cache
    category = categorize_market(market_data)
    
    updated_markets = Map.update(state.market_data, category, [market_data], fn existing ->
      # Replace or add market data
      existing
      |> Enum.reject(fn m -> m.ticker == market_data.ticker end)
      |> List.insert_at(0, market_data)
      |> Enum.take(50)  # Keep latest 50 markets per category
    end)
    
    %{state | market_data: updated_markets}
  end

  defp add_to_historical_data(historical_data, insights) do
    timestamp = DateTime.utc_now()
    
    # Add current insights to historical data
    Map.update(historical_data, timestamp, insights, fn _ -> insights end)
    |> Enum.take(100)  # Keep last 100 data points
    |> Map.new()
  end

  defp calculate_sentiment_confidence(market_data) do
    # Calculate confidence based on market data quality
    total_markets = market_data |> Map.values() |> List.flatten() |> length()
    total_volume = market_data
    |> Map.values()
    |> List.flatten()
    |> Enum.map(&(&1.volume))
    |> Enum.sum()
    
    # More markets and volume = higher confidence
    market_factor = min(total_markets / 20, 1.0)
    volume_factor = min(total_volume / 100_000, 1.0)
    
    (market_factor + volume_factor) / 2
  end

  defp init_sentiment_tracker do
    %{overall: 0.5, last_update: DateTime.utc_now()}
  end

  defp init_volatility_predictor do
    %{models: [], last_training: DateTime.utc_now()}
  end

  defp init_risk_assessor do
    %{thresholds: @default_thresholds, last_calibration: DateTime.utc_now()}
  end
end