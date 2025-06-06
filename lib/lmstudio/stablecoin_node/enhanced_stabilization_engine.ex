defmodule LMStudio.StablecoinNode.EnhancedStabilizationEngine do
  @moduledoc """
  Enhanced stabilization engine that uses Kalshi prediction market data
  to anticipate market movements and proactively adjust monetary policy.
  """

  use GenServer
  require Logger

  alias LMStudio.StablecoinNode.KalshiOracle

  @target_price 1.0
  @peg_tolerance 0.01  # 1% tolerance
  @rebalance_threshold 0.005  # 0.5% threshold for rebalancing
  @max_supply_change 0.15  # Maximum 15% supply change per adjustment
  @prediction_weight 0.3   # Weight of prediction market data in decisions

  defstruct [
    :current_price,
    :target_price,
    :total_supply,
    :collateral_pool,
    :stability_fund,
    :last_adjustment,
    :price_history,
    :deviation_history,
    :rebalance_actions,
    :liquidation_queue,
    :interest_rate,
    :mint_rate,
    :burn_rate,
    :prediction_data,
    :risk_assessment,
    :proactive_adjustments
  ]

  def new do
    %__MODULE__{
      current_price: @target_price,
      target_price: @target_price,
      total_supply: 1_000_000,
      collateral_pool: %{},
      stability_fund: 100_000,
      last_adjustment: DateTime.utc_now(),
      price_history: [],
      deviation_history: [],
      rebalance_actions: [],
      liquidation_queue: [],
      interest_rate: 0.02,
      mint_rate: 1.0,
      burn_rate: 1.0,
      prediction_data: %{},
      risk_assessment: %{},
      proactive_adjustments: []
    }
  end

  def start_link do
    GenServer.start_link(__MODULE__, [], name: __MODULE__)
  end

  def check_peg_with_predictions(engine, current_price, kalshi_data) do
    GenServer.call(__MODULE__, {:check_peg_with_predictions, engine, current_price, kalshi_data})
  end

  def get_enhanced_stability_metrics(engine) do
    %{
      current_price: engine.current_price,
      price_deviation: calculate_deviation(engine.current_price, engine.target_price),
      total_supply: engine.total_supply,
      collateral_ratio: calculate_collateral_ratio(engine),
      stability_fund: engine.stability_fund,
      interest_rate: engine.interest_rate,
      last_adjustment: engine.last_adjustment,
      peg_stability: assess_peg_stability(engine),
      prediction_sentiment: Map.get(engine.prediction_data, :market_sentiment, :neutral),
      volatility_forecast: Map.get(engine.prediction_data, :volatility_24h, 0.5),
      regulatory_risk: Map.get(engine.prediction_data, :regulatory_risk, 0.1),
      stability_score: Map.get(engine.prediction_data, :stability_score, 0.7),
      proactive_mode: length(engine.proactive_adjustments) > 0,
      risk_level: assess_risk_level(engine)
    }
  end

  def init(_) do
    engine = new()
    
    # Schedule periodic stability checks with prediction data
    :timer.send_interval(10_000, self(), :stability_check_with_predictions)
    :timer.send_interval(60_000, self(), :liquidation_check)
    :timer.send_interval(300_000, self(), :rate_adjustment_with_predictions)
    :timer.send_interval(30_000, self(), :update_prediction_data)
    
    {:ok, engine}
  end

  def handle_call({:check_peg_with_predictions, engine, current_price, kalshi_data}, _from, state) do
    new_engine = %{engine | 
      current_price: current_price,
      prediction_data: kalshi_data
    }
    |> update_price_history(current_price)
    |> update_risk_assessment(kalshi_data)
    |> check_and_execute_enhanced_stabilization()
    
    {:reply, new_engine, new_engine}
  end

  def handle_info(:stability_check_with_predictions, state) do
    # Get latest Kalshi data
    kalshi_indicators = try do
      KalshiOracle.get_stability_indicators(%KalshiOracle{})
    rescue
      _ -> %{market_sentiment: :neutral, volatility_24h: 0.5, regulatory_risk: 0.1}
    end
    
    new_state = %{state | prediction_data: kalshi_indicators}
    |> update_risk_assessment(kalshi_indicators)
    |> check_and_execute_enhanced_stabilization()
    
    {:noreply, new_state}
  end

  def handle_info(:update_prediction_data, state) do
    # Periodically update prediction market data
    kalshi_indicators = try do
      KalshiOracle.get_stability_indicators(%KalshiOracle{})
    rescue
      _ -> state.prediction_data
    end
    
    new_state = %{state | prediction_data: kalshi_indicators}
    |> update_risk_assessment(kalshi_indicators)
    
    {:noreply, new_state}
  end

  def handle_info(:rate_adjustment_with_predictions, state) do
    new_state = adjust_interest_rates_with_predictions(state)
    {:noreply, new_state}
  end

  def handle_info(:liquidation_check, state) do
    new_state = process_liquidations_with_predictions(state)
    {:noreply, new_state}
  end

  defp update_price_history(engine, price) do
    new_history = [price | engine.price_history] |> Enum.take(100)
    deviation = calculate_deviation(price, engine.target_price)
    new_deviation_history = [deviation | engine.deviation_history] |> Enum.take(100)
    
    %{engine | 
      price_history: new_history,
      deviation_history: new_deviation_history
    }
  end

  defp update_risk_assessment(engine, kalshi_data) do
    # Enhanced risk assessment using prediction market data
    price_risk = abs(calculate_deviation(engine.current_price, engine.target_price))
    volatility_risk = Map.get(kalshi_data, :volatility_24h, 0.5)
    regulatory_risk = Map.get(kalshi_data, :regulatory_risk, 0.1)
    sentiment_risk = case Map.get(kalshi_data, :market_sentiment, :neutral) do
      :bullish -> 0.6  # Moderate risk from potential bubbles
      :bearish -> 0.8  # High risk from potential crashes
      :neutral -> 0.3  # Low risk
    end
    
    # Composite risk score
    composite_risk = (price_risk * 0.4) + (volatility_risk * 0.3) + 
                    (regulatory_risk * 0.2) + (sentiment_risk * 0.1)
    
    risk_assessment = %{
      price_risk: price_risk,
      volatility_risk: volatility_risk,
      regulatory_risk: regulatory_risk,
      sentiment_risk: sentiment_risk,
      composite_risk: composite_risk,
      risk_level: classify_risk_level(composite_risk),
      last_updated: DateTime.utc_now()
    }
    
    %{engine | risk_assessment: risk_assessment}
  end

  defp check_and_execute_enhanced_stabilization(engine) do
    deviation = calculate_deviation(engine.current_price, engine.target_price)
    volatility_forecast = Map.get(engine.prediction_data, :volatility_24h, 0.5)
    sentiment = Map.get(engine.prediction_data, :market_sentiment, :neutral)
    
    # Proactive adjustments based on predictions
    proactive_threshold = @rebalance_threshold * (1 - @prediction_weight)
    
    cond do
      # High volatility predicted - preemptive tightening
      volatility_forecast > 0.7 and abs(deviation) > proactive_threshold ->
        execute_preemptive_stabilization(engine, :high_volatility)
        
      # Bearish sentiment with approaching threshold
      sentiment == :bearish and abs(deviation) > proactive_threshold * 1.5 ->
        execute_preemptive_stabilization(engine, :bearish_sentiment)
        
      # Standard threshold reached
      abs(deviation) > @rebalance_threshold ->
        execute_standard_stabilization(engine, deviation)
        
      # Current price within tolerance
      abs(deviation) < @rebalance_threshold ->
        engine
        
      true ->
        engine
    end
  end

  defp execute_preemptive_stabilization(engine, reason) do
    Logger.info("Executing preemptive stabilization due to: #{reason}")
    
    case reason do
      :high_volatility ->
        # Reduce volatility by tightening monetary policy
        new_interest_rate = min(engine.interest_rate * 1.2, 0.25)
        new_mint_rate = max(engine.mint_rate * 0.8, 0.3)
        
        action = %{
          type: :preemptive_volatility_control,
          timestamp: DateTime.utc_now(),
          reason: reason,
          old_interest_rate: engine.interest_rate,
          new_interest_rate: new_interest_rate,
          old_mint_rate: engine.mint_rate,
          new_mint_rate: new_mint_rate,
          prediction_trigger: Map.get(engine.prediction_data, :volatility_24h, 0.5)
        }
        
        %{engine |
          interest_rate: new_interest_rate,
          mint_rate: new_mint_rate,
          last_adjustment: DateTime.utc_now(),
          proactive_adjustments: [action | engine.proactive_adjustments],
          rebalance_actions: [action | engine.rebalance_actions]
        }
        
      :bearish_sentiment ->
        # Prepare for potential selling pressure
        burn_amount = trunc(engine.total_supply * 0.05)  # Preemptive 5% supply reduction
        
        action = %{
          type: :preemptive_sentiment_response,
          timestamp: DateTime.utc_now(),
          reason: reason,
          burn_amount: burn_amount,
          market_sentiment: Map.get(engine.prediction_data, :market_sentiment),
          stability_fund_used: min(burn_amount, engine.stability_fund)
        }
        
        %{engine |
          total_supply: max(engine.total_supply - burn_amount, 100_000),
          stability_fund: max(engine.stability_fund - burn_amount, 0),
          last_adjustment: DateTime.utc_now(),
          proactive_adjustments: [action | engine.proactive_adjustments],
          rebalance_actions: [action | engine.rebalance_actions]
        }
    end
  end

  defp execute_standard_stabilization(engine, deviation) do
    cond do
      deviation > @peg_tolerance ->
        execute_depeg_upward_correction_enhanced(engine)
        
      deviation < -@peg_tolerance ->
        execute_depeg_downward_correction_enhanced(engine)
        
      true ->
        engine
    end
  end

  defp execute_depeg_upward_correction_enhanced(engine) do
    Logger.info("Executing enhanced upward depeg correction")
    
    # Enhanced correction using prediction data
    volatility_multiplier = 1 + Map.get(engine.prediction_data, :volatility_24h, 0.5) * 0.5
    sentiment_multiplier = case Map.get(engine.prediction_data, :market_sentiment) do
      :bullish -> 1.3  # More aggressive correction in bull market
      :neutral -> 1.0
      :bearish -> 0.7  # Less aggressive in bear market
    end
    
    base_mint_amount = calculate_enhanced_mint_amount(engine)
    adjusted_mint_amount = trunc(base_mint_amount * volatility_multiplier * sentiment_multiplier)
    
    new_interest_rate = max(engine.interest_rate * 0.85, 0.001)
    new_mint_rate = min(engine.mint_rate * 1.15, 2.0)
    
    action = %{
      type: :enhanced_upward_correction,
      timestamp: DateTime.utc_now(),
      mint_amount: adjusted_mint_amount,
      old_interest_rate: engine.interest_rate,
      new_interest_rate: new_interest_rate,
      old_mint_rate: engine.mint_rate,
      new_mint_rate: new_mint_rate,
      volatility_multiplier: volatility_multiplier,
      sentiment_multiplier: sentiment_multiplier,
      prediction_factors: engine.prediction_data
    }
    
    %{engine |
      total_supply: engine.total_supply + adjusted_mint_amount,
      interest_rate: new_interest_rate,
      mint_rate: new_mint_rate,
      last_adjustment: DateTime.utc_now(),
      rebalance_actions: [action | engine.rebalance_actions]
    }
  end

  defp execute_depeg_downward_correction_enhanced(engine) do
    Logger.info("Executing enhanced downward depeg correction")
    
    # Enhanced correction using prediction data
    volatility_multiplier = 1 + Map.get(engine.prediction_data, :volatility_24h, 0.5) * 0.7
    regulatory_multiplier = 1 + Map.get(engine.prediction_data, :regulatory_risk, 0.1) * 0.5
    
    base_burn_amount = calculate_enhanced_burn_amount(engine)
    adjusted_burn_amount = trunc(base_burn_amount * volatility_multiplier * regulatory_multiplier)
    
    new_interest_rate = min(engine.interest_rate * 1.15, 0.25)
    new_mint_rate = max(engine.mint_rate * 0.85, 0.5)
    new_burn_rate = min(engine.burn_rate * 1.1, 1.5)
    
    action = %{
      type: :enhanced_downward_correction,
      timestamp: DateTime.utc_now(),
      burn_amount: adjusted_burn_amount,
      old_interest_rate: engine.interest_rate,
      new_interest_rate: new_interest_rate,
      old_mint_rate: engine.mint_rate,
      new_mint_rate: new_mint_rate,
      old_burn_rate: engine.burn_rate,
      new_burn_rate: new_burn_rate,
      volatility_multiplier: volatility_multiplier,
      regulatory_multiplier: regulatory_multiplier,
      prediction_factors: engine.prediction_data
    }
    
    %{engine |
      total_supply: max(engine.total_supply - adjusted_burn_amount, 100_000),
      stability_fund: max(engine.stability_fund - adjusted_burn_amount, 0),
      interest_rate: new_interest_rate,
      mint_rate: new_mint_rate,
      burn_rate: new_burn_rate,
      last_adjustment: DateTime.utc_now(),
      rebalance_actions: [action | engine.rebalance_actions]
    }
  end

  defp calculate_enhanced_mint_amount(engine) do
    deviation = calculate_deviation(engine.current_price, engine.target_price)
    base_mint = engine.total_supply * @max_supply_change
    
    # Enhance with prediction data
    market_confidence = Map.get(engine.prediction_data, :stability_score, 0.7)
    confidence_multiplier = 2.0 - market_confidence  # Lower confidence = more aggressive
    
    scaling_factor = min(abs(deviation) / @peg_tolerance, 2.0) * confidence_multiplier
    trunc(base_mint * scaling_factor)
  end

  defp calculate_enhanced_burn_amount(engine) do
    deviation = calculate_deviation(engine.current_price, engine.target_price)
    max_burn = min(engine.stability_fund, engine.total_supply * @max_supply_change)
    
    # Enhance with prediction data
    regulatory_risk = Map.get(engine.prediction_data, :regulatory_risk, 0.1)
    risk_multiplier = 1 + regulatory_risk
    
    scaling_factor = min(abs(deviation) / @peg_tolerance, 2.0) * risk_multiplier
    trunc(max_burn * scaling_factor)
  end

  defp adjust_interest_rates_with_predictions(engine) do
    # Enhanced interest rate adjustment using prediction markets
    volatility_forecast = Map.get(engine.prediction_data, :volatility_24h, 0.5)
    regulatory_uncertainty = Map.get(engine.prediction_data, :regulatory_risk, 0.1)
    market_sentiment = Map.get(engine.prediction_data, :market_sentiment, :neutral)
    
    # Base adjustment
    price_volatility = calculate_price_volatility(engine.price_history)
    deviation_trend = calculate_deviation_trend(engine.deviation_history)
    
    # Prediction-based adjustments
    prediction_factor = case market_sentiment do
      :bullish -> 1.1 + volatility_forecast * 0.2
      :bearish -> 0.9 - volatility_forecast * 0.2
      :neutral -> 1.0
    end
    
    regulatory_factor = 1.0 + regulatory_uncertainty * 0.3
    
    combined_adjustment_factor = cond do
      price_volatility > 0.05 -> 1.1 * prediction_factor * regulatory_factor
      price_volatility < 0.01 -> 0.95 * prediction_factor
      deviation_trend > 0.02 -> 1.05 * prediction_factor * regulatory_factor
      deviation_trend < -0.02 -> 0.95 * prediction_factor
      true -> prediction_factor
    end
    
    new_interest_rate = engine.interest_rate * combined_adjustment_factor
    |> max(0.001)
    |> min(0.30)
    
    if abs(new_interest_rate - engine.interest_rate) > 0.001 do
      Logger.info("Adjusting interest rate with predictions: #{engine.interest_rate} -> #{new_interest_rate}")
      
      action = %{
        type: :prediction_based_rate_adjustment,
        timestamp: DateTime.utc_now(),
        old_rate: engine.interest_rate,
        new_rate: new_interest_rate,
        prediction_factors: %{
          volatility_forecast: volatility_forecast,
          regulatory_uncertainty: regulatory_uncertainty,
          market_sentiment: market_sentiment,
          combined_factor: combined_adjustment_factor
        }
      }
      
      %{engine |
        interest_rate: new_interest_rate,
        rebalance_actions: [action | engine.rebalance_actions]
      }
    else
      engine
    end
  end

  defp process_liquidations_with_predictions(engine) do
    # Enhanced liquidation process considering market predictions
    volatility_forecast = Map.get(engine.prediction_data, :volatility_24h, 0.5)
    
    # Adjust liquidation threshold based on predicted volatility
    base_threshold = 1.5
    adjusted_threshold = base_threshold + (volatility_forecast * 0.3)
    
    positions_to_liquidate = engine.collateral_pool
    |> Enum.filter(fn {_address, position} ->
      collateral_ratio = calculate_position_collateral_ratio(position)
      collateral_ratio < adjusted_threshold
    end)
    
    if length(positions_to_liquidate) > 0 do
      Logger.info("Processing #{length(positions_to_liquidate)} liquidations with enhanced threshold #{adjusted_threshold}")
      
      liquidated_collateral = positions_to_liquidate
      |> Enum.map(fn {_address, position} -> position.collateral_value end)
      |> Enum.sum()
      
      liquidated_debt = positions_to_liquidate
      |> Enum.map(fn {_address, position} -> position.debt_amount end)
      |> Enum.sum()
      
      new_collateral_pool = positions_to_liquidate
      |> Enum.reduce(engine.collateral_pool, fn {address, _position}, pool ->
        Map.delete(pool, address)
      end)
      
      new_stability_fund = engine.stability_fund + liquidated_collateral - liquidated_debt
      
      liquidation_action = %{
        type: :prediction_enhanced_liquidation,
        timestamp: DateTime.utc_now(),
        positions_liquidated: length(positions_to_liquidate),
        collateral_seized: liquidated_collateral,
        debt_cleared: liquidated_debt,
        adjusted_threshold: adjusted_threshold,
        volatility_forecast: volatility_forecast
      }
      
      %{engine |
        collateral_pool: new_collateral_pool,
        stability_fund: new_stability_fund,
        total_supply: engine.total_supply - liquidated_debt,
        rebalance_actions: [liquidation_action | engine.rebalance_actions]
      }
    else
      engine
    end
  end

  # Helper functions (reused from base implementation)
  defp calculate_deviation(current_price, target_price) do
    (current_price - target_price) / target_price
  end

  defp calculate_collateral_ratio(engine) do
    total_collateral = engine.collateral_pool
    |> Enum.map(fn {_address, position} -> position.collateral_value end)
    |> Enum.sum()
    
    total_debt = engine.collateral_pool
    |> Enum.map(fn {_address, position} -> position.debt_amount end)
    |> Enum.sum()
    
    if total_debt > 0, do: total_collateral / total_debt, else: 0
  end

  defp calculate_position_collateral_ratio(position) do
    if position.debt_amount > 0 do
      position.collateral_value / position.debt_amount
    else
      Float.max_finite()
    end
  end

  defp calculate_price_volatility(price_history) when length(price_history) < 10, do: 0.0
  defp calculate_price_volatility(price_history) do
    returns = price_history
    |> Enum.chunk_every(2, 1, :discard)
    |> Enum.map(fn [current, previous] -> 
      if previous != 0, do: (current - previous) / previous, else: 0
    end)
    
    if length(returns) > 0 do
      mean_return = Enum.sum(returns) / length(returns)
      variance = returns
      |> Enum.map(fn return -> :math.pow(return - mean_return, 2) end)
      |> Enum.sum()
      |> Kernel./(length(returns))
      
      :math.sqrt(variance)
    else
      0.0
    end
  end

  defp calculate_deviation_trend(deviation_history) when length(deviation_history) < 5, do: 0.0
  defp calculate_deviation_trend(deviation_history) do
    recent_deviations = Enum.take(deviation_history, 10)
    if length(recent_deviations) > 0 do
      Enum.sum(recent_deviations) / length(recent_deviations)
    else
      0.0
    end
  end

  defp assess_peg_stability(engine) do
    recent_volatility = calculate_price_volatility(Enum.take(engine.price_history, 20))
    recent_deviation = if length(engine.deviation_history) > 0 do
      abs(hd(engine.deviation_history))
    else
      0.0
    end
    
    prediction_stability = Map.get(engine.prediction_data, :stability_score, 0.7)
    
    # Enhanced stability assessment
    combined_stability = (prediction_stability * 0.4) + 
                        ((1.0 - recent_volatility) * 0.3) + 
                        ((1.0 - recent_deviation) * 0.3)
    
    cond do
      combined_stability > 0.8 -> :stable
      combined_stability > 0.6 -> :moderate  
      combined_stability > 0.4 -> :unstable
      true -> :critical
    end
  end

  defp classify_risk_level(composite_risk) do
    cond do
      composite_risk < 0.3 -> :low
      composite_risk < 0.5 -> :medium
      composite_risk < 0.7 -> :high
      true -> :critical
    end
  end

  defp assess_risk_level(engine) do
    Map.get(engine.risk_assessment, :risk_level, :medium)
  end
end