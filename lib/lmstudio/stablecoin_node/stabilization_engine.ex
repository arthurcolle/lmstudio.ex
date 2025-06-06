defmodule LMStudio.StablecoinNode.StabilizationEngine do
  @moduledoc """
  Stabilization engine for maintaining the stablecoin's $1 peg through
  algorithmic monetary policy and market operations.
  """

  use GenServer
  require Logger

  alias LMStudio.StablecoinNode.{Oracle, Blockchain, AIIntelligence}

  @target_price 1.0
  @peg_tolerance 0.01  # 1% tolerance
  @rebalance_threshold 0.005  # 0.5% threshold for rebalancing
  @max_supply_change 0.10  # Maximum 10% supply change per adjustment
  @stability_fee_rate 0.05  # 5% annual stability fee
  @liquidation_ratio 1.5   # 150% collateralization ratio

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
    :ai_intelligence,
    :ml_predictions,
    :market_sentiment,
    :external_factors,
    :smart_adjustments
  ]

  def new do
    %__MODULE__{
      current_price: @target_price,
      target_price: @target_price,
      total_supply: 1_000_000,  # Initial supply
      collateral_pool: %{},
      stability_fund: 100_000,  # Initial stability fund
      last_adjustment: DateTime.utc_now(),
      price_history: [],
      deviation_history: [],
      rebalance_actions: [],
      liquidation_queue: [],
      interest_rate: 0.02,  # 2% base interest rate
      mint_rate: 1.0,
      burn_rate: 1.0,
      ai_intelligence: nil,
      ml_predictions: %{},
      market_sentiment: :neutral,
      external_factors: [],
      smart_adjustments: []
    }
  end

  def start_link do
    GenServer.start_link(__MODULE__, [], name: __MODULE__)
  end

  def check_peg(engine, current_price) do
    GenServer.call(__MODULE__, {:check_peg, engine, current_price})
  end

  def get_stability_metrics(engine) do
    %{
      current_price: engine.current_price,
      price_deviation: calculate_deviation(engine.current_price, engine.target_price),
      total_supply: engine.total_supply,
      collateral_ratio: calculate_collateral_ratio(engine),
      stability_fund: engine.stability_fund,
      interest_rate: engine.interest_rate,
      last_adjustment: engine.last_adjustment,
      peg_stability: assess_peg_stability(engine)
    }
  end

  def init(_) do
    engine = new()
    
    # Initialize AI Intelligence
    {:ok, ai_intelligence} = AIIntelligence.start_link([node_id: "stabilization_engine"])
    enhanced_engine = %{engine | ai_intelligence: ai_intelligence}
    
    # Schedule periodic stability checks with AI enhancement
    :timer.send_interval(10_000, self(), :stability_check)
    :timer.send_interval(60_000, self(), :liquidation_check)
    :timer.send_interval(300_000, self(), :rate_adjustment)
    :timer.send_interval(120_000, self(), :ai_market_analysis)
    :timer.send_interval(180_000, self(), :predictive_adjustment)
    
    Logger.info("Enhanced AI Stabilization Engine initialized")
    {:ok, enhanced_engine}
  end

  def handle_call({:check_peg, engine, current_price}, _from, state) do
    new_engine = %{engine | current_price: current_price}
    |> update_price_history(current_price)
    |> check_and_execute_stabilization()
    
    {:reply, new_engine, new_engine}
  end

  def handle_info(:stability_check, state) do
    new_state = check_and_execute_stabilization(state)
    {:noreply, new_state}
  end

  def handle_info(:liquidation_check, state) do
    new_state = process_liquidations(state)
    {:noreply, new_state}
  end

  def handle_info(:rate_adjustment, state) do
    new_state = adjust_interest_rates(state)
    {:noreply, new_state}
  end

  def handle_info(:ai_market_analysis, state) do
    new_state = perform_ai_market_analysis(state)
    {:noreply, new_state}
  end

  def handle_info(:predictive_adjustment, state) do
    new_state = perform_predictive_adjustment(state)
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

  defp check_and_execute_stabilization(engine) do
    deviation = calculate_deviation(engine.current_price, engine.target_price)
    
    # Get AI recommendation for stabilization
    market_data = %{
      current_price: engine.current_price,
      target_price: engine.target_price,
      total_supply: engine.total_supply,
      volatility: calculate_price_volatility(engine.price_history),
      volume_24h: 1_000_000,  # Would get from real market data
      market_cap: engine.total_supply * engine.current_price,
      trend: determine_price_trend(engine.price_history),
      external_factors: engine.external_factors
    }
    
    current_peg = %{target: engine.target_price}
    
    ai_recommendation = if engine.ai_intelligence do
      AIIntelligence.get_stabilization_recommendation(market_data, current_peg)
    else
      %{"action" => "hold", "urgency" => "low"}
    end
    
    # Combine AI recommendation with traditional approach
    cond do
      abs(deviation) < @rebalance_threshold and ai_recommendation["urgency"] in ["low", "medium"] ->
        # Price is within acceptable range and AI agrees
        engine
        
      deviation > @peg_tolerance or ai_recommendation["action"] in ["burn", "adjust_rates"] ->
        # Price is too high - need to increase supply or decrease demand
        execute_ai_enhanced_upward_correction(engine, ai_recommendation)
        
      deviation < -@peg_tolerance or ai_recommendation["action"] in ["mint", "increase_rates"] ->
        # Price is too low - need to decrease supply or increase demand
        execute_ai_enhanced_downward_correction(engine, ai_recommendation)
        
      true ->
        engine
    end
  end

  defp execute_depeg_upward_correction(engine) do
    Logger.info("Executing upward depeg correction - price too high")
    
    # Strategy 1: Increase supply by minting new tokens
    mint_amount = calculate_mint_amount(engine)
    
    # Strategy 2: Decrease interest rates to reduce demand
    new_interest_rate = max(engine.interest_rate * 0.9, 0.001)
    
    # Strategy 3: Increase mint rate to encourage more minting
    new_mint_rate = min(engine.mint_rate * 1.1, 2.0)
    
    action = %{
      type: :upward_correction,
      timestamp: DateTime.utc_now(),
      mint_amount: mint_amount,
      old_interest_rate: engine.interest_rate,
      new_interest_rate: new_interest_rate,
      old_mint_rate: engine.mint_rate,
      new_mint_rate: new_mint_rate
    }
    
    %{engine |
      total_supply: engine.total_supply + mint_amount,
      interest_rate: new_interest_rate,
      mint_rate: new_mint_rate,
      last_adjustment: DateTime.utc_now(),
      rebalance_actions: [action | engine.rebalance_actions]
    }
  end

  defp execute_depeg_downward_correction(engine) do
    Logger.info("Executing downward depeg correction - price too low")
    
    # Strategy 1: Decrease supply by burning tokens from stability fund
    burn_amount = calculate_burn_amount(engine)
    
    # Strategy 2: Increase interest rates to increase demand
    new_interest_rate = min(engine.interest_rate * 1.1, 0.20)
    
    # Strategy 3: Decrease mint rate to discourage minting
    new_mint_rate = max(engine.mint_rate * 0.9, 0.5)
    
    # Strategy 4: Increase burn incentives
    new_burn_rate = min(engine.burn_rate * 1.05, 1.5)
    
    action = %{
      type: :downward_correction,
      timestamp: DateTime.utc_now(),
      burn_amount: burn_amount,
      old_interest_rate: engine.interest_rate,
      new_interest_rate: new_interest_rate,
      old_mint_rate: engine.mint_rate,
      new_mint_rate: new_mint_rate,
      old_burn_rate: engine.burn_rate,
      new_burn_rate: new_burn_rate
    }
    
    %{engine |
      total_supply: max(engine.total_supply - burn_amount, 100_000),
      stability_fund: max(engine.stability_fund - burn_amount, 0),
      interest_rate: new_interest_rate,
      mint_rate: new_mint_rate,
      burn_rate: new_burn_rate,
      last_adjustment: DateTime.utc_now(),
      rebalance_actions: [action | engine.rebalance_actions]
    }
  end

  defp calculate_mint_amount(engine) do
    deviation = calculate_deviation(engine.current_price, engine.target_price)
    base_mint = engine.total_supply * @max_supply_change
    
    # Scale mint amount based on deviation severity
    scaling_factor = min(abs(deviation) / @peg_tolerance, 2.0)
    trunc(base_mint * scaling_factor)
  end

  defp calculate_burn_amount(engine) do
    deviation = calculate_deviation(engine.current_price, engine.target_price)
    max_burn = min(engine.stability_fund, engine.total_supply * @max_supply_change)
    
    # Scale burn amount based on deviation severity
    scaling_factor = min(abs(deviation) / @peg_tolerance, 2.0)
    trunc(max_burn * scaling_factor)
  end

  defp process_liquidations(engine) do
    # Check for undercollateralized positions
    positions_to_liquidate = engine.collateral_pool
    |> Enum.filter(fn {_address, position} ->
      collateral_ratio = calculate_position_collateral_ratio(position)
      collateral_ratio < @liquidation_ratio
    end)
    
    if length(positions_to_liquidate) > 0 do
      Logger.info("Processing #{length(positions_to_liquidate)} liquidations")
      
      liquidated_collateral = positions_to_liquidate
      |> Enum.map(fn {_address, position} -> position.collateral_value end)
      |> Enum.sum()
      
      liquidated_debt = positions_to_liquidate
      |> Enum.map(fn {_address, position} -> position.debt_amount end)
      |> Enum.sum()
      
      # Remove liquidated positions
      new_collateral_pool = positions_to_liquidate
      |> Enum.reduce(engine.collateral_pool, fn {address, _position}, pool ->
        Map.delete(pool, address)
      end)
      
      # Add liquidated collateral to stability fund
      new_stability_fund = engine.stability_fund + liquidated_collateral - liquidated_debt
      
      liquidation_action = %{
        type: :liquidation,
        timestamp: DateTime.utc_now(),
        positions_liquidated: length(positions_to_liquidate),
        collateral_seized: liquidated_collateral,
        debt_cleared: liquidated_debt
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

  defp adjust_interest_rates(engine) do
    # Adjust interest rates based on price stability and market conditions
    price_volatility = calculate_price_volatility(engine.price_history)
    deviation_trend = calculate_deviation_trend(engine.deviation_history)
    
    adjustment_factor = cond do
      price_volatility > 0.05 -> 1.1  # Increase rates for high volatility
      price_volatility < 0.01 -> 0.95  # Decrease rates for low volatility
      deviation_trend > 0.02 -> 1.05   # Increase rates if consistently above peg
      deviation_trend < -0.02 -> 0.95  # Decrease rates if consistently below peg
      true -> 1.0  # No adjustment
    end
    
    new_interest_rate = engine.interest_rate * adjustment_factor
    |> max(0.001)  # Minimum 0.1%
    |> min(0.25)   # Maximum 25%
    
    if new_interest_rate != engine.interest_rate do
      Logger.info("Adjusting interest rate from #{engine.interest_rate} to #{new_interest_rate}")
      
      action = %{
        type: :interest_rate_adjustment,
        timestamp: DateTime.utc_now(),
        old_rate: engine.interest_rate,
        new_rate: new_interest_rate,
        reason: %{
          volatility: price_volatility,
          trend: deviation_trend,
          adjustment_factor: adjustment_factor
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
    
    cond do
      recent_volatility < 0.01 and recent_deviation < @peg_tolerance -> :stable
      recent_volatility < 0.03 and recent_deviation < @peg_tolerance * 2 -> :moderate
      recent_volatility < 0.05 -> :unstable
      true -> :critical
    end
  end

  # AI-Enhanced stabilization functions
  defp execute_ai_enhanced_upward_correction(engine, ai_recommendation) do
    Logger.info("Executing AI-enhanced upward depeg correction - price too high")
    Logger.info("AI Recommendation: #{inspect(ai_recommendation)}")
    
    # Calculate AI-adjusted parameters
    ai_mint_multiplier = case ai_recommendation["urgency"] do
      "critical" -> 1.5
      "high" -> 1.2
      "medium" -> 1.0
      _ -> 0.8
    end
    
    mint_amount = calculate_ai_enhanced_mint_amount(engine, ai_recommendation, ai_mint_multiplier)
    
    # AI-suggested interest rate adjustment
    ai_rate_adjustment = case ai_recommendation["action"] do
      "adjust_rates" -> 0.85  # More aggressive rate reduction
      _ -> 0.9
    end
    
    new_interest_rate = max(engine.interest_rate * ai_rate_adjustment, 0.001)
    new_mint_rate = min(engine.mint_rate * (1.0 + ai_mint_multiplier * 0.1), 2.0)
    
    action = %{
      type: :ai_enhanced_upward_correction,
      timestamp: DateTime.utc_now(),
      mint_amount: mint_amount,
      old_interest_rate: engine.interest_rate,
      new_interest_rate: new_interest_rate,
      old_mint_rate: engine.mint_rate,
      new_mint_rate: new_mint_rate,
      ai_recommendation: ai_recommendation,
      ai_confidence: Map.get(ai_recommendation, "confidence", 0.5)
    }
    
    %{engine |
      total_supply: engine.total_supply + mint_amount,
      interest_rate: new_interest_rate,
      mint_rate: new_mint_rate,
      last_adjustment: DateTime.utc_now(),
      rebalance_actions: [action | engine.rebalance_actions],
      smart_adjustments: [action | engine.smart_adjustments]
    }
  end

  defp execute_ai_enhanced_downward_correction(engine, ai_recommendation) do
    Logger.info("Executing AI-enhanced downward depeg correction - price too low")
    Logger.info("AI Recommendation: #{inspect(ai_recommendation)}")
    
    # Calculate AI-adjusted parameters
    ai_burn_multiplier = case ai_recommendation["urgency"] do
      "critical" -> 1.5
      "high" -> 1.2
      "medium" -> 1.0
      _ -> 0.8
    end
    
    burn_amount = calculate_ai_enhanced_burn_amount(engine, ai_recommendation, ai_burn_multiplier)
    
    # AI-suggested rate adjustments
    ai_rate_multiplier = case ai_recommendation["action"] do
      "increase_rates" -> 1.15  # More aggressive rate increase
      _ -> 1.1
    end
    
    new_interest_rate = min(engine.interest_rate * ai_rate_multiplier, 0.20)
    new_mint_rate = max(engine.mint_rate * 0.9, 0.5)
    new_burn_rate = min(engine.burn_rate * (1.0 + ai_burn_multiplier * 0.05), 1.5)
    
    action = %{
      type: :ai_enhanced_downward_correction,
      timestamp: DateTime.utc_now(),
      burn_amount: burn_amount,
      old_interest_rate: engine.interest_rate,
      new_interest_rate: new_interest_rate,
      old_mint_rate: engine.mint_rate,
      new_mint_rate: new_mint_rate,
      old_burn_rate: engine.burn_rate,
      new_burn_rate: new_burn_rate,
      ai_recommendation: ai_recommendation,
      ai_confidence: Map.get(ai_recommendation, "confidence", 0.5)
    }
    
    %{engine |
      total_supply: max(engine.total_supply - burn_amount, 100_000),
      stability_fund: max(engine.stability_fund - burn_amount, 0),
      interest_rate: new_interest_rate,
      mint_rate: new_mint_rate,
      burn_rate: new_burn_rate,
      last_adjustment: DateTime.utc_now(),
      rebalance_actions: [action | engine.rebalance_actions],
      smart_adjustments: [action | engine.smart_adjustments]
    }
  end

  defp calculate_ai_enhanced_mint_amount(engine, ai_recommendation, multiplier) do
    base_amount = calculate_mint_amount(engine)
    ai_amount = Map.get(ai_recommendation, "amount", base_amount)
    
    # Blend traditional calculation with AI suggestion
    blended_amount = (base_amount * 0.6 + ai_amount * 0.4) * multiplier
    trunc(blended_amount)
  end

  defp calculate_ai_enhanced_burn_amount(engine, ai_recommendation, multiplier) do
    base_amount = calculate_burn_amount(engine)
    ai_amount = Map.get(ai_recommendation, "amount", base_amount)
    
    # Blend traditional calculation with AI suggestion
    blended_amount = (base_amount * 0.6 + ai_amount * 0.4) * multiplier
    max_burn = min(engine.stability_fund, engine.total_supply * @max_supply_change * multiplier)
    
    trunc(min(blended_amount, max_burn))
  end

  defp perform_ai_market_analysis(engine) do
    if engine.ai_intelligence do
      # Get AI market prediction
      prediction = AIIntelligence.predict_market_movement(3600)  # 1 hour prediction
      
      # Update market sentiment based on AI analysis
      new_sentiment = case Map.get(prediction, "price_direction", "stable") do
        "bullish" -> :bullish
        "bearish" -> :bearish
        _ -> :neutral
      end
      
      # Extract external factors from AI analysis
      new_factors = Map.get(prediction, "risk_factors", [])
      
      # Store ML predictions
      new_predictions = Map.put(engine.ml_predictions, DateTime.utc_now(), prediction)
      |> Enum.take(24)  # Keep last 24 predictions
      |> Map.new()
      
      Logger.debug("AI Market Analysis: sentiment=#{new_sentiment}, factors=#{length(new_factors)}")
      
      %{engine |
        market_sentiment: new_sentiment,
        external_factors: new_factors,
        ml_predictions: new_predictions
      }
    else
      engine
    end
  end

  defp perform_predictive_adjustment(engine) do
    if engine.ai_intelligence and map_size(engine.ml_predictions) > 3 do
      # Analyze recent predictions for patterns
      recent_predictions = engine.ml_predictions
      |> Enum.sort_by(fn {timestamp, _} -> timestamp end, &>=/2)
      |> Enum.take(5)
      |> Enum.map(fn {_, prediction} -> prediction end)
      
      # Check for consistent bearish/bullish predictions
      bullish_count = Enum.count(recent_predictions, &(Map.get(&1, "price_direction") == "bullish"))
      bearish_count = Enum.count(recent_predictions, &(Map.get(&1, "price_direction") == "bearish"))
      
      # Proactive adjustment based on predictions
      cond do
        bullish_count >= 4 ->
          # Strong bullish trend predicted - prepare for potential upward pressure
          perform_proactive_bullish_adjustment(engine)
          
        bearish_count >= 4 ->
          # Strong bearish trend predicted - prepare for potential downward pressure
          perform_proactive_bearish_adjustment(engine)
          
        true ->
          engine
      end
    else
      engine
    end
  end

  defp perform_proactive_bullish_adjustment(engine) do
    Logger.info("Performing proactive adjustment for predicted bullish trend")
    
    # Slightly reduce interest rates to prepare for potential selling pressure
    new_interest_rate = max(engine.interest_rate * 0.98, 0.001)
    
    # Increase mint readiness
    new_mint_rate = min(engine.mint_rate * 1.02, 2.0)
    
    action = %{
      type: :proactive_bullish_adjustment,
      timestamp: DateTime.utc_now(),
      old_interest_rate: engine.interest_rate,
      new_interest_rate: new_interest_rate,
      old_mint_rate: engine.mint_rate,
      new_mint_rate: new_mint_rate,
      reasoning: "AI predicted bullish trend - preparing for upward pressure"
    }
    
    %{engine |
      interest_rate: new_interest_rate,
      mint_rate: new_mint_rate,
      smart_adjustments: [action | engine.smart_adjustments]
    }
  end

  defp perform_proactive_bearish_adjustment(engine) do
    Logger.info("Performing proactive adjustment for predicted bearish trend")
    
    # Slightly increase interest rates to prepare for potential buying pressure
    new_interest_rate = min(engine.interest_rate * 1.02, 0.20)
    
    # Increase burn readiness
    new_burn_rate = min(engine.burn_rate * 1.02, 1.5)
    
    action = %{
      type: :proactive_bearish_adjustment,
      timestamp: DateTime.utc_now(),
      old_interest_rate: engine.interest_rate,
      new_interest_rate: new_interest_rate,
      old_burn_rate: engine.burn_rate,
      new_burn_rate: new_burn_rate,
      reasoning: "AI predicted bearish trend - preparing for downward pressure"
    }
    
    %{engine |
      interest_rate: new_interest_rate,
      burn_rate: new_burn_rate,
      smart_adjustments: [action | engine.smart_adjustments]
    }
  end

  defp determine_price_trend(price_history) when length(price_history) < 5, do: :stable
  defp determine_price_trend(price_history) do
    recent_prices = Enum.take(price_history, 10)
    
    if length(recent_prices) >= 2 do
      [latest | rest] = recent_prices
      average_previous = Enum.sum(rest) / length(rest)
      
      change_percent = (latest - average_previous) / average_previous
      
      cond do
        change_percent > 0.02 -> :bullish
        change_percent < -0.02 -> :bearish
        true -> :stable
      end
    else
      :stable
    end
  end
end