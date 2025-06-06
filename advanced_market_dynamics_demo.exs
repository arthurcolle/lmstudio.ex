#!/usr/bin/env elixir

Mix.install([
  {:jason, "~> 1.4"}
])

defmodule AdvancedMarketDynamicsDemo do
  @moduledoc """
  Complete stablecoin ecosystem with market makers, real-world currency impacts,
  arbitrage opportunities, liquidity pools, and realistic market dynamics.
  
  Models:
  - Market maker agents with different strategies
  - Real-world currency pair impacts (USD, EUR, JPY, etc.)
  - Crypto pair dynamics (BTC, ETH, etc.)
  - Arbitrage opportunities and slippage
  - Liquidity pools and AMM mechanics
  - Order book dynamics with market impact
  - Regulatory and macroeconomic factors
  """

  require Logger
  use GenServer

  # Market constants
  @initial_stablecoin_price 1.0000
  @target_peg 1.0000
  @volatility_threshold 0.005  # 0.5%
  @max_slippage 0.02  # 2%
  @liquidity_pool_fee 0.003  # 0.3%

  defmodule MarketMaker do
    @moduledoc "Individual market maker agent with strategy and capital"
    
    defstruct [
      :id,
      :strategy,  # :conservative, :aggressive, :arbitrage, :high_frequency
      :capital_usd,
      :capital_stablecoin,
      :capital_btc,
      :capital_eth,
      :max_position_size,
      :risk_tolerance,
      :active_orders,
      :pnl,
      :trade_count,
      :last_trade_time
    ]

    def new(id, strategy, initial_capital) do
      %__MODULE__{
        id: id,
        strategy: strategy,
        capital_usd: initial_capital,
        capital_stablecoin: initial_capital * 0.3,
        capital_btc: initial_capital * 0.2 / 45000,
        capital_eth: initial_capital * 0.2 / 3200,
        max_position_size: initial_capital * case strategy do
          :conservative -> 0.1
          :aggressive -> 0.4
          :arbitrage -> 0.6
          :high_frequency -> 0.8
        end,
        risk_tolerance: case strategy do
          :conservative -> 0.002
          :aggressive -> 0.01
          :arbitrage -> 0.005
          :high_frequency -> 0.003
        end,
        active_orders: [],
        pnl: 0.0,
        trade_count: 0,
        last_trade_time: DateTime.utc_now()
      }
    end
  end

  defmodule OrderBook do
    @moduledoc "Order book with bids and asks"
    
    defstruct [
      :pair,  # e.g., "STABLE/USD", "STABLE/BTC"
      :bids,  # [{price, quantity, maker_id}]
      :asks,  # [{price, quantity, maker_id}]
      :last_price,
      :volume_24h,
      :price_impact_model
    ]

    def new(pair, initial_price) do
      %__MODULE__{
        pair: pair,
        bids: generate_initial_orders(:bid, initial_price),
        asks: generate_initial_orders(:ask, initial_price),
        last_price: initial_price,
        volume_24h: 0.0,
        price_impact_model: calculate_price_impact_model(pair)
      }
    end

    defp generate_initial_orders(:bid, price) do
      # Generate realistic bid ladder
      for i <- 1..20 do
        offset = i * 0.0001 + :rand.uniform() * 0.0002
        quantity = 1000 + :rand.uniform(5000)
        {price - offset, quantity, "initial_liquidity"}
      end
      |> Enum.sort_by(&elem(&1, 0), :desc)
    end

    defp generate_initial_orders(:ask, price) do
      # Generate realistic ask ladder
      for i <- 1..20 do
        offset = i * 0.0001 + :rand.uniform() * 0.0002
        quantity = 1000 + :rand.uniform(5000)
        {price + offset, quantity, "initial_liquidity"}
      end
      |> Enum.sort_by(&elem(&1, 0), :asc)
    end

    defp calculate_price_impact_model(pair) do
      # Different pairs have different liquidity characteristics
      case pair do
        "STABLE/USD" -> %{depth: 1_000_000, impact_factor: 0.0001}
        "STABLE/BTC" -> %{depth: 500_000, impact_factor: 0.0003}
        "STABLE/ETH" -> %{depth: 750_000, impact_factor: 0.0002}
        _ -> %{depth: 250_000, impact_factor: 0.0005}
      end
    end
  end

  defmodule LiquidityPool do
    @moduledoc "AMM liquidity pool for stablecoin pairs"
    
    defstruct [
      :pair,
      :token_a_reserve,
      :token_b_reserve,
      :total_shares,
      :fee_rate,
      :providers,  # {address => shares}
      :volume_24h,
      :fees_collected
    ]

    def new(pair, initial_a, initial_b) do
      %__MODULE__{
        pair: pair,
        token_a_reserve: initial_a,
        token_b_reserve: initial_b,
        total_shares: :math.sqrt(initial_a * initial_b),
        fee_rate: @liquidity_pool_fee,
        providers: %{},
        volume_24h: 0.0,
        fees_collected: 0.0
      }
    end

    def get_price(pool) do
      pool.token_b_reserve / pool.token_a_reserve
    end

    def calculate_swap_output(pool, input_amount, input_is_token_a \\ true) do
      {input_reserve, output_reserve} = if input_is_token_a do
        {pool.token_a_reserve, pool.token_b_reserve}
      else
        {pool.token_b_reserve, pool.token_a_reserve}
      end

      # Apply AMM formula: x * y = k (with fees)
      input_with_fee = input_amount * (1 - pool.fee_rate)
      numerator = input_with_fee * output_reserve
      denominator = input_reserve + input_with_fee
      
      output_amount = numerator / denominator
      price_impact = abs(output_amount / output_reserve)
      
      {output_amount, price_impact}
    end
  end

  defmodule GlobalMarket do
    @moduledoc "Global market state including all currencies and external factors"
    
    defstruct [
      :currencies,  # USD, EUR, JPY, GBP exchange rates
      :crypto_prices,  # BTC, ETH, etc.
      :interest_rates,
      :volatility_index,
      :market_sentiment,  # :bullish, :bearish, :neutral
      :regulatory_events,
      :macroeconomic_factors,
      :timestamp
    ]

    def new() do
      %__MODULE__{
        currencies: %{
          "USD" => 1.0000,
          "EUR" => 0.8547,
          "JPY" => 149.2356,
          "GBP" => 0.7823,
          "CHF" => 0.8902,
          "CAD" => 1.3562,
          "AUD" => 1.5234
        },
        crypto_prices: %{
          "BTC" => 45000.0 + :rand.uniform(5000) - 2500,
          "ETH" => 3200.0 + :rand.uniform(800) - 400,
          "ADA" => 1.2 + :rand.uniform() - 0.5,
          "SOL" => 180.0 + :rand.uniform(40) - 20,
          "AVAX" => 75.0 + :rand.uniform(25) - 12.5
        },
        interest_rates: %{
          "USD" => 5.25,
          "EUR" => 4.50,
          "JPY" => -0.10,
          "GBP" => 5.00
        },
        volatility_index: 15.0 + :rand.uniform(20),
        market_sentiment: Enum.random([:bullish, :bearish, :neutral]),
        regulatory_events: [],
        macroeconomic_factors: %{
          inflation_rate: 3.2,
          unemployment_rate: 4.1,
          gdp_growth: 2.1
        },
        timestamp: DateTime.utc_now()
      }
    end

    def update_market_conditions(market) do
      # Simulate realistic market movements
      new_crypto_prices = update_crypto_prices(market.crypto_prices, market.market_sentiment)
      new_currencies = update_currency_rates(market.currencies)
      new_sentiment = update_sentiment(market.market_sentiment, market.volatility_index)
      
      %{market |
        crypto_prices: new_crypto_prices,
        currencies: new_currencies,
        market_sentiment: new_sentiment,
        volatility_index: update_volatility(market.volatility_index),
        timestamp: DateTime.utc_now()
      }
    end

    defp update_crypto_prices(prices, sentiment) do
      sentiment_factor = case sentiment do
        :bullish -> 1.002
        :bearish -> 0.998
        :neutral -> 1.0
      end

      Map.new(prices, fn {symbol, price} ->
        # Different cryptos have different volatilities
        volatility = case symbol do
          "BTC" -> 0.02
          "ETH" -> 0.025
          "ADA" -> 0.04
          "SOL" -> 0.035
          "AVAX" -> 0.04
        end

        change = (:rand.uniform() - 0.5) * volatility * sentiment_factor
        new_price = price * (1 + change)
        {symbol, max(new_price, price * 0.95)}  # 5% downside protection
      end)
    end

    defp update_currency_rates(currencies) do
      Map.new(currencies, fn {currency, rate} ->
        if currency == "USD" do
          {currency, rate}  # USD is base
        else
          volatility = case currency do
            "EUR" -> 0.005
            "JPY" -> 0.008
            "GBP" -> 0.006
            _ -> 0.007
          end
          
          change = (:rand.uniform() - 0.5) * volatility
          {currency, rate * (1 + change)}
        end
      end)
    end

    defp update_sentiment(current, volatility) do
      # Higher volatility increases chance of sentiment change
      change_probability = volatility / 100
      
      if :rand.uniform() < change_probability do
        case current do
          :bullish -> if :rand.uniform() < 0.6, do: :neutral, else: :bearish
          :bearish -> if :rand.uniform() < 0.6, do: :neutral, else: :bullish
          :neutral -> if :rand.uniform() < 0.5, do: :bullish, else: :bearish
        end
      else
        current
      end
    end

    defp update_volatility(current) do
      # Mean reversion with random walk
      target = 20.0
      reversion = (target - current) * 0.1
      random = (:rand.uniform() - 0.5) * 2
      
      max(5.0, min(50.0, current + reversion + random))
    end
  end

  # Main market system state
  defstruct [
    :market_makers,
    :order_books,
    :liquidity_pools,
    :global_market,
    :stablecoin_nodes,
    :arbitrage_opportunities,
    :market_stats,
    :regulatory_compliance,
    :started_at
  ]

  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  def init(_opts) do
    Logger.configure(level: :info)
    
    # Initialize market makers with different strategies
    market_makers = create_market_makers()
    
    # Initialize order books for all trading pairs
    order_books = create_order_books()
    
    # Initialize liquidity pools
    liquidity_pools = create_liquidity_pools()
    
    # Initialize global market
    global_market = GlobalMarket.new()
    
    state = %__MODULE__{
      market_makers: market_makers,
      order_books: order_books,
      liquidity_pools: liquidity_pools,
      global_market: global_market,
      stablecoin_nodes: [],
      arbitrage_opportunities: [],
      market_stats: initialize_market_stats(),
      regulatory_compliance: initialize_compliance(),
      started_at: DateTime.utc_now()
    }

    # Schedule periodic updates
    :timer.send_interval(1000, self(), :update_market_prices)
    :timer.send_interval(2000, self(), :execute_market_maker_strategies)
    :timer.send_interval(5000, self(), :detect_arbitrage_opportunities)
    :timer.send_interval(10000, self(), :update_global_conditions)
    :timer.send_interval(30000, self(), :rebalance_liquidity_pools)
    
    {:ok, state}
  end

  def handle_info(:update_market_prices, state) do
    # Update all order books based on market conditions
    new_order_books = update_all_order_books(state.order_books, state.global_market)
    
    # Calculate new arbitrage opportunities
    new_arbitrage = detect_arbitrage(new_order_books, state.liquidity_pools)
    
    new_state = %{state | 
      order_books: new_order_books,
      arbitrage_opportunities: new_arbitrage
    }
    
    {:noreply, new_state}
  end

  def handle_info(:execute_market_maker_strategies, state) do
    # Each market maker executes their strategy
    {new_market_makers, trades} = execute_all_strategies(
      state.market_makers, 
      state.order_books, 
      state.global_market
    )
    
    # Update order books with new trades
    new_order_books = process_trades(state.order_books, trades)
    
    # Log significant trades
    log_significant_trades(trades)
    
    new_state = %{state | 
      market_makers: new_market_makers,
      order_books: new_order_books
    }
    
    {:noreply, new_state}
  end

  def handle_info(:detect_arbitrage_opportunities, state) do
    # Find and execute arbitrage opportunities
    {new_arbitrage, arbitrage_trades} = find_and_execute_arbitrage(
      state.arbitrage_opportunities,
      state.order_books,
      state.liquidity_pools,
      state.market_makers
    )
    
    if length(arbitrage_trades) > 0 do
      Logger.info("🔄 Executed #{length(arbitrage_trades)} arbitrage trades")
    end
    
    new_state = %{state | arbitrage_opportunities: new_arbitrage}
    {:noreply, new_state}
  end

  def handle_info(:update_global_conditions, state) do
    # Update global market conditions
    new_global_market = GlobalMarket.update_market_conditions(state.global_market)
    
    # Check for regulatory events
    new_compliance = check_regulatory_compliance(state.regulatory_compliance, new_global_market)
    
    # Update market statistics
    new_stats = update_market_statistics(state.market_stats, state.order_books, state.liquidity_pools)
    
    # Log market update
    log_market_update(new_global_market, new_stats)
    
    new_state = %{state | 
      global_market: new_global_market,
      regulatory_compliance: new_compliance,
      market_stats: new_stats
    }
    
    {:noreply, new_state}
  end

  def handle_info(:rebalance_liquidity_pools, state) do
    # Rebalance all liquidity pools
    new_pools = rebalance_all_pools(state.liquidity_pools, state.global_market)
    
    new_state = %{state | liquidity_pools: new_pools}
    {:noreply, new_state}
  end

  def handle_call(:get_market_status, _from, state) do
    status = compile_market_status(state)
    {:reply, status, state}
  end

  # Private implementation functions

  defp create_market_makers() do
    [
      # Institutional market makers
      MarketMaker.new("inst_mm_1", :conservative, 10_000_000),
      MarketMaker.new("inst_mm_2", :conservative, 15_000_000),
      MarketMaker.new("inst_mm_3", :aggressive, 8_000_000),
      
      # Hedge funds
      MarketMaker.new("hedge_1", :aggressive, 25_000_000),
      MarketMaker.new("hedge_2", :arbitrage, 20_000_000),
      
      # High frequency traders
      MarketMaker.new("hft_1", :high_frequency, 5_000_000),
      MarketMaker.new("hft_2", :high_frequency, 7_500_000),
      MarketMaker.new("hft_3", :high_frequency, 6_000_000),
      
      # Retail arbitrageurs
      MarketMaker.new("retail_1", :arbitrage, 100_000),
      MarketMaker.new("retail_2", :arbitrage, 250_000),
      MarketMaker.new("retail_3", :conservative, 75_000)
    ]
  end

  defp create_order_books() do
    pairs = [
      "STABLE/USD",
      "STABLE/BTC", 
      "STABLE/ETH",
      "STABLE/EUR",
      "STABLE/JPY"
    ]

    Map.new(pairs, fn pair ->
      initial_price = case pair do
        "STABLE/USD" -> 1.0000
        "STABLE/BTC" -> 1.0 / 45000
        "STABLE/ETH" -> 1.0 / 3200
        "STABLE/EUR" -> 1.0 / 0.8547
        "STABLE/JPY" -> 1.0 / 149.2356
      end
      
      {pair, OrderBook.new(pair, initial_price)}
    end)
  end

  defp create_liquidity_pools() do
    [
      {"STABLE/USD", LiquidityPool.new("STABLE/USD", 1_000_000, 1_000_000)},
      {"STABLE/BTC", LiquidityPool.new("STABLE/BTC", 500_000, 11.11)},
      {"STABLE/ETH", LiquidityPool.new("STABLE/ETH", 750_000, 234.375)},
    ]
    |> Map.new()
  end

  defp initialize_market_stats() do
    %{
      total_volume_24h: 0.0,
      total_trades_24h: 0,
      average_spread: 0.0,
      price_stability_score: 100.0,
      liquidity_index: 85.0,
      arbitrage_efficiency: 0.95
    }
  end

  defp initialize_compliance() do
    %{
      aml_checks: true,
      kyc_verified_volume: 0.85,
      regulatory_alerts: [],
      compliance_score: 98.5
    }
  end

  defp update_all_order_books(order_books, global_market) do
    Map.new(order_books, fn {pair, book} ->
      # Update book based on global market conditions
      new_book = update_order_book_prices(book, global_market)
      {pair, new_book}
    end)
  end

  defp update_order_book_prices(book, global_market) do
    # Calculate new reference price based on global conditions
    base_price = case book.pair do
      "STABLE/USD" -> adjust_for_sentiment(@target_peg, global_market.market_sentiment)
      "STABLE/BTC" -> @target_peg / global_market.crypto_prices["BTC"]
      "STABLE/ETH" -> @target_peg / global_market.crypto_prices["ETH"]
      "STABLE/EUR" -> @target_peg / global_market.currencies["EUR"]
      "STABLE/JPY" -> @target_peg / global_market.currencies["JPY"]
      _ -> book.last_price
    end

    # Apply volatility and market impact
    volatility_factor = global_market.volatility_index / 100
    price_noise = (:rand.uniform() - 0.5) * volatility_factor * 0.001
    
    new_price = base_price * (1 + price_noise)
    
    %{book | last_price: new_price}
  end

  defp adjust_for_sentiment(base_price, sentiment) do
    case sentiment do
      :bullish -> base_price * 1.0002
      :bearish -> base_price * 0.9998
      :neutral -> base_price
    end
  end

  defp execute_all_strategies(market_makers, order_books, global_market) do
    Enum.map_reduce(market_makers, [], fn maker, trades_acc ->
      {new_maker, new_trades} = execute_strategy(maker, order_books, global_market)
      {new_maker, trades_acc ++ new_trades}
    end)
  end

  defp execute_strategy(maker, order_books, global_market) do
    case maker.strategy do
      :conservative -> execute_conservative_strategy(maker, order_books, global_market)
      :aggressive -> execute_aggressive_strategy(maker, order_books, global_market)
      :arbitrage -> execute_arbitrage_strategy(maker, order_books, global_market)
      :high_frequency -> execute_hf_strategy(maker, order_books, global_market)
    end
  end

  defp execute_conservative_strategy(maker, order_books, _global_market) do
    # Conservative: provide liquidity around current price with tight spreads
    stable_usd_book = order_books["STABLE/USD"]
    current_price = stable_usd_book.last_price
    
    trades = if abs(current_price - @target_peg) < maker.risk_tolerance do
      # Place symmetric orders around peg
      bid_price = current_price - 0.0005
      ask_price = current_price + 0.0005
      quantity = min(maker.max_position_size, maker.capital_usd * 0.1)
      
      [
        %{type: :limit_buy, pair: "STABLE/USD", price: bid_price, quantity: quantity, maker: maker.id},
        %{type: :limit_sell, pair: "STABLE/USD", price: ask_price, quantity: quantity, maker: maker.id}
      ]
    else
      []
    end
    
    new_maker = %{maker | trade_count: maker.trade_count + length(trades)}
    {new_maker, trades}
  end

  defp execute_aggressive_strategy(maker, order_books, global_market) do
    # Aggressive: larger position sizes, momentum following
    stable_usd_book = order_books["STABLE/USD"]
    current_price = stable_usd_book.last_price
    
    # Follow market sentiment
    sentiment_bias = case global_market.market_sentiment do
      :bullish -> 0.7   # 70% buy bias
      :bearish -> 0.3   # 30% buy bias
      :neutral -> 0.5   # 50% buy bias
    end
    
    trades = if :rand.uniform() < sentiment_bias do
      # Buy bias
      quantity = maker.max_position_size * 0.3
      price = current_price + 0.0002  # Slightly above market
      [%{type: :market_buy, pair: "STABLE/USD", quantity: quantity, maker: maker.id}]
    else
      # Sell bias
      quantity = maker.max_position_size * 0.3
      price = current_price - 0.0002  # Slightly below market
      [%{type: :market_sell, pair: "STABLE/USD", quantity: quantity, maker: maker.id}]
    end
    
    new_maker = %{maker | trade_count: maker.trade_count + length(trades)}
    {new_maker, trades}
  end

  defp execute_arbitrage_strategy(maker, order_books, _global_market) do
    # Look for price differences between pairs
    stable_usd_price = order_books["STABLE/USD"].last_price
    stable_btc_price = order_books["STABLE/BTC"].last_price
    btc_usd_implied = stable_btc_price * order_books["STABLE/BTC"].last_price
    
    price_diff = abs(stable_usd_price - btc_usd_implied)
    
    trades = if price_diff > 0.001 do  # 0.1% arbitrage threshold
      quantity = maker.max_position_size * 0.5
      
      if stable_usd_price > btc_usd_implied do
        # Sell STABLE/USD, buy STABLE/BTC
        [
          %{type: :market_sell, pair: "STABLE/USD", quantity: quantity, maker: maker.id},
          %{type: :market_buy, pair: "STABLE/BTC", quantity: quantity, maker: maker.id}
        ]
      else
        # Buy STABLE/USD, sell STABLE/BTC
        [
          %{type: :market_buy, pair: "STABLE/USD", quantity: quantity, maker: maker.id},
          %{type: :market_sell, pair: "STABLE/BTC", quantity: quantity, maker: maker.id}
        ]
      end
    else
      []
    end
    
    new_maker = %{maker | trade_count: maker.trade_count + length(trades)}
    {new_maker, trades}
  end

  defp execute_hf_strategy(maker, order_books, global_market) do
    # High frequency: quick in-and-out trades based on micro-movements
    stable_usd_book = order_books["STABLE/USD"]
    current_price = stable_usd_book.last_price
    
    # Only trade if we detect micro-movement opportunity
    volatility = global_market.volatility_index
    
    trades = if volatility > 20 and :rand.uniform() < 0.3 do
      direction = if :rand.uniform() < 0.5, do: :buy, else: :sell
      quantity = maker.max_position_size * 0.1  # Small position
      
      case direction do
        :buy ->
          [%{type: :market_buy, pair: "STABLE/USD", quantity: quantity, maker: maker.id}]
        :sell ->
          [%{type: :market_sell, pair: "STABLE/USD", quantity: quantity, maker: maker.id}]
      end
    else
      []
    end
    
    new_maker = %{maker | 
      trade_count: maker.trade_count + length(trades),
      last_trade_time: DateTime.utc_now()
    }
    {new_maker, trades}
  end

  defp process_trades(order_books, trades) do
    # Process all trades and update order books
    Enum.reduce(trades, order_books, fn trade, books_acc ->
      book = books_acc[trade.pair]
      {new_book, _execution} = execute_trade(book, trade)
      Map.put(books_acc, trade.pair, new_book)
    end)
  end

  defp execute_trade(book, trade) do
    # Simplified trade execution with price impact
    impact = calculate_market_impact(book, trade.quantity)
    
    new_price = case trade.type do
      type when type in [:market_buy, :limit_buy] ->
        book.last_price * (1 + impact)
      type when type in [:market_sell, :limit_sell] ->
        book.last_price * (1 - impact)
    end
    
    new_book = %{book | 
      last_price: new_price,
      volume_24h: book.volume_24h + trade.quantity
    }
    
    execution = %{
      price: new_price,
      quantity: trade.quantity,
      impact: impact
    }
    
    {new_book, execution}
  end

  defp calculate_market_impact(book, quantity) do
    # Price impact based on order book depth
    depth = book.price_impact_model.depth
    impact_factor = book.price_impact_model.impact_factor
    
    # Square root impact model
    base_impact = impact_factor * :math.sqrt(quantity / depth)
    
    # Add slippage
    slippage = :rand.uniform() * @max_slippage * 0.1
    
    min(base_impact + slippage, @max_slippage)
  end

  defp detect_arbitrage(order_books, liquidity_pools) do
    # Compare prices across order books and liquidity pools
    pairs = Map.keys(order_books)
    
    for pair1 <- pairs,
        pair2 <- pairs,
        pair1 != pair2,
        have_common_asset?(pair1, pair2) do
      
      price1 = order_books[pair1].last_price
      price2 = order_books[pair2].last_price
      
      # Calculate triangular arbitrage opportunity
      calculate_triangular_arbitrage(pair1, pair2, price1, price2, liquidity_pools)
    end
    |> Enum.filter(& &1)  # Remove nil values
    |> Enum.filter(fn arb -> arb.profit_margin > 0.002 end)  # 0.2% minimum
  end

  defp have_common_asset?(pair1, pair2) do
    [base1, quote1] = String.split(pair1, "/")
    [base2, quote2] = String.split(pair2, "/")
    
    base1 == base2 or base1 == quote2 or quote1 == base2 or quote1 == quote2
  end

  defp calculate_triangular_arbitrage(pair1, pair2, price1, price2, _pools) do
    # Simplified triangular arbitrage calculation
    # In reality, this would be much more complex
    profit_margin = abs(price1 - price2) / ((price1 + price2) / 2)
    
    if profit_margin > 0.002 do
      %{
        type: :triangular,
        pairs: [pair1, pair2],
        profit_margin: profit_margin,
        estimated_volume: 10_000,
        expires_at: DateTime.add(DateTime.utc_now(), 30, :second)
      }
    end
  end

  defp find_and_execute_arbitrage(opportunities, order_books, liquidity_pools, market_makers) do
    # Find arbitrageurs
    arbitrageurs = Enum.filter(market_makers, fn mm -> mm.strategy == :arbitrage end)
    
    # Execute valid opportunities
    {executed, remaining} = Enum.split_with(opportunities, fn opp ->
      DateTime.compare(opp.expires_at, DateTime.utc_now()) == :gt
    end)
    
    trades = Enum.flat_map(executed, fn opp ->
      if length(arbitrageurs) > 0 do
        arb = Enum.random(arbitrageurs)
        generate_arbitrage_trades(opp, arb)
      else
        []
      end
    end)
    
    {remaining, trades}
  end

  defp generate_arbitrage_trades(opportunity, arbitrageur) do
    volume = min(opportunity.estimated_volume, arbitrageur.max_position_size)
    
    case opportunity.type do
      :triangular ->
        [pair1, pair2] = opportunity.pairs
        [
          %{type: :market_buy, pair: pair1, quantity: volume, maker: arbitrageur.id},
          %{type: :market_sell, pair: pair2, quantity: volume, maker: arbitrageur.id}
        ]
    end
  end

  defp rebalance_all_pools(pools, global_market) do
    Map.new(pools, fn {pair, pool} ->
      new_pool = rebalance_pool(pool, global_market)
      {pair, new_pool}
    end)
  end

  defp rebalance_pool(pool, _global_market) do
    # Simulate external LPs adding/removing liquidity
    change_factor = 1 + (:rand.uniform() - 0.5) * 0.02  # ±1% change
    
    %{pool |
      token_a_reserve: pool.token_a_reserve * change_factor,
      token_b_reserve: pool.token_b_reserve * change_factor
    }
  end

  defp check_regulatory_compliance(compliance, _global_market) do
    # Simulate regulatory monitoring
    new_alerts = if :rand.uniform() < 0.05 do  # 5% chance of alert
      alert = %{
        type: Enum.random([:volume_spike, :price_manipulation, :wash_trading]),
        severity: Enum.random([:low, :medium, :high]),
        timestamp: DateTime.utc_now()
      }
      [alert | compliance.regulatory_alerts]
    else
      compliance.regulatory_alerts
    end
    
    %{compliance | regulatory_alerts: new_alerts}
  end

  defp update_market_statistics(stats, order_books, liquidity_pools) do
    # Calculate aggregate statistics
    total_volume = order_books
    |> Map.values()
    |> Enum.map(& &1.volume_24h)
    |> Enum.sum()
    
    avg_price = order_books["STABLE/USD"].last_price
    stability_score = calculate_stability_score(avg_price)
    
    %{stats |
      total_volume_24h: total_volume,
      price_stability_score: stability_score,
      liquidity_index: calculate_liquidity_index(liquidity_pools)
    }
  end

  defp calculate_stability_score(current_price) do
    deviation = abs(current_price - @target_peg)
    max(0, 100 - (deviation * 10000))  # Score out of 100
  end

  defp calculate_liquidity_index(pools) do
    # Simplified liquidity index based on pool sizes
    total_liquidity = pools
    |> Map.values()
    |> Enum.map(fn pool -> pool.token_a_reserve + pool.token_b_reserve end)
    |> Enum.sum()
    
    min(100, total_liquidity / 50_000)  # Index out of 100
  end

  defp log_significant_trades(trades) do
    significant = Enum.filter(trades, fn trade ->
      trade.quantity > 50_000  # $50k+ trades
    end)
    
    Enum.each(significant, fn trade ->
      Logger.info("💰 Large trade: #{trade.type} #{Float.round(trade.quantity, 2)} on #{trade.pair} by #{trade.maker}")
    end)
  end

  defp log_market_update(global_market, stats) do
    btc_price = global_market.crypto_prices["BTC"]
    sentiment = global_market.market_sentiment
    stability = stats.price_stability_score
    
    Logger.info("📊 Market Update: BTC $#{Float.round(btc_price, 0)}, Sentiment: #{sentiment}, Stability: #{Float.round(stability, 1)}%")
  end

  defp compile_market_status(state) do
    %{
      total_market_makers: length(state.market_makers),
      active_pairs: map_size(state.order_books),
      liquidity_pools: map_size(state.liquidity_pools),
      arbitrage_opportunities: length(state.arbitrage_opportunities),
      global_market: state.global_market,
      market_stats: state.market_stats,
      uptime: DateTime.diff(DateTime.utc_now(), state.started_at)
    }
  end

  # Public API functions
  def run_demo() do
    IO.puts """
    
    ╔═══════════════════════════════════════════════════════════════════════════════╗
    ║                   🌍 Advanced Market Dynamics Demo 🌍                        ║
    ║                                                                               ║
    ║ Complete stablecoin ecosystem simulation featuring:                           ║
    ║ • 11 Market makers with different strategies                                  ║
    ║ • 5 Trading pairs (USD, BTC, ETH, EUR, JPY)                                  ║
    ║ • 3 AMM liquidity pools                                                       ║
    ║ • Real-world currency and crypto market dynamics                              ║
    ║ • Arbitrage detection and execution                                           ║
    ║ • Regulatory compliance monitoring                                            ║
    ║ • Market impact and slippage modeling                                        ║
    ╚═══════════════════════════════════════════════════════════════════════════════╝
    
    """

    {:ok, _pid} = start_link()
    
    # Wait for initialization
    :timer.sleep(2000)
    
    # Show initial market status
    show_market_dashboard()
    
    # Run monitoring loop
    IO.puts "\n🔄 Market is now running with full dynamics... Press Ctrl+C to stop\n"
    monitor_market_loop()
  end

  def show_market_dashboard() do
    case GenServer.call(__MODULE__, :get_market_status) do
      status ->
        IO.puts """
        
        ╔═══════════════════════════════════════════════════════════════════════════════╗
        ║                            📊 Market Dashboard 📊                            ║
        ╚═══════════════════════════════════════════════════════════════════════════════╝
        
        🏦 Market Structure:
        ├── Market Makers: #{status.total_market_makers}
        ├── Trading Pairs: #{status.active_pairs}
        ├── Liquidity Pools: #{status.liquidity_pools}
        └── Arbitrage Opportunities: #{status.arbitrage_opportunities}
        
        🌐 Global Market:
        ├── BTC Price: $#{Float.round(status.global_market.crypto_prices["BTC"], 0)}
        ├── ETH Price: $#{Float.round(status.global_market.crypto_prices["ETH"], 0)}
        ├── Market Sentiment: #{status.global_market.market_sentiment}
        ├── Volatility Index: #{Float.round(status.global_market.volatility_index, 1)}
        └── USD Interest Rate: #{status.global_market.interest_rates["USD"]}%
        
        📈 Market Statistics:
        ├── 24h Volume: $#{Float.round(status.market_stats.total_volume_24h, 0)}
        ├── Price Stability Score: #{Float.round(status.market_stats.price_stability_score, 1)}/100
        ├── Liquidity Index: #{Float.round(status.market_stats.liquidity_index, 1)}/100
        └── Arbitrage Efficiency: #{Float.round(status.market_stats.arbitrage_efficiency * 100, 1)}%
        
        ⏱️  System Uptime: #{status.uptime} seconds
        """
    end
  end

  defp monitor_market_loop() do
    :timer.sleep(15000)  # Update every 15 seconds
    
    show_market_dashboard()
    monitor_market_loop()
  end
end

# Run the advanced market dynamics demo
AdvancedMarketDynamicsDemo.run_demo()