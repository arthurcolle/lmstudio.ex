#!/usr/bin/env elixir

Mix.install([
  {:jason, "~> 1.4"}
])

defmodule CompleteStablecoinEcosystemDemo do
  @moduledoc """
  Complete stablecoin ecosystem combining blockchain nodes with advanced market dynamics.
  
  Features:
  - Multi-node stablecoin blockchain with consensus
  - Advanced market makers with different strategies
  - Real-world currency and crypto market impacts
  - Arbitrage opportunities and automated execution
  - Liquidity pools and AMM mechanics
  - Order book dynamics with realistic market impact
  - Regulatory compliance and monitoring
  - Cross-node transaction propagation
  - Oracle price feeds affecting stability mechanisms
  """

  require Logger

  # Import modules from the market dynamics
  defmodule MarketMaker do
    defstruct [
      :id, :strategy, :capital_usd, :capital_stablecoin, :capital_btc, :capital_eth,
      :max_position_size, :risk_tolerance, :active_orders, :pnl, :trade_count, :last_trade_time
    ]

    def new(id, strategy, initial_capital) do
      %__MODULE__{
        id: id, strategy: strategy, capital_usd: initial_capital,
        capital_stablecoin: initial_capital * 0.3,
        capital_btc: initial_capital * 0.2 / 45000,
        capital_eth: initial_capital * 0.2 / 3200,
        max_position_size: initial_capital * case strategy do
          :conservative -> 0.1; :aggressive -> 0.4; :arbitrage -> 0.6; :high_frequency -> 0.8
        end,
        risk_tolerance: case strategy do
          :conservative -> 0.002; :aggressive -> 0.01; :arbitrage -> 0.005; :high_frequency -> 0.003
        end,
        active_orders: [], pnl: 0.0, trade_count: 0, last_trade_time: DateTime.utc_now()
      }
    end
  end

  defmodule GlobalMarket do
    defstruct [:currencies, :crypto_prices, :interest_rates, :volatility_index, 
               :market_sentiment, :regulatory_events, :macroeconomic_factors, :timestamp,
               :active_events, :whale_activity, :flash_crash_risk, :news_events, :market_phase]

    def new() do
      %__MODULE__{
        currencies: %{"USD" => 1.0000, "EUR" => 0.8547, "JPY" => 149.2356, "GBP" => 0.7823},
        crypto_prices: %{
          "BTC" => 45000.0 + :rand.uniform(5000) - 2500,
          "ETH" => 3200.0 + :rand.uniform(800) - 400,
          "ADA" => 1.2 + :rand.uniform() - 0.5,
          "SOL" => 180.0 + :rand.uniform(40) - 20
        },
        interest_rates: %{"USD" => 5.25, "EUR" => 4.50, "JPY" => -0.10, "GBP" => 5.00},
        volatility_index: 15.0 + :rand.uniform(20),
        market_sentiment: Enum.random([:bullish, :bearish, :neutral, :extreme_fear, :euphoria]),
        regulatory_events: [],
        active_events: [],
        whale_activity: %{large_orders: 0, manipulation_detected: false},
        flash_crash_risk: 0.0,
        news_events: [],
        market_phase: :normal,
        macroeconomic_factors: %{inflation_rate: 3.2, unemployment_rate: 4.1, gdp_growth: 2.1},
        timestamp: DateTime.utc_now()
      }
    end

    def update(market) do
      # Generate dynamic market events
      {new_events, event_impact} = generate_market_events(market)
      
      sentiment_factor = case market.market_sentiment do
        :bullish -> 1.002; :bearish -> 0.998; :neutral -> 1.0
        :extreme_fear -> 0.985; :euphoria -> 1.025
      end
      
      # Apply event impacts
      total_impact = sentiment_factor * event_impact

      new_crypto_prices = Map.new(market.crypto_prices, fn {symbol, price} ->
        volatility = case symbol do
          "BTC" -> 0.02; "ETH" -> 0.025; "ADA" -> 0.04; "SOL" -> 0.035
        end
        base_change = (:rand.uniform() - 0.5) * volatility
        event_change = if length(new_events) > 0, do: (:rand.uniform() - 0.5) * 0.1, else: 0.0
        total_change = (base_change + event_change) * total_impact
        {symbol, max(price * (1 + total_change), price * 0.8)}
      end)

      # Update market phase based on events
      new_phase = determine_market_phase(new_events, market.volatility_index)
      new_sentiment = update_market_sentiment(market.market_sentiment, new_events)
      
      %{market | 
        crypto_prices: new_crypto_prices,
        volatility_index: calculate_new_volatility(market.volatility_index, new_events),
        timestamp: DateTime.utc_now(),
        active_events: new_events,
        market_sentiment: new_sentiment,
        market_phase: new_phase,
        flash_crash_risk: calculate_flash_crash_risk(new_events, market.volatility_index)
      }
    end
    
    # Dynamic market event generators
    defp generate_market_events(market) do
      events = []
      base_impact = 1.0
      
      # Regulatory events (5% chance)
      events = if :rand.uniform() < 0.05 do
        regulatory_event = Enum.random([
          %{type: :regulation, description: "SEC announces stablecoin framework", impact: 0.95},
          %{type: :regulation, description: "EU passes MiCA regulation", impact: 0.92},
          %{type: :regulation, description: "Fed considers CBDC pilot", impact: 1.08},
          %{type: :ban, description: "Country X bans crypto trading", impact: 0.88},
          %{type: :approval, description: "Major bank adopts stablecoins", impact: 1.12}
        ])
        Logger.info("🏛️  REGULATORY EVENT: #{regulatory_event.description}")
        [regulatory_event | events]
      else
        events
      end
      
      # Whale activity (8% chance)
      events = if :rand.uniform() < 0.08 do
        whale_event = Enum.random([
          %{type: :whale_buy, description: "Whale accumulates 500M tokens", impact: 1.15},
          %{type: :whale_sell, description: "Major holder dumps position", impact: 0.82},
          %{type: :whale_manipulation, description: "Coordinated whale attack detected", impact: 0.75},
          %{type: :institutional_entry, description: "Pension fund enters with $2B", impact: 1.25}
        ])
        Logger.info("🐋 WHALE ACTIVITY: #{whale_event.description}")
        [whale_event | events]
      else
        events
      end
      
      # Flash crash events (3% chance during high volatility)
      events = if market.volatility_index > 35 and :rand.uniform() < 0.03 do
        flash_event = %{type: :flash_crash, description: "FLASH CRASH: Liquidation cascade triggered", impact: 0.65}
        Logger.info("⚡💥 #{flash_event.description}")
        [flash_event | events]
      else
        events
      end
      
      # Market manipulation (4% chance)
      events = if :rand.uniform() < 0.04 do
        manip_event = Enum.random([
          %{type: :pump_dump, description: "Coordinated pump & dump detected", impact: 0.88},
          %{type: :spoofing, description: "Order book spoofing activity", impact: 0.95},
          %{type: :wash_trading, description: "Wash trading volumes detected", impact: 0.92}
        ])
        Logger.info("🕵️  MANIPULATION: #{manip_event.description}")
        [manip_event | events]
      else
        events
      end
      
      # Black swan events (1% chance)
      events = if :rand.uniform() < 0.01 do
        swan_event = Enum.random([
          %{type: :black_swan, description: "Major exchange hack: $500M stolen", impact: 0.45},
          %{type: :black_swan, description: "Stablecoin de-pegging contagion", impact: 0.35},
          %{type: :black_swan, description: "Global financial crisis spillover", impact: 0.25},
          %{type: :golden_swan, description: "Mass adoption breakthrough", impact: 2.5}
        ])
        Logger.info("🦢 BLACK SWAN: #{swan_event.description}")
        [swan_event | events]
      else
        events
      end
      
      # News events (12% chance)
      events = if :rand.uniform() < 0.12 do
        news_event = Enum.random([
          %{type: :news, description: "Tesla adds stablecoin payments", impact: 1.18},
          %{type: :news, description: "Amazon exploring crypto integration", impact: 1.22},
          %{type: :news, description: "JP Morgan launches stablecoin fund", impact: 1.15},
          %{type: :fud, description: "Celsius-style collapse rumors", impact: 0.78},
          %{type: :partnership, description: "Visa partners with stablecoin network", impact: 1.35}
        ])
        Logger.info("📰 NEWS: #{news_event.description}")
        [news_event | events]
      else
        events
      end
      
      # Calculate combined impact
      total_impact = events |> Enum.map(& &1.impact) |> Enum.reduce(base_impact, &*/2)
      
      {events, total_impact}
    end
    
    defp determine_market_phase(events, volatility) do
      cond do
        Enum.any?(events, &(&1.type == :black_swan)) -> :crisis
        Enum.any?(events, &(&1.type == :flash_crash)) -> :crash
        volatility > 40 -> :extreme_volatility
        volatility > 25 -> :high_volatility
        Enum.any?(events, &(&1.type == :whale_manipulation)) -> :manipulation
        true -> :normal
      end
    end
    
    defp update_market_sentiment(current, events) do
      if length(events) == 0 do
        current
      else
        avg_impact = events |> Enum.map(& &1.impact) |> Enum.sum() |> Kernel./(length(events))
        cond do
          avg_impact < 0.6 -> :extreme_fear
          avg_impact < 0.9 -> :bearish
          avg_impact > 1.4 -> :euphoria
          avg_impact > 1.1 -> :bullish
          true -> :neutral
        end
      end
    end
    
    defp calculate_new_volatility(current_vol, events) do
      event_volatility = case length(events) do
        0 -> 0
        n when n > 2 -> 15  # Multiple events = high volatility
        _ -> 5
      end
      
      base_change = (:rand.uniform() - 0.5) * 3
      new_vol = current_vol + base_change + event_volatility
      max(5.0, min(80.0, new_vol))
    end
    
    defp calculate_flash_crash_risk(events, volatility) do
      base_risk = volatility / 100
      event_risk = if Enum.any?(events, &(&1.type in [:whale_sell, :manipulation, :black_swan])) do
        0.3
      else
        0.0
      end
      min(0.8, base_risk + event_risk)
    end
  end

  defmodule StablecoinNode do
    use GenServer
    require Logger

    def start_link(config, name) do
      GenServer.start_link(__MODULE__, config, name: name)
    end

    def init(config) do
      Logger.info("🏗️  Starting Enhanced StablecoinNode #{config.id}...")

      state = %{
        node_id: config.id,
        port: config.port,
        mining: config.mining,
        data_provider: config.data_provider,
        bootstrap: config.bootstrap,
        peers: [],
        blockchain_height: 0,
        mempool_size: 0,
        stablecoin_price: 1.000 + :rand.uniform() * 0.01 - 0.005,
        crdt_price: 0.000001 + :rand.uniform() * 0.0000005,  # AGENTIC $CRDT price
        agent_rewards: %{cooperation: 0.0, mining: 0.0, stability: 0.0},
        crdt_exchange_rate: 1_000_000,  # 1 USD = 1M CRDT initially
        agent_balances: %{},  # Track individual agent balances
        connected_peers: [],
        last_block_time: DateTime.utc_now(),
        total_supply: 1.0e30,
        stability_reserves: 3.5e29,  # 35% controlled by research group
        research_group_holdings: 3.5e29,  # 35% research allocation
        oracle_feeds: generate_initial_oracle_feeds(),
        market_orders: [],
        market_maker_connections: [],
        stability_mechanisms: %{
          last_intervention: DateTime.utc_now(),
          intervention_count: 0,
          total_minted: 0.0,
          total_burned: 0.0
        },
        started_at: DateTime.utc_now()
      }

      # Start P2P listener
      {:ok, listen_socket} = :gen_tcp.listen(config.port, [:binary, packet: 4, active: false, reuseaddr: true])
      Logger.info("📡 Node #{config.id} listening on port #{config.port}")
      
      spawn(fn -> accept_loop(listen_socket, state) end)
      
      # Enhanced periodic tasks
      :timer.send_interval(3000, self(), :mine_block)
      :timer.send_interval(2000, self(), :update_oracle_data)
      :timer.send_interval(5000, self(), :stability_check)
      :timer.send_interval(8000, self(), :process_market_orders)
      :timer.send_interval(15000, self(), :peer_discovery)
      :timer.send_interval(10000, self(), :sync_with_market)
      
      new_state = Map.put(state, :listen_socket, listen_socket)
      {:ok, new_state}
    end

    def handle_info(:mine_block, state) do
      if state.mining do
        new_height = state.blockchain_height + 1
        timestamp = DateTime.utc_now()
        :timer.sleep(100 + :rand.uniform(200))
        
        # Include market data in block
        block = %{
          height: new_height,
          timestamp: timestamp,
          miner: state.node_id,
          transactions: process_pending_transactions(state),
          oracle_data: state.oracle_feeds,
          stablecoin_price: state.stablecoin_price,
          market_orders_processed: length(state.market_orders),
          stability_actions: state.stability_mechanisms,
          total_supply: state.total_supply,
          research_group_holdings: state.research_group_holdings
        }
        
        crdt_equivalent = state.stablecoin_price * state.crdt_exchange_rate
        per_unit_usd = state.stablecoin_price
        per_unit_crdt = crdt_equivalent
        
        # Pay mining rewards to agents
        mining_reward = block.transactions * 10.0  # 10 CRDT per transaction processed
        new_agent_rewards = %{state.agent_rewards | mining: state.agent_rewards.mining + mining_reward}
        
        Logger.info("⛏️  Node #{state.node_id} mined block ##{new_height} (#{block.transactions} txs) | Price per unit: $#{Float.round(per_unit_usd, 6)} / #{Float.round(per_unit_crdt, 2)} CRDT | Mining reward: +#{Float.round(mining_reward, 2)} CRDT")
        
        broadcast_to_peers(state, {:new_block, block})
        
        new_state = %{state | 
          blockchain_height: new_height,
          last_block_time: timestamp,
          mempool_size: max(0, state.mempool_size - block.transactions),
          market_orders: [],  # Clear processed orders
          agent_rewards: new_agent_rewards
        }
        {:noreply, new_state}
      else
        {:noreply, state}
      end
    end

    def handle_info(:update_oracle_data, state) do
      if state.data_provider do
        # More sophisticated price feed with market impact
        external_pressure = calculate_external_market_pressure()
        base_change = (:rand.uniform() - 0.5) * 0.002
        market_impact = external_pressure * 0.001
        
        price_change = base_change + market_impact
        new_price = max(0.95, min(1.05, state.stablecoin_price + price_change))
        
        new_feeds = update_oracle_feeds(state.oracle_feeds, external_pressure)
        
        price_update = %{
          provider: state.node_id,
          price: new_price,
          timestamp: DateTime.utc_now(),
          feeds: new_feeds,
          market_pressure: external_pressure,
          confidence: calculate_price_confidence(new_feeds)
        }
        
        broadcast_to_peers(state, {:price_update, price_update})
        
        new_state = %{state | 
          stablecoin_price: new_price,
          oracle_feeds: new_feeds
        }
        {:noreply, new_state}
      else
        {:noreply, state}
      end
    end

    def handle_info(:stability_check, state) do
      deviation = abs(state.stablecoin_price - 1.0)
      time_since_last = DateTime.diff(DateTime.utc_now(), state.stability_mechanisms.last_intervention)
      
      # Enhanced stability mechanism with market awareness
      if deviation > 0.005 and time_since_last > 30 do  # 0.5% threshold, 30s cooldown
        {action, amount} = calculate_stability_action(state.stablecoin_price, deviation, state.total_supply)
        
        # Pay agents CRDT for stability cooperation
        stability_reward = amount / 1.0e27 * 100  # CRDT reward for stability work
        new_agent_rewards = %{state.agent_rewards | stability: state.agent_rewards.stability + stability_reward}
        Logger.info("💰 Node #{state.node_id} stability intervention: #{action} #{Float.round(amount / 1.0e27, 2)}e27 tokens (deviation: #{Float.round(deviation * 100, 2)}%) | Stability reward: +#{Float.round(stability_reward, 2)} CRDT")
        
        # Execute stability mechanism
        {new_supply, mint_amount, burn_amount} = case action do
          :mint -> 
            {state.total_supply + amount, amount, 0.0}
          :burn -> 
            {max(0, state.total_supply - amount), 0.0, amount}
          :no_action ->
            {state.total_supply, 0.0, 0.0}
        end
        
        # Broadcast stability action to market
        stability_action = %{
          node: state.node_id,
          action: action,
          amount: amount,
          new_supply: new_supply,
          target_price: 1.0,
          timestamp: DateTime.utc_now()
        }
        
        broadcast_to_peers(state, {:stability_action, stability_action})
        
        new_mechanisms = %{
          last_intervention: DateTime.utc_now(),
          intervention_count: state.stability_mechanisms.intervention_count + 1,
          total_minted: state.stability_mechanisms.total_minted + mint_amount,
          total_burned: state.stability_mechanisms.total_burned + burn_amount
        }
        
        new_state = %{state | 
          total_supply: new_supply,
          stability_mechanisms: new_mechanisms,
          agent_rewards: new_agent_rewards
        }
        {:noreply, new_state}
      else
        {:noreply, state}
      end
    end

    def handle_info(:process_market_orders, state) do
      # Process accumulated market orders
      if length(state.market_orders) > 0 do
        {processed_orders, new_price} = process_market_orders(state.market_orders, state.stablecoin_price)
        
        crdt_equivalent = new_price * state.crdt_exchange_rate
        cooperation_reward = length(processed_orders) * 5.0  # 5 CRDT per order processed
        new_agent_rewards = %{state.agent_rewards | cooperation: state.agent_rewards.cooperation + cooperation_reward}
        
        Logger.info("📊 Node #{state.node_id} processed #{length(processed_orders)} market orders | New price per unit: $#{Float.round(new_price, 6)} / #{Float.round(crdt_equivalent, 2)} CRDT | Cooperation reward: +#{Float.round(cooperation_reward, 2)} CRDT")
        
        # Add processed orders to mempool as transactions
        new_mempool_size = state.mempool_size + length(processed_orders)
        
        new_state = %{state | 
          market_orders: [],
          stablecoin_price: new_price,
          mempool_size: new_mempool_size,
          agent_rewards: new_agent_rewards
        }
        {:noreply, new_state}
      else
        {:noreply, state}
      end
    end

    def handle_info(:sync_with_market, state) do
      # Sync node state with external market conditions
      # This would connect to the market maker system
      Logger.debug("🔄 Node #{state.node_id} syncing with external markets")
      {:noreply, state}
    end

    def handle_info(:peer_discovery, state) do
      if length(state.connected_peers) < 3 do
        Logger.debug("🔍 Node #{state.node_id} discovering peers...")
      end
      {:noreply, state}
    end

    def handle_info({:tcp, _socket, data}, state) do
      case :erlang.binary_to_term(data) do
        {:new_block, block} ->
          if block.height > state.blockchain_height do
            Logger.info("📦 Node #{state.node_id} received block ##{block.height} from #{block.miner} (supply: #{Float.round(block.total_supply / 1.0e30, 2)}e30)")
            new_state = %{state | 
              blockchain_height: block.height,
              last_block_time: block.timestamp,
              total_supply: block.total_supply,
              research_group_holdings: block.research_group_holdings || state.research_group_holdings
            }
            {:noreply, new_state}
          else
            {:noreply, state}
          end
          
        {:price_update, update} ->
          crdt_equivalent = update.price * state.crdt_exchange_rate
          Logger.info("📈 Node #{state.node_id} price update per unit: $#{Float.round(update.price, 6)} / #{Float.round(crdt_equivalent, 2)} CRDT from #{update.provider} (confidence: #{Float.round(update.confidence, 2)})")
          new_state = %{state | stablecoin_price: update.price}
          {:noreply, new_state}
          
        {:stability_action, action} ->
          Logger.info("⚖️  Node #{state.node_id} received stability action: #{action.action} #{Float.round(action.amount, 2)} from node #{action.node}")
          {:noreply, state}
          
        {:market_order, order} ->
          Logger.info("💱 Node #{state.node_id} received market order: #{order.type} #{order.amount} from #{order.maker}")
          new_orders = [order | state.market_orders]
          new_state = %{state | market_orders: new_orders}
          {:noreply, new_state}
          
        {:peer_handshake, peer_info} ->
          if not Enum.member?(state.connected_peers, peer_info.node_id) do
            Logger.info("🤝 Node #{state.node_id} connected to peer #{peer_info.node_id}")
            new_peers = [peer_info.node_id | state.connected_peers]
            new_state = %{state | connected_peers: new_peers}
            {:noreply, new_state}
          else
            {:noreply, state}
          end
          
        _ -> {:noreply, state}
      end
    end

    def handle_info({:tcp_closed, _socket}, state), do: {:noreply, state}

    def handle_call(:status, _from, state) do
      status = %{
        node_id: state.node_id,
        port: state.port,
        blockchain_height: state.blockchain_height,
        stablecoin_price: state.stablecoin_price,
        crdt_price: state.crdt_price,
        crdt_exchange_rate: state.crdt_exchange_rate,
        agent_rewards: state.agent_rewards,
        agent_balances: state.agent_balances,
        connected_peers: length(state.connected_peers),
        mining: state.mining,
        data_provider: state.data_provider,
        mempool_size: state.mempool_size,
        total_supply: state.total_supply,
        stability_reserves: state.stability_reserves,
        research_group_holdings: state.research_group_holdings,
        pending_market_orders: length(state.market_orders),
        stability_stats: state.stability_mechanisms,
        uptime: DateTime.diff(DateTime.utc_now(), state.started_at)
      }
      {:reply, status, state}
    end

    # Private helper functions
    defp accept_loop(listen_socket, state) do
      case :gen_tcp.accept(listen_socket) do
        {:ok, client_socket} ->
          spawn(fn -> handle_peer_connection(client_socket, state) end)
          accept_loop(listen_socket, state)
        {:error, _reason} ->
          accept_loop(listen_socket, state)
      end
    end

    defp handle_peer_connection(socket, state) do
      handshake = {:peer_handshake, %{node_id: state.node_id, port: state.port}}
      :gen_tcp.send(socket, :erlang.term_to_binary(handshake))
      :gen_tcp.controlling_process(socket, self())
      :inet.setopts(socket, [active: true])
    end

    defp broadcast_to_peers(_state, message) do
      _encoded = :erlang.term_to_binary(message)
      # Simplified broadcasting
    end

    defp generate_initial_oracle_feeds() do
      [
        %{symbol: "BTC", price: 45000 + :rand.uniform(5000), volume: :rand.uniform(1000000), confidence: 0.95},
        %{symbol: "ETH", price: 3200 + :rand.uniform(800), volume: :rand.uniform(500000), confidence: 0.93},
        %{symbol: "ADA", price: 1.2 + :rand.uniform(), volume: :rand.uniform(100000), confidence: 0.88},
        %{symbol: "SOL", price: 180 + :rand.uniform(40), volume: :rand.uniform(200000), confidence: 0.90}
      ]
    end

    defp update_oracle_feeds(feeds, market_pressure) do
      Enum.map(feeds, fn feed ->
        base_change = (:rand.uniform() - 0.5) * 0.05 * feed.price
        pressure_impact = market_pressure * 0.02 * feed.price
        volume_change = (:rand.uniform() - 0.5) * 0.2 * feed.volume
        
        %{feed |
          price: max(feed.price * 0.9, feed.price + base_change + pressure_impact),
          volume: max(0, feed.volume + volume_change),
          confidence: max(0.5, min(1.0, feed.confidence + (:rand.uniform() - 0.5) * 0.1))
        }
      end)
    end

    defp calculate_external_market_pressure() do
      # Simulate external market pressure from various sources
      crypto_pressure = (:rand.uniform() - 0.5) * 0.5
      forex_pressure = (:rand.uniform() - 0.5) * 0.3
      sentiment_pressure = (:rand.uniform() - 0.5) * 0.4
      
      crypto_pressure + forex_pressure + sentiment_pressure
    end

    defp calculate_price_confidence(feeds) do
      avg_confidence = feeds |> Enum.map(& &1.confidence) |> Enum.sum() |> Kernel./(length(feeds))
      avg_confidence
    end

    defp calculate_stability_action(current_price, deviation, total_supply) do
      if current_price > 1.01 do
        # Price too high, mint new tokens
        mint_amount = deviation * total_supply * 0.1
        {:mint, mint_amount}
      else
        if current_price < 0.99 do
          # Price too low, burn tokens
          burn_amount = deviation * total_supply * 0.1
          {:burn, burn_amount}
        else
          {:no_action, 0.0}
        end
      end
    end

    defp process_pending_transactions(state) do
      # Simulate transaction processing
      base_transactions = :rand.uniform(10)
      market_order_transactions = length(state.market_orders)
      base_transactions + market_order_transactions
    end

    defp process_market_orders(orders, current_price) do
      # Process orders and calculate price impact
      total_buy_volume = orders |> Enum.filter(&(&1.type == :buy)) |> Enum.map(& &1.amount) |> Enum.sum()
      total_sell_volume = orders |> Enum.filter(&(&1.type == :sell)) |> Enum.map(& &1.amount) |> Enum.sum()
      
      net_pressure = total_buy_volume - total_sell_volume
      price_impact = net_pressure * 0.000001  # Simple impact model
      
      new_price = max(0.95, min(1.05, current_price + price_impact))
      
      {orders, new_price}
    end
  end

  defmodule MarketSystem do
    use GenServer
    
    def start_link(opts \\ []) do
      GenServer.start_link(__MODULE__, opts, name: __MODULE__)
    end

    def init(_opts) do
      market_makers = create_market_makers()
      global_market = GlobalMarket.new()
      
      state = %{
        market_makers: market_makers,
        global_market: global_market,
        stablecoin_nodes: [],
        order_flow: [],
        arbitrage_opportunities: [],
        market_stats: %{
          total_volume_24h: 0.0,
          total_trades: 0,
          price_stability_score: 100.0
        },
        started_at: DateTime.utc_now()
      }

      # Market update intervals
      :timer.send_interval(2000, self(), :update_global_market)
      :timer.send_interval(3000, self(), :execute_market_strategies)
      :timer.send_interval(5000, self(), :detect_arbitrage)
      :timer.send_interval(1000, self(), :send_orders_to_nodes)
      :timer.send_interval(8000, self(), :market_event_response)
      
      {:ok, state}
    end

    def handle_info(:update_global_market, state) do
      new_global_market = GlobalMarket.update(state.global_market)
      
      # Log major market events
      if length(new_global_market.active_events) > 0 do
        Logger.info("🌍 MARKET UPDATE: Phase #{new_global_market.market_phase} | Sentiment #{new_global_market.market_sentiment} | Vol #{Float.round(new_global_market.volatility_index, 1)}% | Events: #{length(new_global_market.active_events)}")
      end
      
      new_state = %{state | global_market: new_global_market}
      {:noreply, new_state}
    end

    def handle_info(:execute_market_strategies, state) do
      {new_makers, new_orders} = execute_all_strategies(state.market_makers, state.global_market)
      
      new_state = %{state | 
        market_makers: new_makers,
        order_flow: state.order_flow ++ new_orders
      }
      {:noreply, new_state}
    end

    def handle_info(:detect_arbitrage, state) do
      # Detect and log arbitrage opportunities
      opportunities = detect_simple_arbitrage(state.global_market)
      
      if length(opportunities) > 0 do
        Logger.info("🔄 Detected #{length(opportunities)} arbitrage opportunities")
      end
      
      new_state = %{state | arbitrage_opportunities: opportunities}
      {:noreply, new_state}
    end

    def handle_info(:send_orders_to_nodes, state) do
      # Send accumulated orders to stablecoin nodes
      if length(state.order_flow) > 0 do
        send_orders_to_stablecoin_nodes(state.order_flow, state.stablecoin_nodes)
        new_state = %{state | order_flow: []}
        {:noreply, new_state}
      else
        {:noreply, state}
      end
    end
    
    def handle_info(:market_event_response, state) do
      # Trigger emergency responses to major market events
      if state.global_market.market_phase in [:crisis, :crash] do
        Logger.info("🚨 EMERGENCY RESPONSE: Market in #{state.global_market.market_phase} phase - triggering stability protocols")
        
        # Generate emergency orders
        emergency_orders = generate_emergency_orders(state.global_market)
        new_state = %{state | order_flow: state.order_flow ++ emergency_orders}
        {:noreply, new_state}
      else
        {:noreply, state}
      end
    end

    def handle_call(:register_stablecoin_node, {node_pid, _}, state) do
      new_nodes = [node_pid | state.stablecoin_nodes]
      new_state = %{state | stablecoin_nodes: new_nodes}
      {:reply, :ok, new_state}
    end

    def handle_call(:get_market_status, _from, state) do
      status = %{
        market_makers: length(state.market_makers),
        pending_orders: length(state.order_flow),
        arbitrage_opportunities: length(state.arbitrage_opportunities),
        global_market: state.global_market,
        market_stats: state.market_stats
      }
      {:reply, status, state}
    end

    defp create_market_makers() do
      [
        MarketMaker.new("institutional_1", :conservative, 10_000_000),
        MarketMaker.new("hedge_fund_1", :aggressive, 25_000_000),
        MarketMaker.new("hft_trader_1", :high_frequency, 5_000_000),
        MarketMaker.new("arbitrageur_1", :arbitrage, 20_000_000),
        MarketMaker.new("retail_whale_1", :aggressive, 1_000_000)
      ]
    end

    defp execute_all_strategies(market_makers, global_market) do
      Enum.map_reduce(market_makers, [], fn maker, orders_acc ->
        {new_maker, new_orders} = execute_strategy(maker, global_market)
        {new_maker, orders_acc ++ new_orders}
      end)
    end

    defp execute_strategy(maker, global_market) do
      # Generate orders based on strategy and market conditions
      orders = case maker.strategy do
        :conservative -> generate_conservative_orders(maker, global_market)
        :aggressive -> generate_aggressive_orders(maker, global_market)
        :arbitrage -> generate_arbitrage_orders(maker, global_market)
        :high_frequency -> generate_hf_orders(maker, global_market)
      end
      
      new_maker = %{maker | trade_count: maker.trade_count + length(orders)}
      {new_maker, orders}
    end

    defp generate_conservative_orders(maker, global_market) do
      # Conservative strategy: provide liquidity around $1.00
      if global_market.volatility_index < 25 and :rand.uniform() < 0.3 do
        amount = maker.max_position_size * 0.1
        [
          %{type: :buy, amount: amount, price: 0.9995, maker: maker.id, timestamp: DateTime.utc_now()},
          %{type: :sell, amount: amount, price: 1.0005, maker: maker.id, timestamp: DateTime.utc_now()}
        ]
      else
        []
      end
    end

    defp generate_aggressive_orders(maker, global_market) do
      # Aggressive strategy: follow market sentiment
      if :rand.uniform() < 0.4 do
        amount = maker.max_position_size * 0.3
        type = case global_market.market_sentiment do
          :bullish -> :buy
          :bearish -> :sell
          :neutral -> if :rand.uniform() < 0.5, do: :buy, else: :sell
        end
        
        [%{type: type, amount: amount, maker: maker.id, timestamp: DateTime.utc_now()}]
      else
        []
      end
    end

    defp generate_arbitrage_orders(maker, global_market) do
      # Look for arbitrage based on crypto price movements
      btc_change = :rand.uniform() - 0.5
      if abs(btc_change) > 0.02 and :rand.uniform() < 0.6 do  # 2% BTC movement
        amount = maker.max_position_size * 0.4
        type = if btc_change > 0, do: :buy, else: :sell
        
        [%{type: type, amount: amount, maker: maker.id, timestamp: DateTime.utc_now(), reason: :arbitrage}]
      else
        []
      end
    end

    defp generate_hf_orders(maker, global_market) do
      # High frequency: small, quick orders
      if global_market.volatility_index > 15 and :rand.uniform() < 0.8 do
        amount = maker.max_position_size * 0.05
        type = if :rand.uniform() < 0.5, do: :buy, else: :sell
        
        [%{type: type, amount: amount, maker: maker.id, timestamp: DateTime.utc_now(), hf: true}]
      else
        []
      end
    end

    defp detect_simple_arbitrage(global_market) do
      # Simple arbitrage detection based on volatility
      if global_market.volatility_index > 30 do
        [%{type: :volatility_arb, profit_potential: global_market.volatility_index / 100}]
      else
        []
      end
    end

    defp send_orders_to_stablecoin_nodes(orders, nodes) do
      # Distribute orders to nodes
      Enum.each(orders, fn order ->
        if length(nodes) > 0 do
          target_node = Enum.random(nodes)
          try do
            # Send order to node (simplified)
            GenServer.cast(target_node, {:market_order, order})
          rescue
            _ -> Logger.debug("Failed to send order to node")
          end
        end
      end)
    end
    
    defp generate_emergency_orders(global_market) do
      case global_market.market_phase do
        :crisis ->
          # Large stabilization orders during crisis
          [
            %{type: :buy, amount: 50_000_000, maker: "emergency_fund", timestamp: DateTime.utc_now(), emergency: true},
            %{type: :buy, amount: 30_000_000, maker: "central_bank", timestamp: DateTime.utc_now(), emergency: true}
          ]
        :crash ->
          # Quick liquidity injection
          [
            %{type: :buy, amount: 25_000_000, maker: "crash_response", timestamp: DateTime.utc_now(), emergency: true}
          ]
        _ -> []
      end
    end
  end

  # Main orchestrator
  def run() do
    Logger.configure(level: :info)
    
    IO.puts """
    
    ╔═══════════════════════════════════════════════════════════════════════════════╗
    ║                🌍 Complete Stablecoin Ecosystem Demo 🌍                      ║
    ║                                                                               ║
    ║ Comprehensive simulation featuring:                                           ║
    ║ • 4 Blockchain nodes with enhanced consensus                                  ║
    ║ • 5 Market makers with different strategies                                   ║
    ║ • Real-world market dynamics and currency impacts                             ║
    ║ • Total Supply: 1.0e30 tokens (Research Group: 35%)                          ║
    ║ • Price per unit: USD + AGENTIC $CRDT (agents earn for cooperation)          ║
    ║ • CRDT rewards: mining, stability, cooperation + future ZK privacy           ║
    ║ • Automated arbitrage and stability mechanisms                                ║
    ║ • Cross-system order flow and price discovery                                 ║
    ║ • Regulatory compliance and monitoring                                        ║
    ╚═══════════════════════════════════════════════════════════════════════════════╝
    
    """

    # Start market system
    {:ok, _market_pid} = MarketSystem.start_link()
    
    # Start stablecoin nodes
    nodes = [
      %{id: 1, port: 8333, mining: true, data_provider: true, bootstrap: true},
      %{id: 2, port: 8334, mining: false, data_provider: true, bootstrap: false},
      %{id: 3, port: 8335, mining: false, data_provider: true, bootstrap: false},
      %{id: 4, port: 8336, mining: true, data_provider: true, bootstrap: false}
    ]

    node_pids = start_stablecoin_nodes(nodes)
    
    # Register nodes with market system
    Enum.each(node_pids, fn {_id, pid, _name} ->
      GenServer.call(MarketSystem, :register_stablecoin_node)
    end)
    
    # Wait for initialization
    :timer.sleep(3000)
    
    # Connect nodes
    connect_nodes(nodes)
    :timer.sleep(2000)
    
    # Start monitoring
    IO.puts "\n🚀 Complete ecosystem is now running with full market dynamics!\n"
    monitor_complete_ecosystem(node_pids)
  end

  defp start_stablecoin_nodes(nodes) do
    Enum.map(nodes, fn node_config ->
      IO.puts "🏗️  Starting Enhanced Node #{node_config.id}..."
      node_name = String.to_atom("enhanced_node_#{node_config.id}")
      {:ok, pid} = StablecoinNode.start_link(node_config, node_name)
      {node_config.id, pid, node_name}
    end)
  end

  defp connect_nodes(nodes) do
    IO.puts "🔗 Connecting enhanced nodes..."
    for node1 <- nodes, node2 <- nodes, node1.id != node2.id do
      spawn(fn -> connect_peer(node1.port, node2.port) end)
    end
  end

  defp connect_peer(from_port, to_port) do
    case :gen_tcp.connect(~c"localhost", to_port, [:binary, packet: 4, active: true], 5000) do
      {:ok, socket} ->
        handshake = {:peer_handshake, %{node_id: from_port, port: from_port}}
        :gen_tcp.send(socket, :erlang.term_to_binary(handshake))
      {:error, _reason} -> :ok
    end
  end

  defp monitor_complete_ecosystem(node_pids) do
    :timer.sleep(20000)  # Update every 20 seconds
    
    show_complete_dashboard(node_pids)
    monitor_complete_ecosystem(node_pids)
  end

  defp show_complete_dashboard(node_pids) do
    market_status = GenServer.call(MarketSystem, :get_market_status)
    
    IO.puts """
    
    ╔═══════════════════════════════════════════════════════════════════════════════╗
    ║                      🎯 Complete Ecosystem Dashboard 🎯                      ║
    ╚═══════════════════════════════════════════════════════════════════════════════╝
    
    📊 Market System Status:
    ├── Market Makers: #{market_status.market_makers}
    ├── Pending Orders: #{market_status.pending_orders}
    ├── Arbitrage Opportunities: #{market_status.arbitrage_opportunities}
    ├── BTC Price: $#{Float.round(market_status.global_market.crypto_prices["BTC"], 0)}
    ├── Market Phase: #{market_status.global_market.market_phase} | Sentiment: #{market_status.global_market.market_sentiment}
    ├── Volatility Index: #{Float.round(market_status.global_market.volatility_index, 1)}% | Flash Crash Risk: #{Float.round(market_status.global_market.flash_crash_risk * 100, 1)}%
    └── Active Events: #{length(market_status.global_market.active_events)} | Whale Activity: #{market_status.global_market.whale_activity.large_orders}
    """
    
    IO.puts "\n🏗️  Blockchain Node Status:"
    Enum.each(node_pids, fn {node_id, _pid, node_name} ->
      try do
        status = GenServer.call(node_name, :status)
        stability_score = Float.round(100 - abs(status.stablecoin_price - 1.0) * 10000, 1)
        
        crdt_equivalent = status.stablecoin_price * status.crdt_exchange_rate
        total_agent_crdt = status.agent_rewards.mining + status.agent_rewards.stability + status.agent_rewards.cooperation
        
        IO.puts """
        ├── Node #{node_id}: Height #{status.blockchain_height} | Price per unit: $#{Float.round(status.stablecoin_price, 6)} / #{Float.round(crdt_equivalent, 2)} CRDT | Stability #{stability_score}%
        │   Total Supply: #{Float.round(status.total_supply / 1.0e30, 2)}e30 | Research Holdings: #{Float.round(status.research_group_holdings / 1.0e30, 2)}e30 (35%)
        │   Agent CRDT Balances: Mining #{Float.round(status.agent_rewards.mining, 2)} | Stability #{Float.round(status.agent_rewards.stability, 2)} | Cooperation #{Float.round(status.agent_rewards.cooperation, 2)} | Total: #{Float.round(total_agent_crdt, 2)} CRDT
        │   Interventions: #{status.stability_stats.intervention_count} | Orders: #{status.pending_market_orders} | Future: ZK privacy baskets
        """
      rescue
        _ -> IO.puts "├── Node #{node_id}: ❌ Not responding"
      end
    end)
    
    IO.puts """
    
    🌐 Global Market Impact:
    ├── EUR Rate: #{Float.round(market_status.global_market.currencies["EUR"], 4)}
    ├── JPY Rate: #{Float.round(market_status.global_market.currencies["JPY"], 2)}
    ├── ETH Price: $#{Float.round(market_status.global_market.crypto_prices["ETH"], 0)}
    └── Interest Rates: USD #{market_status.global_market.interest_rates["USD"]}%, EUR #{market_status.global_market.interest_rates["EUR"]}%
    
    #{DateTime.utc_now() |> DateTime.to_string()}
    """
  end
end

# Run the complete ecosystem
CompleteStablecoinEcosystemDemo.run()