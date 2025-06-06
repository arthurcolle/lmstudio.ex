defmodule LMStudio.StablecoinNode.Oracle do
  @moduledoc """
  Oracle system for aggregating price data from top 250 cryptocurrencies
  and managing data provider rewards and rankings.
  """

  use GenServer
  require Logger

  @top_cryptos [
    "bitcoin", "ethereum", "tether", "bnb", "solana", "usdc", "xrp", "dogecoin",
    "cardano", "avalanche", "tron", "chainlink", "polygon", "polkadot", "internet-computer",
    "uniswap", "litecoin", "near", "dai", "bitcoin-cash", "ethereum-classic", "stellar",
    "cosmos", "monero", "okb", "filecoin", "hedera", "cronos", "vechain", "algorand",
    "quant", "fantom", "aave", "theta", "elrond", "flow", "tezos", "decentraland",
    "sandbox", "axie-infinity", "chiliz", "helium", "kucoin-shares", "neo", "bittorrent",
    "eos", "iota", "gala", "huobi-token", "maker", "pancakeswap", "compound"
    # ... continuing to 250 cryptos (generated dynamically)
  ]

  @data_providers [
    %{id: "coinbase", name: "Coinbase Pro", weight: 0.15, reliability: 0.98},
    %{id: "binance", name: "Binance", weight: 0.15, reliability: 0.97},
    %{id: "kraken", name: "Kraken", weight: 0.12, reliability: 0.96},
    %{id: "coinmarketcap", name: "CoinMarketCap", weight: 0.10, reliability: 0.95},
    %{id: "coingecko", name: "CoinGecko", weight: 0.10, reliability: 0.94},
    %{id: "huobi", name: "Huobi Global", weight: 0.08, reliability: 0.93},
    %{id: "okx", name: "OKX", weight: 0.08, reliability: 0.92},
    %{id: "bybit", name: "Bybit", weight: 0.07, reliability: 0.91},
    %{id: "kucoin", name: "KuCoin", weight: 0.06, reliability: 0.90},
    %{id: "gate", name: "Gate.io", weight: 0.05, reliability: 0.89},
    %{id: "mexc", name: "MEXC", weight: 0.04, reliability: 0.88},
    %{id: "bitfinex", name: "Bitfinex", weight: 0.03, reliability: 0.87}
  ]

  defstruct [
    :price_feeds,
    :data_providers,
    :stablecoin_price,
    :last_update,
    :provider_scores,
    :aggregated_prices,
    :rewards_pool
  ]

  def new do
    %__MODULE__{
      price_feeds: %{},
      data_providers: @data_providers,
      stablecoin_price: 1.0,
      last_update: DateTime.utc_now(),
      provider_scores: init_provider_scores(),
      aggregated_prices: %{},
      rewards_pool: 0
    }
  end

  def start_link do
    GenServer.start_link(__MODULE__, [], name: __MODULE__)
  end

  def sync_price_feeds(oracle) do
    GenServer.call(__MODULE__, {:sync_price_feeds, oracle})
  end

  def get_stablecoin_price(oracle) do
    oracle.stablecoin_price
  end

  def get_latest_data(oracle) do
    %{
      prices: oracle.aggregated_prices,
      stablecoin_price: oracle.stablecoin_price,
      timestamp: oracle.last_update,
      data_providers: oracle.data_providers
    }
  end

  def get_top_data_providers(oracle, count) do
    oracle.provider_scores
    |> Enum.sort_by(fn {_id, score} -> -score.total_score end)
    |> Enum.take(count)
    |> Enum.map(fn {id, score} ->
      provider = Enum.find(oracle.data_providers, &(&1.id == id))
      Map.put(score, :provider, provider)
    end)
  end

  def calculate_rewards(oracle_data, top_count) do
    oracle_data
    |> Map.get(:data_providers, [])
    |> Enum.take(top_count)
    |> Enum.map(&create_reward_transaction/1)
  end

  def init(_) do
    oracle = new()
    # Schedule price updates every 30 seconds
    :timer.send_interval(30_000, self(), :update_prices)
    {:ok, oracle}
  end

  def handle_call({:sync_price_feeds, _oracle}, _from, state) do
    case fetch_all_price_data() do
      {:ok, price_data} ->
        new_state = %{state |
          price_feeds: price_data,
          aggregated_prices: aggregate_prices(price_data),
          stablecoin_price: calculate_stablecoin_price(price_data),
          last_update: DateTime.utc_now(),
          provider_scores: update_provider_scores(state.provider_scores, price_data)
        }
        {:reply, {:ok, new_state}, new_state}
      {:error, reason} ->
        Logger.error("Failed to sync price feeds: #{inspect(reason)}")
        {:reply, {:error, reason}, state}
    end
  end

  def handle_info(:update_prices, state) do
    case sync_price_feeds(state) do
      {:ok, new_state} ->
        {:noreply, new_state}
      {:error, _reason} ->
        {:noreply, state}
    end
  end

  defp fetch_all_price_data do
    tasks = @data_providers
    |> Enum.map(fn provider ->
      Task.async(fn -> fetch_provider_data(provider) end)
    end)

    results = Task.await_many(tasks, 10_000)
    
    successful_results = Enum.filter(results, fn
      {:ok, _} -> true
      _ -> false
    end)

    if length(successful_results) >= 3 do
      aggregated_data = successful_results
      |> Enum.map(fn {:ok, data} -> data end)
      |> merge_provider_data()
      
      {:ok, aggregated_data}
    else
      {:error, :insufficient_data_providers}
    end
  end

  defp fetch_provider_data(provider) do
    case provider.id do
      "coinbase" -> fetch_coinbase_data()
      "binance" -> fetch_binance_data()
      "kraken" -> fetch_kraken_data()
      "coinmarketcap" -> fetch_cmc_data()
      "coingecko" -> fetch_coingecko_data()
      _ -> fetch_mock_data(provider)
    end
  end

  defp fetch_coinbase_data do
    # Simulate Coinbase Pro API call
    simulate_api_call("coinbase", fn ->
      get_all_cryptos()
      |> Enum.take(100)  # Coinbase has fewer pairs
      |> Enum.map(fn crypto ->
        {crypto, %{
          price: simulate_price(crypto),
          volume_24h: :rand.uniform(1_000_000),
          timestamp: DateTime.utc_now(),
          source: "coinbase"
        }}
      end)
      |> Map.new()
    end)
  end

  defp fetch_binance_data do
    # Simulate Binance API call
    simulate_api_call("binance", fn ->
      get_all_cryptos()
      |> Enum.take(200)  # Binance has more pairs
      |> Enum.map(fn crypto ->
        {crypto, %{
          price: simulate_price(crypto),
          volume_24h: :rand.uniform(2_000_000),
          timestamp: DateTime.utc_now(),
          source: "binance"
        }}
      end)
      |> Map.new()
    end)
  end

  defp fetch_kraken_data do
    simulate_api_call("kraken", fn ->
      get_all_cryptos()
      |> Enum.take(80)
      |> Enum.map(fn crypto ->
        {crypto, %{
          price: simulate_price(crypto),
          volume_24h: :rand.uniform(800_000),
          timestamp: DateTime.utc_now(),
          source: "kraken"
        }}
      end)
      |> Map.new()
    end)
  end

  defp fetch_cmc_data do
    simulate_api_call("coinmarketcap", fn ->
      get_all_cryptos()
      |> Enum.map(fn crypto ->
        {crypto, %{
          price: simulate_price(crypto),
          market_cap: :rand.uniform(100_000_000_000),
          volume_24h: :rand.uniform(1_500_000),
          timestamp: DateTime.utc_now(),
          source: "coinmarketcap"
        }}
      end)
      |> Map.new()
    end)
  end

  defp fetch_coingecko_data do
    simulate_api_call("coingecko", fn ->
      get_all_cryptos()
      |> Enum.map(fn crypto ->
        {crypto, %{
          price: simulate_price(crypto),
          market_cap: :rand.uniform(100_000_000_000),
          volume_24h: :rand.uniform(1_200_000),
          timestamp: DateTime.utc_now(),
          source: "coingecko"
        }}
      end)
      |> Map.new()
    end)
  end

  defp fetch_mock_data(provider) do
    simulate_api_call(provider.id, fn ->
      get_all_cryptos()
      |> Enum.take(150)
      |> Enum.map(fn crypto ->
        {crypto, %{
          price: simulate_price(crypto),
          volume_24h: :rand.uniform(1_000_000),
          timestamp: DateTime.utc_now(),
          source: provider.id
        }}
      end)
      |> Map.new()
    end)
  end

  defp simulate_api_call(provider_id, data_fn) do
    # Simulate network latency and potential failures
    :timer.sleep(:rand.uniform(1000))
    
    reliability = get_provider_reliability(provider_id)
    if :rand.uniform() <= reliability do
      {:ok, data_fn.()}
    else
      {:error, :provider_timeout}
    end
  end

  defp simulate_price(crypto) do
    base_prices = %{
      "bitcoin" => 45000,
      "ethereum" => 2800,
      "tether" => 1.0,
      "bnb" => 320,
      "solana" => 110,
      "usdc" => 1.0,
      "xrp" => 0.52,
      "dogecoin" => 0.08,
      "cardano" => 0.48,
      "avalanche" => 38
    }
    
    base_price = Map.get(base_prices, crypto, :rand.uniform(1000))
    # Add some realistic volatility
    volatility = ((:rand.uniform() - 0.5) * 0.1) + 1
    base_price * volatility
  end

  defp merge_provider_data(provider_data_list) do
    all_cryptos = provider_data_list
    |> Enum.flat_map(&Map.keys/1)
    |> Enum.uniq()

    all_cryptos
    |> Enum.map(fn crypto ->
      prices = provider_data_list
      |> Enum.filter(&Map.has_key?(&1, crypto))
      |> Enum.map(&get_in(&1, [crypto, :price]))
      
      if length(prices) >= 2 do
        {crypto, %{
          prices: prices,
          sources: get_sources_for_crypto(provider_data_list, crypto),
          weighted_price: calculate_weighted_price(provider_data_list, crypto),
          timestamp: DateTime.utc_now()
        }}
      else
        nil
      end
    end)
    |> Enum.filter(&(&1 != nil))
    |> Map.new()
  end

  defp get_sources_for_crypto(provider_data_list, crypto) do
    provider_data_list
    |> Enum.filter(&Map.has_key?(&1, crypto))
    |> Enum.map(&get_in(&1, [crypto, :source]))
  end

  defp calculate_weighted_price(provider_data_list, crypto) do
    prices_with_weights = provider_data_list
    |> Enum.filter(&Map.has_key?(&1, crypto))
    |> Enum.map(fn data ->
      source = get_in(data, [crypto, :source])
      price = get_in(data, [crypto, :price])
      weight = get_provider_weight(source)
      {price, weight}
    end)

    total_weight = prices_with_weights |> Enum.map(&elem(&1, 1)) |> Enum.sum()
    weighted_sum = prices_with_weights
    |> Enum.map(fn {price, weight} -> price * weight end)
    |> Enum.sum()

    if total_weight > 0, do: weighted_sum / total_weight, else: 0
  end

  defp aggregate_prices(price_feeds) do
    price_feeds
    |> Enum.map(fn {crypto, data} ->
      {crypto, data.weighted_price}
    end)
    |> Map.new()
  end

  defp calculate_stablecoin_price(price_feeds) do
    if map_size(price_feeds) == 0 do
      1.0
    else
      # Calculate market-cap weighted average of top 250 cryptos
      total_market_caps = price_feeds
      |> Enum.map(fn {crypto, _data} ->
        # Simulate market cap based on price and circulating supply
        price = get_in(price_feeds, [crypto, :weighted_price]) || 0
        supply = simulate_circulating_supply(crypto)
        price * supply
      end)
      |> Enum.sum()

      weighted_prices = price_feeds
      |> Enum.map(fn {crypto, data} ->
        price = data.weighted_price
        supply = simulate_circulating_supply(crypto)
        market_cap = price * supply
        weight = if total_market_caps > 0, do: market_cap / total_market_caps, else: 0
        price * weight
      end)
      |> Enum.sum()

      # Normalize to maintain $1 peg
      if weighted_prices > 0, do: weighted_prices / 1000, else: 1.0
    end
  end

  defp simulate_circulating_supply(crypto) do
    supplies = %{
      "bitcoin" => 19_500_000,
      "ethereum" => 120_000_000,
      "tether" => 83_000_000_000,
      "bnb" => 166_000_000,
      "solana" => 400_000_000
    }
    
    Map.get(supplies, crypto, :rand.uniform(1_000_000_000))
  end

  defp get_provider_weight(provider_id) do
    provider = Enum.find(@data_providers, &(&1.id == provider_id))
    if provider, do: provider.weight, else: 0.01
  end

  defp get_provider_reliability(provider_id) do
    provider = Enum.find(@data_providers, &(&1.id == provider_id))
    if provider, do: provider.reliability, else: 0.5
  end

  defp init_provider_scores do
    @data_providers
    |> Enum.map(fn provider ->
      {provider.id, %{
        total_score: 0,
        accuracy_score: 0,
        uptime_score: 0,
        speed_score: 0,
        reward_earned: 0
      }}
    end)
    |> Map.new()
  end

  defp update_provider_scores(current_scores, price_data) do
    @data_providers
    |> Enum.map(fn provider ->
      score = Map.get(current_scores, provider.id, %{total_score: 0})
      new_score = calculate_provider_score(provider, price_data, score)
      {provider.id, new_score}
    end)
    |> Map.new()
  end

  defp calculate_provider_score(provider, price_data, current_score) do
    # Calculate accuracy based on how close provider's prices are to consensus
    accuracy = calculate_accuracy_score(provider.id, price_data)
    uptime = provider.reliability
    speed = calculate_speed_score(provider.id)
    
    total_score = (accuracy * 0.5) + (uptime * 0.3) + (speed * 0.2)
    
    %{
      total_score: total_score,
      accuracy_score: accuracy,
      uptime_score: uptime,
      speed_score: speed,
      reward_earned: current_score.reward_earned || 0
    }
  end

  defp calculate_accuracy_score(provider_id, price_data) do
    # Compare provider prices to weighted consensus
    :rand.uniform()  # Placeholder - would calculate actual accuracy
  end

  defp calculate_speed_score(_provider_id) do
    # Measure response time
    :rand.uniform()  # Placeholder
  end

  defp create_reward_transaction(provider_data) do
    reward_amount = calculate_reward_amount(provider_data)
    
    %{
      type: :data_provider_reward,
      provider_id: provider_data.provider.id,
      amount: reward_amount,
      timestamp: DateTime.utc_now(),
      block_reward: true
    }
  end

  defp calculate_reward_amount(provider_data) do
    base_reward = 10.0
    score_multiplier = provider_data.total_score
    base_reward * score_multiplier
  end

  defp get_all_cryptos do
    @top_cryptos ++ generate_additional_cryptos(200)
  end

  defp generate_additional_cryptos(count) do
    1..count
    |> Enum.map(fn i -> "crypto-#{i}" end)
  end
end