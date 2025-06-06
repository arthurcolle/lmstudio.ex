defmodule LMStudio.StablecoinNode.KalshiMarketClient do
  @moduledoc """
  Production-ready Kalshi market integration client for real-time prediction market data.
  
  This module handles:
  - Authentication with Kalshi API
  - Real-time market data fetching
  - WebSocket streaming for live updates
  - Rate limiting and error handling
  - Market data processing and normalization
  """

  use GenServer
  require Logger

  @kalshi_api_base "https://trading-api.kalshi.com/trade-api/v2"
  @kalshi_demo_base "https://demo-api.kalshi.co/trade-api/v2"
  @websocket_url "wss://trading-api.kalshi.com/trade-api/ws/v2"
  @demo_websocket_url "wss://demo-api.kalshi.co/trade-api/ws/v2"
  
  @rate_limit_per_minute 100
  @max_retries 3
  @request_timeout 10_000
  @heartbeat_interval 30_000

  defstruct [
    :api_key,
    :user_id,
    :password,
    :access_token,
    :websocket_pid,
    :base_url,
    :websocket_url,
    :subscribed_markets,
    :market_cache,
    :rate_limiter,
    :last_heartbeat,
    :connection_status,
    :demo_mode
  ]

  # Market categories we're interested in for stablecoin stability
  @crypto_market_categories [
    "CRYPTO",
    "ECONOMY", 
    "REGULATION",
    "FINANCE"
  ]

  @crypto_related_keywords [
    "bitcoin", "btc", "ethereum", "eth", "tether", "usdt", "usdc", "crypto",
    "cryptocurrency", "stablecoin", "defi", "regulation", "sec", "cftc",
    "federal", "government", "treasury", "fed", "inflation", "dollar"
  ]

  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  def new(opts \\ []) do
    %__MODULE__{
      api_key: Keyword.get(opts, :api_key),
      user_id: Keyword.get(opts, :user_id),
      password: Keyword.get(opts, :password),
      demo_mode: Keyword.get(opts, :demo_mode, true),
      subscribed_markets: MapSet.new(),
      market_cache: %{},
      rate_limiter: init_rate_limiter(),
      connection_status: :disconnected
    }
  end

  # Public API
  def authenticate(client \\ __MODULE__, credentials) do
    GenServer.call(client, {:authenticate, credentials})
  end

  def get_crypto_markets(client \\ __MODULE__) do
    GenServer.call(client, :get_crypto_markets, 15_000)
  end

  def subscribe_to_market(client \\ __MODULE__, market_ticker) do
    GenServer.call(client, {:subscribe_to_market, market_ticker})
  end

  def get_market_data(client \\ __MODULE__, market_ticker) do
    GenServer.call(client, {:get_market_data, market_ticker})
  end

  def get_portfolio(client \\ __MODULE__) do
    GenServer.call(client, :get_portfolio)
  end

  def place_order(client \\ __MODULE__, order_params) do
    GenServer.call(client, {:place_order, order_params})
  end

  def get_connection_status(client \\ __MODULE__) do
    GenServer.call(client, :get_connection_status)
  end

  # GenServer callbacks
  def init(opts) do
    client = new(opts)
    
    client = %{client |
      base_url: if(client.demo_mode, do: @kalshi_demo_base, else: @kalshi_api_base),
      websocket_url: if(client.demo_mode, do: @demo_websocket_url, else: @websocket_url)
    }

    # Schedule periodic tasks
    :timer.send_interval(@heartbeat_interval, self(), :heartbeat)
    :timer.send_interval(60_000, self(), :reset_rate_limiter)
    
    Logger.info("Kalshi Market Client initialized (demo: #{client.demo_mode})")
    {:ok, client}
  end

  def handle_call({:authenticate, credentials}, _from, state) do
    case authenticate_with_kalshi(state, credentials) do
      {:ok, access_token, user_id} ->
        new_state = %{state |
          access_token: access_token,
          user_id: user_id,
          connection_status: :authenticated
        }
        
        # Start WebSocket connection after authentication
        spawn_link(fn -> start_websocket_connection(new_state) end)
        
        Logger.info("Successfully authenticated with Kalshi")
        {:reply, {:ok, :authenticated}, new_state}
        
      {:error, reason} ->
        Logger.error("Kalshi authentication failed: #{inspect(reason)}")
        {:reply, {:error, reason}, state}
    end
  end

  def handle_call(:get_crypto_markets, _from, state) do
    case check_rate_limit(state) do
      :ok ->
        case fetch_crypto_related_markets(state) do
          {:ok, markets} ->
            new_state = update_market_cache(state, markets)
            {:reply, {:ok, markets}, new_state}
          {:error, reason} ->
            {:reply, {:error, reason}, state}
        end
      :rate_limited ->
        {:reply, {:error, :rate_limited}, state}
    end
  end

  def handle_call({:subscribe_to_market, market_ticker}, _from, state) do
    case subscribe_to_market_updates(state, market_ticker) do
      :ok ->
        new_subscriptions = MapSet.put(state.subscribed_markets, market_ticker)
        new_state = %{state | subscribed_markets: new_subscriptions}
        Logger.info("Subscribed to market: #{market_ticker}")
        {:reply, :ok, new_state}
      {:error, reason} ->
        {:reply, {:error, reason}, state}
    end
  end

  def handle_call({:get_market_data, market_ticker}, _from, state) do
    case check_rate_limit(state) do
      :ok ->
        case fetch_market_details(state, market_ticker) do
          {:ok, market_data} ->
            {:reply, {:ok, market_data}, state}
          {:error, reason} ->
            {:reply, {:error, reason}, state}
        end
      :rate_limited ->
        {:reply, {:error, :rate_limited}, state}
    end
  end

  def handle_call(:get_portfolio, _from, state) do
    case check_rate_limit(state) do
      :ok ->
        case fetch_portfolio(state) do
          {:ok, portfolio} ->
            {:reply, {:ok, portfolio}, state}
          {:error, reason} ->
            {:reply, {:error, reason}, state}
        end
      :rate_limited ->
        {:reply, {:error, :rate_limited}, state}
    end
  end

  def handle_call({:place_order, order_params}, _from, state) do
    case check_rate_limit(state) do
      :ok ->
        case place_kalshi_order(state, order_params) do
          {:ok, order_response} ->
            Logger.info("Order placed successfully: #{inspect(order_response)}")
            {:reply, {:ok, order_response}, state}
          {:error, reason} ->
            {:reply, {:error, reason}, state}
        end
      :rate_limited ->
        {:reply, {:error, :rate_limited}, state}
    end
  end

  def handle_call(:get_connection_status, _from, state) do
    {:reply, state.connection_status, state}
  end

  def handle_info(:heartbeat, state) do
    new_state = %{state | last_heartbeat: DateTime.utc_now()}
    
    # Send heartbeat to WebSocket if connected
    if state.websocket_pid && Process.alive?(state.websocket_pid) do
      send(state.websocket_pid, :ping)
    end
    
    {:noreply, new_state}
  end

  def handle_info(:reset_rate_limiter, state) do
    new_state = %{state | rate_limiter: init_rate_limiter()}
    {:noreply, new_state}
  end

  def handle_info({:websocket_message, message}, state) do
    case process_websocket_message(message) do
      {:market_update, market_data} ->
        new_state = update_market_cache(state, [market_data])
        broadcast_market_update(market_data)
        {:noreply, new_state}
        
      {:error, reason} ->
        Logger.warning("WebSocket message processing error: #{inspect(reason)}")
        {:noreply, state}
        
      _ ->
        {:noreply, state}
    end
  end

  def handle_info({:websocket_closed, reason}, state) do
    Logger.warning("WebSocket connection closed: #{inspect(reason)}")
    new_state = %{state | 
      websocket_pid: nil,
      connection_status: :websocket_disconnected
    }
    
    # Attempt reconnection after delay
    :timer.send_after(5000, self(), :reconnect_websocket)
    {:noreply, new_state}
  end

  def handle_info(:reconnect_websocket, state) do
    if state.access_token do
      spawn_link(fn -> start_websocket_connection(state) end)
    end
    {:noreply, state}
  end

  # Private functions
  defp authenticate_with_kalshi(state, credentials) do
    auth_body = %{
      email: credentials[:email] || state.user_id,
      password: credentials[:password] || state.password
    }

    headers = [
      {"Content-Type", "application/json"},
      {"Accept", "application/json"}
    ]

    case make_http_request(:post, "#{state.base_url}/login", Jason.encode!(auth_body), headers) do
      {:ok, %{status: 200, body: body}} ->
        case Jason.decode(body) do
          {:ok, %{"token" => token, "member_id" => member_id}} ->
            {:ok, token, member_id}
          {:ok, response} ->
            Logger.error("Unexpected auth response: #{inspect(response)}")
            {:error, :invalid_response}
          {:error, _} ->
            {:error, :json_decode_error}
        end
        
      {:ok, %{status: status, body: body}} ->
        Logger.error("Auth failed with status #{status}: #{body}")
        {:error, {:http_error, status}}
        
      {:error, reason} ->
        {:error, reason}
    end
  end

  defp fetch_crypto_related_markets(state) do
    if not state.access_token do
      {:error, :not_authenticated}
    else
      headers = [
        {"Authorization", "Bearer #{state.access_token}"},
        {"Accept", "application/json"}
      ]

      # First, get all active markets
      case make_http_request(:get, "#{state.base_url}/markets?limit=200&status=open", "", headers) do
        {:ok, %{status: 200, body: body}} ->
          case Jason.decode(body) do
            {:ok, %{"markets" => markets}} ->
              crypto_markets = filter_crypto_markets(markets)
              {:ok, crypto_markets}
            {:error, _} ->
              {:error, :json_decode_error}
          end
          
        {:ok, %{status: status}} ->
          {:error, {:http_error, status}}
          
        {:error, reason} ->
          {:error, reason}
      end
    end
  end

  defp filter_crypto_markets(markets) do
    markets
    |> Enum.filter(&is_crypto_related_market?/1)
    |> Enum.map(&normalize_market_data/1)
  end

  defp is_crypto_related_market?(market) do
    title = String.downcase(market["title"] || "")
    subtitle = String.downcase(market["subtitle"] || "")
    category = String.downcase(market["category"] || "")
    
    # Check if market contains crypto-related keywords
    text_content = "#{title} #{subtitle} #{category}"
    
    Enum.any?(@crypto_related_keywords, fn keyword ->
      String.contains?(text_content, keyword)
    end) or category in Enum.map(@crypto_market_categories, &String.downcase/1)
  end

  defp normalize_market_data(market) do
    %{
      ticker: market["ticker"],
      title: market["title"],
      subtitle: market["subtitle"],
      category: market["category"],
      status: market["status"],
      close_time: market["close_time"],
      yes_price: get_in(market, ["yes_ask"]) || 0,
      no_price: get_in(market, ["no_ask"]) || 0,
      volume: market["volume"] || 0,
      open_interest: market["open_interest"] || 0,
      probability: calculate_probability(market),
      last_update: DateTime.utc_now(),
      raw_data: market
    }
  end

  defp calculate_probability(market) do
    yes_price = get_in(market, ["yes_ask"]) || get_in(market, ["yes_bid"]) || 0
    # Kalshi prices are in cents, so divide by 100 to get probability
    yes_price / 100.0
  end

  defp fetch_market_details(state, market_ticker) do
    if not state.access_token do
      {:error, :not_authenticated}
    else
      headers = [
        {"Authorization", "Bearer #{state.access_token}"},
        {"Accept", "application/json"}
      ]

      case make_http_request(:get, "#{state.base_url}/markets/#{market_ticker}", "", headers) do
        {:ok, %{status: 200, body: body}} ->
          case Jason.decode(body) do
            {:ok, %{"market" => market}} ->
              {:ok, normalize_market_data(market)}
            {:error, _} ->
              {:error, :json_decode_error}
          end
          
        {:ok, %{status: status}} ->
          {:error, {:http_error, status}}
          
        {:error, reason} ->
          {:error, reason}
      end
    end
  end

  defp fetch_portfolio(state) do
    if not state.access_token do
      {:error, :not_authenticated}
    else
      headers = [
        {"Authorization", "Bearer #{state.access_token}"},
        {"Accept", "application/json"}
      ]

      case make_http_request(:get, "#{state.base_url}/portfolio", "", headers) do
        {:ok, %{status: 200, body: body}} ->
          case Jason.decode(body) do
            {:ok, portfolio} ->
              {:ok, normalize_portfolio_data(portfolio)}
            {:error, _} ->
              {:error, :json_decode_error}
          end
          
        {:ok, %{status: status}} ->
          {:error, {:http_error, status}}
          
        {:error, reason} ->
          {:error, reason}
      end
    end
  end

  defp normalize_portfolio_data(portfolio) do
    %{
      balance: portfolio["balance"] || 0,
      payout: portfolio["payout"] || 0,
      positions: normalize_positions(portfolio["positions"] || []),
      last_update: DateTime.utc_now()
    }
  end

  defp normalize_positions(positions) do
    Enum.map(positions, fn position ->
      %{
        market_ticker: position["market_ticker"],
        side: position["side"],
        position_id: position["position_id"],
        quantity: position["quantity"] || 0,
        cost_basis: position["cost_basis"] || 0,
        market_value: position["market_value"] || 0,
        unrealized_pnl: position["unrealized_pnl"] || 0
      }
    end)
  end

  defp place_kalshi_order(state, order_params) do
    if not state.access_token do
      {:error, :not_authenticated}
    else
      headers = [
        {"Authorization", "Bearer #{state.access_token}"},
        {"Content-Type", "application/json"},
        {"Accept", "application/json"}
      ]

      order_body = %{
        ticker: order_params[:ticker],
        client_order_id: order_params[:client_order_id] || generate_order_id(),
        side: order_params[:side], # "yes" or "no"
        action: order_params[:action], # "buy" or "sell"
        count: order_params[:count],
        type: order_params[:type] || "market",
        yes_price: order_params[:yes_price],
        no_price: order_params[:no_price],
        expiration_time: order_params[:expiration_time]
      }

      case make_http_request(:post, "#{state.base_url}/orders", Jason.encode!(order_body), headers) do
        {:ok, %{status: 201, body: body}} ->
          case Jason.decode(body) do
            {:ok, order_response} ->
              {:ok, normalize_order_response(order_response)}
            {:error, _} ->
              {:error, :json_decode_error}
          end
          
        {:ok, %{status: status, body: body}} ->
          Logger.error("Order placement failed with status #{status}: #{body}")
          {:error, {:http_error, status}}
          
        {:error, reason} ->
          {:error, reason}
      end
    end
  end

  defp normalize_order_response(order_response) do
    %{
      order_id: order_response["order_id"],
      status: order_response["status"],
      ticker: order_response["ticker"],
      side: order_response["side"],
      action: order_response["action"],
      count: order_response["count"],
      filled: order_response["filled"] || 0,
      remaining: order_response["remaining"] || 0,
      created_time: order_response["created_time"],
      last_update: DateTime.utc_now()
    }
  end

  defp start_websocket_connection(state) do
    case :websocket_client.start_link("#{state.websocket_url}", __MODULE__.WebSocketHandler, [state]) do
      {:ok, pid} ->
        send(self(), {:websocket_connected, pid})
        {:ok, pid}
      {:error, reason} ->
        Logger.error("Failed to start WebSocket connection: #{inspect(reason)}")
        {:error, reason}
    end
  end

  defp subscribe_to_market_updates(state, market_ticker) do
    if state.websocket_pid && Process.alive?(state.websocket_pid) do
      subscribe_message = %{
        id: generate_request_id(),
        cmd: "subscribe",
        params: %{
          channels: [%{
            name: "ticker",
            market_tickers: [market_ticker]
          }]
        }
      }
      
      send(state.websocket_pid, {:send_message, Jason.encode!(subscribe_message)})
      :ok
    else
      {:error, :websocket_not_connected}
    end
  end

  defp process_websocket_message(message) do
    case Jason.decode(message) do
      {:ok, %{"type" => "ticker", "msg" => ticker_data}} ->
        market_data = normalize_ticker_data(ticker_data)
        {:market_update, market_data}
        
      {:ok, %{"type" => "error", "msg" => error_msg}} ->
        {:error, error_msg}
        
      {:ok, data} ->
        Logger.debug("Received WebSocket message: #{inspect(data)}")
        :ok
        
      {:error, _} ->
        {:error, :json_decode_error}
    end
  end

  defp normalize_ticker_data(ticker_data) do
    %{
      ticker: ticker_data["ticker"],
      yes_price: ticker_data["yes_ask"] || 0,
      no_price: ticker_data["no_ask"] || 0,
      yes_bid: ticker_data["yes_bid"] || 0,
      no_bid: ticker_data["no_bid"] || 0,
      volume: ticker_data["volume"] || 0,
      probability: (ticker_data["yes_ask"] || 0) / 100.0,
      last_update: DateTime.utc_now()
    }
  end

  defp broadcast_market_update(market_data) do
    # Broadcast to interested processes
    Registry.dispatch(LMStudio.PubSub, "kalshi_updates", fn entries ->
      for {pid, _} <- entries, do: send(pid, {:kalshi_market_update, market_data})
    end)
  end

  defp update_market_cache(state, markets) do
    new_cache = Enum.reduce(markets, state.market_cache, fn market, cache ->
      Map.put(cache, market.ticker, market)
    end)
    
    %{state | market_cache: new_cache}
  end

  defp make_http_request(method, url, body, headers, retries \\ 0) do
    case HTTPoison.request(method, url, body, headers, [timeout: @request_timeout]) do
      {:ok, response} ->
        {:ok, response}
        
      {:error, %HTTPoison.Error{reason: reason}} when retries < @max_retries ->
        Logger.warning("HTTP request failed (#{retries + 1}/#{@max_retries}): #{inspect(reason)}")
        :timer.sleep(1000 * (retries + 1))  # Exponential backoff
        make_http_request(method, url, body, headers, retries + 1)
        
      {:error, reason} ->
        {:error, reason}
    end
  end

  defp check_rate_limit(state) do
    current_count = Map.get(state.rate_limiter, :current_count, 0)
    if current_count < @rate_limit_per_minute do
      :ok
    else
      :rate_limited
    end
  end

  defp init_rate_limiter do
    %{
      current_count: 0,
      last_reset: DateTime.utc_now()
    }
  end

  defp generate_order_id do
    "STBL_" <> (:crypto.strong_rand_bytes(8) |> Base.encode16())
  end

  defp generate_request_id do
    :crypto.strong_rand_bytes(16) |> Base.encode16()
  end
end

defmodule LMStudio.StablecoinNode.KalshiMarketClient.WebSocketHandler do
  @moduledoc """
  WebSocket handler for Kalshi real-time market data.
  """
  
  @behaviour :websocket_client

  def init([state]) do
    {:once, state}
  end

  def onconnect(_req, state) do
    Logger.info("Connected to Kalshi WebSocket")
    send(state.websocket_pid, {:websocket_connected, self()})
    {:ok, state}
  end

  def ondisconnect(reason, state) do
    Logger.warning("Disconnected from Kalshi WebSocket: #{inspect(reason)}")
    send(state.websocket_pid, {:websocket_closed, reason})
    {:ok, state}
  end

  def websocket_handle({:text, message}, _conn_state, state) do
    send(state.websocket_pid, {:websocket_message, message})
    {:ok, state}
  end

  def websocket_handle({:pong, _}, _conn_state, state) do
    {:ok, state}
  end

  def websocket_info({:send_message, message}, _conn_state, state) do
    {:reply, {:text, message}, state}
  end

  def websocket_info(:ping, _conn_state, state) do
    {:reply, :ping, state}
  end

  def websocket_info(_message, _conn_state, state) do
    {:ok, state}
  end

  def websocket_terminate(reason, _conn_state, state) do
    Logger.info("WebSocket terminated: #{inspect(reason)}")
    send(state.websocket_pid, {:websocket_closed, reason})
    :ok
  end
end