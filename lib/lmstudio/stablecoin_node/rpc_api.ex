defmodule LMStudio.StablecoinNode.RpcApi do
  @moduledoc """
  JSON-RPC API server for the stablecoin node, providing HTTP endpoints
  for wallet operations, blockchain queries, and node management.
  """

  use Plug.Router
  require Logger

  alias LMStudio.StablecoinNode
  alias LMStudio.StablecoinNode.{Blockchain, Oracle, Consensus, Mempool, Wallet}

  plug Plug.Parsers, parsers: [:json], json_decoder: Jason
  plug :match
  plug :dispatch

  # Node information endpoints
  post "/rpc" do
    handle_rpc_request(conn)
  end

  get "/health" do
    send_resp(conn, 200, Jason.encode!(%{status: "healthy", timestamp: DateTime.utc_now()}))
  end

  get "/status" do
    status = StablecoinNode.status()
    send_resp(conn, 200, Jason.encode!(status))
  end

  defp handle_rpc_request(conn) do
    case conn.body_params do
      %{"method" => method, "params" => params, "id" => id} ->
        response = handle_rpc_method(method, params)
        rpc_response = %{
          jsonrpc: "2.0",
          id: id,
          result: response
        }
        send_resp(conn, 200, Jason.encode!(rpc_response))
        
      %{"method" => method, "params" => params} ->
        response = handle_rpc_method(method, params)
        rpc_response = %{
          jsonrpc: "2.0",
          result: response
        }
        send_resp(conn, 200, Jason.encode!(rpc_response))
        
      invalid_request ->
        error_response = %{
          jsonrpc: "2.0",
          error: %{
            code: -32600,
            message: "Invalid Request",
            data: invalid_request
          }
        }
        send_resp(conn, 400, Jason.encode!(error_response))
    end
  rescue
    error ->
      Logger.error("RPC error: #{inspect(error)}")
      error_response = %{
        jsonrpc: "2.0",
        error: %{
          code: -32603,
          message: "Internal error",
          data: %{error: inspect(error)}
        }
      }
      send_resp(conn, 500, Jason.encode!(error_response))
  end

  # Blockchain methods
  defp handle_rpc_method("getBlockchainInfo", _params) do
    %{
      height: StablecoinNode.height(),
      stablecoin_price: StablecoinNode.stablecoin_price(),
      data_providers: StablecoinNode.data_providers(),
      node_status: StablecoinNode.status()
    }
  end

  defp handle_rpc_method("getBlockHeight", _params) do
    StablecoinNode.height()
  end

  defp handle_rpc_method("getStablecoinPrice", _params) do
    StablecoinNode.stablecoin_price()
  end

  # Wallet methods
  defp handle_rpc_method("createAddress", _params) do
    # In a real implementation, this would use a managed wallet instance
    wallet = Wallet.new()
    {address, _new_wallet} = Wallet.generate_address(wallet)
    %{address: address}
  end

  defp handle_rpc_method("getBalance", %{"address" => address}) do
    balance = StablecoinNode.get_balance(address)
    %{address: address, balance: balance}
  end

  defp handle_rpc_method("sendTransaction", %{"from" => from, "to" => to, "amount" => amount} = params) do
    fee = Map.get(params, "fee", 0.01)
    
    case StablecoinNode.submit_transaction(%{
      from: from,
      to: to,
      amount: amount,
      fee: fee,
      type: :transfer,
      timestamp: DateTime.utc_now(),
      nonce: get_address_nonce(from)
    }) do
      :ok ->
        %{status: "submitted", message: "Transaction submitted to mempool"}
        
      {:error, reason} ->
        %{status: "error", message: "Transaction failed: #{reason}"}
    end
  end

  # Oracle methods
  defp handle_rpc_method("getOracleData", _params) do
    # Return current oracle data including top 250 crypto prices
    %{
      top_cryptos: get_top_crypto_prices(),
      data_providers: StablecoinNode.data_providers(),
      last_update: DateTime.utc_now(),
      stablecoin_price: StablecoinNode.stablecoin_price()
    }
  end

  defp handle_rpc_method("getDataProviders", _params) do
    StablecoinNode.data_providers()
  end

  # Consensus methods
  defp handle_rpc_method("getValidators", _params) do
    # Return current validator set
    %{
      active_validators: get_active_validators(),
      total_stake: get_total_staked(),
      current_leader: get_current_consensus_leader()
    }
  end

  defp handle_rpc_method("delegateStake", %{"validator" => validator, "amount" => amount}) do
    # In a real implementation, this would use the user's wallet
    %{
      status: "submitted",
      message: "Stake delegation submitted",
      validator: validator,
      amount: amount
    }
  end

  # Mining/Staking methods
  defp handle_rpc_method("startMining", _params) do
    # Start mining process
    %{status: "started", message: "Mining process started"}
  end

  defp handle_rpc_method("stopMining", _params) do
    # Stop mining process
    %{status: "stopped", message: "Mining process stopped"}
  end

  # Mempool methods
  defp handle_rpc_method("getMempoolInfo", _params) do
    %{
      size: get_mempool_size(),
      total_fees: get_mempool_total_fees(),
      pending_transactions: get_pending_transaction_count()
    }
  end

  defp handle_rpc_method("getPendingTransactions", params) do
    limit = Map.get(params, "limit", 50)
    get_pending_transactions(limit)
  end

  # Network methods
  defp handle_rpc_method("getNetworkInfo", _params) do
    %{
      peer_count: get_peer_count(),
      connected_peers: get_connected_peers(),
      network_height: StablecoinNode.height(),
      sync_status: get_sync_status()
    }
  end

  defp handle_rpc_method("getPeers", _params) do
    get_connected_peers()
  end

  defp handle_rpc_method("addPeer", %{"address" => address, "port" => port}) do
    # Add a new peer connection
    %{
      status: "connecting",
      message: "Attempting to connect to peer",
      peer: "#{address}:#{port}"
    }
  end

  # Stabilization methods
  defp handle_rpc_method("getStabilizationMetrics", _params) do
    %{
      current_price: StablecoinNode.stablecoin_price(),
      target_price: 1.0,
      price_deviation: calculate_price_deviation(),
      total_supply: get_total_supply(),
      stability_fund: get_stability_fund_balance(),
      last_adjustment: get_last_stabilization_adjustment(),
      peg_stability: assess_peg_stability()
    }
  end

  # Analytics methods
  defp handle_rpc_method("getAnalytics", params) do
    timeframe = Map.get(params, "timeframe", "24h")
    
    %{
      price_history: get_price_history(timeframe),
      volume_history: get_volume_history(timeframe),
      block_production_stats: get_block_production_stats(timeframe),
      validator_performance: get_validator_performance(timeframe),
      data_provider_accuracy: get_data_provider_accuracy(timeframe)
    }
  end

  # Administrative methods
  defp handle_rpc_method("getNodeMetrics", _params) do
    %{
      uptime: get_node_uptime(),
      memory_usage: get_memory_usage(),
      cpu_usage: get_cpu_usage(),
      disk_usage: get_disk_usage(),
      network_throughput: get_network_throughput()
    }
  end

  # Error handling for unknown methods
  defp handle_rpc_method(unknown_method, _params) do
    throw({:error, %{
      code: -32601,
      message: "Method not found",
      data: %{method: unknown_method}
    }})
  end

  # Helper functions (these would connect to actual subsystems)
  defp get_address_nonce(_address), do: 0
  defp get_top_crypto_prices, do: %{}
  defp get_active_validators, do: []
  defp get_total_staked, do: 0
  defp get_current_consensus_leader, do: nil
  defp get_mempool_size, do: 0
  defp get_mempool_total_fees, do: 0
  defp get_pending_transaction_count, do: 0
  defp get_pending_transactions(_limit), do: []
  defp get_peer_count, do: 0
  defp get_connected_peers, do: []
  defp get_sync_status, do: "synced"
  defp calculate_price_deviation, do: 0.0
  defp get_total_supply, do: 1_000_000
  defp get_stability_fund_balance, do: 100_000
  defp get_last_stabilization_adjustment, do: DateTime.utc_now()
  defp assess_peg_stability, do: "stable"
  defp get_price_history(_timeframe), do: []
  defp get_volume_history(_timeframe), do: []
  defp get_block_production_stats(_timeframe), do: %{}
  defp get_validator_performance(_timeframe), do: %{}
  defp get_data_provider_accuracy(_timeframe), do: %{}
  defp get_node_uptime, do: 3600
  defp get_memory_usage, do: %{used: 512, total: 1024}
  defp get_cpu_usage, do: 25.5
  defp get_disk_usage, do: %{used: 10, total: 100}
  defp get_network_throughput, do: %{in: 1000, out: 800}
end