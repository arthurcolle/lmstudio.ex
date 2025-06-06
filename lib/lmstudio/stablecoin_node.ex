defmodule LMStudio.StablecoinNode do
  @moduledoc """
  Full blockchain node implementation for a stablecoin pegged to top 250 cryptocurrencies.
  Features oracle-based price feeds, consensus mechanism, and data provider rewards.
  """

  use GenServer
  require Logger

  alias LMStudio.StablecoinNode.{
    Blockchain,
    Oracle,
    Consensus,
    P2P,
    Mempool,
    Wallet,
    StabilizationEngine
  }

  @doc "Starts the stablecoin node"
  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  @doc "Gets current node status"
  def status do
    GenServer.call(__MODULE__, :status)
  end

  @doc "Gets current blockchain height"
  def height do
    GenServer.call(__MODULE__, :height)
  end

  @doc "Gets current stablecoin price"
  def stablecoin_price do
    GenServer.call(__MODULE__, :stablecoin_price)
  end

  @doc "Gets top data providers and their rewards"
  def data_providers do
    GenServer.call(__MODULE__, :data_providers)
  end

  @doc "Submits a transaction to the mempool"
  def submit_transaction(tx) do
    GenServer.call(__MODULE__, {:submit_transaction, tx})
  end

  @doc "Gets balance for an address"
  def get_balance(address) do
    GenServer.call(__MODULE__, {:get_balance, address})
  end

  def init(opts) do
    Logger.info("Starting StablecoinNode...")

    state = %{
      blockchain: Blockchain.new(),
      oracle: Oracle.new(),
      consensus: Consensus.new(),
      p2p: P2P.new(),
      mempool: Mempool.new(),
      wallet: Wallet.new(),
      stabilization_engine: StabilizationEngine.new(),
      node_id: generate_node_id(),
      peers: [],
      mining: Keyword.get(opts, :mining, false),
      data_provider: Keyword.get(opts, :data_provider, false),
      started_at: DateTime.utc_now()
    }

    # Start subsystems
    {:ok, _} = Oracle.start_link()
    {:ok, _} = Consensus.start_link()
    {:ok, _} = P2P.start_link()
    {:ok, _} = Mempool.start_link()
    {:ok, _} = StabilizationEngine.start_link()

    # Schedule periodic tasks
    :timer.send_interval(1000, self(), :sync_oracle_data)
    :timer.send_interval(5000, self(), :process_mempool)
    :timer.send_interval(10000, self(), :mine_block)
    :timer.send_interval(30000, self(), :sync_peers)

    Logger.info("StablecoinNode started with ID: #{state.node_id}")
    {:ok, state}
  end

  def handle_call(:status, _from, state) do
    status = %{
      node_id: state.node_id,
      height: Blockchain.height(state.blockchain),
      peers: length(state.peers),
      mempool_size: Mempool.size(state.mempool),
      mining: state.mining,
      data_provider: state.data_provider,
      uptime: DateTime.diff(DateTime.utc_now(), state.started_at),
      stablecoin_price: Oracle.get_stablecoin_price(state.oracle)
    }
    {:reply, status, state}
  end

  def handle_call(:height, _from, state) do
    {:reply, Blockchain.height(state.blockchain), state}
  end

  def handle_call(:stablecoin_price, _from, state) do
    {:reply, Oracle.get_stablecoin_price(state.oracle), state}
  end

  def handle_call(:data_providers, _from, state) do
    providers = Oracle.get_top_data_providers(state.oracle, 12)
    {:reply, providers, state}
  end

  def handle_call({:submit_transaction, tx}, _from, state) do
    case Mempool.add_transaction(state.mempool, tx) do
      {:ok, new_mempool} ->
        new_state = %{state | mempool: new_mempool}
        {:reply, :ok, new_state}
      {:error, reason} ->
        {:reply, {:error, reason}, state}
    end
  end

  def handle_call({:get_balance, address}, _from, state) do
    balance = Blockchain.get_balance(state.blockchain, address)
    {:reply, balance, state}
  end

  def handle_info(:sync_oracle_data, state) do
    case Oracle.sync_price_feeds(state.oracle) do
      {:ok, new_oracle} ->
        # Check if stabilization is needed
        new_stabilization_engine = StabilizationEngine.check_peg(
          state.stabilization_engine,
          Oracle.get_stablecoin_price(new_oracle)
        )
        
        new_state = %{state | 
          oracle: new_oracle,
          stabilization_engine: new_stabilization_engine
        }
        {:noreply, new_state}
      {:error, _reason} ->
        {:noreply, state}
    end
  end

  def handle_info(:process_mempool, state) do
    case Mempool.get_pending_transactions(state.mempool) do
      [] ->
        {:noreply, state}
      transactions ->
        validated_txs = Enum.filter(transactions, &validate_transaction(&1, state.blockchain))
        new_mempool = Mempool.remove_transactions(state.mempool, validated_txs)
        new_state = %{state | mempool: new_mempool}
        {:noreply, new_state}
    end
  end

  def handle_info(:mine_block, state) do
    if state.mining do
      case mine_new_block(state) do
        {:ok, block, new_blockchain} ->
          Logger.info("Mined new block ##{block.height}")
          # Broadcast block to peers
          P2P.broadcast_block(state.p2p, block)
          
          new_state = %{state | blockchain: new_blockchain}
          {:noreply, new_state}
        {:error, _reason} ->
          {:noreply, state}
      end
    else
      {:noreply, state}
    end
  end

  def handle_info(:sync_peers, state) do
    new_peers = P2P.discover_peers(state.p2p)
    new_state = %{state | peers: new_peers}
    {:noreply, new_state}
  end

  defp generate_node_id do
    :crypto.strong_rand_bytes(32) |> Base.encode16()
  end

  defp validate_transaction(tx, blockchain) do
    Blockchain.validate_transaction(blockchain, tx)
  end

  defp mine_new_block(state) do
    transactions = Mempool.get_pending_transactions(state.mempool)
    oracle_data = Oracle.get_latest_data(state.oracle)
    data_provider_rewards = calculate_data_provider_rewards(oracle_data)
    
    Blockchain.mine_block(
      state.blockchain,
      transactions ++ data_provider_rewards,
      oracle_data
    )
  end

  defp calculate_data_provider_rewards(oracle_data) do
    Oracle.calculate_rewards(oracle_data, 12)
  end
end