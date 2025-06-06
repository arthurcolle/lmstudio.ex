defmodule LMStudio.StablecoinNode.Mempool do
  @moduledoc """
  Transaction mempool for managing pending transactions before they're included in blocks.
  """

  use GenServer
  require Logger

  @max_mempool_size 10_000
  @max_transaction_age 3600  # 1 hour in seconds
  @cleanup_interval 300_000  # 5 minutes

  defstruct [
    :transactions,
    :transaction_index,
    :fee_index,
    :nonce_index,
    :size,
    :total_fees
  ]

  def new do
    %__MODULE__{
      transactions: %{},
      transaction_index: %{},  # tx_id -> transaction
      fee_index: %{},          # fee_rate -> [tx_ids]
      nonce_index: %{},        # address -> nonce -> tx_id
      size: 0,
      total_fees: 0
    }
  end

  def start_link do
    GenServer.start_link(__MODULE__, [], name: __MODULE__)
  end

  def add_transaction(_mempool, transaction) do
    GenServer.call(__MODULE__, {:add_transaction, transaction})
  end

  def get_pending_transactions(_mempool) do
    GenServer.call(__MODULE__, :get_pending_transactions)
  end

  def remove_transactions(_mempool, transactions) do
    GenServer.call(__MODULE__, {:remove_transactions, transactions})
  end

  def size(mempool) do
    mempool.size
  end

  def get_transactions_by_fee(_mempool, count) do
    GenServer.call(__MODULE__, {:get_transactions_by_fee, count})
  end

  def get_transaction(_mempool, tx_id) do
    GenServer.call(__MODULE__, {:get_transaction, tx_id})
  end

  def init(_) do
    mempool = new()
    # Schedule periodic cleanup
    :timer.send_interval(@cleanup_interval, self(), :cleanup_expired)
    {:ok, mempool}
  end

  def handle_call({:add_transaction, transaction}, _from, state) do
    case validate_transaction_for_mempool(transaction, state) do
      :ok ->
        new_state = add_transaction_to_mempool(state, transaction)
        {:reply, {:ok, new_state}, new_state}
        
      {:error, reason} ->
        {:reply, {:error, reason}, state}
    end
  end

  def handle_call(:get_pending_transactions, _from, state) do
    transactions = state.transaction_index |> Map.values()
    {:reply, transactions, state}
  end

  def handle_call({:remove_transactions, transactions}, _from, state) do
    new_state = Enum.reduce(transactions, state, fn transaction, acc_state ->
      remove_transaction_from_mempool(acc_state, transaction.id)
    end)
    {:reply, new_state, new_state}
  end

  def handle_call({:get_transactions_by_fee, count}, _from, state) do
    # Get transactions sorted by fee rate (highest first)
    transactions = state.fee_index
    |> Enum.sort_by(fn {fee_rate, _tx_ids} -> -fee_rate end)
    |> Enum.flat_map(fn {_fee_rate, tx_ids} -> tx_ids end)
    |> Enum.take(count)
    |> Enum.map(fn tx_id -> Map.get(state.transaction_index, tx_id) end)
    |> Enum.filter(&(&1 != nil))
    
    {:reply, transactions, state}
  end

  def handle_call({:get_transaction, tx_id}, _from, state) do
    transaction = Map.get(state.transaction_index, tx_id)
    {:reply, transaction, state}
  end

  def handle_info(:cleanup_expired, state) do
    new_state = cleanup_expired_transactions(state)
    {:noreply, new_state}
  end

  defp validate_transaction_for_mempool(transaction, state) do
    cond do
      state.size >= @max_mempool_size ->
        {:error, :mempool_full}
        
      Map.has_key?(state.transaction_index, transaction.id) ->
        {:error, :duplicate_transaction}
        
      transaction_too_old?(transaction) ->
        {:error, :transaction_expired}
        
      invalid_nonce?(transaction, state) ->
        {:error, :invalid_nonce}
        
      insufficient_fee?(transaction) ->
        {:error, :insufficient_fee}
        
      true ->
        :ok
    end
  end

  defp add_transaction_to_mempool(state, transaction) do
    fee_rate = calculate_fee_rate(transaction)
    
    # Add to transaction index
    new_transaction_index = Map.put(state.transaction_index, transaction.id, transaction)
    
    # Add to fee index
    new_fee_index = Map.update(state.fee_index, fee_rate, [transaction.id], fn tx_ids ->
      [transaction.id | tx_ids]
    end)
    
    # Add to nonce index (for address nonce tracking)
    new_nonce_index = if transaction.from do
      Map.update(state.nonce_index, transaction.from, %{transaction.nonce => transaction.id}, fn nonce_map ->
        Map.put(nonce_map, transaction.nonce, transaction.id)
      end)
    else
      state.nonce_index
    end
    
    Logger.info("Added transaction #{transaction.id} to mempool (fee_rate: #{fee_rate})")
    
    %{state |
      transaction_index: new_transaction_index,
      fee_index: new_fee_index,
      nonce_index: new_nonce_index,
      size: state.size + 1,
      total_fees: state.total_fees + transaction.fee
    }
  end

  defp remove_transaction_from_mempool(state, tx_id) do
    case Map.get(state.transaction_index, tx_id) do
      nil ->
        state
        
      transaction ->
        fee_rate = calculate_fee_rate(transaction)
        
        # Remove from transaction index
        new_transaction_index = Map.delete(state.transaction_index, tx_id)
        
        # Remove from fee index
        new_fee_index = Map.update(state.fee_index, fee_rate, [], fn tx_ids ->
          List.delete(tx_ids, tx_id)
        end)
        |> Enum.reject(fn {_fee_rate, tx_ids} -> Enum.empty?(tx_ids) end)
        |> Map.new()
        
        # Remove from nonce index
        new_nonce_index = if transaction.from do
          Map.update(state.nonce_index, transaction.from, %{}, fn nonce_map ->
            Map.delete(nonce_map, transaction.nonce)
          end)
        else
          state.nonce_index
        end
        
        Logger.debug("Removed transaction #{tx_id} from mempool")
        
        %{state |
          transaction_index: new_transaction_index,
          fee_index: new_fee_index,
          nonce_index: new_nonce_index,
          size: state.size - 1,
          total_fees: state.total_fees - transaction.fee
        }
    end
  end

  defp cleanup_expired_transactions(state) do
    current_time = DateTime.utc_now()
    cutoff_time = DateTime.add(current_time, -@max_transaction_age, :second)
    
    expired_transactions = state.transaction_index
    |> Enum.filter(fn {_tx_id, transaction} ->
      DateTime.compare(transaction.timestamp, cutoff_time) == :lt
    end)
    |> Enum.map(fn {tx_id, _transaction} -> tx_id end)
    
    if length(expired_transactions) > 0 do
      Logger.info("Cleaning up #{length(expired_transactions)} expired transactions")
      
      Enum.reduce(expired_transactions, state, fn tx_id, acc_state ->
        remove_transaction_from_mempool(acc_state, tx_id)
      end)
    else
      state
    end
  end

  defp transaction_too_old?(transaction) do
    current_time = DateTime.utc_now()
    age = DateTime.diff(current_time, transaction.timestamp)
    age > @max_transaction_age
  end

  defp invalid_nonce?(transaction, state) do
    # Check if this nonce is already used for this address
    case Map.get(state.nonce_index, transaction.from) do
      nil ->
        false  # First transaction from this address
        
      nonce_map ->
        Map.has_key?(nonce_map, transaction.nonce)
    end
  end

  defp insufficient_fee?(transaction) do
    min_fee = calculate_minimum_fee(transaction)
    transaction.fee < min_fee
  end

  defp calculate_fee_rate(transaction) do
    # Calculate fee per byte (simplified - would use actual transaction size)
    transaction_size = estimate_transaction_size(transaction)
    if transaction_size > 0 do
      transaction.fee / transaction_size
    else
      0
    end
  end

  defp estimate_transaction_size(transaction) do
    # Simplified transaction size estimation
    base_size = 250  # Base transaction overhead
    
    # Add size for inputs/outputs
    input_size = if transaction.from, do: 150, else: 0
    output_size = if transaction.to, do: 34, else: 0
    
    # Add size for additional data
    data_size = if transaction.data do
      byte_size(:erlang.term_to_binary(transaction.data))
    else
      0
    end
    
    base_size + input_size + output_size + data_size
  end

  defp calculate_minimum_fee(transaction) do
    # Minimum fee calculation based on transaction size and network conditions
    transaction_size = estimate_transaction_size(transaction)
    base_fee_rate = 0.001  # Base fee per byte
    
    max(transaction_size * base_fee_rate, 0.01)  # Minimum absolute fee
  end
end