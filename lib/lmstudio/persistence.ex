defmodule LMStudio.Persistence do
  @moduledoc """
  Persistent storage for evolution system with ETS and file-based persistence.
  
  Provides both in-memory (ETS) and disk persistence for:
  - Agent states and grids
  - Evolution cycles and metrics
  - Generated code and patterns
  - Learning history and insights
  """
  
  use GenServer
  require Logger
  
  @table_name :lmstudio_persistence
  @storage_dir "priv/evolution_storage"
  @checkpoint_interval 30_000  # 30 seconds
  
  # Client API
  
  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end
  
  def store(key, value, opts \\ []) do
    persist_to_disk = Keyword.get(opts, :persist, true)
    GenServer.call(__MODULE__, {:store, key, value, persist_to_disk})
  end
  
  def get(key, default \\ nil) do
    # Ensure the GenServer is running
    case Process.whereis(__MODULE__) do
      nil ->
        # If the service isn't running, start it first
        case start_link() do
          {:ok, _pid} -> get_from_ets(key, default)
          {:error, {:already_started, _pid}} -> get_from_ets(key, default)
          _ -> default
        end
      _pid ->
        get_from_ets(key, default)
    end
  end

  defp get_from_ets(key, default) do
    case :ets.lookup(@table_name, key) do
      [{^key, value}] -> value
      [] -> default
    end
  end
  
  def delete(key) do
    GenServer.call(__MODULE__, {:delete, key})
  end
  
  def list_keys(pattern \\ :_) do
    # Ensure the GenServer is running
    case Process.whereis(__MODULE__) do
      nil ->
        # If the service isn't running, start it first
        case start_link() do
          {:ok, _pid} -> list_keys_from_ets(pattern)
          {:error, {:already_started, _pid}} -> list_keys_from_ets(pattern)
          _ -> []
        end
      _pid ->
        list_keys_from_ets(pattern)
    end
  end

  defp list_keys_from_ets(pattern) do
    if pattern == :_ do
      :ets.select(@table_name, [{{:"$1", :"$2"}, [], [:"$1"]}])
    else
      :ets.select(@table_name, [{{pattern, :"$2"}, [], [pattern]}])
    end
  end
  
  def export_to_file(filename) do
    GenServer.call(__MODULE__, {:export, filename})
  end
  
  def import_from_file(filename) do
    GenServer.call(__MODULE__, {:import, filename})
  end
  
  def get_stats do
    GenServer.call(__MODULE__, :get_stats)
  end
  
  def checkpoint_now do
    GenServer.cast(__MODULE__, :checkpoint)
  end
  
  # Server Callbacks
  
  @impl true
  def init(_opts) do
    # Create ETS table for fast in-memory access
    table = :ets.new(@table_name, [:named_table, :public, :set, {:read_concurrency, true}])
    
    # Ensure storage directory exists
    File.mkdir_p!(@storage_dir)
    
    # Load existing data from disk
    load_from_disk()
    
    # Schedule periodic checkpointing
    :timer.send_interval(@checkpoint_interval, :checkpoint)
    
    state = %{
      table: table,
      storage_dir: @storage_dir,
      last_checkpoint: DateTime.utc_now(),
      operations_since_checkpoint: 0
    }
    
    Logger.info("Persistence system initialized with ETS table and storage dir: #{@storage_dir}")
    {:ok, state}
  end
  
  @impl true
  def handle_call({:store, key, value, persist_to_disk}, _from, state) do
    # Store in ETS
    :ets.insert(@table_name, {key, value})
    
    # Optionally persist to disk immediately
    if persist_to_disk do
      persist_key_to_disk(key, value)
    end
    
    new_state = %{state | operations_since_checkpoint: state.operations_since_checkpoint + 1}
    {:reply, :ok, new_state}
  end
  
  @impl true
  def handle_call({:delete, key}, _from, state) do
    :ets.delete(@table_name, key)
    delete_key_from_disk(key)
    
    new_state = %{state | operations_since_checkpoint: state.operations_since_checkpoint + 1}
    {:reply, :ok, new_state}
  end
  
  @impl true
  def handle_call({:export, filename}, _from, state) do
    result = export_all_data(filename)
    {:reply, result, state}
  end
  
  @impl true
  def handle_call({:import, filename}, _from, state) do
    result = import_all_data(filename)
    {:reply, result, state}
  end
  
  @impl true
  def handle_call(:get_stats, _from, state) do
    stats = %{
      total_keys: :ets.info(@table_name, :size),
      last_checkpoint: state.last_checkpoint,
      operations_since_checkpoint: state.operations_since_checkpoint,
      memory_usage: :ets.info(@table_name, :memory),
      storage_dir: state.storage_dir
    }
    {:reply, stats, state}
  end
  
  @impl true
  def handle_cast(:checkpoint, state) do
    perform_checkpoint()
    new_state = %{
      state | 
      last_checkpoint: DateTime.utc_now(),
      operations_since_checkpoint: 0
    }
    {:noreply, new_state}
  end
  
  @impl true
  def handle_info(:checkpoint, state) do
    if state.operations_since_checkpoint > 0 do
      perform_checkpoint()
      new_state = %{
        state | 
        last_checkpoint: DateTime.utc_now(),
        operations_since_checkpoint: 0
      }
      {:noreply, new_state}
    else
      {:noreply, state}
    end
  end
  
  # Private Functions
  
  defp load_from_disk do
    index_file = Path.join(@storage_dir, "index.etf")
    
    if File.exists?(index_file) do
      case File.read(index_file) do
        {:ok, binary} ->
          case :erlang.binary_to_term(binary) do
            keys when is_list(keys) ->
              Enum.each(keys, fn key ->
                key_binary = :erlang.term_to_binary(key)
                key_hash = :crypto.hash(:sha256, key_binary) |> Base.encode16() |> String.slice(0, 32)
                key_file = Path.join(@storage_dir, "#{key_hash}.etf")
                if File.exists?(key_file) do
                  case File.read(key_file) do
                    {:ok, value_binary} when byte_size(value_binary) > 0 ->
                      try do
                        value = :erlang.binary_to_term(value_binary)
                        :ets.insert(@table_name, {key, value})
                      rescue
                        ArgumentError ->
                          Logger.warning("Failed to decode binary for key #{inspect(key)}, skipping")
                      end
                    {:ok, ""} ->
                      Logger.warning("Empty file for key #{inspect(key)}, skipping")
                    {:error, reason} ->
                      Logger.warning("Failed to load key #{inspect(key)}: #{inspect(reason)}")
                  end
                end
              end)
              Logger.info("Loaded #{length(keys)} keys from disk")
            _ ->
              Logger.warning("Invalid index file format")
          end
        {:error, reason} ->
          Logger.warning("Failed to read index file: #{inspect(reason)}")
      end
    else
      Logger.info("No existing persistence data found")
    end
  end
  
  defp persist_key_to_disk(key, value) do
    # Convert key to binary for safe encoding and hash to avoid long filenames
    key_binary = :erlang.term_to_binary(key)
    key_hash = :crypto.hash(:sha256, key_binary) |> Base.encode16() |> String.slice(0, 32)
    key_file = Path.join(@storage_dir, "#{key_hash}.etf")
    binary = :erlang.term_to_binary(value, [:compressed])
    
    case File.write(key_file, binary) do
      :ok -> :ok
      {:error, reason} ->
        Logger.error("Failed to persist key #{inspect(key)}: #{inspect(reason)}")
    end
  end
  
  defp delete_key_from_disk(key) do
    key_binary = :erlang.term_to_binary(key)
    key_hash = :crypto.hash(:sha256, key_binary) |> Base.encode16() |> String.slice(0, 32)
    key_file = Path.join(@storage_dir, "#{key_hash}.etf")
    File.rm(key_file)
  end
  
  defp perform_checkpoint do
    Logger.debug("Performing checkpoint...")
    
    # Get all keys from ETS
    all_keys = :ets.select(@table_name, [{{:"$1", :_}, [], [:"$1"]}])
    
    # Write index file
    index_file = Path.join(@storage_dir, "index.etf")
    index_binary = :erlang.term_to_binary(all_keys, [:compressed])
    File.write!(index_file, index_binary)
    
    # Persist all data
    :ets.foldl(fn {key, value}, acc ->
      persist_key_to_disk(key, value)
      acc + 1
    end, 0, @table_name)
    
    Logger.debug("Checkpoint completed for #{length(all_keys)} keys")
  end
  
  defp export_all_data(filename) do
    try do
      all_data = :ets.tab2list(@table_name)
      binary = :erlang.term_to_binary(all_data, [:compressed])
      
      case File.write(filename, binary) do
        :ok -> {:ok, "Exported #{length(all_data)} entries to #{filename}"}
        {:error, reason} -> {:error, reason}
      end
    rescue
      error -> {:error, error}
    end
  end
  
  defp import_all_data(filename) do
    try do
      case File.read(filename) do
        {:ok, binary} ->
          data = :erlang.binary_to_term(binary)
          :ets.delete_all_objects(@table_name)
          :ets.insert(@table_name, data)
          {:ok, "Imported #{length(data)} entries from #{filename}"}
        {:error, reason} ->
          {:error, reason}
      end
    rescue
      error -> {:error, error}
    end
  end
end

defmodule LMStudio.Persistence.Helpers do
  @moduledoc """
  Helper functions for common persistence operations in the evolution system.
  """
  
  alias LMStudio.Persistence
  
  # Agent State Persistence
  
  def save_agent_state(agent_name, state) do
    Persistence.store({:agent_state, agent_name}, state, persist: true)
  end
  
  def load_agent_state(agent_name) do
    Persistence.get({:agent_state, agent_name})
  end
  
  def save_agent_grid(agent_name, grid_data) do
    Persistence.store({:agent_grid, agent_name}, grid_data, persist: true)
  end
  
  def load_agent_grid(agent_name) do
    Persistence.get({:agent_grid, agent_name}, %{})
  end
  
  # Evolution Cycle Persistence
  
  def save_evolution_cycle(cycle_id, cycle_data) do
    Persistence.store({:evolution_cycle, cycle_id}, cycle_data, persist: true)
  end
  
  def load_evolution_cycles do
    Persistence.list_keys({:evolution_cycle, :_})
    |> Enum.map(fn key -> Persistence.get(key) end)
    |> Enum.filter(&(&1 != nil))
  end
  
  def save_global_insights(insights) do
    Persistence.store(:global_insights, insights, persist: true)
  end
  
  def load_global_insights do
    Persistence.get(:global_insights, [])
  end
  
  # Generated Code Persistence
  
  def save_generated_code(code_id, code_data) do
    Persistence.store({:generated_code, code_id}, code_data, persist: true)
  end
  
  def load_generated_code(code_id) do
    Persistence.get({:generated_code, code_id})
  end
  
  def list_generated_code do
    Persistence.list_keys({:generated_code, :_})
    |> Enum.map(fn {_, code_id} -> {code_id, Persistence.get({:generated_code, code_id})} end)
  end
  
  # Performance Metrics
  
  def save_performance_metrics(agent_name, metrics) do
    Persistence.store({:performance_metrics, agent_name}, metrics)
  end
  
  def load_performance_metrics(agent_name) do
    Persistence.get({:performance_metrics, agent_name}, [])
  end
  
  # Knowledge Base
  
  def save_knowledge_pattern(pattern_id, pattern_data) do
    Persistence.store({:knowledge_pattern, pattern_id}, pattern_data, persist: true)
  end
  
  def load_knowledge_pattern(pattern_id) do
    Persistence.get({:knowledge_pattern, pattern_id})
  end
  
  def list_knowledge_patterns do
    Persistence.list_keys({:knowledge_pattern, :_})
    |> Enum.map(fn {_, pattern_id} -> {pattern_id, Persistence.get({:knowledge_pattern, pattern_id})} end)
  end
  
  # Batch Operations
  
  def backup_all_agent_data(agent_name) do
    timestamp = DateTime.utc_now() |> DateTime.to_iso8601()
    filename = "priv/backups/agent_#{agent_name}_#{timestamp}.backup"
    File.mkdir_p!(Path.dirname(filename))
    
    agent_data = %{
      state: load_agent_state(agent_name),
      grid: load_agent_grid(agent_name),
      performance: load_performance_metrics(agent_name)
    }
    
    binary = :erlang.term_to_binary(agent_data, [:compressed])
    File.write(filename, binary)
  end
  
  def restore_agent_data(agent_name, backup_file) do
    case File.read(backup_file) do
      {:ok, binary} ->
        agent_data = :erlang.binary_to_term(binary)
        
        if agent_data.state, do: save_agent_state(agent_name, agent_data.state)
        if agent_data.grid, do: save_agent_grid(agent_name, agent_data.grid)
        if agent_data.performance, do: save_performance_metrics(agent_name, agent_data.performance)
        
        {:ok, "Restored agent data for #{agent_name}"}
      {:error, reason} ->
        {:error, reason}
    end
  end
end