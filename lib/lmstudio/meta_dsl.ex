defmodule LMStudio.MetaDSL do
  @moduledoc """
  Self-Modifying MetaDSL Cognitive System for Elixir
  ================================================

  A continuously evolving system that uses MetaDSL to mutate its own prompts,
  reasoning patterns, and cognitive strategies through recursive self-improvement.

  This system combines:
  - MetaDSL for prompt mutation
  - Cognitive agents with thinking capabilities  
  - Self-modification through grid mutations
  - Continuous learning and evolution using OTP
  """

  # ========== Mutation Types ==========

  defmodule MutationType do
    @moduledoc "Types of mutations for self-modification"
    
    @type t :: :append | :replace | :delete | :insert | :compress | :expand | 
               :evolve | :merge | :fork | :mutate_strategy
    
    def types do
      [:append, :replace, :delete, :insert, :compress, :expand, 
       :evolve, :merge, :fork, :mutate_strategy]
    end
    
    def from_string(str) when is_binary(str) do
      case str do
        "append" -> {:ok, :append}
        "replace" -> {:ok, :replace}
        "delete" -> {:ok, :delete}
        "insert" -> {:ok, :insert}
        "compress" -> {:ok, :compress}
        "expand" -> {:ok, :expand}
        "evolve" -> {:ok, :evolve}
        "merge" -> {:ok, :merge}
        "fork" -> {:ok, :fork}
        "mutate_strategy" -> {:ok, :mutate_strategy}
        _ -> {:error, :invalid_mutation_type}
      end
    end
  end

  defmodule Mutation do
    @moduledoc "Represents a mutation operation"
    
    @type t :: %__MODULE__{
      type: MutationType.t(),
      target: String.t(),
      content: String.t() | nil,
      position: integer() | nil,
      metadata: map(),
      confidence: float(),
      reasoning: String.t() | nil,
      timestamp: DateTime.t()
    }
    
    defstruct [
      :type,
      :target,
      :content,
      :position,
      metadata: %{},
      confidence: 1.0,
      reasoning: nil,
      timestamp: nil
    ]
    
    def new(type, target, opts \\ []) do
      %__MODULE__{
        type: type,
        target: target,
        content: Keyword.get(opts, :content),
        position: Keyword.get(opts, :position),
        metadata: Keyword.get(opts, :metadata, %{}),
        confidence: Keyword.get(opts, :confidence, 1.0),
        reasoning: Keyword.get(opts, :reasoning),
        timestamp: DateTime.utc_now()
      }
    end
  end

  # ========== Self-Modifying Grid GenServer ==========

  defmodule SelfModifyingGrid do
    @moduledoc """
    GenServer that maintains a grid of data that can modify itself based on performance feedback.
    
    The grid stores key-value pairs and tracks mutations, performance metrics, and evolution.
    Enhanced with persistence capabilities for continuous learning.
    """
    
    use GenServer
    require Logger
    alias LMStudio.Persistence.Helpers
    
    @type state :: %{
      data: map(),
      mutation_history: [Mutation.t()],
      performance_metrics: [float()],
      evolution_generation: integer(),
      meta_prompts: map(),
      max_history: integer(),
      grid_id: String.t(),
      auto_persist: boolean(),
      last_persist_time: DateTime.t()
    }
    
    # Client API
    
    def start_link(opts \\ []) do
      initial_data = Keyword.get(opts, :initial_data, %{})
      grid_id = Keyword.get(opts, :grid_id, generate_grid_id())
      auto_persist = Keyword.get(opts, :auto_persist, true)
      
      init_opts = %{
        initial_data: initial_data,
        grid_id: grid_id,
        auto_persist: auto_persist
      }
      
      # Use grid_id as unique name if no name provided
      case Keyword.get(opts, :name) do
        nil -> GenServer.start_link(__MODULE__, init_opts)
        name -> GenServer.start_link(__MODULE__, init_opts, name: name)
      end
    end
    
    def get_data(pid) do
      GenServer.call(pid, :get_data)
    end
    
    def get_state(pid) do
      GenServer.call(pid, :get_state)
    end
    
    def mutate(pid, mutation) do
      GenServer.call(pid, {:mutate, mutation})
    end
    
    def add_performance(pid, score) when is_number(score) do
      GenServer.cast(pid, {:add_performance, score})
    end
    
    def analyze_patterns(pid) do
      GenServer.call(pid, :analyze_patterns)
    end
    
    def suggest_mutations(pid) do
      GenServer.call(pid, :suggest_mutations)
    end
    
    def render_grid(pid) do
      GenServer.call(pid, :render_grid)
    end
    
    def persist_now(pid) do
      GenServer.cast(pid, :persist_now)
    end
    
    def load_from_persistence(pid, grid_id) do
      GenServer.call(pid, {:load_from_persistence, grid_id})
    end
    
    def export_grid(pid, filename) do
      GenServer.call(pid, {:export_grid, filename})
    end
    
    def import_grid(pid, filename) do
      GenServer.call(pid, {:import_grid, filename})
    end
    
    # Server Callbacks
    
    @impl true
    def init(opts) do
      grid_id = opts.grid_id
      auto_persist = opts.auto_persist
      initial_data = opts.initial_data
      
      # Try to load from persistence first
      {loaded_data, loaded_state} = load_persisted_state(grid_id, initial_data)
      
      state = Map.merge(%{
        data: loaded_data,
        mutation_history: [],
        performance_metrics: [],
        evolution_generation: 0,
        meta_prompts: %{
          "mutation_strategy" => "I analyze and evolve my own reasoning patterns",
          "learning_objective" => "Continuously improve through recursive self-modification", 
          "evolution_policy" => "Favor mutations that increase insight depth and coherence"
        },
        max_history: 1000,
        grid_id: grid_id,
        auto_persist: auto_persist,
        last_persist_time: DateTime.utc_now()
      }, loaded_state)
      
      # Schedule periodic persistence if auto_persist is enabled
      if auto_persist do
        :timer.send_interval(30_000, :persist_state)
      end
      
      Logger.info("SelfModifyingGrid #{grid_id} initialized with #{map_size(state.data)} data entries")
      {:ok, state}
    end
    
    @impl true
    def handle_call(:get_data, _from, state) do
      {:reply, state.data, state}
    end
    
    @impl true
    def handle_call(:get_state, _from, state) do
      {:reply, state, state}
    end
    
    @impl true
    def handle_call({:mutate, mutation}, _from, state) do
      case apply_mutation(state, mutation) do
        {:ok, new_state} ->
          Logger.info("Applied mutation: #{mutation.type} on #{mutation.target}")
          {:reply, :ok, new_state}
        {:error, reason} ->
          Logger.error("Mutation failed: #{inspect(reason)}")
          {:reply, {:error, reason}, state}
      end
    end
    
    @impl true
    def handle_call(:analyze_patterns, _from, state) do
      analysis = perform_pattern_analysis(state)
      {:reply, analysis, state}
    end
    
    @impl true
    def handle_call(:suggest_mutations, _from, state) do
      suggestions = generate_mutation_suggestions(state)
      {:reply, suggestions, state}
    end
    
    @impl true
    def handle_call(:render_grid, _from, state) do
      rendered = render_grid_content(state.data)
      {:reply, rendered, state}
    end
    
    @impl true
    def handle_call({:load_from_persistence, grid_id}, _from, state) do
      case load_grid_from_persistence(grid_id) do
        {:ok, loaded_state} ->
          new_state = Map.merge(state, loaded_state)
          Logger.info("Loaded grid #{grid_id} from persistence")
          {:reply, {:ok, :loaded}, new_state}
        {:error, reason} ->
          Logger.warning("Failed to load grid #{grid_id}: #{inspect(reason)}")
          {:reply, {:error, reason}, state}
      end
    end
    
    @impl true
    def handle_call({:export_grid, filename}, _from, state) do
      result = export_grid_to_file(state, filename)
      {:reply, result, state}
    end
    
    @impl true
    def handle_call({:import_grid, filename}, _from, state) do
      case import_grid_from_file(filename) do
        {:ok, imported_state} ->
          new_state = Map.merge(state, imported_state)
          if state.auto_persist do
            persist_grid_state(new_state)
          end
          {:reply, {:ok, :imported}, new_state}
        {:error, reason} ->
          {:reply, {:error, reason}, state}
      end
    end
    
    @impl true
    def handle_cast({:add_performance, score}, state) do
      new_metrics = [score | state.performance_metrics]
      |> Enum.take(state.max_history)
      
      new_state = %{state | performance_metrics: new_metrics}
      
      # Auto persist if performance change is significant
      if state.auto_persist and should_persist_on_performance_change?(score, state.performance_metrics) do
        persist_grid_state(new_state)
      end
      
      {:noreply, new_state}
    end
    
    @impl true
    def handle_cast(:persist_now, state) do
      persist_grid_state(state)
      new_state = %{state | last_persist_time: DateTime.utc_now()}
      Logger.debug("Grid #{state.grid_id} persisted on demand")
      {:noreply, new_state}
    end
    
    @impl true
    def handle_info(:persist_state, state) do
      if state.auto_persist and should_persist?(state) do
        persist_grid_state(state)
        new_state = %{state | last_persist_time: DateTime.utc_now()}
        {:noreply, new_state}
      else
        {:noreply, state}
      end
    end
    
    # Private Functions
    
    defp apply_mutation(state, %Mutation{type: :append, target: target, content: content}) do
      current_value = Map.get(state.data, target, "")
      new_data = Map.put(state.data, target, current_value <> content)
      update_state_after_mutation(state, new_data, nil)
    end
    
    defp apply_mutation(state, %Mutation{type: :replace, target: target, content: content}) do
      case content do
        nil -> {:error, :content_required_for_replace}
        content ->
          new_data = Enum.reduce(state.data, %{}, fn {key, value}, acc ->
            new_value = if String.contains?(value, target) do
              String.replace(value, target, content)
            else
              value
            end
            Map.put(acc, key, new_value)
          end)
          update_state_after_mutation(state, new_data, nil)
      end
    end
    
    defp apply_mutation(state, %Mutation{type: :delete, target: target}) do
      new_data = Map.delete(state.data, target)
      update_state_after_mutation(state, new_data, nil)
    end
    
    defp apply_mutation(state, %Mutation{type: :evolve, target: target, content: evolution_prompt}) do
      case evolve_content(state, target, evolution_prompt) do
        {:ok, new_data} -> update_state_after_mutation(state, new_data, nil)
        error -> error
      end
    end
    
    defp apply_mutation(state, %Mutation{type: :mutate_strategy, target: target, content: content, confidence: confidence}) do
      # Use confidence to determine mutation content
      new_content = content || (if confidence > 0.8, do: "High-confidence strategic evolution", else: "Adaptive strategy refinement")
      new_meta_prompts = Map.put(state.meta_prompts, target, new_content)
      update_state_after_mutation(state, state.data, new_meta_prompts)
    end
    
    defp apply_mutation(state, %Mutation{type: :compress, target: target}) do
      case Map.get(state.data, target) do
        nil -> {:error, :target_not_found}
        content ->
          compressed = compress_content(content)
          new_data = Map.put(state.data, target, compressed)
          update_state_after_mutation(state, new_data, nil)
      end
    end
    
    defp apply_mutation(_state, mutation) do
      {:error, {:unsupported_mutation_type, mutation.type}}
    end
    
    defp update_state_after_mutation(state, new_data, new_meta_prompts) do
      new_state = %{
        state |
        data: new_data,
        meta_prompts: new_meta_prompts || state.meta_prompts,
        evolution_generation: state.evolution_generation + 1,
        mutation_history: limit_history([%{mutation: true} | state.mutation_history], state.max_history)
      }
      
      # Auto persist after mutations if enabled
      if state.auto_persist do
        persist_grid_state(new_state)
      end
      
      {:ok, new_state}
    end
    
    defp evolve_content(state, key, evolution_prompt) do
      case Map.get(state.data, key) do
        nil -> {:error, :key_not_found}
        current_content ->
          fitness = calculate_fitness(state.performance_metrics)
          
          new_content = if fitness > 0.7 do
            # Successful pattern - enhance and expand
            "#{current_content}\n[Evolution #{state.evolution_generation + 1}]: #{evolution_prompt}"
          else
            # Poor performance - try different approach
            "[Mutation #{state.evolution_generation + 1}]: #{evolution_prompt}\n#{String.slice(current_content, 0, 100)}..."
          end
          
          new_data = Map.put(state.data, key, new_content)
          {:ok, new_data}
      end
    end
    
    defp compress_content(content) when byte_size(content) > 500 do
      # Simple compression: keep first 200 chars + "..." + last 200 chars
      first_part = String.slice(content, 0, 200)
      last_part = String.slice(content, -200, 200)
      "[Compressed]: #{first_part}...#{last_part}"
    end
    defp compress_content(content), do: content
    
    defp calculate_fitness([]), do: 0.5
    defp calculate_fitness(metrics) do
      metrics
      |> Enum.take(5)
      |> Enum.sum()
      |> then(&(&1 / length(Enum.take(metrics, 5))))
    end
    
    defp perform_pattern_analysis(state) do
      mutation_count = length(state.mutation_history)
      avg_performance = calculate_fitness(state.performance_metrics)
      
      %{
        total_mutations: mutation_count,
        evolution_generation: state.evolution_generation,
        avg_performance: avg_performance,
        data_keys: Map.keys(state.data),
        recent_performance: Enum.take(state.performance_metrics, 5)
      }
    end
    
    defp generate_mutation_suggestions(state) do
      analysis = perform_pattern_analysis(state)
      suggestions = []
      
      # If performance is low, suggest strategy mutation
      suggestions = if analysis.avg_performance < 0.5 do
        [Mutation.new(:mutate_strategy, "evolution_policy", 
          content: "Explore more diverse reasoning patterns through creative mutations",
          reasoning: "Low performance suggests need for strategy change") | suggestions]
      else
        suggestions
      end
      
      # If certain keys are large, suggest compression
      large_keys = Enum.filter(state.data, fn {_key, content} -> 
        byte_size(content) > 2000 
      end)
      
      compression_suggestions = Enum.map(large_keys, fn {key, _content} ->
        Mutation.new(:compress, key,
          reasoning: "Key '#{key}' consuming too much space")
      end)
      
      suggestions ++ compression_suggestions
    end
    
    defp render_grid_content(data) do
      data
      |> Enum.map(fn {key, value} -> "<#{key}>#{value}</#{key}>" end)
      |> Enum.join("\n")
    end
    
    defp limit_history(list, max_size) do
      Enum.take(list, max_size)
    end
    
    # ========== Persistence Functions ==========
    
    defp generate_grid_id do
      "grid_#{System.system_time(:millisecond)}_#{:crypto.strong_rand_bytes(4) |> Base.encode16()}"
    end
    
    defp load_persisted_state(grid_id, default_data) do
      case Helpers.load_agent_grid(grid_id) do
        nil -> 
          {default_data, %{}}
        persisted_state ->
          data = Map.get(persisted_state, :data, default_data)
          state_without_data = Map.delete(persisted_state, :data)
          {data, state_without_data}
      end
    end
    
    defp persist_grid_state(state) do
      persistable_state = %{
        data: state.data,
        mutation_history: state.mutation_history,
        performance_metrics: state.performance_metrics,
        evolution_generation: state.evolution_generation,
        meta_prompts: state.meta_prompts,
        grid_id: state.grid_id,
        last_persist_time: DateTime.utc_now()
      }
      
      Helpers.save_agent_grid(state.grid_id, persistable_state)
    end
    
    defp load_grid_from_persistence(grid_id) do
      case Helpers.load_agent_grid(grid_id) do
        nil -> {:error, :not_found}
        state -> {:ok, state}
      end
    end
    
    defp should_persist?(state) do
      # Persist if it's been more than 5 minutes since last persist
      # or if there have been significant changes
      time_threshold = 5 * 60 * 1000  # 5 minutes in milliseconds
      time_diff = DateTime.diff(DateTime.utc_now(), state.last_persist_time, :millisecond)
      
      time_diff > time_threshold or 
      state.evolution_generation > 0 or
      length(state.mutation_history) > 10
    end
    
    defp should_persist_on_performance_change?(new_score, previous_scores) do
      case previous_scores do
        [] -> true
        [last_score | _] ->
          # Persist if performance change is significant (>20% change)
          abs(new_score - last_score) > 0.2
      end
    end
    
    defp export_grid_to_file(state, filename) do
      try do
        export_data = %{
          version: "1.0",
          grid_id: state.grid_id,
          exported_at: DateTime.utc_now(),
          data: state.data,
          mutation_history: state.mutation_history,
          performance_metrics: state.performance_metrics,
          evolution_generation: state.evolution_generation,
          meta_prompts: state.meta_prompts
        }
        
        binary = :erlang.term_to_binary(export_data, [:compressed])
        
        case File.write(filename, binary) do
          :ok -> {:ok, "Grid exported to #{filename}"}
          {:error, reason} -> {:error, reason}
        end
      rescue
        error -> {:error, error}
      end
    end
    
    defp import_grid_from_file(filename) do
      try do
        case File.read(filename) do
          {:ok, binary} ->
            import_data = :erlang.binary_to_term(binary)
            
            # Extract state components
            state = %{
              data: import_data.data,
              mutation_history: import_data.mutation_history || [],
              performance_metrics: import_data.performance_metrics || [],
              evolution_generation: import_data.evolution_generation || 0,
              meta_prompts: import_data.meta_prompts || %{}
            }
            
            {:ok, state}
          {:error, reason} ->
            {:error, reason}
        end
      rescue
        error -> {:error, error}
      end
    end
  end
end