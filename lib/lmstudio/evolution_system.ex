defmodule LMStudio.EvolutionSystem do
  @moduledoc """
  Supervisor and coordinator for the continuous evolution system.
  
  Manages multiple cognitive agents and facilitates cross-pollination
  of successful patterns between agents.
  """
  
  use Supervisor
  require Logger
  alias LMStudio.CognitiveAgent
  alias LMStudio.MetaDSL.Mutation
  alias LMStudio.Persistence.Helpers
  alias LMStudio.ErlangKnowledgeBase
  
  @agent_types [
    {"Explorer", "explores new reasoning patterns through creative mutations"},
    {"Optimizer", "optimizes existing patterns for efficiency and clarity"}, 
    {"Synthesizer", "combines and merges successful patterns into new forms"}
  ]
  
  def start_link(opts \\ []) do
    Supervisor.start_link(__MODULE__, opts, name: __MODULE__)
  end
  
  @impl true
  def init(opts) do
    num_agents = Keyword.get(opts, :num_agents, 3)
    model = Keyword.get(opts, :model, "deepseek-r1-0528-qwen3-8b-mlx")
    
    # Build children list dynamically based on what's already running
    children = []
    
    # Only start Registry if it's not already running
    children = case Process.whereis(LMStudio.AgentRegistry) do
      nil ->
        [{Registry, keys: :unique, name: LMStudio.AgentRegistry} | children]
      _pid ->
        children
    end
    
    # Add Task Supervisor
    children = [{Task.Supervisor, name: LMStudio.EvolutionTaskSupervisor} | children]
    
    # Add Persistence if not already started
    children = if Process.whereis(LMStudio.Persistence) do
      children
    else
      [LMStudio.Persistence | children]
    end
    
    # Add agents to supervision tree
    agent_children = create_agent_specs(num_agents, model)
    
    all_children = children ++ agent_children ++ [
      {LMStudio.EvolutionCoordinator, [num_agents: num_agents]}
    ]
    
    Supervisor.init(all_children, strategy: :one_for_one)
  end
  
  defp create_agent_specs(num_agents, model) do
    1..num_agents
    |> Enum.map(fn i ->
      {name, _role} = Enum.at(@agent_types, rem(i - 1, length(@agent_types)))
      agent_name = "#{name}_#{i}"
      
      Supervisor.child_spec(
        {CognitiveAgent, [
          name: agent_name,
          model: model,
          thinking_enabled: true
        ]},
        id: String.to_atom(agent_name)
      )
    end)
  end
  
  def get_agent_names do
    LMStudio.EvolutionCoordinator.get_agent_names()
  end
  
  def run_evolution_cycle(topic \\ nil) do
    LMStudio.EvolutionCoordinator.run_evolution_cycle(topic)
  end
  
  def start_continuous_evolution(base_topic \\ nil) do
    LMStudio.EvolutionCoordinator.start_continuous_evolution(base_topic)
  end
  
  def stop_continuous_evolution do
    LMStudio.EvolutionCoordinator.stop_continuous_evolution()
  end
  
  def get_system_state do
    LMStudio.EvolutionCoordinator.get_system_state()
  end
  
  def generate_code_from_insights(insights \\ nil) do
    LMStudio.EvolutionCoordinator.generate_code_from_insights(insights)
  end
  
  def get_learning_trajectory do
    LMStudio.EvolutionCoordinator.get_learning_trajectory()
  end
end

defmodule LMStudio.EvolutionCoordinator do
  @moduledoc """
  Coordinates evolution cycles and cross-pollination between agents.
  """
  
  use GenServer
  require Logger
  alias LMStudio.CognitiveAgent
  alias LMStudio.MetaDSL.Mutation
  alias LMStudio.Persistence.Helpers
  alias LMStudio.ErlangKnowledgeBase
  alias LMStudio.CodeGeneration
  
  @agent_types [
    {"Explorer", "explores new reasoning patterns through creative mutations"},
    {"Optimizer", "optimizes existing patterns for efficiency and clarity"}, 
    {"Synthesizer", "combines and merges successful patterns into new forms"}
  ]
  
  @topics [
    "recursive self-improvement through meta-cognitive reflection",
    "the emergence of consciousness from information processing", 
    "evolving reasoning strategies through self-modification",
    "the nature of intelligence and self-awareness",
    "optimization of thought patterns through mutation"
  ]
  
  def start_link(opts) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end
  
  def get_agent_names do
    GenServer.call(__MODULE__, :get_agent_names)
  end
  
  def run_evolution_cycle(topic) do
    GenServer.call(__MODULE__, {:run_evolution_cycle, topic}, 30_000)
  end
  
  def start_continuous_evolution(base_topic) do
    GenServer.cast(__MODULE__, {:start_continuous_evolution, base_topic})
  end
  
  def stop_continuous_evolution do
    GenServer.cast(__MODULE__, :stop_continuous_evolution)
  end
  
  def get_system_state do
    GenServer.call(__MODULE__, :get_system_state)
  end
  
  def generate_code_from_insights(insights) do
    GenServer.call(__MODULE__, {:generate_code_from_insights, insights}, 60_000)
  end
  
  def get_learning_trajectory do
    GenServer.call(__MODULE__, :get_learning_trajectory)
  end
  
  @impl true
  def init(opts) do
    num_agents = Keyword.get(opts, :num_agents, 3)
    
    # Generate agent names based on types
    agent_names = 1..num_agents
    |> Enum.map(fn i ->
      {name, _role} = Enum.at(@agent_types, rem(i - 1, length(@agent_types)))
      "#{name}_#{i}"
    end)
    
    # Load persisted state if available
    {loaded_history, loaded_insights, loaded_cycles} = load_evolution_state()
    
    state = %{
      agent_names: agent_names,
      evolution_cycles: loaded_cycles,
      conversation_history: loaded_history,
      global_insights: loaded_insights,
      continuous_evolution_task: nil,
      topic_index: 0,
      generated_code_count: 0,
      pattern_usage_stats: %{},
      learning_trajectory: []
    }
    
    Logger.info("EvolutionCoordinator initialized with #{length(agent_names)} agents")
    {:ok, state}
  end
  
  @impl true
  def handle_call(:get_agent_names, _from, state) do
    {:reply, state.agent_names, state}
  end
  
  @impl true
  def handle_call({:run_evolution_cycle, topic}, _from, state) do
    case execute_evolution_cycle(state, topic) do
      {:ok, new_state} ->
        {:reply, :ok, new_state}
      {:error, reason} ->
        Logger.error("Evolution cycle failed: #{inspect(reason)}")
        {:reply, {:error, reason}, state}
    end
  end
  
  @impl true
  def handle_call(:get_system_state, _from, state) do
    system_state = build_system_state(state)
    {:reply, system_state, state}
  end
  
  @impl true
  def handle_call({:generate_code_from_insights, insights}, _from, state) do
    case generate_code_from_learning(insights || state.global_insights, state) do
      {:ok, generated_code, new_state} ->
        {:reply, {:ok, generated_code}, new_state}
      {:error, reason} ->
        {:reply, {:error, reason}, state}
    end
  end
  
  @impl true
  def handle_call(:get_learning_trajectory, _from, state) do
    {:reply, state.learning_trajectory, state}
  end
  
  @impl true
  def handle_cast({:start_continuous_evolution, base_topic}, state) do
    if state.continuous_evolution_task do
      Task.Supervisor.terminate_child(LMStudio.EvolutionTaskSupervisor, state.continuous_evolution_task)
    end
    
    task = Task.Supervisor.async_nolink(LMStudio.EvolutionTaskSupervisor, fn ->
      run_continuous_evolution_loop(state, base_topic)
    end)
    
    new_state = %{state | continuous_evolution_task: task}
    Logger.info("Started continuous evolution")
    {:noreply, new_state}
  end
  
  @impl true
  def handle_cast(:stop_continuous_evolution, state) do
    if state.continuous_evolution_task do
      Task.Supervisor.terminate_child(LMStudio.EvolutionTaskSupervisor, state.continuous_evolution_task)
    end
    
    new_state = %{state | continuous_evolution_task: nil}
    Logger.info("Stopped continuous evolution")
    {:noreply, new_state}
  end
  
  @impl true
  def handle_info({ref, _result}, state) when is_reference(ref) do
    # Continuous evolution task completed
    Process.demonitor(ref, [:flush])
    new_state = %{state | continuous_evolution_task: nil}
    Logger.info("Continuous evolution task completed")
    {:noreply, new_state}
  end
  
  @impl true
  def handle_info({:DOWN, _ref, :process, _pid, reason}, state) do
    Logger.warning("Continuous evolution task crashed: #{inspect(reason)}")
    new_state = %{state | continuous_evolution_task: nil}
    {:noreply, new_state}
  end
  
  # Private Functions
  
  defp execute_evolution_cycle(state, topic) do
    current_topic = topic || Enum.at(@topics, rem(state.topic_index, length(@topics)))
    
    Logger.info("=== Evolution Cycle #{state.evolution_cycles + 1} ===")
    Logger.info("Topic: #{current_topic}")
    
    # Build context from recent interactions
    context = build_conversation_context(state.conversation_history)
    
    # Process with each agent
    results = Enum.map(state.agent_names, fn agent_name ->
      case CognitiveAgent.process_query(agent_name, current_topic, context) do
        {:ok, result} ->
          Logger.info("[#{agent_name}] Processed query - Mutations: #{result.mutations_applied}, Performance: #{Float.round(result.performance_score, 3)}")
          
          # Record interaction
          interaction = %{
            cycle: state.evolution_cycles + 1,
            agent: agent_name,
            content: result.response,
            thinking: result.thinking,
            performance: result.performance_score,
            timestamp: DateTime.utc_now()
          }
          
          # Extract insights
          insights = extract_insights(result.thinking, agent_name, state.evolution_cycles + 1)
          
          {:ok, interaction, insights}
          
        {:error, reason} ->
          Logger.error("[#{agent_name}] Failed to process query: #{inspect(reason)}")
          {:error, reason}
      end
    end)
    
    # Collect successful results
    successful_results = Enum.filter(results, fn
      {:ok, _, _} -> true
      _ -> false
    end)
    
    if length(successful_results) > 0 do
      {interactions, insights_lists} = Enum.unzip(
        Enum.map(successful_results, fn {:ok, interaction, insights} -> {interaction, insights} end)
      )
      
      new_conversation_history = interactions ++ state.conversation_history
      |> Enum.take(100)  # Keep last 100 interactions
      
      new_global_insights = List.flatten(insights_lists) ++ state.global_insights
      |> Enum.take(50)  # Keep last 50 insights
      
      # Record learning trajectory
      learning_entry = %{
        cycle: state.evolution_cycles + 1,
        topic: current_topic,
        insights_generated: length(List.flatten(insights_lists)),
        performance_avg: calculate_avg_performance(interactions),
        patterns_identified: identify_patterns(interactions),
        timestamp: DateTime.utc_now()
      }
      
      new_learning_trajectory = [learning_entry | state.learning_trajectory]
      |> Enum.take(100)  # Keep last 100 learning entries
      
      new_state = %{
        state |
        evolution_cycles: state.evolution_cycles + 1,
        conversation_history: new_conversation_history,
        global_insights: new_global_insights,
        topic_index: state.topic_index + 1,
        learning_trajectory: new_learning_trajectory
      }
      
      # Persist evolution state
      persist_evolution_state(new_state)
      
      # Trigger cross-pollination every 3 cycles
      if rem(new_state.evolution_cycles, 3) == 0 do
        spawn(fn -> perform_cross_pollination(state.agent_names) end)
      end
      
      # Trigger autonomous evolution every 5 cycles
      if rem(new_state.evolution_cycles, 5) == 0 do
        spawn(fn -> trigger_autonomous_evolution(state.agent_names) end)
      end
      
      # Generate code from insights every 10 cycles
      if rem(new_state.evolution_cycles, 10) == 0 do
        spawn(fn -> 
          case generate_code_from_learning(new_global_insights, new_state) do
            {:ok, _code, _updated_state} ->
              Logger.info("Generated code from insights at cycle #{new_state.evolution_cycles}")
            {:error, reason} ->
              Logger.warning("Failed to generate code from insights: #{inspect(reason)}")
          end
        end)
      end
      
      {:ok, new_state}
    else
      {:error, :all_agents_failed}
    end
  end
  
  defp build_conversation_context(history) do
    history
    |> Enum.take(3)
    |> Enum.map(fn interaction ->
      "#{interaction.agent}: #{String.slice(interaction.content, 0, 100)}..."
    end)
    |> Enum.join("\n")
  end
  
  defp extract_insights(thinking, agent_name, cycle) do
    insight_keywords = ["insight", "realize", "understand", "discover", "learn"]
    
    if Enum.any?(insight_keywords, &String.contains?(String.downcase(thinking), &1)) do
      [%{
        agent: agent_name,
        content: thinking,
        cycle: cycle,
        timestamp: DateTime.utc_now()
      }]
    else
      []
    end
  end
  
  defp perform_cross_pollination(agent_names) do
    Logger.info("=== Cross-Pollination Phase ===")
    
    # Get performance data for all agents
    agent_performances = Enum.map(agent_names, fn name ->
      case CognitiveAgent.get_state(name) do
        state when is_map(state) ->
          {name, state.interaction_count, state.insight_count}
        _ ->
          {name, 0, 0}
      end
    end)
    
    # Find best performing agent
    {best_agent, _, _} = Enum.max_by(agent_performances, fn {_name, interactions, insights} ->
      interactions + insights * 2  # Weight insights more heavily
    end)
    
    Logger.info("Best performing agent: #{best_agent}")
    
    # Share knowledge from best agent to others
    case CognitiveAgent.get_grid_data(best_agent) do
      grid_data when is_map(grid_data) ->
        knowledge_to_share = Map.get(grid_data, "knowledge", "")
        
        if byte_size(knowledge_to_share) > 0 do
          other_agents = Enum.reject(agent_names, &(&1 == best_agent))
          
          Enum.each(other_agents, fn agent_name ->
            _mutation = Mutation.new(:append, "knowledge",
              content: "\n[Inherited from #{best_agent}]: #{String.slice(knowledge_to_share, 0, 200)}",
              reasoning: "Cross-pollination of successful patterns"
            )
            
            # This would need to access the agent's grid directly
            # For now, we'll log the intention
            Logger.debug("Would share knowledge from #{best_agent} to #{agent_name}")
          end)
        end
        
      _ ->
        Logger.warning("Could not retrieve grid data from #{best_agent}")
    end
  end
  
  defp trigger_autonomous_evolution(agent_names) do
    Logger.info("=== Autonomous Evolution Phase ===")
    
    # Trigger autonomous evolution for each agent
    Enum.each(agent_names, fn agent_name ->
      CognitiveAgent.evolve_autonomously(agent_name, 2)
    end)
  end
  
  defp build_system_state(state) do
    %{
      evolution_cycles: state.evolution_cycles,
      total_interactions: length(state.conversation_history),
      global_insights: length(state.global_insights),
      agents: length(state.agent_names),
      continuous_evolution_running: not is_nil(state.continuous_evolution_task),
      recent_insights: Enum.take(state.global_insights, 3)
    }
  end
  
  defp run_continuous_evolution_loop(state, base_topic) do
    Logger.info("=== Starting Continuous Evolution Loop ===")
    
    try do
      Stream.repeatedly(fn -> :continue end)
      |> Enum.reduce_while(state, fn _, current_state ->
        topic = base_topic || Enum.at(@topics, rem(current_state.topic_index, length(@topics)))
        
        case execute_evolution_cycle(current_state, topic) do
          {:ok, new_state} ->
            if rem(new_state.evolution_cycles, 10) == 0 do
              log_system_state(new_state)
            end
            
            Process.sleep(2000)  # Pause between cycles
            {:cont, new_state}
            
          {:error, reason} ->
            Logger.error("Evolution cycle failed, stopping: #{inspect(reason)}")
            {:halt, current_state}
        end
      end)
    rescue
      error ->
        Logger.error("Continuous evolution loop crashed: #{inspect(error)}")
    end
  end
  
  defp log_system_state(state) do
    Logger.info("""
    === System State (Cycle #{state.evolution_cycles}) ===
    Agents: #{length(state.agent_names)}
    Total interactions: #{length(state.conversation_history)}
    Global insights: #{length(state.global_insights)}
    Generated code modules: #{state.generated_code_count}
    Learning trajectory entries: #{length(state.learning_trajectory)}
    Recent insight: #{if length(state.global_insights) > 0, do: List.first(state.global_insights).content |> String.slice(0, 100), else: "None"}...
    ===============================================
    """)
  end
  
  # ========== Persistence and Learning Functions ==========
  
  defp load_evolution_state do
    conversation_history = Helpers.load_global_insights() |> get_conversations_from_insights()
    global_insights = Helpers.load_global_insights()
    evolution_cycles = get_persisted_cycle_count()
    
    {conversation_history, global_insights, evolution_cycles}
  end
  
  defp persist_evolution_state(state) do
    Helpers.save_global_insights(state.global_insights)
    Helpers.save_evolution_cycle(state.evolution_cycles, %{
      cycle: state.evolution_cycles,
      timestamp: DateTime.utc_now(),
      agent_count: length(state.agent_names),
      insights_count: length(state.global_insights),
      learning_trajectory: state.learning_trajectory
    })
  end
  
  defp get_conversations_from_insights(insights) do
    # Extract conversation history from insights if available
    insights
    |> Enum.take(20)
    |> Enum.map(fn insight ->
      %{
        cycle: insight.cycle,
        agent: insight.agent,
        content: insight.content,
        thinking: "",
        performance: 0.7,
        timestamp: insight.timestamp
      }
    end)
  end
  
  defp get_persisted_cycle_count do
    case Helpers.load_evolution_cycles() do
      [] -> 0
      cycles -> 
        cycles
        |> Enum.map(&(&1.cycle))
        |> Enum.max()
    end
  end
  
  defp calculate_avg_performance(interactions) do
    if length(interactions) > 0 do
      interactions
      |> Enum.map(&(&1.performance))
      |> Enum.sum()
      |> then(&(&1 / length(interactions)))
    else
      0.0
    end
  end
  
  defp identify_patterns(interactions) do
    # Analyze interactions to identify recurring patterns
    patterns = []
    
    # Pattern 1: High-performing topics
    high_performance = Enum.filter(interactions, &(&1.performance > 0.8))
    patterns = if length(high_performance) > 0 do
      ["high_performance_agents"] ++ patterns
    else
      patterns
    end
    
    # Pattern 2: Insight generation trends
    insight_rich = Enum.filter(interactions, fn interaction ->
      String.contains?(String.downcase(interaction.thinking), "insight") or
      String.contains?(String.downcase(interaction.thinking), "discover")
    end)
    patterns = if length(insight_rich) > length(interactions) / 2 do
      ["insight_generation_trend"] ++ patterns
    else
      patterns
    end
    
    # Pattern 3: Evolution and mutation mentions
    evolution_focused = Enum.filter(interactions, fn interaction ->
      String.contains?(String.downcase(interaction.content), "evolve") or
      String.contains?(String.downcase(interaction.content), "mutate")
    end)
    patterns = if length(evolution_focused) > 0 do
      ["evolution_focus"] ++ patterns
    else
      patterns
    end
    
    patterns
  end
  
  defp generate_code_from_learning(insights, state) do
    try do
      # Analyze insights to determine what kind of code to generate
      analysis = analyze_insights_for_code_generation(insights)
      
      pattern_name = analysis.recommended_pattern
      
      # Use the knowledge base to get the best pattern
      context = %{
        use_case: analysis.use_case,
        scale: analysis.scale,
        fault_tolerance: analysis.fault_tolerance
      }
      
      recommendations = ErlangKnowledgeBase.get_pattern_recommendations(context)
          selected_pattern = List.first(recommendations) || pattern_name
          
          case ErlangKnowledgeBase.generate_code_from_pattern(selected_pattern) do
            {:ok, generated_code} ->
              # Save the generated code
              code_id = "generated_#{System.system_time()}_#{selected_pattern}"
              CodeGeneration.save_generated_code(code_id, %{
                code: generated_code,
                pattern: selected_pattern,
                insights_used: length(insights),
                generation_cycle: state.evolution_cycles,
                analysis: analysis
              })
              
              # Update pattern usage stats
              updated_stats = Map.update(state.pattern_usage_stats, selected_pattern, 1, &(&1 + 1))
              new_state = %{
                state | 
                generated_code_count: state.generated_code_count + 1,
                pattern_usage_stats: updated_stats
              }
              
              Logger.info("Generated #{selected_pattern} code from #{length(insights)} insights")
              {:ok, generated_code, new_state}
              
            {:error, reason} ->
              {:error, reason}
          end
    rescue
      error ->
        Logger.error("Code generation failed: #{inspect(error)}")
        {:error, error}
    end
  end
  
  defp analyze_insights_for_code_generation(insights) do
    # Analyze insights to determine what patterns and code to generate
    use_case_keywords = extract_keywords_from_insights(insights, ["state", "worker", "event", "service", "process"])
    scale_indicators = extract_keywords_from_insights(insights, ["scale", "concurrent", "distributed", "pool"])
    fault_tolerance_keywords = extract_keywords_from_insights(insights, ["fault", "error", "crash", "restart", "supervision"])
    
    # Determine use case
    use_case = cond do
      "state" in use_case_keywords -> "state management"
      "worker" in use_case_keywords -> "worker processing" 
      "event" in use_case_keywords -> "event handling"
      "service" in use_case_keywords -> "service implementation"
      true -> "general processing"
    end
    
    # Determine scale
    scale = cond do
      "distributed" in scale_indicators -> :large
      "concurrent" in scale_indicators or "pool" in scale_indicators -> :medium
      true -> :small
    end
    
    # Determine fault tolerance needs
    fault_tolerance = cond do
      length(fault_tolerance_keywords) > 2 -> :high
      length(fault_tolerance_keywords) > 0 -> :medium
      true -> :low
    end
    
    # Determine recommended pattern
    recommended_pattern = case {use_case, scale, fault_tolerance} do
      {"state management", _, _} -> :gen_server_with_state
      {"worker processing", :medium, _} -> :task_supervisor_pattern
      {"worker processing", :large, _} -> :distributed_worker
      {"event handling", _, _} -> :gen_event
      {"service implementation", _, :high} -> :supervisor_one_for_one
      _ -> :gen_server_with_state
    end
    
    %{
      use_case: use_case,
      scale: scale,
      fault_tolerance: fault_tolerance,
      recommended_pattern: recommended_pattern,
      keywords_found: use_case_keywords ++ scale_indicators ++ fault_tolerance_keywords
    }
  end
  
  defp extract_keywords_from_insights(insights, keywords) do
    insights
    |> Enum.flat_map(fn insight ->
      content = insight.content || ""
      Enum.filter(keywords, fn keyword ->
        String.contains?(String.downcase(content), keyword)
      end)
    end)
    |> Enum.uniq()
  end
end