#!/usr/bin/env elixir

defmodule StandaloneEvolutionDemo do
  @moduledoc """
  Standalone Continuous Evolution and Error Correction Demo
  =======================================================
  
  This demonstrates a self-contained system that:
  1. Keeps running and mutating continuously
  2. Handles errors and self-corrects
  3. Evolves its behavior over time
  4. Maintains state across iterations
  """

  require Logger

  defmodule SelfModifyingAgent do
    use GenServer
    
    defstruct [
      :name,
      :state_data,
      :mutation_count,
      :error_count,
      :success_count,
      :evolution_generation,
      :performance_history,
      :strategies
    ]

    def start_link(name) do
      GenServer.start_link(__MODULE__, name, name: {:global, name})
    end

    def init(name) do
      initial_state = %__MODULE__{
        name: name,
        state_data: %{
          "knowledge" => "I am a self-modifying agent learning to evolve",
          "strategy" => "Explore and adapt through trial and error",
          "goal" => "Continuously improve through self-modification"
        },
        mutation_count: 0,
        error_count: 0,
        success_count: 0,
        evolution_generation: 1,
        performance_history: [],
        strategies: [
          "conservative_mutation",
          "aggressive_exploration", 
          "error_correction_focus",
          "performance_optimization"
        ]
      }
      
      Logger.info("🧠 #{name} agent initialized with generation #{initial_state.evolution_generation}")
      
      # Schedule first mutation
      Process.send_after(self(), :mutate, 2000)
      
      {:ok, initial_state}
    end

    def handle_info(:mutate, state) do
      new_state = perform_mutation(state)
      
      # Schedule next mutation with adaptive timing
      next_mutation_time = calculate_next_mutation_time(new_state)
      Process.send_after(self(), :mutate, next_mutation_time)
      
      {:noreply, new_state}
    end

    def handle_info(:error_check, state) do
      new_state = perform_error_correction(state)
      
      # Schedule next error check
      Process.send_after(self(), :error_check, 5000)
      
      {:noreply, new_state}
    end

    def handle_call(:get_state, _from, state) do
      {:reply, state, state}
    end

    def handle_call(:force_error, _from, state) do
      # Simulate an error for testing error correction
      Logger.warning("💥 #{state.name}: Forced error injected for testing")
      
      new_state = %{state | 
        error_count: state.error_count + 1,
        performance_history: [0.1 | state.performance_history]
      }
      
      corrected_state = apply_error_correction(new_state, :forced_error)
      
      {:reply, :error_injected, corrected_state}
    end

    defp perform_mutation(state) do
      try do
        # Choose a mutation strategy based on recent performance
        strategy = choose_mutation_strategy(state)
        
        # Apply the mutation
        mutated_data = apply_mutation_strategy(state.state_data, strategy)
        
        # Calculate performance score
        performance = calculate_performance_score(state, mutated_data)
        
        # Update state
        new_state = %{state |
          state_data: mutated_data,
          mutation_count: state.mutation_count + 1,
          success_count: if(performance > 0.5, do: state.success_count + 1, else: state.success_count),
          performance_history: [performance | state.performance_history] |> Enum.take(20),
          evolution_generation: if(performance > 0.7, do: state.evolution_generation + 1, else: state.evolution_generation)
        }
        
        Logger.info("🧬 #{state.name}: Mutation #{new_state.mutation_count} (#{strategy}) - Performance: #{Float.round(performance, 3)}")
        
        # If performance is consistently low, trigger error correction
        if avg_performance(new_state.performance_history) < 0.3 do
          Logger.warning("⚠️  #{state.name}: Low performance detected, triggering error correction")
          apply_error_correction(new_state, :low_performance)
        else
          new_state
        end
        
      rescue
        error ->
          Logger.error("💥 #{state.name}: Mutation failed: #{inspect(error)}")
          
          new_state = %{state | error_count: state.error_count + 1}
          apply_error_correction(new_state, error)
      end
    end

    defp choose_mutation_strategy(state) do
      recent_performance = avg_performance(Enum.take(state.performance_history, 5))
      
      cond do
        recent_performance > 0.7 -> 
          # Doing well, be conservative
          "conservative_mutation"
        recent_performance < 0.3 -> 
          # Doing poorly, try aggressive changes
          "aggressive_exploration"
        state.error_count > state.success_count ->
          # Too many errors, focus on stability
          "error_correction_focus"
        true ->
          # Balanced approach
          "performance_optimization"
      end
    end

    defp apply_mutation_strategy(data, strategy) do
      case strategy do
        "conservative_mutation" ->
          # Small, safe changes
          new_knowledge = data["knowledge"] <> " [Conservative insight: Small changes compound over time]"
          Map.put(data, "knowledge", new_knowledge)
          
        "aggressive_exploration" ->
          # Bold, risky changes
          Map.merge(data, %{
            "strategy" => "Explore bold new approaches and learn from failures",
            "exploration_level" => "high",
            "risk_tolerance" => "elevated"
          })
          
        "error_correction_focus" ->
          # Focus on stability and error handling
          Map.merge(data, %{
            "strategy" => "Prioritize stability and robust error handling",
            "error_handling" => "enhanced",
            "stability_focus" => "maximum"
          })
          
        "performance_optimization" ->
          # Optimize for better performance
          optimized_knowledge = data["knowledge"] <> " [Optimization: Focus on high-impact changes]"
          Map.put(data, "knowledge", optimized_knowledge)
      end
    end

    defp calculate_performance_score(state, mutated_data) do
      base_score = 0.5
      
      # Bonus for knowledge growth
      knowledge_size = byte_size(mutated_data["knowledge"] || "")
      knowledge_bonus = min(knowledge_size / 1000, 0.2)
      
      # Bonus for low error rate
      error_rate = if state.mutation_count > 0, do: state.error_count / state.mutation_count, else: 0
      error_penalty = error_rate * 0.3
      
      # Bonus for evolution generation
      evolution_bonus = min(state.evolution_generation * 0.05, 0.2)
      
      # Random factor to simulate real-world unpredictability
      random_factor = (:rand.uniform() - 0.5) * 0.1
      
      final_score = base_score + knowledge_bonus + evolution_bonus - error_penalty + random_factor
      max(0.0, min(1.0, final_score))
    end

    defp perform_error_correction(state) do
      if state.error_count > 0 and state.error_count > state.success_count * 0.3 do
        Logger.info("🔧 #{state.name}: Performing scheduled error correction")
        apply_error_correction(state, :scheduled_maintenance)
      else
        state
      end
    end

    defp apply_error_correction(state, error_type) do
      Logger.info("🛠️  #{state.name}: Applying error correction for #{inspect(error_type)}")
      
      corrected_data = case error_type do
        :low_performance ->
          # Reset to a known good state
          %{
            "knowledge" => "Reset to stable foundation, learning from past mistakes",
            "strategy" => "Focus on proven approaches while carefully exploring",
            "error_recovery" => "low_performance_reset_applied"
          }
          
        :forced_error ->
          # Add error handling capabilities
          Map.merge(state.state_data, %{
            "error_handling" => "enhanced_after_forced_error",
            "resilience" => "improved",
            "recovery_strategy" => "adaptive_response"
          })
          
        _ ->
          # General error correction
          Map.merge(state.state_data, %{
            "error_handling" => "general_correction_applied",
            "stability" => "enhanced"
          })
      end
      
      # Performance boost from error correction
      correction_performance = 0.6 + :rand.uniform() * 0.2
      
      %{state |
        state_data: corrected_data,
        performance_history: [correction_performance | state.performance_history] |> Enum.take(20),
        evolution_generation: state.evolution_generation + 1
      }
    end

    defp calculate_next_mutation_time(state) do
      base_time = 3000
      
      # Faster mutations if performing well
      performance_factor = avg_performance(state.performance_history)
      time_adjustment = if performance_factor > 0.6, do: -1000, else: 500
      
      # Slower mutations if too many errors
      error_factor = if state.error_count > state.success_count, do: 2000, else: 0
      
      max(1000, base_time + time_adjustment + error_factor)
    end

    defp avg_performance([]), do: 0.5
    defp avg_performance(performance_list) do
      performance_list
      |> Enum.take(10)
      |> Enum.sum()
      |> then(&(&1 / length(Enum.take(performance_list, 10))))
    end

    # Public API
    def get_state(agent_name) do
      GenServer.call({:global, agent_name}, :get_state)
    end

    def inject_error(agent_name) do
      GenServer.call({:global, agent_name}, :force_error)
    end
  end

  def run do
    IO.puts """
    🚀 Starting Standalone Continuous Evolution Demo
    ===============================================
    
    This demo creates self-modifying agents that:
    • Continuously mutate and evolve
    • Handle errors and self-correct
    • Adapt their behavior over time
    • Learn from successes and failures
    
    Press Ctrl+C to stop the demo.
    """

    # Start multiple agents
    agent_names = ["Explorer", "Optimizer", "Innovator"]
    
    # Start each agent
    agents = Enum.map(agent_names, fn name ->
      case SelfModifyingAgent.start_link(name) do
        {:ok, pid} ->
          Logger.info("✅ Started agent: #{name}")
          {name, pid}
        {:error, reason} ->
          Logger.error("❌ Failed to start agent #{name}: #{inspect(reason)}")
          nil
      end
    end)
    |> Enum.filter(&(&1 != nil))

    if length(agents) > 0 do
      IO.puts "\n🎯 #{length(agents)} agents started successfully!"
      
      # Start monitoring loop
      spawn_link(fn -> monitoring_loop(agents) end)
      
      # Start error injection for testing
      spawn_link(fn -> error_injection_loop(agents) end)
      
      # Show status updates
      status_loop(agents)
    else
      IO.puts "❌ No agents could be started. Exiting."
    end
  end

  defp monitoring_loop(agents) do
    receive after
      10_000 ->
        monitor_agents(agents)
    end
    
    monitoring_loop(agents)
  end

  defp monitor_agents(agents) do
    IO.puts "\n📊 Agent Status Report:"
    IO.puts "========================"
    
    Enum.each(agents, fn {name, _pid} ->
      try do
        state = SelfModifyingAgent.get_state(name)
        
        avg_perf = avg_performance(state.performance_history)
        
        IO.puts """
        🤖 #{name}:
          • Mutations: #{state.mutation_count}
          • Errors: #{state.error_count}
          • Successes: #{state.success_count}
          • Generation: #{state.evolution_generation}
          • Avg Performance: #{Float.round(avg_perf, 3)}
          • Status: #{agent_status(state)}
        """
        
      rescue
        error ->
          IO.puts "❌ #{name}: Failed to get status - #{inspect(error)}"
      end
    end)
  end

  defp error_injection_loop(agents) do
    receive after
      15_000 ->
        # Randomly inject an error for testing
        if :rand.uniform() < 0.3 and length(agents) > 0 do
          {name, _pid} = Enum.random(agents)
          
          try do
            SelfModifyingAgent.inject_error(name)
            Logger.info("🧪 Injected test error in #{name}")
          rescue
            error ->
              Logger.warning("Failed to inject error: #{inspect(error)}")
          end
        end
    end
    
    error_injection_loop(agents)
  end

  defp status_loop(agents) do
    receive after
      30_000 ->
        show_evolution_summary(agents)
    end
    
    status_loop(agents)
  end

  defp show_evolution_summary(agents) do
    IO.puts "\n🧬 Evolution Summary:"
    IO.puts "==================="
    
    total_mutations = Enum.reduce(agents, 0, fn {name, _pid}, acc ->
      try do
        state = SelfModifyingAgent.get_state(name)
        acc + state.mutation_count
      rescue
        _ -> acc
      end
    end)
    
    total_errors = Enum.reduce(agents, 0, fn {name, _pid}, acc ->
      try do
        state = SelfModifyingAgent.get_state(name)
        acc + state.error_count
      rescue
        _ -> acc
      end
    end)
    
    IO.puts """
    • Total Mutations: #{total_mutations}
    • Total Errors: #{total_errors}
    • Error Rate: #{if total_mutations > 0, do: Float.round(total_errors / total_mutations * 100, 1), else: 0}%
    • Active Agents: #{length(agents)}
    
    System continues evolving...
    """
  end

  defp agent_status(state) do
    recent_performance = avg_performance(Enum.take(state.performance_history, 3))
    
    cond do
      recent_performance > 0.7 -> "🟢 Performing Well"
      recent_performance > 0.4 -> "🟡 Stable"  
      recent_performance > 0.2 -> "🟠 Struggling"
      true -> "🔴 Needs Attention"
    end
  end

  defp avg_performance([]), do: 0.5
  defp avg_performance(performance_list) do
    performance_list
    |> Enum.take(10)
    |> Enum.sum()
    |> then(&(&1 / length(Enum.take(performance_list, 10))))
  end
end

# Start the demo
Logger.configure(level: :info)
StandaloneEvolutionDemo.run()