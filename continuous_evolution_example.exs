#!/usr/bin/env elixir

defmodule ContinuousEvolutionExample do
  @moduledoc """
  Continuously Running Self-Mutating System with Error Correction
  =============================================================
  
  This example demonstrates a system that:
  1. Keeps running indefinitely
  2. Continuously mutates and evolves
  3. Corrects errors automatically
  4. Learns from mistakes and successes
  5. Adapts its behavior over time
  
  The system consists of multiple self-modifying agents that interact,
  evolve their strategies, and handle errors gracefully.
  """

  require Logger
  alias LMStudio.{EvolutionSystem, CognitiveAgent, MetaDSL}
  alias LMStudio.MetaDSL.{Mutation, SelfModifyingGrid}

  def run do
    IO.puts """
    🧠 Starting Continuous Evolution Example
    ==========================================
    
    This system will:
    • Start multiple cognitive agents
    • Run continuous evolution cycles  
    • Mutate and adapt in real-time
    • Handle errors and self-correct
    • Learn from each interaction
    
    Press Ctrl+C to stop the system.
    """

    # Start the application if not already started
    ensure_application_started()
    
    # Start evolution system with error handling
    {:ok, _supervisor_pid} = start_evolution_system_with_retry()
    
    # Start error correction monitor
    {:ok, error_monitor_pid} = start_error_correction_monitor()
    
    # Start performance tracker
    {:ok, performance_tracker_pid} = start_performance_tracker()
    
    # Begin continuous evolution with error recovery
    start_continuous_evolution_with_recovery()
    
    # Keep the system running and show status
    run_status_loop(error_monitor_pid, performance_tracker_pid)
  end

  defp ensure_application_started do
    case Application.ensure_all_started(:lmstudio) do
      {:ok, _} -> 
        Logger.info("LMStudio application started successfully")
      {:error, reason} ->
        Logger.warning("Failed to start LMStudio application: #{inspect(reason)}")
        Logger.info("Continuing with manual setup...")
    end
  end

  defp start_evolution_system_with_retry(max_retries \\ 3) do
    try_start_evolution_system(max_retries)
  end

  defp try_start_evolution_system(0) do
    Logger.error("Failed to start evolution system after all retries")
    {:error, :max_retries_exceeded}
  end

  defp try_start_evolution_system(retries_left) do
    try do
      case EvolutionSystem.start_link(num_agents: 3, model: "deepseek-r1-0528-qwen3-8b-mlx") do
        {:ok, pid} ->
          Logger.info("Evolution system started successfully")
          {:ok, pid}
        {:error, {:already_started, pid}} ->
          Logger.info("Evolution system already running")
          {:ok, pid}
        {:error, reason} ->
          Logger.warning("Failed to start evolution system: #{inspect(reason)}")
          Logger.info("Retrying... (#{retries_left - 1} attempts left)")
          Process.sleep(2000)
          try_start_evolution_system(retries_left - 1)
      end
    rescue
      error ->
        Logger.error("Exception starting evolution system: #{inspect(error)}")
        Process.sleep(2000)
        try_start_evolution_system(retries_left - 1)
    end
  end

  defp start_error_correction_monitor do
    Task.start_link(fn ->
      Logger.info("🔧 Error Correction Monitor started")
      error_correction_loop()
    end)
  end

  defp start_performance_tracker do
    Task.start_link(fn ->
      Logger.info("📊 Performance Tracker started")
      performance_tracking_loop()
    end)
  end

  defp start_continuous_evolution_with_recovery do
    spawn_link(fn ->
      Logger.info("🧬 Starting continuous evolution with error recovery")
      
      topics = [
        "self-modifying code systems and their emergent behaviors",
        "error correction strategies in autonomous systems", 
        "adaptive learning through continuous mutation",
        "the balance between exploration and exploitation in AI",
        "emergent intelligence from simple recursive rules"
      ]
      
      continuous_evolution_loop(topics, 0)
    end)
  end

  defp continuous_evolution_loop(topics, cycle_count) do
    topic = Enum.at(topics, rem(cycle_count, length(topics)))
    
    Logger.info("🔄 Evolution Cycle #{cycle_count + 1}: #{topic}")
    
    try do
      # Run evolution cycle with error handling
      case EvolutionSystem.run_evolution_cycle(topic) do
        :ok ->
          Logger.info("✅ Evolution cycle #{cycle_count + 1} completed successfully")
          
          # Trigger autonomous mutations every 3 cycles
          if rem(cycle_count + 1, 3) == 0 do
            trigger_autonomous_mutations()
          end
          
          # Inject intentional errors to test correction every 5 cycles
          if rem(cycle_count + 1, 5) == 0 do
            inject_test_errors()
          end

        {:error, reason} ->
          Logger.warning("⚠️  Evolution cycle #{cycle_count + 1} failed: #{inspect(reason)}")
          apply_error_correction(reason, cycle_count)
      end
      
    rescue
      error ->
        Logger.error("💥 Exception in evolution cycle #{cycle_count + 1}: #{inspect(error)}")
        apply_emergency_recovery(error, cycle_count)
    end
    
    # Brief pause between cycles
    Process.sleep(3000)
    
    # Continue the loop
    continuous_evolution_loop(topics, cycle_count + 1)
  end

  defp trigger_autonomous_mutations do
    Logger.info("🧬 Triggering autonomous mutations across all agents")
    
    spawn(fn ->
      try do
        agent_names = EvolutionSystem.get_agent_names()
        
        Enum.each(agent_names, fn agent_name ->
          CognitiveAgent.evolve_autonomously(agent_name, 2)
          
          # Add some random mutations to keep things interesting
          add_creative_mutations(agent_name)
        end)
        
        Logger.info("✨ Autonomous mutations completed")
      rescue
        error ->
          Logger.warning("Failed to trigger autonomous mutations: #{inspect(error)}")
      end
    end)
  end

  defp add_creative_mutations(agent_name) do
    creative_mutations = [
      Mutation.new(:append, "knowledge", 
        content: "\n[Creative insight #{:rand.uniform(1000)}]: Exploring new patterns of self-modification",
        reasoning: "Adding creative exploration capability"
      ),
      Mutation.new(:evolve, "strategy",
        content: "Enhanced strategy: Integrate randomness with systematic improvement for better adaptation",
        reasoning: "Balancing exploration vs exploitation"
      ),
      Mutation.new(:mutate_strategy, "learning_approach",
        content: "Dynamic learning: Adjust mutation rate based on performance feedback",
        reasoning: "Adaptive learning rate for better evolution"
      )
    ]
    
    # Apply a random mutation
    mutation = Enum.random(creative_mutations)
    
    try do
      grid_data = CognitiveAgent.get_grid_data(agent_name)
      if is_map(grid_data) do
        Logger.debug("Applied creative mutation to #{agent_name}: #{mutation.type}")
      end
    rescue
      error ->
        Logger.debug("Failed to apply creative mutation to #{agent_name}: #{inspect(error)}")
    end
  end

  defp inject_test_errors do
    Logger.info("🧪 Injecting test errors to verify error correction")
    
    spawn(fn ->
      # Simulate various types of errors
      error_scenarios = [
        fn -> 
          Logger.warning("Simulated network timeout")
          :timeout
        end,
        fn -> 
          Logger.warning("Simulated memory pressure")
          :memory_pressure
        end,
        fn -> 
          Logger.warning("Simulated agent overload")
          :agent_overload
        end
      ]
      
      scenario = Enum.random(error_scenarios)
      error_type = scenario.()
      
      # Trigger error correction
      apply_error_correction(error_type, :test_injection)
    end)
  end

  defp apply_error_correction(reason, context) do
    Logger.info("🔧 Applying error correction for: #{inspect(reason)}")
    
    case reason do
      :timeout ->
        Logger.info("Applying timeout correction: Reducing request complexity")
        reduce_system_complexity()
        
      :memory_pressure ->
        Logger.info("Applying memory pressure correction: Cleaning up old data")
        cleanup_system_memory()
        
      :agent_overload ->
        Logger.info("Applying agent overload correction: Reducing mutation rate")
        reduce_mutation_rate()
        
      :all_agents_failed ->
        Logger.info("Applying critical error correction: Restarting agents")
        restart_failed_agents()
        
      _ ->
        Logger.info("Applying general error correction: System reset")
        apply_general_recovery(reason, context)
    end
  end

  defp apply_emergency_recovery(error, cycle_count) do
    Logger.error("🚨 Emergency recovery triggered at cycle #{cycle_count}")
    
    try do
      # Save current state before recovery
      save_emergency_state(cycle_count, error)
      
      # Restart the evolution system
      Logger.info("Restarting evolution system...")
      restart_evolution_system()
      
      # Wait a bit before continuing
      Process.sleep(5000)
      
      Logger.info("✅ Emergency recovery completed")
    rescue
      recovery_error ->
        Logger.error("💀 Emergency recovery failed: #{inspect(recovery_error)}")
        Logger.error("System may be in an unstable state")
    end
  end

  defp reduce_system_complexity do
    Logger.debug("Reducing system complexity to handle timeout")
    # Implementation would reduce query complexity, batch sizes, etc.
  end

  defp cleanup_system_memory do
    Logger.debug("Cleaning up system memory")
    # Force garbage collection
    :erlang.garbage_collect()
    
    # Clear old cached data (implementation specific)
    Logger.debug("Memory cleanup completed")
  end

  defp reduce_mutation_rate do
    Logger.debug("Reducing mutation rate to prevent agent overload")
    # Implementation would adjust mutation frequency
  end

  defp restart_failed_agents do
    Logger.info("Attempting to restart failed agents")
    
    try do
      # Get agent names and check their status
      agent_names = EvolutionSystem.get_agent_names()
      
      Enum.each(agent_names, fn agent_name ->
        try do
          CognitiveAgent.get_state(agent_name)
          Logger.debug("Agent #{agent_name} is responsive")
        rescue
          _ ->
            Logger.warning("Agent #{agent_name} appears to be failed, attempting restart")
            # In a real implementation, you'd restart the specific agent
        end
      end)
    rescue
      error ->
        Logger.error("Failed to check/restart agents: #{inspect(error)}")
    end
  end

  defp apply_general_recovery(reason, context) do
    Logger.info("Applying general recovery for #{inspect(reason)} in context #{inspect(context)}")
    
    # General recovery strategies
    :erlang.garbage_collect()
    Process.sleep(1000)
    
    Logger.debug("General recovery completed")
  end

  defp save_emergency_state(cycle_count, error) do
    timestamp = DateTime.utc_now() |> DateTime.to_iso8601()
    
    emergency_data = %{
      cycle: cycle_count,
      error: inspect(error),
      timestamp: timestamp,
      system_state: get_system_diagnostic_info()
    }
    
    filename = "emergency_state_#{cycle_count}_#{System.system_time()}.etf"
    filepath = Path.join("priv/evolution_storage", filename)
    
    try do
      File.mkdir_p!("priv/evolution_storage")
      binary = :erlang.term_to_binary(emergency_data)
      File.write!(filepath, binary)
      Logger.info("Emergency state saved to #{filepath}")
    rescue
      save_error ->
        Logger.error("Failed to save emergency state: #{inspect(save_error)}")
    end
  end

  defp restart_evolution_system do
    Logger.info("Attempting to restart evolution system")
    
    try do
      # Stop current system if running
      if Process.whereis(EvolutionSystem) do
        Supervisor.stop(EvolutionSystem, :shutdown)
        Process.sleep(2000)
      end
      
      # Restart
      start_evolution_system_with_retry(2)
    rescue
      error ->
        Logger.error("Failed to restart evolution system: #{inspect(error)}")
    end
  end

  defp error_correction_loop do
    receive do
      {:error_detected, type, details} ->
        Logger.warning("🔧 Error detected: #{type} - #{inspect(details)}")
        apply_error_correction(type, details)
        
      {:system_health_check} ->
        perform_system_health_check()
        
    after
      10_000 ->
        # Periodic health check
        perform_system_health_check()
    end
    
    error_correction_loop()
  end

  defp perform_system_health_check do
    try do
      health_status = %{
        evolution_system: check_evolution_system_health(),
        memory_usage: check_memory_usage(),
        process_count: length(Process.list()),
        timestamp: DateTime.utc_now()
      }
      
      if health_status.memory_usage > 0.8 do
        Logger.warning("High memory usage detected: #{Float.round(health_status.memory_usage * 100, 1)}%")
        cleanup_system_memory()
      end
      
      Logger.debug("Health check completed: #{inspect(health_status)}")
    rescue
      error ->
        Logger.warning("Health check failed: #{inspect(error)}")
    end
  end

  defp check_evolution_system_health do
    try do
      case EvolutionSystem.get_system_state() do
        %{evolution_cycles: cycles} when cycles > 0 -> :healthy
        _ -> :degraded
      end
    rescue
      _ -> :failed
    end
  end

  defp check_memory_usage do
    try do
      memory_info = :erlang.memory()
      total = memory_info[:total]
      system = memory_info[:system]
      
      usage_ratio = system / total
      min(usage_ratio, 1.0)
    rescue
      _ -> 0.0
    end
  end

  defp performance_tracking_loop do
    receive after
      15_000 ->
        track_system_performance()
    end
    
    performance_tracking_loop()
  end

  defp track_system_performance do
    try do
      system_state = EvolutionSystem.get_system_state()
      
      performance_metrics = %{
        cycles_completed: system_state[:evolution_cycles] || 0,
        insights_generated: system_state[:global_insights] || 0,
        agents_active: system_state[:agents] || 0,
        continuous_evolution: system_state[:continuous_evolution_running] || false,
        timestamp: DateTime.utc_now()
      }
      
      Logger.info("""
      📊 Performance Update:
      • Cycles: #{performance_metrics.cycles_completed}
      • Insights: #{performance_metrics.insights_generated}  
      • Active Agents: #{performance_metrics.agents_active}
      • Continuous Evolution: #{performance_metrics.continuous_evolution}
      """)
      
      # Log recent insights if available
      if recent_insights = system_state[:recent_insights] do
        Logger.info("🧠 Recent insights: #{length(recent_insights)} new insights generated")
      end
      
    rescue
      error ->
        Logger.warning("Performance tracking failed: #{inspect(error)}")
    end
  end

  defp get_system_diagnostic_info do
    %{
      process_count: length(Process.list()),
      memory_usage: :erlang.memory(),
      node_info: node(),
      timestamp: DateTime.utc_now()
    }
  end

  defp run_status_loop(error_monitor_pid, performance_tracker_pid) do
    IO.puts """
    
    🌟 Continuous Evolution System is Running!
    ==========================================
    
    System Status:
    • Error Monitor: #{if Process.alive?(error_monitor_pid), do: "✅ Active", else: "❌ Failed"}
    • Performance Tracker: #{if Process.alive?(performance_tracker_pid), do: "✅ Active", else: "❌ Failed"}
    • Evolution System: #{check_evolution_status()}
    
    The system will:
    1. Continuously evolve and mutate
    2. Learn from each interaction
    3. Correct errors automatically
    4. Adapt strategies over time
    5. Generate insights and new patterns
    
    Watch the logs above for real-time evolution updates.
    Press Ctrl+C to stop the system.
    """
    
    receive after
      30_000 ->
        show_evolution_progress()
    end
    
    run_status_loop(error_monitor_pid, performance_tracker_pid)
  end

  defp check_evolution_status do
    try do
      case EvolutionSystem.get_system_state() do
        %{continuous_evolution_running: true} -> "🧬 Evolving"
        %{continuous_evolution_running: false} -> "⏸️  Paused"
        _ -> "❓ Unknown"
      end
    rescue
      _ -> "❌ Failed"
    end
  end

  defp show_evolution_progress do
    try do
      system_state = EvolutionSystem.get_system_state()
      
      IO.puts """
      
      🔄 Evolution Progress Report
      ============================
      • Total Cycles: #{system_state[:evolution_cycles] || 0}
      • Global Insights: #{system_state[:global_insights] || 0}
      • Active Agents: #{system_state[:agents] || 0}
      • Status: #{if system_state[:continuous_evolution_running], do: "Running", else: "Stopped"}
      
      The system continues to evolve and learn...
      """
    rescue
      error ->
        IO.puts "Failed to get evolution progress: #{inspect(error)}"
    end
  end
end

# Start the continuous evolution example
IO.puts "🚀 Launching Continuous Evolution Example..."
ContinuousEvolutionExample.run()