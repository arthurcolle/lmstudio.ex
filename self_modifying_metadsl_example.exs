#!/usr/bin/env elixir

# Self-Modifying MetaDSL Cognitive System Example
# ===============================================
# 
# This demonstrates a continuously evolving system that uses MetaDSL to mutate 
# its own prompts, reasoning patterns, and cognitive strategies through recursive 
# self-improvement.
#
# Usage: mix run self_modifying_metadsl_example.exs

# Dependencies are managed by mix.exs in this project

defmodule MetaDSLDemo do
  @moduledoc """
  Demonstration of the Self-Modifying MetaDSL system.
  """
  
  require Logger
  alias LMStudio.{EvolutionSystem, CognitiveAgent}
  alias LMStudio.MetaDSL.{SelfModifyingGrid, Mutation}

  def run_demo do
    IO.puts("""
    
    ╔══════════════════════════════════════════════════════════════╗
    ║              Self-Modifying MetaDSL System                   ║
    ║                    Elixir Implementation                     ║
    ╚══════════════════════════════════════════════════════════════╝
    
    This system demonstrates:
    • Cognitive agents that can think and modify themselves
    • MetaDSL for prompt and behavior mutations
    • Cross-pollination between agents
    • Continuous evolution and learning
    
    """)

    IO.puts("Starting demonstration...")
    
    # Example 1: Single Grid Mutation
    IO.puts("\n" <> String.duplicate("=", 60))
    IO.puts("Example 1: Basic Self-Modifying Grid")
    IO.puts(String.duplicate("=", 60))
    
    demonstrate_basic_grid()
    
    # Example 2: Single Cognitive Agent
    IO.puts("\n" <> String.duplicate("=", 60))
    IO.puts("Example 2: Single Cognitive Agent with Thinking")
    IO.puts(String.duplicate("=", 60))
    
    demonstrate_single_agent()
    
    # Example 3: Multi-Agent Evolution System
    IO.puts("\n" <> String.duplicate("=", 60))
    IO.puts("Example 3: Multi-Agent Evolution System")
    IO.puts(String.duplicate("=", 60))
    
    demonstrate_evolution_system()
    
    IO.puts("\n" <> String.duplicate("=", 60))
    IO.puts("Demonstration Complete!")
    IO.puts(String.duplicate("=", 60))
  end
  
  defp demonstrate_basic_grid do
    IO.puts("Creating a self-modifying grid...")
    
    # Start a grid with initial data
    initial_data = %{
      "identity" => "I am a learning system",
      "knowledge" => "I start with basic knowledge",
      "strategy" => "I learn through mutations"
    }
    
    {:ok, grid_pid} = SelfModifyingGrid.start_link(initial_data: initial_data)
    
    IO.puts("Initial grid state:")
    initial_rendered = SelfModifyingGrid.render_grid(grid_pid)
    IO.puts(initial_rendered)
    
    # Apply some mutations
    IO.puts("\nApplying mutations...")
    
    mutations = [
      Mutation.new(:append, "knowledge", content: "\nI have learned about self-modification"),
      Mutation.new(:evolve, "strategy", content: "Enhanced strategy through experience"),
      Mutation.new(:append, "identity", content: " with growing capabilities")
    ]
    
    Enum.each(mutations, fn mutation ->
      case SelfModifyingGrid.mutate(grid_pid, mutation) do
        {:ok, :mutated} ->
          IO.puts("✓ Applied #{mutation.type} mutation to #{mutation.target}")
        {:error, reason} ->
          IO.puts("✗ Failed to apply mutation: #{inspect(reason)}")
      end
    end)
    
    # Add some performance scores
    [0.8, 0.9, 0.7, 0.85]
    |> Enum.each(&SelfModifyingGrid.add_performance(grid_pid, &1))
    
    IO.puts("\nGrid state after mutations:")
    final_rendered = SelfModifyingGrid.render_grid(grid_pid)
    IO.puts(final_rendered)
    
    # Analyze patterns
    analysis = SelfModifyingGrid.analyze_patterns(grid_pid)
    IO.puts("\nPattern analysis:")
    IO.inspect(analysis, pretty: true)
    
    # Get suggestions
    suggestions = SelfModifyingGrid.suggest_mutations(grid_pid)
    IO.puts("\nSuggested mutations: #{length(suggestions)}")
    
    GenServer.stop(grid_pid)
  end
  
  defp demonstrate_single_agent do
    IO.puts("Starting Registry for agent names...")
    case Registry.start_link(keys: :unique, name: LMStudio.AgentRegistry) do
      {:ok, _registry_pid} -> 
        IO.puts("✓ Registry started successfully")
      {:error, {:already_started, _pid}} ->
        IO.puts("✓ Registry already running")
    end
    
    IO.puts("Creating a cognitive agent...")
    
    agent_opts = [
      name: "DemoAgent", 
      model: "deepseek-r1-0528-qwen3-8b-mlx",
      thinking_enabled: true
    ]
    
    case CognitiveAgent.start_link(agent_opts) do
      {:ok, _agent_pid} ->
        IO.puts("✓ Agent 'DemoAgent' created successfully")
        
        # Test queries
        test_queries = [
          "What is the nature of self-improvement?",
          "How can I evolve my reasoning patterns?",
          "What mutations would help me learn better?"
        ]
        
        Enum.with_index(test_queries, 1)
        |> Enum.each(fn {query, index} ->
          IO.puts("\n--- Query #{index}: #{query} ---")
          
          case CognitiveAgent.process_query("DemoAgent", query) do
            {:ok, result} ->
              IO.puts("Response: #{String.slice(result.response, 0, 200)}...")
              if byte_size(result.thinking) > 0 do
                IO.puts("Thinking: #{String.slice(result.thinking, 0, 150)}...")
              end
              IO.puts("Mutations applied: #{result.mutations_applied}")
              IO.puts("Performance score: #{Float.round(result.performance_score, 3)}")
              
            {:error, reason} ->
              IO.puts("Error: #{inspect(reason)}")
          end
          
          Process.sleep(1000)  # Brief pause between queries
        end)
        
        # Show agent state
        IO.puts("\n--- Final Agent State ---")
        state = CognitiveAgent.get_state("DemoAgent")
        IO.puts("Interactions: #{state.interaction_count}")
        IO.puts("Insights: #{state.insight_count}")
        IO.puts("Recent conversations: #{length(state.conversation_history)}")
        
        # Show grid data
        grid_data = CognitiveAgent.get_grid_data("DemoAgent")
        IO.puts("\n--- Agent's Evolved Grid Data ---")
        Enum.each(grid_data, fn {key, value} ->
          IO.puts("#{key}: #{String.slice(value, 0, 100)}...")
        end)
        
        # Trigger autonomous evolution
        IO.puts("\n--- Triggering Autonomous Evolution ---")
        CognitiveAgent.evolve_autonomously("DemoAgent", 2)
        Process.sleep(5000)  # Wait for evolution to complete
        
        final_state = CognitiveAgent.get_state("DemoAgent")
        IO.puts("Final interactions: #{final_state.interaction_count}")
        IO.puts("Final insights: #{final_state.insight_count}")
        
      {:error, reason} ->
        IO.puts("✗ Failed to create agent: #{inspect(reason)}")
    end
  end
  
  defp demonstrate_evolution_system do
    IO.puts("Starting the Evolution System with multiple agents...")
    
    case EvolutionSystem.start_link(num_agents: 3, model: "deepseek-r1-0528-qwen3-8b-mlx") do
      {:ok, _supervisor_pid} ->
        IO.puts("✓ Evolution System started successfully")
        
        Process.sleep(2000)  # Allow system to initialize
        
        # Get agent names
        agent_names = EvolutionSystem.get_agent_names()
        IO.puts("Active agents: #{inspect(agent_names)}")
        
        # Run a few evolution cycles
        topics = [
          "the nature of recursive self-improvement",
          "consciousness emerging from information processing",
          "optimization of reasoning through mutations"
        ]
        
        Enum.with_index(topics, 1)
        |> Enum.each(fn {topic, cycle} ->
          IO.puts("\n=== Evolution Cycle #{cycle} ===")
          IO.puts("Topic: #{topic}")
          
          case EvolutionSystem.run_evolution_cycle(topic) do
            :ok ->
              IO.puts("✓ Evolution cycle #{cycle} completed successfully")
              
              # Show system state
              system_state = EvolutionSystem.get_system_state()
              IO.puts("System State:")
              IO.puts("  Cycles: #{system_state.evolution_cycles}")
              IO.puts("  Interactions: #{system_state.total_interactions}")
              IO.puts("  Insights: #{system_state.global_insights}")
              
            {:error, reason} ->
              IO.puts("✗ Evolution cycle #{cycle} failed: #{inspect(reason)}")
          end
          
          Process.sleep(2000)  # Pause between cycles
        end)
        
        # Demonstrate continuous evolution briefly
        IO.puts("\n=== Starting Brief Continuous Evolution ===")
        EvolutionSystem.start_continuous_evolution("the evolution of artificial intelligence")
        
        IO.puts("Running continuous evolution for 10 seconds...")
        Process.sleep(10_000)
        
        EvolutionSystem.stop_continuous_evolution()
        IO.puts("Stopped continuous evolution")
        
        # Final system state
        final_state = EvolutionSystem.get_system_state()
        IO.puts("\n=== Final System State ===")
        IO.inspect(final_state, pretty: true)
        
      {:error, reason} ->
        IO.puts("✗ Failed to start Evolution System: #{inspect(reason)}")
    end
  end
end

# Interactive Demo Menu
defmodule InteractiveDemo do
  def run do
    IO.puts("""
    
    ╔══════════════════════════════════════════════════════════════╗
    ║             Interactive MetaDSL Demo Menu                    ║
    ╚══════════════════════════════════════════════════════════════╝
    
    Choose a demonstration:
    
    1. Full automated demo (all examples)
    2. Basic grid mutations only
    3. Single cognitive agent only  
    4. Multi-agent evolution system only
    5. Custom query to single agent
    6. Start continuous evolution (Ctrl+C to stop)
    
    """)
    
    choice = IO.gets("Enter your choice (1-6): ") |> String.trim()
    
    case choice do
      "1" -> MetaDSLDemo.run_demo()
      "2" -> MetaDSLDemo.demonstrate_basic_grid()
      "3" -> MetaDSLDemo.demonstrate_single_agent()
      "4" -> MetaDSLDemo.demonstrate_evolution_system()
      "5" -> run_custom_query()
      "6" -> run_continuous_evolution()
      _ -> 
        IO.puts("Invalid choice. Please run the script again.")
    end
  end
  
  defp run_custom_query do
    IO.puts("Starting single agent for custom query...")
    
    case Registry.start_link(keys: :unique, name: LMStudio.AgentRegistry) do
      {:ok, _registry_pid} -> 
        IO.puts("✓ Registry started successfully")
      {:error, {:already_started, _pid}} ->
        IO.puts("✓ Registry already running")
    end
    
    case CognitiveAgent.start_link([name: "CustomAgent", thinking_enabled: true]) do
      {:ok, _pid} ->
        query = IO.gets("\nEnter your query: ") |> String.trim()
        
        case CognitiveAgent.process_query("CustomAgent", query) do
          {:ok, result} ->
            IO.puts("\n--- Agent Response ---")
            IO.puts(result.response)
            
            if byte_size(result.thinking) > 0 do
              IO.puts("\n--- Agent Thinking ---")
              IO.puts(result.thinking)
            end
            
            IO.puts("\n--- Metadata ---")
            IO.puts("Mutations applied: #{result.mutations_applied}")
            IO.puts("Performance score: #{Float.round(result.performance_score, 3)}")
            
          {:error, reason} ->
            IO.puts("Error: #{inspect(reason)}")
        end
        
      {:error, reason} ->
        IO.puts("Failed to start agent: #{inspect(reason)}")
    end
  end
  
  defp run_continuous_evolution do
    IO.puts("Starting continuous evolution system...")
    IO.puts("Press Ctrl+C to stop")
    
    case EvolutionSystem.start_link(num_agents: 2) do
      {:ok, _pid} ->
        Process.sleep(2000)
        EvolutionSystem.start_continuous_evolution()
        
        # Keep running until interrupted
        Stream.repeatedly(fn -> Process.sleep(5000) end)
        |> Stream.each(fn _ ->
          state = EvolutionSystem.get_system_state()
          IO.puts("Cycles: #{state.evolution_cycles}, Interactions: #{state.total_interactions}, Insights: #{state.global_insights}")
        end)
        |> Enum.take(1000)  # Effectively infinite
        
      {:error, reason} ->
        IO.puts("Failed to start evolution system: #{inspect(reason)}")
    end
  end
end

# Check if we should run interactively or automatically
case System.argv() do
  ["--interactive"] -> InteractiveDemo.run()
  [] -> MetaDSLDemo.run_demo()
  _ -> 
    IO.puts("Usage: mix run self_modifying_metadsl_example.exs [--interactive]")
end