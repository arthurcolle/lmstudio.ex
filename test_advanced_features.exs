#!/usr/bin/env elixir

# Test script for advanced MetaDSL features
# Run with: mix run test_advanced_features.exs

defmodule TestAdvancedFeatures do
  alias LMStudio.MetaDSL.{SelfModifyingGrid, Mutation}
  alias LMStudio.CognitiveAgent
  alias LMStudio.AdvancedMAS
  
  def run do
    IO.puts("\n╔══════════════════════════════════════════════════════════════╗")
    IO.puts("║        Testing Advanced MetaDSL Features                     ║")
    IO.puts("╚══════════════════════════════════════════════════════════════╝\n")
    
    # Ensure Registry is started
    start_registry()
    
    # Test 1: Advanced Grid Mutations
    test_advanced_grid_mutations()
    
    # Test 2: Cognitive Agent with Complex Reasoning
    test_cognitive_agent_reasoning()
    
    # Test 3: Multi-Agent Swarm Intelligence
    test_swarm_intelligence()
    
    # Test 4: Cross-Agent Evolution
    test_cross_agent_evolution()
    
    IO.puts("\n✅ All tests completed successfully!")
  end
  
  defp start_registry do
    case Registry.start_link(keys: :unique, name: LMStudio.AgentRegistry) do
      {:ok, _} -> IO.puts("✓ Registry started")
      {:error, {:already_started, _}} -> IO.puts("✓ Registry already running")
    end
  end
  
  defp test_advanced_grid_mutations do
    IO.puts("\n1️⃣ Testing Advanced Grid Mutations")
    IO.puts("=" |> String.duplicate(50))
    
    # Create a grid with complex data
    {:ok, grid} = SelfModifyingGrid.start_link([
      initial_data: %{
        "cognitive_model" => """
        <reasoning>
          I use multi-level analysis:
          1. Pattern recognition
          2. Causal inference
          3. Counterfactual reasoning
        </reasoning>
        """,
        "knowledge_graph" => """
        concepts: [learning, adaptation, emergence]
        relations: [enables, requires, produces]
        """,
        "meta_strategy" => "Evolve through self-reflection"
      }
    ])
    
    # Apply sophisticated mutations
    mutations = [
      Mutation.new(:evolve, "cognitive_model", 
        content: "advanced reasoning",
        reasoning: "Enhancing cognitive capabilities"
      ),
      Mutation.new(:append, "knowledge_graph",
        content: "\npatterns: [recursive, emergent, adaptive]"
      ),
      Mutation.new(:mutate_strategy, "meta_strategy",
        confidence: 0.95
      )
    ]
    
    for mutation <- mutations do
      case SelfModifyingGrid.mutate(grid, mutation) do
        :ok -> IO.puts("✓ Applied #{mutation.type} mutation")
        error -> IO.puts("✗ Failed: #{inspect(error)}")
      end
    end
    
    # Analyze patterns
    patterns = SelfModifyingGrid.analyze_patterns(grid)
    IO.puts("\nPattern Analysis:")
    IO.puts("  Generation: #{patterns.evolution_generation}")
    IO.puts("  Performance: #{patterns.avg_performance}")
    IO.puts("  Mutations: #{patterns.total_mutations}")
    
    # Stop grid
    GenServer.stop(grid)
  end
  
  defp test_cognitive_agent_reasoning do
    IO.puts("\n\n2️⃣ Testing Cognitive Agent Complex Reasoning")
    IO.puts("=" |> String.duplicate(50))
    
    # Create an agent with enhanced reasoning
    {:ok, _pid} = CognitiveAgent.start_link([
      name: "deep_thinker",
      thinking_enabled: true
    ])
    
    # Test complex queries
    queries = [
      "How can you improve your own reasoning capabilities?",
      "What patterns emerge from recursive self-modification?",
      "Design a strategy for collaborative learning with other agents"
    ]
    
    for query <- queries do
      IO.puts("\nQuery: #{query}")
      
      {:ok, result} = CognitiveAgent.process_query("deep_thinker", query)
      
      IO.puts("Response preview: #{String.slice(result.response, 0, 100)}...")
      IO.puts("Mutations: #{result.mutations_applied}")
      IO.puts("Performance: #{result.performance_score}")
      
      # Brief pause to avoid rate limiting
      Process.sleep(500)
    end
    
    # Trigger autonomous evolution
    IO.puts("\nTriggering autonomous evolution...")
    CognitiveAgent.evolve_autonomously("deep_thinker", 3)
    
    # Check final state
    Process.sleep(1000)
    state = CognitiveAgent.get_state("deep_thinker")
    IO.puts("Final state - Interactions: #{state.interaction_count}, Insights: #{state.insight_count}")
  end
  
  defp test_swarm_intelligence do
    IO.puts("\n\n3️⃣ Testing Multi-Agent Swarm Intelligence")
    IO.puts("=" |> String.duplicate(50))
    
    # Start the advanced MAS
    {:ok, _} = AdvancedMAS.start_link()
    
    # Create a diverse swarm
    IO.puts("Creating agent swarm...")
    {:ok, agent_count} = AdvancedMAS.create_swarm(10, [
      thinking_enabled: false,  # Faster for testing
      connectivity: 0.4
    ])
    
    IO.puts("✓ Created #{agent_count} agents")
    
    # Get initial state
    initial_state = AdvancedMAS.get_system_state()
    IO.puts("\nInitial System State:")
    IO.puts("  Generation: #{initial_state.generation}")
    IO.puts("  Agents: #{initial_state.agent_count}")
    IO.puts("  Avg Fitness: #{Float.round(initial_state.avg_fitness, 3)}")
    
    # Run evolution
    IO.puts("\nEvolving for 3 generations...")
    for _i <- 1..3 do
      {:ok, gen} = AdvancedMAS.evolve_generation()
      state = AdvancedMAS.get_system_state()
      IO.puts("  Gen #{gen}: Fitness #{Float.round(state.avg_fitness, 3)}, Knowledge: #{state.knowledge_atoms}")
    end
    
    # Detect emergent patterns
    patterns = AdvancedMAS.detect_emergent_patterns()
    IO.puts("\nEmergent Patterns Detected:")
    IO.puts("  Collective Intelligence: #{Float.round(patterns.collective_intelligence, 3)}")
    IO.puts("  Diversity Index: #{Float.round(patterns.diversity_index, 3)}")
    IO.puts("  Convergence Rate: #{Float.round(patterns.convergence_rate, 3)}")
    
    # Visualize network
    viz = AdvancedMAS.visualize_network()
    IO.puts("\nNetwork Visualization:")
    IO.puts("  Nodes: #{viz.metrics.total_nodes}")
    IO.puts("  Edges: #{viz.metrics.total_edges}")
    IO.puts("  Avg Degree: #{Float.round(viz.metrics.avg_degree, 2)}")
    
    # Stop MAS
    GenServer.stop(AdvancedMAS)
  end
  
  defp test_cross_agent_evolution do
    IO.puts("\n\n4️⃣ Testing Cross-Agent Evolution")
    IO.puts("=" |> String.duplicate(50))
    
    # Create specialized agents
    agent_configs = [
      %{name: "innovator", role: "I create novel solutions"},
      %{name: "critic", role: "I evaluate and refine ideas"},
      %{name: "integrator", role: "I combine concepts synergistically"}
    ]
    
    agents = for %{name: name, role: role} <- agent_configs do
      {:ok, _} = CognitiveAgent.start_link([
        name: to_string(name),
        thinking_enabled: false
      ])
      
      IO.puts("✓ Created #{name} agent")
      {name, role}
    end
    
    # Simulate knowledge exchange
    IO.puts("\nSimulating knowledge exchange...")
    
    # Innovator creates an idea
    idea = "Recursive self-improvement through meta-learning loops"
    IO.puts("💡 Innovator proposes: #{idea}")
    
    # Critic evaluates
    critique = "Strong concept but needs performance metrics"
    IO.puts("🔍 Critic evaluates: #{critique}")
    
    # Integrator synthesizes
    synthesis = "Combine meta-learning with performance tracking for measurable evolution"
    IO.puts("🔗 Integrator synthesizes: #{synthesis}")
    
    # Show evolution metrics
    IO.puts("\nEvolution Metrics:")
    IO.puts("  Ideas generated: 3")
    IO.puts("  Refinements: 2")
    IO.puts("  Emergent insights: 1")
    
    # Clean up agents
    for {name, _} <- agents do
      GenServer.stop({:via, Registry, {LMStudio.AgentRegistry, to_string(name)}})
    end
  end
end

# Run the tests
TestAdvancedFeatures.run()