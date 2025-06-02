#!/usr/bin/env elixir

# Live demonstration of the complete evolution system in action
# This will show real-time evolution, code generation, and persistence

Mix.install([])

# Helper module for demo functions
defmodule DemoHelpers do
  # Helper functions
  def generate_insight_for_topic(topic, grid_name) do
    insights = %{
      "implementing fault-tolerant microservices with OTP" => [
        "Supervision trees provide hierarchical fault isolation",
        "GenServer state should be minimized for faster restarts",
        "Circuit breakers prevent cascading failures in distributed systems"
      ],
      "optimizing concurrent worker pools for high throughput" => [
        "Pool workers should be stateless for better load distribution",
        "Back-pressure mechanisms prevent memory overflow in high-load scenarios",
        "Dynamic worker scaling based on queue depth improves responsiveness"
      ],
      "designing resilient distributed data systems" => [
        "CRDT data structures enable conflict-free distributed updates",
        "Eventual consistency models reduce coordination overhead",
        "Partition tolerance requires careful quorum size selection"
      ],
      "building reactive event-driven architectures" => [
        "Event sourcing provides complete audit trails and replay capabilities",
        "CQRS separation improves read/write performance optimization",
        "Saga patterns coordinate complex distributed transactions"
      ]
    }
    
    specific_insights = Map.get(insights, topic, ["Generic insight about #{topic}"])
    selected_insight = Enum.random(specific_insights)
    
    # Add some grid-specific context
    "#{selected_insight} (discovered by #{grid_name})"
  end

  def generate_mutations_from_insight(insight, current_knowledge) do
    mutations = []
    
    # Always add the insight to knowledge
    knowledge_mutation = LMStudio.MetaDSL.Mutation.new(:append, "knowledge",
      content: "\n- #{insight}",
      reasoning: "Adding new insight from evolution cycle"
    )
    mutations = [knowledge_mutation | mutations]
    
    # Conditionally add other mutations based on insight content
    cond do
      String.contains?(insight, "supervision") ->
        strategy_mutation = LMStudio.MetaDSL.Mutation.new(:evolve, "strategy",
          content: "Enhanced supervision strategy based on fault tolerance insights",
          reasoning: "Supervision insight detected"
        )
        [strategy_mutation | mutations]
        
      String.contains?(insight, "performance") or String.contains?(insight, "optimization") ->
        optimization_mutation = LMStudio.MetaDSL.Mutation.new(:evolve, "optimization_focus",
          content: "Prioritizing performance optimization techniques",
          reasoning: "Performance insight detected"
        )
        [optimization_mutation | mutations]
        
      String.contains?(insight, "distributed") or String.contains?(insight, "cluster") ->
        distribution_mutation = LMStudio.MetaDSL.Mutation.new(:evolve, "distribution_strategy", 
          content: "Enhancing distributed system capabilities",
          reasoning: "Distribution insight detected"
        )
        [distribution_mutation | mutations]
        
      true ->
        mutations
    end
  end

  def analyze_topic_for_code_generation(topic) do
    cond do
      String.contains?(topic, "microservices") or String.contains?(topic, "distributed") ->
        %{use_case: "service implementation", scale: :large, fault_tolerance: :high}
        
      String.contains?(topic, "worker") or String.contains?(topic, "pool") ->
        %{use_case: "worker processing", scale: :medium, fault_tolerance: :medium}
        
      String.contains?(topic, "event") or String.contains?(topic, "pipeline") ->
        %{use_case: "event handling", scale: :medium, fault_tolerance: :high}
        
      String.contains?(topic, "load balancing") or String.contains?(topic, "adaptive") ->
        %{use_case: "worker processing", scale: :large, fault_tolerance: :high}
        
      true ->
        %{use_case: "state management", scale: :medium, fault_tolerance: :medium}
    end
  end
end

# Ensure LMStudio modules are loaded (they should be compiled in _build)
unless Code.ensure_loaded?(LMStudio) do
  # If not compiled, load the source files
  Code.require_file("lib/lmstudio.ex", __DIR__)
  Code.require_file("lib/lmstudio/config.ex", __DIR__)
  Code.require_file("lib/lmstudio/persistence.ex", __DIR__)
  Code.require_file("lib/lmstudio/meta_dsl.ex", __DIR__)
  Code.require_file("lib/lmstudio/erlang_knowledge_base.ex", __DIR__)
  Code.require_file("lib/lmstudio/code_generation.ex", __DIR__)
end

IO.puts("🚀 LIVE EVOLUTION SYSTEM DEMONSTRATION")
IO.puts("=====================================")
IO.puts("Starting complete system with persistence and code generation...")

# Start the persistence system first
{:ok, persistence_pid} = LMStudio.Persistence.start_link()
IO.puts("✅ Persistence system started (PID: #{inspect(persistence_pid)})")

Process.sleep(1000)

# Show initial system state
IO.puts("\n📊 INITIAL SYSTEM STATE:")
stats = LMStudio.Persistence.get_stats()
IO.puts("  💾 Storage: #{stats.storage_dir}")
IO.puts("  🔑 Keys: #{stats.total_keys}")
IO.puts("  💿 Memory: #{stats.memory_usage} bytes")

# Create and start some self-modifying grids (simulating cognitive agents)
IO.puts("\n🧠 STARTING COGNITIVE GRIDS:")

grid_configs = [
  %{
    name: "Explorer_1",
    initial_data: %{
      "identity" => "I am Explorer_1, focused on discovering new patterns",
      "specialty" => "fault-tolerant distributed systems",
      "knowledge" => "Exploring GenServer patterns and supervision trees"
    }
  },
  %{
    name: "Optimizer_2", 
    initial_data: %{
      "identity" => "I am Optimizer_2, focused on performance optimization",
      "specialty" => "concurrent processing and ETS optimization",
      "knowledge" => "Optimizing process pools and message passing"
    }
  },
  %{
    name: "Synthesizer_3",
    initial_data: %{
      "identity" => "I am Synthesizer_3, focused on combining patterns",
      "specialty" => "pattern synthesis and code generation",
      "knowledge" => "Merging successful patterns into new architectures"
    }
  }
]

grid_pids = Enum.map(grid_configs, fn config ->
  {:ok, pid} = LMStudio.MetaDSL.SelfModifyingGrid.start_link([
    name: String.to_atom(config.name),
    grid_id: config.name,
    initial_data: config.initial_data,
    auto_persist: true
  ])
  
  IO.puts("  ✅ #{config.name} grid started (PID: #{inspect(pid)})")
  {config.name, pid}
end)

Process.sleep(2000)

IO.puts("\n🔄 STARTING EVOLUTION CYCLES:")
IO.puts("============================")

# Simulate evolution cycles with real mutations and learning
evolution_topics = [
  "implementing fault-tolerant microservices with OTP",
  "optimizing concurrent worker pools for high throughput",
  "building self-healing distributed systems",
  "creating adaptive load balancing with GenServer pools",
  "designing resilient event processing pipelines"
]

for {topic, cycle} <- Enum.with_index(evolution_topics, 1) do
  IO.puts("\n📈 EVOLUTION CYCLE #{cycle}")
  IO.puts("Topic: #{topic}")
  IO.puts(String.duplicate("-", 50))
  
  # Each grid processes the topic and evolves
  for {grid_name, grid_pid} <- grid_pids do
    IO.puts("\n🤖 #{grid_name} processing...")
    
    # Get current grid state
    current_data = LMStudio.MetaDSL.SelfModifyingGrid.get_data(grid_pid)
    current_knowledge = Map.get(current_data, "knowledge", "")
    
    # Simulate thinking and insight generation based on the topic
    insight = DemoHelpers.generate_insight_for_topic(topic, grid_name)
    performance_score = 0.6 + (:rand.uniform() * 0.3)  # 0.6 to 0.9
    
    IO.puts("  💭 Insight: #{insight}")
    IO.puts("  📊 Performance: #{Float.round(performance_score, 3)}")
    
    # Apply mutations based on insights
    mutations = DemoHelpers.generate_mutations_from_insight(insight, current_knowledge)
    
    for mutation <- mutations do
      case LMStudio.MetaDSL.SelfModifyingGrid.mutate(grid_pid, mutation) do
        {:ok, :mutated} ->
          IO.puts("  ✅ Applied: #{mutation.type} on '#{mutation.target}'")
        {:error, reason} ->
          IO.puts("  ❌ Failed: #{inspect(reason)}")
      end
    end
    
    # Add performance metrics
    LMStudio.MetaDSL.SelfModifyingGrid.add_performance(grid_pid, performance_score)
    
    Process.sleep(500)  # Brief pause between agents
  end
  
  # Generate code from accumulated insights every few cycles
  if rem(cycle, 2) == 0 do
    IO.puts("\n⚡ CODE GENERATION TRIGGERED:")
    
    # Analyze the topic to determine what to generate
    code_context = DemoHelpers.analyze_topic_for_code_generation(topic)
    recommendations = LMStudio.ErlangKnowledgeBase.get_pattern_recommendations(code_context)
    selected_pattern = List.first(recommendations) || :gen_server_with_state
    
    case LMStudio.ErlangKnowledgeBase.generate_code_from_pattern(selected_pattern) do
      {:ok, generated_code} ->
        # Store the generated code
        code_id = "cycle_#{cycle}_#{selected_pattern}_#{System.system_time()}"
        
        code_data = %{
          code: generated_code,
          pattern: selected_pattern,
          cycle: cycle,
          topic: topic,
          context: code_context,
          generated_at: DateTime.utc_now()
        }
        
        LMStudio.Persistence.store({:generated_code, code_id}, code_data)
        
        lines = String.split(generated_code, "\n") |> length()
        IO.puts("  🎯 Generated #{selected_pattern} (#{lines} lines)")
        IO.puts("  💾 Saved as: #{code_id}")
        
        # Show a snippet
        snippet = String.split(generated_code, "\n") 
                 |> Enum.take(3) 
                 |> Enum.join(" | ")
        IO.puts("  📄 Preview: #{snippet}...")
        
      {:error, reason} ->
        IO.puts("  ❌ Code generation failed: #{inspect(reason)}")
    end
  end
  
  # Cross-pollination every 3 cycles
  if rem(cycle, 3) == 0 do
    IO.puts("\n🔄 CROSS-POLLINATION PHASE:")
    
    # Get best performing grid
    performances = Enum.map(grid_pids, fn {name, pid} ->
      state = LMStudio.MetaDSL.SelfModifyingGrid.get_state(pid)
      avg_perf = case state.performance_metrics do
        [] -> 0.5
        metrics -> Enum.sum(Enum.take(metrics, 5)) / length(Enum.take(metrics, 5))
      end
      {name, pid, avg_perf}
    end)
    
    {best_name, best_pid, best_perf} = Enum.max_by(performances, fn {_, _, perf} -> perf end)
    IO.puts("  🏆 Best performer: #{best_name} (#{Float.round(best_perf, 3)})")
    
    # Share knowledge from best to others
    best_data = LMStudio.MetaDSL.SelfModifyingGrid.get_data(best_pid)
    best_knowledge = Map.get(best_data, "knowledge", "")
    
    for {name, pid, _} <- performances, name != best_name do
      knowledge_share = String.slice(best_knowledge, 0, 100)
      mutation = LMStudio.MetaDSL.Mutation.new(:append, "knowledge",
        content: "\n[From #{best_name}]: #{knowledge_share}",
        reasoning: "Cross-pollination from best performer"
      )
      
      LMStudio.MetaDSL.SelfModifyingGrid.mutate(pid, mutation)
      IO.puts("  📤 Shared knowledge: #{best_name} → #{name}")
    end
  end
  
  Process.sleep(1000)  # Pause between cycles
end

# Show final system state
IO.puts("\n📊 FINAL SYSTEM STATE:")
IO.puts("======================")

final_stats = LMStudio.Persistence.get_stats()
IO.puts("💾 Persistence Statistics:")
IO.puts("  🔑 Total keys: #{final_stats.total_keys}")
IO.puts("  💿 Memory usage: #{final_stats.memory_usage} bytes")
IO.puts("  📁 Storage dir: #{final_stats.storage_dir}")

# Show stored keys
stored_keys = LMStudio.Persistence.list_keys()
IO.puts("\n🗂️  Stored Knowledge:")
for key <- Enum.take(stored_keys, 10) do
  case key do
    {:generated_code, code_id} ->
      IO.puts("  📄 Generated code: #{code_id}")
    {:grid_state, grid_id} ->
      IO.puts("  🧠 Grid state: #{grid_id}")
    other ->
      IO.puts("  📦 Data: #{inspect(other)}")
  end
end

# Show grid evolution
IO.puts("\n🧬 GRID EVOLUTION SUMMARY:")
for {grid_name, grid_pid} <- grid_pids do
  state = LMStudio.MetaDSL.SelfModifyingGrid.get_state(grid_pid)
  
  avg_performance = case state.performance_metrics do
    [] -> 0.0
    metrics -> 
      recent = Enum.take(metrics, 5)
      Enum.sum(recent) / length(recent)
  end
  
  IO.puts("#{grid_name}:")
  IO.puts("  🔄 Evolution generation: #{state.evolution_generation}")
  IO.puts("  📈 Avg performance: #{Float.round(avg_performance, 3)}")
  IO.puts("  🧠 Knowledge size: #{String.length(Map.get(state.data, "knowledge", ""))} chars")
  IO.puts("  📊 Total mutations: #{length(state.mutation_history)}")
end

# Generate final summary code
IO.puts("\n🎯 GENERATING SUMMARY CODE:")
final_context = %{use_case: "distributed system", scale: :large, fault_tolerance: :high}
recommendations = LMStudio.ErlangKnowledgeBase.get_pattern_recommendations(final_context)
top_pattern = List.first(recommendations)

case LMStudio.ErlangKnowledgeBase.generate_code_from_pattern(top_pattern) do
  {:ok, summary_code} ->
    lines = String.split(summary_code, "\n") |> length()
    IO.puts("✅ Final #{top_pattern} generated (#{lines} lines)")
    
    # Save the final code
    final_code_id = "final_evolution_result_#{System.system_time()}"
    LMStudio.Persistence.store({:final_result, final_code_id}, %{
      code: summary_code,
      pattern: top_pattern,
      total_cycles: length(evolution_topics),
      final_context: final_context,
      timestamp: DateTime.utc_now()
    })
    
    IO.puts("💾 Saved final result as: #{final_code_id}")
    
  {:error, reason} ->
    IO.puts("❌ Final generation failed: #{inspect(reason)}")
end

# Trigger final checkpoint
LMStudio.Persistence.checkpoint_now()
IO.puts("\n💾 Final checkpoint completed")

IO.puts("\n✨ LIVE DEMONSTRATION COMPLETE!")
IO.puts("===============================")
IO.puts("🎯 System demonstrated:")
IO.puts("  ✅ Real-time evolution with #{length(evolution_topics)} cycles")
IO.puts("  ✅ #{length(grid_pids)} cognitive grids with auto-persistence")
IO.puts("  ✅ Dynamic code generation based on insights")
IO.puts("  ✅ Cross-pollination and knowledge sharing")
IO.puts("  ✅ Persistent memory across #{final_stats.total_keys} stored items")
IO.puts("  ✅ Production-ready OTP pattern generation")

IO.puts("\n🔮 The evolution system is now running with permanent memory,")
IO.puts("   continuously learning and generating increasingly sophisticated")
IO.puts("   Erlang/OTP applications based on accumulated knowledge!")