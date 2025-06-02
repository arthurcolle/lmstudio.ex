#!/usr/bin/env elixir

# Live demonstration of the complete evolution system in action
defmodule LiveDemo do
  def run do
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

    # Create and start some self-modifying grids
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

    # Run evolution cycles
    evolution_topics = [
      "implementing fault-tolerant microservices with OTP",
      "optimizing concurrent worker pools for high throughput",
      "building self-healing distributed systems"
    ]

    for {topic, cycle} <- Enum.with_index(evolution_topics, 1) do
      IO.puts("\n📈 EVOLUTION CYCLE #{cycle}")
      IO.puts("Topic: #{topic}")
      IO.puts(String.duplicate("-", 50))
      
      # Each grid processes the topic and evolves
      for {grid_name, grid_pid} <- grid_pids do
        IO.puts("\n🤖 #{grid_name} processing...")
        
        # Generate insight and apply mutations
        insight = generate_insight_for_topic(topic, grid_name)
        performance_score = 0.6 + (:rand.uniform() * 0.3)  # 0.6 to 0.9
        
        IO.puts("  💭 Insight: #{insight}")
        IO.puts("  📊 Performance: #{Float.round(performance_score, 3)}")
        
        # Apply mutations
        mutations = generate_mutations_from_insight(insight)
        
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
        
        Process.sleep(500)
      end
      
      # Generate code every 2 cycles
      if rem(cycle, 2) == 0 do
        IO.puts("\n⚡ CODE GENERATION TRIGGERED:")
        
        code_context = analyze_topic_for_code_generation(topic)
        recommendations = LMStudio.ErlangKnowledgeBase.get_pattern_recommendations(code_context)
        selected_pattern = List.first(recommendations) || :gen_server_with_state
        
        case LMStudio.ErlangKnowledgeBase.generate_code_from_pattern(selected_pattern) do
          {:ok, generated_code} ->
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
            
            # Show snippet
            snippet = String.split(generated_code, "\n") 
                     |> Enum.take(3) 
                     |> Enum.join(" | ")
            IO.puts("  📄 Preview: #{snippet}...")
            
          {:error, reason} ->
            IO.puts("  ❌ Code generation failed: #{inspect(reason)}")
        end
      end
      
      Process.sleep(1000)
    end

    # Show final results
    show_final_results(grid_pids)
  end

  defp generate_insight_for_topic(topic, grid_name) do
    insights = %{
      "implementing fault-tolerant microservices with OTP" => [
        "Supervision trees provide hierarchical fault isolation",
        "GenServer state should be minimized for faster restarts",
        "Circuit breakers prevent cascading failures"
      ],
      "optimizing concurrent worker pools for high throughput" => [
        "Process pools reduce creation overhead significantly", 
        "ETS tables enable lock-free concurrent data access",
        "Message batching improves throughput under high load"
      ],
      "building self-healing distributed systems" => [
        "Node monitoring enables automatic failover strategies",
        "Health checks should be lightweight and frequent",
        "Graceful degradation maintains service availability"
      ]
    }
    
    topic_insights = Map.get(insights, topic, ["General OTP pattern optimization"])
    base_insight = Enum.random(topic_insights)
    
    case grid_name do
      "Explorer_1" -> "Discovered: #{base_insight}"
      "Optimizer_2" -> "Optimized: #{base_insight}"
      "Synthesizer_3" -> "Synthesized: #{base_insight}"
      _ -> base_insight
    end
  end

  defp generate_mutations_from_insight(insight) do
    knowledge_mutation = LMStudio.MetaDSL.Mutation.new(:append, "knowledge",
      content: "\n- #{insight}",
      reasoning: "Adding new insight from evolution cycle"
    )
    
    additional_mutations = cond do
      String.contains?(insight, "supervision") ->
        [LMStudio.MetaDSL.Mutation.new(:evolve, "strategy",
          content: "Enhanced supervision strategy",
          reasoning: "Supervision insight detected")]
      
      String.contains?(insight, "performance") ->
        [LMStudio.MetaDSL.Mutation.new(:evolve, "optimization_focus",
          content: "Prioritizing performance optimization",
          reasoning: "Performance insight detected")]
      
      true -> []
    end
    
    [knowledge_mutation | additional_mutations]
  end

  defp analyze_topic_for_code_generation(topic) do
    cond do
      String.contains?(topic, "microservices") or String.contains?(topic, "distributed") ->
        %{use_case: "service implementation", scale: :large, fault_tolerance: :high}
      
      String.contains?(topic, "worker") or String.contains?(topic, "pool") ->
        %{use_case: "worker processing", scale: :medium, fault_tolerance: :medium}
      
      true ->
        %{use_case: "state management", scale: :medium, fault_tolerance: :medium}
    end
  end

  defp show_final_results(grid_pids) do
    IO.puts("\n📊 FINAL SYSTEM STATE:")
    IO.puts("======================")

    final_stats = LMStudio.Persistence.get_stats()
    IO.puts("💾 Persistence Statistics:")
    IO.puts("  🔑 Total keys: #{final_stats.total_keys}")
    IO.puts("  💿 Memory usage: #{final_stats.memory_usage} bytes")

    # Show grid evolution
    IO.puts("\n🧬 GRID EVOLUTION SUMMARY:")
    for {grid_name, grid_pid} <- grid_pids do
      state = LMStudio.MetaDSL.SelfModifyingGrid.get_state(grid_pid)
      
      avg_performance = case state.performance_metrics do
        [] -> 0.0
        metrics -> 
          recent = Enum.take(metrics, 3)
          Enum.sum(recent) / length(recent)
      end
      
      IO.puts("#{grid_name}:")
      IO.puts("  🔄 Evolution generation: #{state.evolution_generation}")
      IO.puts("  📈 Avg performance: #{Float.round(avg_performance, 3)}")
      IO.puts("  🧠 Knowledge size: #{String.length(Map.get(state.data, "knowledge", ""))} chars")
      IO.puts("  📊 Total mutations: #{length(state.mutation_history)}")
    end

    # Generate final summary
    IO.puts("\n🎯 GENERATING FINAL CODE:")
    final_context = %{use_case: "distributed system", scale: :large, fault_tolerance: :high}
    recommendations = LMStudio.ErlangKnowledgeBase.get_pattern_recommendations(final_context)
    top_pattern = List.first(recommendations)

    case LMStudio.ErlangKnowledgeBase.generate_code_from_pattern(top_pattern) do
      {:ok, summary_code} ->
        lines = String.split(summary_code, "\n") |> length()
        IO.puts("✅ Final #{top_pattern} generated (#{lines} lines)")
        
        final_code_id = "final_evolution_result_#{System.system_time()}"
        LMStudio.Persistence.store({:final_result, final_code_id}, %{
          code: summary_code,
          pattern: top_pattern,
          timestamp: DateTime.utc_now()
        })
        
        IO.puts("💾 Saved final result as: #{final_code_id}")
        
      {:error, reason} ->
        IO.puts("❌ Final generation failed: #{inspect(reason)}")
    end

    LMStudio.Persistence.checkpoint_now()
    IO.puts("\n💾 Final checkpoint completed")

    IO.puts("\n✨ LIVE DEMONSTRATION COMPLETE!")
    IO.puts("===============================")
    IO.puts("🎯 System demonstrated:")
    IO.puts("  ✅ Real-time evolution with cognitive grids")
    IO.puts("  ✅ Dynamic code generation from insights") 
    IO.puts("  ✅ Persistent memory and knowledge accumulation")
    IO.puts("  ✅ Pattern-based OTP application generation")
    IO.puts("  ✅ Performance tracking and optimization")

    IO.puts("\n🔮 The evolution system is now running with permanent memory!")
  end
end

# Run the demonstration
LiveDemo.run()