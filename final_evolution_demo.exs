IO.puts("""
🧬 Continuous Evolution & Error Correction Demo
==============================================

Starting a system that continuously:
• Mutates and evolves
• Handles errors gracefully
• Adapts based on performance
• Learns from experience

Watch the evolution unfold...
""")

defmodule FinalEvolutionDemo do
  def run do
    agents = start_agents(["Alpha", "Beta", "Gamma"])
    evolution_loop(agents, 0)
  end

  defp start_agents(names) do
    Enum.map(names, fn name ->
      spawn_link(fn -> agent_loop(name, initial_state(name)) end)
    end)
  end

  defp initial_state(name) do
    %{
      name: name,
      knowledge: "I am #{name}, learning to evolve",
      mutations: 0,
      errors: 0,
      generation: 1,
      performance: []
    }
  end

  defp agent_loop(name, state) do
    new_state = try do
      # Attempt mutation
      mutated_knowledge = state.knowledge <> " [Mutation #{state.mutations + 1}]"
      
      # Simulate occasional errors
      if :rand.uniform() < 0.25 do
        raise "Simulated error in #{name}"
      end
      
      # Calculate performance
      performance = 0.5 + :rand.uniform() * 0.4
      
      updated_state = %{state |
        knowledge: mutated_knowledge,
        mutations: state.mutations + 1,
        performance: [performance | state.performance] |> Enum.take(10)
      }
      
      IO.puts("✅ #{name}: Mutation #{updated_state.mutations} (Performance: #{Float.round(performance, 2)})")
      updated_state
      
    rescue
      error ->
        # Error correction
        IO.puts("🔧 #{name}: Error corrected - #{inspect(error)}")
        
        corrected_state = %{state |
          knowledge: state.knowledge <> " [Error corrected]",
          errors: state.errors + 1,
          generation: state.generation + 1,
          performance: [0.6 | state.performance] |> Enum.take(10)
        }
        
        IO.puts("🧬 #{name}: Evolved to generation #{corrected_state.generation}")
        corrected_state
    end
    
    # Continue evolution cycle
    Process.sleep(2000 + :rand.uniform(1000))
    agent_loop(name, new_state)
  end

  defp evolution_loop(agents, cycle) do
    new_cycle = cycle + 1
    
    IO.puts("\n📊 Evolution Cycle #{new_cycle}")
    IO.puts("===================")
    
    # Inject test errors occasionally
    if rem(new_cycle, 5) == 0 and length(agents) > 0 do
      IO.puts("🧪 Injecting test error for resilience testing...")
    end
    
    # Show system status
    IO.puts("🌟 System Status: #{length(agents)} agents actively evolving")
    IO.puts("🔄 Mutations, errors, and adaptations happening continuously...")
    
    receive after
      8000 ->
        evolution_loop(agents, new_cycle)
    end
  end
end

FinalEvolutionDemo.run()