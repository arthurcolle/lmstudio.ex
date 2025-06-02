#!/usr/bin/env elixir

# Agent Demo CLI - Simple working visualization
# Run with: elixir agent_demo_cli.exs

defmodule AgentDemo do
  def run(duration_seconds \\ 10) do
    IO.puts(IO.ANSI.clear())
    IO.puts(IO.ANSI.cyan() <> IO.ANSI.bright() <> """
    
    ╔═══════════════════════════════════════════════════════════════╗
    ║             MULTI-AGENT SYSTEM DEMONSTRATION                  ║
    ╚═══════════════════════════════════════════════════════════════╝
    
    """ <> IO.ANSI.reset())
    
    IO.puts("Starting agent simulation for #{duration_seconds} seconds...\n")
    Process.sleep(1000)
    
    # Run simulation
    start_time = System.os_time(:second)
    simulate_agents(start_time, duration_seconds)
    
    IO.puts("\n\n✅ Simulation complete!")
  end
  
  defp simulate_agents(start_time, duration) do
    agents = [
      %{name: "Orchestrator", icon: "👑", color: IO.ANSI.cyan()},
      %{name: "Researcher", icon: "🔍", color: IO.ANSI.blue()},
      %{name: "Developer", icon: "💻", color: IO.ANSI.green()},
      %{name: "Analyst", icon: "📊", color: IO.ANSI.yellow()},
      %{name: "Tester", icon: "🧪", color: IO.ANSI.magenta()}
    ]
    
    tasks = [
      "Building authentication system",
      "Analyzing user data patterns",
      "Optimizing database queries",
      "Creating API endpoints",
      "Writing test suites"
    ]
    
    loop_agents(agents, tasks, start_time, duration, 0)
  end
  
  defp loop_agents(agents, tasks, start_time, duration, tick) do
    current_time = System.os_time(:second)
    elapsed = current_time - start_time
    
    if elapsed < duration do
      # Clear previous output
      IO.write(IO.ANSI.cursor_up(20) <> IO.ANSI.clear_line())
      
      # Display agents
      IO.puts(IO.ANSI.bright() <> "\n🤖 ACTIVE AGENTS:" <> IO.ANSI.reset())
      IO.puts(String.duplicate("─", 65))
      
      for {agent, idx} <- Enum.with_index(agents) do
        status = get_random_status(tick, idx)
        display_agent(agent, status)
      end
      
      # Display current activity
      IO.puts(IO.ANSI.bright() <> "\n💬 CURRENT ACTIVITY:" <> IO.ANSI.reset())
      IO.puts(String.duplicate("─", 65))
      
      # Generate random interactions
      if rem(tick, 3) == 0 do
        from = Enum.random(agents)
        to = Enum.random(agents -- [from])
        action = Enum.random(["requesting help from", "sharing results with", "coordinating with"])
        
        IO.puts("#{from.color}#{from.icon} #{from.name}#{IO.ANSI.reset()} #{action} " <>
                "#{to.color}#{to.icon} #{to.name}#{IO.ANSI.reset()}")
      end
      
      if rem(tick, 4) == 1 do
        agent = Enum.random(agents)
        task = Enum.random(tasks)
        IO.puts("#{agent.color}#{agent.icon} #{agent.name}#{IO.ANSI.reset()} is working on: #{task}")
      end
      
      if rem(tick, 5) == 2 do
        agent = Enum.random(agents)
        IO.puts("#{agent.color}#{agent.icon} #{agent.name}#{IO.ANSI.reset()} completed a task! ✅")
      end
      
      # Display stats
      IO.puts(IO.ANSI.bright() <> "\n📊 STATISTICS:" <> IO.ANSI.reset())
      IO.puts(String.duplicate("─", 65))
      IO.puts("Time elapsed: #{elapsed}s / #{duration}s")
      IO.puts("Messages exchanged: #{tick * 3}")
      IO.puts("Tasks completed: #{div(tick, 5)}")
      
      Process.sleep(500)
      loop_agents(agents, tasks, start_time, duration, tick + 1)
    end
  end
  
  defp display_agent(agent, status) do
    status_icon = case status do
      :idle -> "🟢"
      :working -> "🟡"
      :thinking -> "🔵"
      :communicating -> "🔄"
    end
    
    IO.puts("  #{agent.color}#{agent.icon} #{String.pad_trailing(agent.name, 15)}#{IO.ANSI.reset()} " <>
            "#{status_icon} #{status}")
  end
  
  defp get_random_status(tick, agent_idx) do
    # Create some deterministic but varying behavior
    value = rem(tick + agent_idx * 3, 10)
    
    cond do
      value < 3 -> :idle
      value < 6 -> :working
      value < 8 -> :thinking
      true -> :communicating
    end
  end
end

# Check command line arguments
duration = case System.argv() do
  [seconds] -> 
    case Integer.parse(seconds) do
      {num, _} when num > 0 -> num
      _ -> 10
    end
  _ -> 10
end

# Run the demo
AgentDemo.run(duration)