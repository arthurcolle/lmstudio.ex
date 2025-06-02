#!/usr/bin/env elixir

# Agent Visualization CLI
# A simpler, working visualization of multi-agent interactions

defmodule AgentViz do
  @update_interval 500  # milliseconds
  
  defmodule Colors do
    def red(t), do: IO.ANSI.red() <> t <> IO.ANSI.reset()
    def green(t), do: IO.ANSI.green() <> t <> IO.ANSI.reset()
    def yellow(t), do: IO.ANSI.yellow() <> t <> IO.ANSI.reset()
    def blue(t), do: IO.ANSI.blue() <> t <> IO.ANSI.reset()
    def magenta(t), do: IO.ANSI.magenta() <> t <> IO.ANSI.reset()
    def cyan(t), do: IO.ANSI.cyan() <> t <> IO.ANSI.reset()
    def white(t), do: IO.ANSI.white() <> t <> IO.ANSI.reset()
    def bright(t), do: IO.ANSI.bright() <> t <> IO.ANSI.reset()
    def dim(t), do: IO.ANSI.light_black() <> t <> IO.ANSI.reset()
  end
  
  def run do
    IO.puts(Colors.bright(Colors.cyan("""
    
    ╔═════════════════════════════════════════════════════════════╗
    ║          MULTI-AGENT SYSTEM VISUALIZATION                   ║
    ╚═════════════════════════════════════════════════════════════╝
    
    """)))
    
    # Initialize state
    state = %{
      agents: create_initial_agents(),
      messages: [],
      tasks: create_initial_tasks(),
      tick: 0,
      completed_tasks: 0
    }
    
    # Run simulation loop
    loop(state)
  end
  
  defp create_initial_agents do
    [
      %{id: "COORD-001", name: "Coordinator", type: :coordinator, status: :idle, color: &Colors.cyan/1},
      %{id: "RES-001", name: "Researcher", type: :researcher, status: :idle, color: &Colors.blue/1},
      %{id: "DEV-001", name: "Developer", type: :developer, status: :idle, color: &Colors.green/1},
      %{id: "TEST-001", name: "Tester", type: :tester, status: :idle, color: &Colors.yellow/1},
      %{id: "ARCH-001", name: "Architect", type: :architect, status: :idle, color: &Colors.magenta/1}
    ]
  end
  
  defp create_initial_tasks do
    [
      %{id: 1, name: "Design API endpoints", assigned_to: nil, status: :pending},
      %{id: 2, name: "Research best practices", assigned_to: nil, status: :pending},
      %{id: 3, name: "Implement user auth", assigned_to: nil, status: :pending},
      %{id: 4, name: "Write test cases", assigned_to: nil, status: :pending},
      %{id: 5, name: "Optimize database", assigned_to: nil, status: :pending}
    ]
  end
  
  defp loop(state) do
    # Clear screen
    IO.write(IO.ANSI.clear() <> IO.ANSI.home())
    
    # Display header
    display_header(state)
    
    # Display agents
    display_agents(state.agents)
    
    # Display tasks
    display_tasks(state.tasks)
    
    # Display messages
    display_messages(state.messages)
    
    # Display stats
    display_stats(state)
    
    # Update state
    new_state = update_state(state)
    
    # Check for exit
    if check_exit() do
      IO.puts("\n\n" <> Colors.bright("Goodbye! 👋"))
    else
      Process.sleep(@update_interval)
      loop(new_state)
    end
  end
  
  defp display_header(state) do
    IO.puts(Colors.dim("Tick: #{state.tick} | Press Ctrl+C to exit"))
    IO.puts(String.duplicate("─", 65))
  end
  
  defp display_agents(agents) do
    IO.puts("\n" <> Colors.bright("🤖 AGENTS"))
    IO.puts(String.duplicate("─", 65))
    
    for agent <- agents do
      status_icon = case agent.status do
        :idle -> Colors.green("●")
        :working -> Colors.yellow("◐")
        :thinking -> Colors.blue("◑")
        :communicating -> Colors.cyan("↔")
        _ -> Colors.dim("○")
      end
      
      name = agent.color.(String.pad_trailing(agent.name, 12))
      IO.puts("  #{status_icon} #{name} [#{agent.id}] - #{agent.status}")
    end
  end
  
  defp display_tasks(tasks) do
    IO.puts("\n" <> Colors.bright("📋 TASKS"))
    IO.puts(String.duplicate("─", 65))
    
    for task <- Enum.take(tasks, 5) do
      status_icon = case task.status do
        :pending -> Colors.dim("○")
        :in_progress -> Colors.yellow("◐")
        :completed -> Colors.green("✓")
        _ -> Colors.red("✗")
      end
      
      assigned = if task.assigned_to, do: " → #{task.assigned_to}", else: ""
      task_name = String.pad_trailing(task.name, 25)
      
      IO.puts("  #{status_icon} #{task_name}#{assigned}")
    end
  end
  
  defp display_messages(messages) do
    IO.puts("\n" <> Colors.bright("💬 RECENT MESSAGES"))
    IO.puts(String.duplicate("─", 65))
    
    recent_messages = messages |> Enum.take(5)
    
    if Enum.empty?(recent_messages) do
      IO.puts(Colors.dim("  No messages yet..."))
    else
      for msg <- recent_messages do
        from_color = get_agent_color(msg.from)
        to_color = get_agent_color(msg.to)
        
        IO.puts("  #{from_color.(msg.from)} → #{to_color.(msg.to)}: #{Colors.dim(msg.content)}")
      end
    end
  end
  
  defp display_stats(state) do
    IO.puts("\n" <> Colors.bright("📊 STATISTICS"))
    IO.puts(String.duplicate("─", 65))
    
    total_agents = length(state.agents)
    active_agents = Enum.count(state.agents, &(&1.status != :idle))
    pending_tasks = Enum.count(state.tasks, &(&1.status == :pending))
    
    IO.puts("  Total Agents: #{Colors.bright(to_string(total_agents))}")
    IO.puts("  Active Agents: #{Colors.green(to_string(active_agents))}")
    IO.puts("  Pending Tasks: #{Colors.yellow(to_string(pending_tasks))}")
    IO.puts("  Completed Tasks: #{Colors.cyan(to_string(state.completed_tasks))}")
    IO.puts("  Total Messages: #{Colors.magenta(to_string(length(state.messages)))}")
  end
  
  defp update_state(state) do
    state
    |> Map.update!(:tick, &(&1 + 1))
    |> maybe_assign_task()
    |> maybe_update_agent_status()
    |> maybe_send_message()
    |> maybe_complete_task()
    |> maybe_add_agent()
    |> maybe_add_task()
  end
  
  defp maybe_assign_task(state) do
    idle_agents = Enum.filter(state.agents, &(&1.status == :idle))
    pending_tasks = Enum.filter(state.tasks, &(&1.status == :pending))
    
    if length(idle_agents) > 0 and length(pending_tasks) > 0 and :rand.uniform() > 0.7 do
      agent = Enum.random(idle_agents)
      task = hd(pending_tasks)
      
      # Update agent status
      agents = Enum.map(state.agents, fn a ->
        if a.id == agent.id, do: %{a | status: :working}, else: a
      end)
      
      # Update task
      tasks = Enum.map(state.tasks, fn t ->
        if t.id == task.id, do: %{t | status: :in_progress, assigned_to: agent.name}, else: t
      end)
      
      # Add message
      message = %{
        from: "Coordinator",
        to: agent.name,
        content: "Assigned task: #{task.name}",
        timestamp: :os.system_time(:second)
      }
      
      %{state | agents: agents, tasks: tasks, messages: [message | state.messages]}
    else
      state
    end
  end
  
  defp maybe_update_agent_status(state) do
    agents = Enum.map(state.agents, fn agent ->
      if agent.status == :working and :rand.uniform() > 0.8 do
        new_status = Enum.random([:thinking, :communicating, :working])
        %{agent | status: new_status}
      else
        agent
      end
    end)
    
    %{state | agents: agents}
  end
  
  defp maybe_send_message(state) do
    working_agents = Enum.filter(state.agents, &(&1.status in [:working, :communicating]))
    
    if length(working_agents) > 1 and :rand.uniform() > 0.85 do
      sender = Enum.random(working_agents)
      receiver = Enum.random(state.agents -- [sender])
      
      messages = [
        "Need help with this implementation",
        "Found an interesting approach",
        "Can you review this?",
        "Making good progress",
        "Running into some issues",
        "Almost done with my part",
        "This looks promising",
        "Let's sync up on this"
      ]
      
      message = %{
        from: sender.name,
        to: receiver.name,
        content: Enum.random(messages),
        timestamp: :os.system_time(:second)
      }
      
      %{state | messages: [message | state.messages] |> Enum.take(20)}
    else
      state
    end
  end
  
  defp maybe_complete_task(state) do
    in_progress_tasks = Enum.filter(state.tasks, &(&1.status == :in_progress))
    
    if length(in_progress_tasks) > 0 and :rand.uniform() > 0.9 do
      task = Enum.random(in_progress_tasks)
      
      # Update task
      tasks = Enum.map(state.tasks, fn t ->
        if t.id == task.id, do: %{t | status: :completed}, else: t
      end)
      
      # Update agent
      agents = Enum.map(state.agents, fn a ->
        if a.name == task.assigned_to, do: %{a | status: :idle}, else: a
      end)
      
      # Add completion message
      message = %{
        from: task.assigned_to || "Unknown",
        to: "Coordinator",
        content: "Completed: #{task.name}",
        timestamp: :os.system_time(:second)
      }
      
      %{state | 
        agents: agents, 
        tasks: tasks, 
        messages: [message | state.messages],
        completed_tasks: state.completed_tasks + 1
      }
    else
      state
    end
  end
  
  defp maybe_add_agent(state) do
    if length(state.agents) < 8 and :rand.uniform() > 0.98 do
      types = [:researcher, :developer, :tester, :analyst]
      type = Enum.random(types)
      id = "#{String.upcase(to_string(type))}-#{:rand.uniform(999)}"
      
      new_agent = %{
        id: id,
        name: "#{type}_#{:rand.uniform(99)}",
        type: type,
        status: :idle,
        color: get_type_color(type)
      }
      
      message = %{
        from: "System",
        to: "All",
        content: "New agent joined: #{new_agent.name}",
        timestamp: :os.system_time(:second)
      }
      
      %{state | 
        agents: state.agents ++ [new_agent],
        messages: [message | state.messages]
      }
    else
      state
    end
  end
  
  defp maybe_add_task(state) do
    if length(state.tasks) < 10 and :rand.uniform() > 0.95 do
      task_names = [
        "Implement caching layer",
        "Optimize queries",
        "Add logging system",
        "Create documentation",
        "Setup monitoring",
        "Refactor module",
        "Add error handling",
        "Improve performance"
      ]
      
      new_task = %{
        id: :rand.uniform(9999),
        name: Enum.random(task_names),
        assigned_to: nil,
        status: :pending
      }
      
      %{state | tasks: state.tasks ++ [new_task]}
    else
      state
    end
  end
  
  defp get_agent_color(name) do
    cond do
      String.contains?(name, "Coord") -> &Colors.cyan/1
      String.contains?(name, "Research") -> &Colors.blue/1
      String.contains?(name, "Dev") -> &Colors.green/1
      String.contains?(name, "Test") -> &Colors.yellow/1
      String.contains?(name, "Arch") -> &Colors.magenta/1
      true -> &Colors.white/1
    end
  end
  
  defp get_type_color(type) do
    case type do
      :coordinator -> &Colors.cyan/1
      :researcher -> &Colors.blue/1
      :developer -> &Colors.green/1
      :tester -> &Colors.yellow/1
      :architect -> &Colors.magenta/1
      :analyst -> &Colors.red/1
      _ -> &Colors.white/1
    end
  end
  
  defp check_exit do
    # Simple check - in real app would handle signals properly
    false
  end
end

# Run the visualization
AgentViz.run()