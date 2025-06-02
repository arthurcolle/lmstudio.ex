#!/usr/bin/env elixir
# Multi-Agent System Interactive CLI with Fixed Panes
# Shows agents being created, communicating, and collaborating in real-time
# Features fixed window panes and interactive communication

defmodule MultiAgentCLI do
  @moduledoc """
  Interactive CLI for visualizing multi-agent system interactions
  with fixed window panes and real-time updates
  """
  
  # Terminal control sequences and window management
  defmodule Terminal do
    def clear(), do: IO.puts(IO.ANSI.clear())
    def home(), do: IO.puts(IO.ANSI.home())
    def hide_cursor(), do: IO.write("\e[?25l")
    def show_cursor(), do: IO.write("\e[?25h")
    def move_cursor(row, col), do: IO.write("\e[#{row};#{col}H")
    def save_cursor(), do: IO.write("\e[s")
    def restore_cursor(), do: IO.write("\e[u")
    
    def get_terminal_size() do
      case System.cmd("tput", ["cols"]) do
        {cols, 0} -> 
          case System.cmd("tput", ["lines"]) do
            {lines, 0} -> 
              {String.trim(cols) |> String.to_integer(), 
               String.trim(lines) |> String.to_integer()}
            _ -> {80, 24}
          end
        _ -> {80, 24}
      end
    end
  end
  
  # Color helpers
  def color(color, text) do
    apply(IO.ANSI, color, []) <> text <> IO.ANSI.reset()
  end
  
  def color(style, color, text) do
    apply(IO.ANSI, style, []) <> apply(IO.ANSI, color, []) <> text <> IO.ANSI.reset()
  end
  
  def start() do
    Terminal.clear()
    Terminal.hide_cursor()
    
    # Get terminal size for layout
    {width, height} = Terminal.get_terminal_size()
    
    # Initialize layout dimensions
    layout = %{
      width: width,
      height: height,
      agent_pane: {1, 1, div(width, 2) - 1, div(height, 2) - 1},
      message_pane: {div(width, 2) + 1, 1, width - 1, div(height, 2) - 1},
      stats_pane: {1, div(height, 2) + 1, div(width, 2) - 1, height - 3},
      task_pane: {div(width, 2) + 1, div(height, 2) + 1, width - 1, height - 3},
      input_pane: {1, height - 1, width - 1, height}
    }
    
    # Initialize agents with proper state
    agents = init_agents()
    
    # Draw initial layout
    draw_layout(layout)
    
    # Start simulation
    try do
      run_simulation(agents, 0, layout, [])
    catch
      :exit, _ -> cleanup_and_exit()
    end
  rescue
    _ -> 
      Terminal.show_cursor()
      Terminal.clear()
  end
  
  defp init_agents() do
    [
      %{id: 1, name: "Orchestrator", type: :orchestrator, status: :idle, 
        task: nil, color: :cyan, messages_sent: 0, messages_received: 0,
        position: {5, 3}},
      %{id: 2, name: "Researcher", type: :researcher, status: :idle, 
        task: nil, color: :blue, messages_sent: 0, messages_received: 0,
        position: {20, 3}},
      %{id: 3, name: "Developer", type: :developer, status: :idle, 
        task: nil, color: :green, messages_sent: 0, messages_received: 0,
        position: {5, 8}},
      %{id: 4, name: "Analyst", type: :analyst, status: :idle, 
        task: nil, color: :yellow, messages_sent: 0, messages_received: 0,
        position: {20, 8}},
      %{id: 5, name: "Tester", type: :tester, status: :idle, 
        task: nil, color: :magenta, messages_sent: 0, messages_received: 0,
        position: {12, 5}}
    ]
  end
  
  defp run_simulation(agents, tick, layout, messages) do
    # Update display with fixed panes
    draw_agent_pane(agents, layout.agent_pane)
    draw_message_pane(messages, layout.message_pane)
    draw_stats_pane(agents, tick, layout.stats_pane)
    draw_task_pane(agents, layout.task_pane)
    draw_input_prompt(layout.input_pane)
    
    # Simulate agent interactions
    {updated_agents, action_messages} = simulate_tick(agents, tick)
    all_messages = action_messages ++ messages |> Enum.take(50)
    
    # Continue or stop
    Process.sleep(100)
    
    # Check for SIGINT
    receive do
      :interrupt -> cleanup_and_exit()
    after
      0 ->
        if tick < 300 do  # Run for 30 seconds
          run_simulation(updated_agents, tick + 1, layout, all_messages)
        else
          cleanup_and_exit()
        end
    end
  end
  
  defp draw_layout(layout) do
    # Draw all pane borders
    draw_box("Agents", layout.agent_pane)
    draw_box("Messages", layout.message_pane)
    draw_box("Statistics", layout.stats_pane)
    draw_box("Task Queue", layout.task_pane)
  end
  
  defp draw_box(title, {x1, y1, x2, y2}) do
    width = x2 - x1 + 1
    _height = y2 - y1 + 1
    
    # Top border
    Terminal.move_cursor(y1, x1)
    IO.write("┌─ #{title} " <> String.duplicate("─", width - String.length(title) - 5) <> "┐")
    
    # Side borders
    for y <- (y1 + 1)..(y2 - 1) do
      Terminal.move_cursor(y, x1)
      IO.write("│")
      Terminal.move_cursor(y, x2)
      IO.write("│")
    end
    
    # Bottom border
    Terminal.move_cursor(y2, x1)
    IO.write("└" <> String.duplicate("─", width - 2) <> "┘")
  end
  
  defp draw_agent_pane(agents, {x1, y1, x2, y2}) do
    # Draw agents in their positions
    agents
    |> Enum.each(fn agent ->
      {ax, ay} = agent.position
      if ax >= x1 + 2 and ax <= x2 - 10 and ay >= y1 + 1 and ay <= y2 - 1 do
        Terminal.move_cursor(ay, ax)
        
        status_icon = case agent.status do
          :idle -> "●"
          :working -> "◐"
          :communicating -> "◑"
          :thinking -> "◒"
        end
        
        IO.write(color(agent.color, "#{status_icon} #{agent.name}"))
      end
    end)
  end
  
  defp draw_message_pane(messages, {x1, y1, x2, y2}) do
    # Clear message area
    for y <- (y1 + 1)..(y2 - 1) do
      Terminal.move_cursor(y, x1 + 1)
      IO.write(String.duplicate(" ", x2 - x1 - 1))
    end
    
    # Display recent messages
    messages
    |> Enum.take(y2 - y1 - 2)
    |> Enum.with_index()
    |> Enum.each(fn {msg, idx} ->
      Terminal.move_cursor(y1 + 1 + idx, x1 + 2)
      truncated = String.slice(msg, 0, x2 - x1 - 4)
      IO.write(truncated)
    end)
  end
  
  defp draw_stats_pane(agents, tick, {x1, y1, _x2, _y2}) do
    active_count = Enum.count(agents, fn a -> a.status != :idle end)
    total_messages = agents |> Enum.map(& &1.messages_sent) |> Enum.sum()
    
    stats = [
      "Time: #{div(tick, 10)}s",
      "Active: #{active_count}/#{length(agents)}",
      "Messages: #{total_messages}",
      "Tick: #{tick}"
    ]
    
    stats
    |> Enum.with_index()
    |> Enum.each(fn {stat, idx} ->
      Terminal.move_cursor(y1 + 1 + idx, x1 + 2)
      IO.write(stat)
    end)
  end
  
  defp draw_task_pane(agents, {x1, y1, x2, y2}) do
    tasks = agents
    |> Enum.filter(& &1.task)
    |> Enum.map(fn a -> "#{a.name}: #{a.task}" end)
    |> Enum.take(y2 - y1 - 2)
    
    # Clear area first
    for y <- (y1 + 1)..(y2 - 1) do
      Terminal.move_cursor(y, x1 + 1)
      IO.write(String.duplicate(" ", x2 - x1 - 1))
    end
    
    tasks
    |> Enum.with_index()
    |> Enum.each(fn {task, idx} ->
      Terminal.move_cursor(y1 + 1 + idx, x1 + 2)
      truncated = String.slice(task, 0, x2 - x1 - 4)
      IO.write(truncated)
    end)
  end
  
  defp draw_input_prompt({x1, y1, _x2, _y2}) do
    Terminal.move_cursor(y1, x1)
    IO.write("[Running simulation... Ctrl+C to quit]")
  end
  
  defp simulate_tick(agents, _tick) do
    _messages = []
    
    updated_agents = agents
    |> Enum.map(fn agent ->
      case :rand.uniform(20) do
        1 -> 
          # Start working
          task = random_task()
          new_messages = ["#{agent.name} started: #{task}"]
          {%{agent | status: :working, task: task}, new_messages}
          
        2 -> 
          # Start thinking
          {%{agent | status: :thinking}, ["#{agent.name} is analyzing..."]}
          
        3 -> 
          # Communicate with another agent
          target = Enum.random(agents)
          if target.id != agent.id do
            msg = "#{agent.name} → #{target.name}: #{random_message()}"
            updated_agent = %{agent | 
              status: :communicating, 
              messages_sent: agent.messages_sent + 1
            }
            {updated_agent, [msg]}
          else
            {agent, []}
          end
          
        4..5 when agent.status == :working -> 
          # Complete task
          msg = "✓ #{agent.name} completed: #{agent.task}"
          {%{agent | status: :idle, task: nil}, [msg]}
          
        _ -> 
          {agent, []}
      end
    end)
    |> Enum.reduce({[], []}, fn
      {agent, new_msgs}, {agents_acc, msgs_acc} ->
        {[agent | agents_acc], new_msgs ++ msgs_acc}
      agent, {agents_acc, msgs_acc} when is_map(agent) ->
        {[agent | agents_acc], msgs_acc}
    end)
    
    {Enum.reverse(elem(updated_agents, 0)), elem(updated_agents, 1)}
  end
  
  defp random_task() do
    tasks = [
      "Analyzing system performance",
      "Optimizing database queries",
      "Implementing new feature",
      "Reviewing code changes",
      "Running test suite",
      "Deploying to staging",
      "Monitoring system health",
      "Refactoring legacy code",
      "Writing documentation",
      "Debugging issue ##{:rand.uniform(100)}"
    ]
    Enum.random(tasks)
  end
  
  defp random_message() do
    messages = [
      "Need help with this task",
      "Can you review my changes?",
      "Found an optimization opportunity",
      "Tests are passing",
      "Deployment ready",
      "Performance improved by 20%",
      "Bug fixed in module X",
      "Documentation updated",
      "Ready for code review",
      "System metrics look good"
    ]
    Enum.random(messages)
  end
  
  defp cleanup_and_exit() do
    Terminal.show_cursor()
    Terminal.clear()
    IO.puts("\n✅ Multi-Agent System terminated gracefully")
  end
end

# Run the CLI
MultiAgentCLI.start()