#!/usr/bin/env elixir

# Agent Swarm CLI - Watch agents collaborate in real-time
# This version uses proper terminal handling and concurrent processes

defmodule AgentSwarmCLI do
  use GenServer
  
  # Terminal control sequences
  defmodule Term do
    def clear, do: IO.write("\e[2J\e[H")
    def move(row, col), do: IO.write("\e[#{row};#{col}H")
    def hide_cursor, do: IO.write("\e[?25l")
    def show_cursor, do: IO.write("\e[?25h")
    def save_cursor, do: IO.write("\e[s")
    def restore_cursor, do: IO.write("\e[u")
    
    # Colors
    def black(text), do: "\e[30m#{text}\e[0m"
    def red(text), do: "\e[31m#{text}\e[0m"
    def green(text), do: "\e[32m#{text}\e[0m"
    def yellow(text), do: "\e[33m#{text}\e[0m"
    def blue(text), do: "\e[34m#{text}\e[0m"
    def magenta(text), do: "\e[35m#{text}\e[0m"
    def cyan(text), do: "\e[36m#{text}\e[0m"
    def white(text), do: "\e[37m#{text}\e[0m"
    def bright_black(text), do: "\e[90m#{text}\e[0m"
    def bright_red(text), do: "\e[91m#{text}\e[0m"
    def bright_green(text), do: "\e[92m#{text}\e[0m"
    def bright_yellow(text), do: "\e[93m#{text}\e[0m"
    def bright_blue(text), do: "\e[94m#{text}\e[0m"
    def bright_magenta(text), do: "\e[95m#{text}\e[0m"
    def bright_cyan(text), do: "\e[96m#{text}\e[0m"
    def bright_white(text), do: "\e[97m#{text}\e[0m"
    
    def bold(text), do: "\e[1m#{text}\e[0m"
    def dim(text), do: "\e[2m#{text}\e[0m"
    def italic(text), do: "\e[3m#{text}\e[0m"
    def underline(text), do: "\e[4m#{text}\e[0m"
    def blink(text), do: "\e[5m#{text}\e[0m"
    def reverse(text), do: "\e[7m#{text}\e[0m"
  end
  
  # Agent process
  defmodule Agent do
    use GenServer
    
    defstruct [:id, :name, :type, :status, :skills, :memory, :current_task, :color_fn]
    
    # Agent types and their capabilities
    @agent_types %{
      orchestrator: %{
        skills: [:planning, :delegation, :monitoring, :coordination],
        color: &Term.bright_cyan/1,
        icon: "👑"
      },
      researcher: %{
        skills: [:web_search, :data_analysis, :summarization, :fact_checking],
        color: &Term.bright_blue/1,
        icon: "🔍"
      },
      engineer: %{
        skills: [:coding, :debugging, :optimization, :architecture],
        color: &Term.bright_green/1,
        icon: "⚡"
      },
      analyst: %{
        skills: [:data_processing, :visualization, :statistics, :reporting],
        color: &Term.bright_yellow/1,
        icon: "📊"
      },
      creative: %{
        skills: [:ideation, :design, :writing, :storytelling],
        color: &Term.bright_magenta/1,
        icon: "🎨"
      },
      specialist: %{
        skills: [:domain_expertise, :validation, :quality_assurance, :testing],
        color: &Term.bright_red/1,
        icon: "🎯"
      }
    }
    
    def start_link(type, name) do
      GenServer.start_link(__MODULE__, {type, name})
    end
    
    def init({type, name}) do
      agent_config = Map.get(@agent_types, type, @agent_types.specialist)
      
      state = %Agent{
        id: generate_id(),
        name: name,
        type: type,
        status: :initializing,
        skills: agent_config.skills,
        memory: [],
        current_task: nil,
        color_fn: agent_config.color
      }
      
      # Simulate initialization
      Process.send_after(self(), :initialized, :rand.uniform(2000))
      
      {:ok, state}
    end
    
    def get_state(agent_pid) do
      GenServer.call(agent_pid, :get_state)
    end
    
    def assign_task(agent_pid, task) do
      GenServer.cast(agent_pid, {:assign_task, task})
    end
    
    def send_message(agent_pid, from, message) do
      GenServer.cast(agent_pid, {:receive_message, from, message})
    end
    
    # Callbacks
    def handle_call(:get_state, _from, state) do
      {:reply, state, state}
    end
    
    def handle_cast({:assign_task, task}, state) do
      # Notify UI of status change
      notify_ui({:agent_update, self(), :working})
      
      # Simulate task processing
      Process.send_after(self(), :process_task, 1000 + :rand.uniform(3000))
      
      {:noreply, %{state | current_task: task, status: :working}}
    end
    
    def handle_cast({:receive_message, from, message}, state) do
      # Add to memory
      memory_entry = {DateTime.utc_now(), from, message}
      new_memory = [memory_entry | state.memory] |> Enum.take(10)
      
      # Notify UI of message
      notify_ui({:message, self(), from, message})
      
      # Sometimes respond
      if :rand.uniform() > 0.7 do
        Process.send_after(self(), {:respond_to, from}, 500 + :rand.uniform(1500))
      end
      
      {:noreply, %{state | memory: new_memory}}
    end
    
    def handle_info(:initialized, state) do
      notify_ui({:agent_update, self(), :idle})
      broadcast_presence(state)
      {:noreply, %{state | status: :idle}}
    end
    
    def handle_info(:process_task, state) do
      # Simulate different work phases
      case :rand.uniform(3) do
        1 ->
          # Need collaboration
          notify_ui({:agent_update, self(), :collaborating})
          request_collaboration(state)
          Process.send_after(self(), :complete_task, 2000)
          
        2 ->
          # Need research
          notify_ui({:agent_update, self(), :researching})
          Process.send_after(self(), :complete_task, 3000)
          
        _ ->
          # Can complete independently
          Process.send_after(self(), :complete_task, 1500)
      end
      
      {:noreply, state}
    end
    
    def handle_info(:complete_task, state) do
      if state.current_task do
        notify_ui({:task_completed, self(), state.current_task})
        notify_ui({:agent_update, self(), :idle})
        broadcast_achievement(state)
      end
      
      {:noreply, %{state | current_task: nil, status: :idle}}
    end
    
    def handle_info({:respond_to, target}, state) do
      responses = [
        "I can help with that!",
        "Interesting approach, let me analyze...",
        "Have you considered this alternative?",
        "Great progress! Here's my input...",
        "I'll run some tests on that.",
        "Let me check my knowledge base...",
        "Collaborating on this now!"
      ]
      
      response = Enum.random(responses)
      send(target, {:receive_message, self(), response})
      
      {:noreply, state}
    end
    
    # Private functions
    defp generate_id do
      :crypto.strong_rand_bytes(4) |> Base.encode16() |> String.downcase()
    end
    
    defp notify_ui(message) do
      case Process.whereis(:ui_server) do
        nil -> :ok
        pid -> send(pid, message)
      end
    end
    
    defp broadcast_presence(state) do
      notify_ui({:broadcast, self(), "#{state.name} is online and ready!"})
    end
    
    defp broadcast_achievement(state) do
      achievements = [
        "completed the analysis!",
        "found an optimization!",
        "resolved the issue!",
        "generated the solution!",
        "validated the approach!"
      ]
      
      message = "#{state.name} #{Enum.random(achievements)}"
      notify_ui({:broadcast, self(), message})
    end
    
    defp request_collaboration(state) do
      notify_ui({:broadcast, self(), "#{state.name} is seeking collaboration on: #{state.current_task.name}"})
    end
  end
  
  # Main UI Server
  def start_link do
    GenServer.start_link(__MODULE__, [], name: :ui_server)
  end
  
  def init(_) do
    # Set up terminal
    Term.hide_cursor()
    Term.clear()
    
    # Initial state
    state = %{
      agents: %{},
      messages: [],
      tasks: generate_initial_tasks(),
      stats: %{
        total_messages: 0,
        tasks_completed: 0,
        collaborations: 0
      },
      tick: 0,
      running: true
    }
    
    # Start the update loop
    schedule_update()
    
    # Spawn initial agents
    spawn_initial_agents()
    
    {:ok, state}
  end
  
  # Public API
  def add_agent(type, name) do
    GenServer.cast(:ui_server, {:add_agent, type, name})
  end
  
  def stop do
    GenServer.stop(:ui_server)
  end
  
  # Callbacks
  def handle_cast({:add_agent, type, name}, state) do
    {:ok, pid} = Agent.start_link(type, name)
    agent_info = %{
      pid: pid,
      type: type,
      name: name,
      status: :initializing,
      position: {10 + map_size(state.agents) * 15, 5 + rem(map_size(state.agents), 4) * 10}
    }
    
    new_agents = Map.put(state.agents, pid, agent_info)
    {:noreply, %{state | agents: new_agents}}
  end
  
  def handle_info(:update, state) do
    if state.running do
      # Update display
      draw_ui(state)
      
      # Schedule next update
      schedule_update()
      
      # Increment tick and potentially trigger events
      new_state = %{state | tick: state.tick + 1}
      new_state = maybe_trigger_event(new_state)
      
      {:noreply, new_state}
    else
      {:noreply, state}
    end
  end
  
  def handle_info({:agent_update, pid, status}, state) do
    new_agents = update_in(state.agents[pid], fn agent ->
      %{agent | status: status}
    end)
    
    {:noreply, %{state | agents: new_agents}}
  end
  
  def handle_info({:message, from, to, content}, state) do
    message = %{
      from: get_agent_name(state, from),
      to: get_agent_name(state, to),
      content: content,
      timestamp: DateTime.utc_now()
    }
    
    new_messages = [message | state.messages] |> Enum.take(20)
    new_stats = update_in(state.stats.total_messages, &(&1 + 1))
    
    {:noreply, %{state | messages: new_messages, stats: new_stats}}
  end
  
  def handle_info({:broadcast, from, message}, state) do
    broadcast = %{
      from: get_agent_name(state, from),
      to: "All Agents",
      content: message,
      timestamp: DateTime.utc_now()
    }
    
    new_messages = [broadcast | state.messages] |> Enum.take(20)
    
    {:noreply, %{state | messages: new_messages}}
  end
  
  def handle_info({:task_completed, agent_pid, task}, state) do
    new_stats = update_in(state.stats.tasks_completed, &(&1 + 1))
    
    # Remove completed task and maybe add new one
    new_tasks = List.delete(state.tasks, task)
    new_tasks = if length(new_tasks) < 5 do
      [generate_random_task() | new_tasks]
    else
      new_tasks
    end
    
    {:noreply, %{state | tasks: new_tasks, stats: new_stats}}
  end
  
  def terminate(_, _state) do
    Term.clear()
    Term.show_cursor()
    Term.move(1, 1)
    IO.puts("Thank you for using Agent Swarm CLI!")
    :ok
  end
  
  # Private functions
  defp schedule_update do
    Process.send_after(self(), :update, 100)  # 10 FPS
  end
  
  defp spawn_initial_agents do
    agents = [
      {:orchestrator, "MasterMind"},
      {:researcher, "Scholar"},
      {:engineer, "Builder"},
      {:analyst, "DataWiz"},
      {:creative, "Visionary"}
    ]
    
    for {type, name} <- agents do
      Process.sleep(200)
      add_agent(type, name)
    end
  end
  
  defp generate_initial_tasks do
    [
      %{id: 1, name: "Design scalable microservices architecture", priority: :high},
      %{id: 2, name: "Analyze user behavior patterns", priority: :medium},
      %{id: 3, name: "Optimize database performance", priority: :high},
      %{id: 4, name: "Create API documentation", priority: :low},
      %{id: 5, name: "Implement real-time notifications", priority: :medium}
    ]
  end
  
  defp generate_random_task do
    tasks = [
      "Implement authentication system",
      "Debug production issue",
      "Create data visualization",
      "Optimize algorithm performance",
      "Design user interface",
      "Conduct security audit",
      "Refactor legacy code",
      "Set up monitoring alerts",
      "Train machine learning model",
      "Write technical documentation"
    ]
    
    %{
      id: :rand.uniform(10000),
      name: Enum.random(tasks),
      priority: Enum.random([:low, :medium, :high])
    }
  end
  
  defp maybe_trigger_event(state) do
    cond do
      rem(state.tick, 50) == 0 and :rand.uniform() > 0.5 ->
        # Add new agent
        types = [:researcher, :engineer, :analyst, :creative, :specialist]
        type = Enum.random(types)
        name = generate_agent_name(type)
        add_agent(type, name)
        state
        
      rem(state.tick, 30) == 0 ->
        # Assign task to idle agent
        idle_agents = state.agents
        |> Enum.filter(fn {_pid, info} -> info.status == :idle end)
        |> Enum.map(fn {pid, _info} -> pid end)
        
        if length(idle_agents) > 0 and length(state.tasks) > 0 do
          agent_pid = Enum.random(idle_agents)
          task = hd(state.tasks)
          Agent.assign_task(agent_pid, task)
        end
        state
        
      true ->
        state
    end
  end
  
  defp generate_agent_name(type) do
    prefixes = %{
      researcher: ["Deep", "Smart", "Wise", "Expert"],
      engineer: ["Tech", "Code", "Build", "Dev"],
      analyst: ["Data", "Insight", "Logic", "Stats"],
      creative: ["Art", "Design", "Dream", "Spark"],
      specialist: ["Pro", "Master", "Elite", "Prime"]
    }
    
    suffixes = ["Bot", "AI", "Mind", "Core", "Agent", "System"]
    
    prefix = Enum.random(Map.get(prefixes, type, ["Generic"]))
    suffix = Enum.random(suffixes)
    "#{prefix}#{suffix}"
  end
  
  defp get_agent_name(state, pid) do
    case Map.get(state.agents, pid) do
      nil -> "Unknown"
      agent -> agent.name
    end
  end
  
  defp draw_ui(state) do
    # Header
    Term.move(1, 1)
    IO.write(Term.bold(Term.bright_cyan("╔═══════════════════════════════════════════════════════════════════════════════════════════════════════════════╗")))
    Term.move(2, 1)
    IO.write(Term.bold(Term.bright_cyan("║                                      AGENT SWARM COMMAND CENTER                                               ║")))
    Term.move(3, 1)
    IO.write(Term.bold(Term.bright_cyan("╚═══════════════════════════════════════════════════════════════════════════════════════════════════════════════╝")))
    
    # Draw agents
    draw_agents(state.agents, 5)
    
    # Draw message log
    draw_messages(state.messages, 5, 65)
    
    # Draw task queue
    draw_tasks(state.tasks, 20, 1)
    
    # Draw stats
    draw_stats(state.stats, 20, 35)
    
    # Status bar
    Term.move(30, 1)
    IO.write(Term.dim("Tick: #{state.tick} | Agents: #{map_size(state.agents)} | Press Ctrl+C to exit"))
  end
  
  defp draw_agents(agents, start_row) do
    for {pid, info} <- agents do
      agent_state = try do
        Agent.get_state(pid)
      catch
        :exit, _ -> %Agent{type: info.type, name: info.name, status: :offline}
      end
      
      {col, row} = info.position
      config = Map.get(Agent.agent_types(), agent_state.type, %{icon: "?", color: &Term.white/1})
      
      # Draw agent box
      Term.move(row, col)
      status_indicator = case agent_state.status do
        :idle -> Term.green("●")
        :working -> Term.yellow("◐")
        :collaborating -> Term.cyan("◑")
        :researching -> Term.blue("◒")
        :initializing -> Term.bright_black("○")
        _ -> Term.red("✗")
      end
      
      IO.write(config.color.(config.icon <> " " <> info.name))
      
      Term.move(row + 1, col)
      IO.write("#{status_indicator} #{agent_state.status}")
      
      if agent_state.current_task do
        Term.move(row + 2, col)
        task_text = agent_state.current_task.name
        |> String.slice(0, 20)
        |> Term.dim()
        IO.write(task_text)
      end
    end
  end
  
  defp draw_messages(messages, start_row, start_col) do
    Term.move(start_row - 1, start_col)
    IO.write(Term.bold("📨 Message Stream"))
    
    messages
    |> Enum.take(12)
    |> Enum.with_index()
    |> Enum.each(fn {msg, index} ->
      Term.move(start_row + index, start_col)
      
      time = Calendar.strftime(msg.timestamp, "%H:%M:%S")
      from_color = get_agent_color(msg.from)
      
      formatted = "#{Term.dim(time)} #{from_color.(msg.from)} → #{msg.to}: "
      IO.write(formatted)
      
      # Truncate message if needed
      max_length = 40
      content = if String.length(msg.content) > max_length do
        String.slice(msg.content, 0, max_length) <> "..."
      else
        msg.content
      end
      
      IO.write(Term.dim(content))
    end)
  end
  
  defp draw_tasks(tasks, start_row, start_col) do
    Term.move(start_row - 1, start_col)
    IO.write(Term.bold("📋 Task Queue"))
    
    tasks
    |> Enum.take(5)
    |> Enum.with_index()
    |> Enum.each(fn {task, index} ->
      Term.move(start_row + index, start_col)
      
      priority_color = case task.priority do
        :high -> &Term.red/1
        :medium -> &Term.yellow/1
        :low -> &Term.green/1
      end
      
      IO.write(priority_color.("●") <> " " <> String.slice(task.name, 0, 30))
    end)
  end
  
  defp draw_stats(stats, start_row, start_col) do
    Term.move(start_row - 1, start_col)
    IO.write(Term.bold("📊 Statistics"))
    
    Term.move(start_row, start_col)
    IO.write("Messages: #{Term.bright_cyan(to_string(stats.total_messages))}")
    
    Term.move(start_row + 1, start_col)
    IO.write("Tasks Done: #{Term.bright_green(to_string(stats.tasks_completed))}")
    
    Term.move(start_row + 2, start_col)
    IO.write("Collaborations: #{Term.bright_yellow(to_string(stats.collaborations))}")
  end
  
  defp get_agent_color(name) do
    cond do
      String.contains?(name, "Master") -> &Term.bright_cyan/1
      String.contains?(name, "Scholar") or String.contains?(name, "Research") -> &Term.bright_blue/1
      String.contains?(name, "Build") or String.contains?(name, "Dev") -> &Term.bright_green/1
      String.contains?(name, "Data") or String.contains?(name, "Analyst") -> &Term.bright_yellow/1
      String.contains?(name, "Vision") or String.contains?(name, "Creative") -> &Term.bright_magenta/1
      true -> &Term.white/1
    end
  end
end

# Main entry point
defmodule CLI do
  def run do
    IO.puts(AgentSwarmCLI.Term.bold(AgentSwarmCLI.Term.bright_cyan("""
    
     █████╗  ██████╗ ███████╗███╗   ██╗████████╗    ███████╗██╗    ██╗ █████╗ ██████╗ ███╗   ███╗
    ██╔══██╗██╔════╝ ██╔════╝████╗  ██║╚══██╔══╝    ██╔════╝██║    ██║██╔══██╗██╔══██╗████╗ ████║
    ███████║██║  ███╗█████╗  ██╔██╗ ██║   ██║       ███████╗██║ █╗ ██║███████║██████╔╝██╔████╔██║
    ██╔══██║██║   ██║██╔══╝  ██║╚██╗██║   ██║       ╚════██║██║███╗██║██╔══██║██╔══██╗██║╚██╔╝██║
    ██║  ██║╚██████╔╝███████╗██║ ╚████║   ██║       ███████║╚███╔███╔╝██║  ██║██║  ██║██║ ╚═╝ ██║
    ╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚═══╝   ╚═╝       ╚══════╝ ╚══╝╚══╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝     ╚═╝
    
    """)))
    
    IO.puts("Initializing agent swarm...")
    Process.sleep(1000)
    
    # Start the UI server
    {:ok, _pid} = AgentSwarmCLI.start_link()
    
    # Keep the main process alive
    Process.sleep(:infinity)
  end
end

# Handle termination gracefully
Process.flag(:trap_exit, true)

# Run the CLI
CLI.run()