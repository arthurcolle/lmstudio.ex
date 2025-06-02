#!/usr/bin/env elixir

# Ensure LMStudio modules are loaded (they should be compiled in _build)
unless Code.ensure_loaded?(LMStudio) do
  # If not compiled, load the source files
  Code.require_file("lib/lmstudio.ex", __DIR__)
  Code.require_file("lib/lmstudio/config.ex", __DIR__)
  Code.require_file("lib/lmstudio/legitimate_mas.ex", __DIR__)
end

# Multi-Agent System Real-time Visualization
# Interactive visualization showing agents, messages, and collaboration

defmodule ACL do
  defstruct [
    :performative,
    :sender,
    :receiver,
    :reply_to,
    :content,
    :language,
    :encoding,
    :ontology,
    :protocol,
    :conversation_id,
    :reply_with,
    :in_reply_to,
    :reply_by
  ]
end

defmodule MASVisualization do
  use GenServer
  
  alias LMStudio.LegitimateMAS
  
  # Terminal control module
  defmodule Term do
    def clear, do: IO.write("\e[2J\e[H")
    def move(row, col), do: IO.write("\e[#{row};#{col}H")
    def hide_cursor, do: IO.write("\e[?25l")
    def show_cursor, do: IO.write("\e[?25h")
    
    # Colors
    def color(text, :black), do: "\e[30m#{text}\e[0m"
    def color(text, :red), do: "\e[31m#{text}\e[0m"
    def color(text, :green), do: "\e[32m#{text}\e[0m"
    def color(text, :yellow), do: "\e[33m#{text}\e[0m"
    def color(text, :blue), do: "\e[34m#{text}\e[0m"
    def color(text, :magenta), do: "\e[35m#{text}\e[0m"
    def color(text, :cyan), do: "\e[36m#{text}\e[0m"
    def color(text, :white), do: "\e[37m#{text}\e[0m"
    def color(text, :bright_red), do: "\e[91m#{text}\e[0m"
    def color(text, :bright_green), do: "\e[92m#{text}\e[0m"
    def color(text, :bright_yellow), do: "\e[93m#{text}\e[0m"
    def color(text, :bright_blue), do: "\e[94m#{text}\e[0m"
    def color(text, :bright_magenta), do: "\e[95m#{text}\e[0m"
    def color(text, :bright_cyan), do: "\e[96m#{text}\e[0m"
    
    def bold(text), do: "\e[1m#{text}\e[0m"
    def dim(text), do: "\e[2m#{text}\e[0m"
    def underline(text), do: "\e[4m#{text}\e[0m"
    def blink(text), do: "\e[5m#{text}\e[0m"
  end
  
  # Visualization state
  defstruct [
    :mas_pid,
    :agents,
    :messages,
    :blackboard_entries,
    :contracts,
    :metrics,
    :layout,
    :tick
  ]
  
  def start_link(mas_pid) do
    GenServer.start_link(__MODULE__, mas_pid, name: __MODULE__)
  end
  
  def init(mas_pid) do
    # Setup terminal
    Term.hide_cursor()
    Term.clear()
    
    # Create layout
    layout = create_layout()
    
    state = %__MODULE__{
      mas_pid: mas_pid,
      agents: %{},
      messages: [],
      blackboard_entries: [],
      contracts: %{},
      metrics: %{
        total_messages: 0,
        beliefs_updated: 0,
        goals_achieved: 0,
        contracts_completed: 0
      },
      layout: layout,
      tick: 0
    }
    
    # Start visualization loop
    schedule_update()
    
    # Subscribe to MAS events
    subscribe_to_events(mas_pid)
    
    {:ok, state}
  end
  
  # Public API to create agents
  def create_demo_agents(mas_pid) do
    Process.send(self(), {:create_demo_agents, mas_pid}, [])
  end
  
  # Event handlers
  def handle_info(:update, state) do
    # Update display
    render(state)
    
    # Schedule next update
    schedule_update()
    
    # Increment tick
    new_state = %{state | tick: state.tick + 1}
    
    # Trigger demo events
    new_state = maybe_trigger_demo_event(new_state)
    
    {:noreply, new_state}
  end
  
  def handle_info({:create_demo_agents, mas_pid}, state) do
    spawn(fn ->
      # Create director agent
      GenServer.call(mas_pid, {:create_agent, :director, [
        role: :director,
        capabilities: [:strategic_planning, :resource_allocation, :oversight],
        initial_goals: [
          %{id: "dir_goal_1", type: :strategic_planning, priority: :high}
        ]
      ]})
      
      # Create manager agents
      for i <- 1..2 do
        GenServer.call(mas_pid, {:create_agent, :manager, [
          role: :manager,
          capabilities: [:task_management, :team_coordination, :reporting],
          initial_goals: [
            %{id: "man_goal_#{i}", type: :task_management, priority: :medium}
          ]
        ]})
      end
      
      # Create worker agents
      for i <- 1..3 do
        GenServer.call(mas_pid, {:create_agent, :worker, [
          role: :worker,
          capabilities: [:execute_tasks, :task_execution, :reporting],
          initial_goals: [
            %{id: "work_goal_#{i}", type: :task_execution, priority: :medium}
          ]
        ]})
      end
    end)
    
    {:noreply, state}
  end
  
  def handle_info({:agent_created, agent_info}, state) do
    new_agents = Map.put(state.agents, agent_info.id, enrich_agent_info(agent_info))
    {:noreply, %{state | agents: new_agents}}
  end
  
  def handle_info({:message_sent, %ACL{} = message}, state) do
    new_messages = [format_message(message) | state.messages] |> Enum.take(20)
    new_metrics = update_in(state.metrics.total_messages, &(&1 + 1))
    
    {:noreply, %{state | messages: new_messages, metrics: new_metrics}}
  end
  
  def handle_info({:blackboard_update, category, key, value}, state) do
    entry = %{
      category: category,
      key: key,
      value: summarize_value(value),
      timestamp: DateTime.utc_now()
    }
    
    new_entries = [entry | state.blackboard_entries] |> Enum.take(10)
    {:noreply, %{state | blackboard_entries: new_entries}}
  end
  
  def handle_info({:belief_updated, agent_id, _belief_type}, state) do
    new_metrics = update_in(state.metrics.beliefs_updated, &(&1 + 1))
    new_state = %{state | metrics: new_metrics}
    
    # Update agent status
    if agent = Map.get(state.agents, agent_id) do
      updated_agent = %{agent | last_activity: DateTime.utc_now()}
      new_agents = Map.put(state.agents, agent_id, updated_agent)
      {:noreply, %{new_state | agents: new_agents}}
    else
      {:noreply, new_state}
    end
  end
  
  def handle_info({:contract_update, contract_id, status}, state) do
    contract = %{
      id: contract_id,
      status: status,
      timestamp: DateTime.utc_now()
    }
    
    new_contracts = Map.put(state.contracts, contract_id, contract)
    new_metrics = if status == :completed do
      update_in(state.metrics.contracts_completed, &(&1 + 1))
    else
      state.metrics
    end
    
    {:noreply, %{state | contracts: new_contracts, metrics: new_metrics}}
  end
  
  def handle_info(_msg, state) do
    {:noreply, state}
  end
  
  # Private helpers
  defp schedule_update do
    Process.send_after(self(), :update, 100)
  end
  
  defp subscribe_to_events(_mas_pid) do
    # Subscribe to actual MAS events by monitoring the GenServer
    # For now, we'll start simulating events after a delay
    Process.send_after(self(), :start_simulation, 1000)
    :ok
  end
  
  defp create_layout do
    %{
      header: {1, 1, 120, 3},
      agents_panel: {5, 1, 55, 20},
      messages_panel: {5, 60, 60, 12},
      blackboard_panel: {18, 60, 60, 8},
      metrics_panel: {26, 1, 55, 6},
      contracts_panel: {26, 60, 60, 6}
    }
  end
  
  defp render(state) do
    render_header(state)
    render_agents_panel(state)
    render_messages_panel(state)
    render_blackboard_panel(state)
    render_metrics_panel(state)
    render_contracts_panel(state)
  end
  
  defp render_header(state) do
    {row, col, _width, _height} = state.layout.header
    Term.move(row, col)
    IO.write(Term.bold(Term.color("🌟 MULTI-AGENT SYSTEM VISUALIZATION", :bright_cyan)))
    Term.move(row + 1, col)
    IO.write(Term.dim("━" |> String.duplicate(120)))
    Term.move(row + 2, col)
    IO.write(Term.color("Tick: #{state.tick} | Agents: #{map_size(state.agents)} | Messages: #{state.metrics.total_messages}", :white))
  end
  
  defp render_agents_panel(state) do
    {row, col, _width, height} = state.layout.agents_panel
    
    # Panel header
    Term.move(row, col)
    IO.write(Term.bold(Term.color("📋 AGENTS", :bright_yellow)))
    Term.move(row + 1, col)
    IO.write(Term.dim("─" |> String.duplicate(50)))
    
    # Agent list
    agents = state.agents |> Map.values() |> Enum.sort_by(& &1.type)
    
    agents
    |> Enum.take(height - 3)
    |> Enum.with_index()
    |> Enum.each(fn {agent, idx} ->
      Term.move(row + 2 + idx, col)
      render_agent(agent)
    end)
  end
  
  defp render_agent(agent) do
    icon = case agent.type do
      :director -> "👔"
      :manager -> "📊"
      :worker -> "⚙️"
      _ -> "🤖"
    end
    
    color = case agent.type do
      :director -> :bright_magenta
      :manager -> :bright_blue
      :worker -> :bright_green
      _ -> :white
    end
    
    status = if agent[:last_activity] && 
              DateTime.diff(DateTime.utc_now(), agent.last_activity) < 5 do
      Term.color("●", :bright_green)
    else
      Term.color("●", :dim)
    end
    
    IO.write("#{icon} #{status} #{Term.color(agent.name || agent.id, color)}")
    if agent[:role], do: IO.write(" [#{agent.role}]")
  end
  
  defp render_messages_panel(state) do
    {row, col, _width, height} = state.layout.messages_panel
    
    # Panel header
    Term.move(row, col)
    IO.write(Term.bold(Term.color("💬 MESSAGES (FIPA ACL)", :bright_cyan)))
    Term.move(row + 1, col)
    IO.write(Term.dim("─" |> String.duplicate(55)))
    
    # Messages
    state.messages
    |> Enum.take(height - 3)
    |> Enum.with_index()
    |> Enum.each(fn {msg, idx} ->
      Term.move(row + 2 + idx, col)
      IO.write(msg)
    end)
  end
  
  defp render_blackboard_panel(state) do
    {row, col, _width, height} = state.layout.blackboard_panel
    
    # Panel header
    Term.move(row, col)
    IO.write(Term.bold(Term.color("📝 BLACKBOARD", :bright_green)))
    Term.move(row + 1, col)
    IO.write(Term.dim("─" |> String.duplicate(55)))
    
    # Entries
    state.blackboard_entries
    |> Enum.take(height - 3)
    |> Enum.with_index()
    |> Enum.each(fn {entry, idx} ->
      Term.move(row + 2 + idx, col)
      IO.write("#{Term.color(entry.category, :yellow)}/#{entry.key}: #{entry.value}")
    end)
  end
  
  defp render_metrics_panel(state) do
    {row, col, _width, _height} = state.layout.metrics_panel
    
    # Panel header
    Term.move(row, col)
    IO.write(Term.bold(Term.color("📊 METRICS", :bright_magenta)))
    Term.move(row + 1, col)
    IO.write(Term.dim("─" |> String.duplicate(50)))
    
    # Metrics
    Term.move(row + 2, col)
    IO.write("Messages: #{state.metrics.total_messages} | ")
    IO.write("Beliefs Updated: #{state.metrics.beliefs_updated} | ")
    IO.write("Goals: #{state.metrics.goals_achieved}")
    
    Term.move(row + 3, col)
    IO.write("Contracts: #{map_size(state.contracts)} | ")
    IO.write("Completed: #{state.metrics.contracts_completed}")
  end
  
  defp render_contracts_panel(state) do
    {row, col, _width, height} = state.layout.contracts_panel
    
    # Panel header
    Term.move(row, col)
    IO.write(Term.bold(Term.color("📜 CONTRACT NET", :bright_yellow)))
    Term.move(row + 1, col)
    IO.write(Term.dim("─" |> String.duplicate(55)))
    
    # Contracts
    state.contracts
    |> Map.values()
    |> Enum.sort_by(& &1.timestamp, {:desc, DateTime})
    |> Enum.take(height - 3)
    |> Enum.with_index()
    |> Enum.each(fn {contract, idx} ->
      Term.move(row + 2 + idx, col)
      status_color = case contract.status do
        :collecting_proposals -> :yellow
        :evaluating -> :cyan
        :awarded -> :blue
        :completed -> :green
        _ -> :white
      end
      IO.write("Contract #{contract.id}: #{Term.color(contract.status, status_color)}")
    end)
  end
  
  defp enrich_agent_info(agent_info) do
    Map.merge(agent_info, %{
      last_activity: DateTime.utc_now(),
      status: :active
    })
  end
  
  defp format_message(%ACL{} = msg) do
    perf_color = case msg.performative do
      :inform -> :blue
      :request -> :yellow
      :cfp -> :magenta
      :propose -> :green
      :accept_proposal -> :bright_green
      :reject_proposal -> :red
      _ -> :white
    end
    
    "#{Term.color(msg.performative, perf_color)} #{msg.sender}→#{msg.receiver}"
  end
  
  defp summarize_value(value) when is_binary(value), do: value
  defp summarize_value(value), do: inspect(value, limit: 20)
  
  defp maybe_trigger_demo_event(state) do
    cond do
      rem(state.tick, 30) == 0 ->
        # Simulate message
        simulate_message()
        state
        
      rem(state.tick, 50) == 0 ->
        # Simulate blackboard update
        simulate_blackboard_update()
        state
        
      rem(state.tick, 100) == 0 ->
        # Simulate contract
        simulate_contract()
        state
        
      rem(state.tick, 20) == 0 ->
        # Simulate belief update
        simulate_belief_update(state)
        state
        
      true ->
        state
    end
  end
  
  defp simulate_message do
    messages = [
      %ACL{
        performative: :inform,
        sender: "manager_001",
        receiver: "worker_001",
        content: %{task_status: :in_progress}
      },
      %ACL{
        performative: :request,
        sender: "director_001",
        receiver: "manager_001",
        content: %{action: :provide_status_report}
      },
      %ACL{
        performative: :cfp,
        sender: "manager_002",
        receiver: "all_workers",
        content: %{task: "implement_feature_x", deadline: "2024-01-15"}
      },
      %ACL{
        performative: :propose,
        sender: "worker_002",
        receiver: "manager_002",
        content: %{can_complete_by: "2024-01-14", resources_needed: [:time, :api_access]}
      }
    ]
    
    msg = Enum.random(messages)
    Process.send(self(), {:message_sent, msg}, [])
  end
  
  defp simulate_blackboard_update do
    updates = [
      {:problems, :resource_conflict, "Database connection pool exhausted"},
      {:partial_solutions, :cache_implementation, "Redis cache layer added"},
      {:constraints, :performance_requirement, "Response time < 200ms"},
      {:hypotheses, :scaling_strategy, "Horizontal scaling with load balancer"}
    ]
    
    {category, key, value} = Enum.random(updates)
    Process.send(self(), {:blackboard_update, category, key, value}, [])
  end
  
  defp simulate_contract do
    contract_id = :crypto.strong_rand_bytes(4) |> Base.encode16()
    status = Enum.random([:collecting_proposals, :evaluating, :awarded, :completed])
    
    Process.send(self(), {:contract_update, contract_id, status}, [])
  end
  
  defp simulate_belief_update(state) do
    if map_size(state.agents) > 0 do
      agent_id = state.agents |> Map.keys() |> Enum.random()
      Process.send(self(), {:belief_updated, agent_id, :environment}, [])
    end
  end
end

defmodule VisualizationDemo do
  def run do
    IO.puts(MASVisualization.Term.bold(MASVisualization.Term.color("""
    
    🌟 MULTI-AGENT SYSTEM VISUALIZATION
    ==================================
    
    Watch a legitimate MAS in action with:
    • BDI agents making decisions
    • FIPA ACL message exchanges  
    • Blackboard collaboration
    • Contract Net Protocol
    
    Starting visualization...
    """, :bright_cyan)))
    
    Process.sleep(2000)
    
    # Start MAS first
    {:ok, mas_pid} = LMStudio.LegitimateMAS.start_link()
    
    # Start visualization with MAS pid
    {:ok, _vis_pid} = MASVisualization.start_link(mas_pid)
    
    # Create demo agents after a delay to ensure everything is running
    Process.sleep(500)
    MASVisualization.create_demo_agents(mas_pid)
    
    # Keep running
    Process.sleep(:infinity)
  end
end

# Handle termination
Process.flag(:trap_exit, true)

# Run the visualization
VisualizationDemo.run()