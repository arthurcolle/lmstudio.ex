#!/usr/bin/env elixir

# Multi-Agent System Real-time Visualization (Standalone)
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
    :agents,
    :messages,
    :blackboard_entries,
    :contracts,
    :metrics,
    :layout,
    :tick
  ]
  
  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end
  
  def init(_opts) do
    # Setup terminal
    Term.hide_cursor()
    Term.clear()
    
    # Create layout
    layout = create_layout()
    
    state = %__MODULE__{
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
    
    # Spawn initial agents
    spawn_demo_agents()
    
    {:ok, state}
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
    
    # Update agent visualization
    new_agents = if Map.has_key?(state.agents, agent_id) do
      update_in(state.agents[agent_id], fn agent ->
        Map.merge(agent, %{last_belief_update: DateTime.utc_now(), thinking: true})
      end)
    else
      state.agents
    end
    
    # Clear thinking indicator after delay
    Process.send_after(self(), {:clear_thinking, agent_id}, 1000)
    
    {:noreply, %{state | agents: new_agents, metrics: new_metrics}}
  end
  
  def handle_info({:clear_thinking, agent_id}, state) do
    new_agents = if Map.has_key?(state.agents, agent_id) do
      update_in(state.agents[agent_id], fn agent ->
        Map.put(agent, :thinking, false)
      end)
    else
      state.agents
    end
    {:noreply, %{state | agents: new_agents}}
  end
  
  def handle_info({:contract_update, contract_id, status}, state) do
    new_contracts = Map.put(state.contracts, contract_id, status)
    
    new_metrics = if status == :completed do
      update_in(state.metrics.contracts_completed, &(&1 + 1))
    else
      state.metrics
    end
    
    {:noreply, %{state | contracts: new_contracts, metrics: new_metrics}}
  end
  
  def terminate(_reason, _state) do
    Term.clear()
    Term.show_cursor()
    Term.move(1, 1)
    :ok
  end
  
  # Private functions
  defp schedule_update do
    Process.send_after(self(), :update, 100)  # 10 FPS
  end
  
  defp spawn_demo_agents do
    agents = [
      %{id: "director_001", type: :director, name: "Strategic Director", role: :director},
      %{id: "manager_001", type: :manager, name: "Project Manager", role: :manager},
      %{id: "manager_002", type: :manager, name: "Tech Lead", role: :manager},
      %{id: "worker_001", type: :worker, name: "Backend Dev", role: :worker},
      %{id: "worker_002", type: :worker, name: "Frontend Dev", role: :worker},
      %{id: "worker_003", type: :worker, name: "QA Engineer", role: :worker}
    ]
    
    for agent <- agents do
      Process.send_after(self(), {:agent_created, agent}, :rand.uniform(2000))
    end
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
    {row, col, width, _height} = state.layout.header
    
    Term.move(row, col)
    IO.write(Term.bold(Term.color("╔" <> String.duplicate("═", width - 2) <> "╗", :bright_cyan)))
    
    Term.move(row + 1, col)
    title = "MULTI-AGENT SYSTEM - REAL-TIME VISUALIZATION"
    padding = div(width - String.length(title) - 2, 2)
    IO.write(Term.bold(Term.color("║" <> String.pad_leading("", padding) <> title <> String.pad_trailing("", width - String.length(title) - padding - 2) <> "║", :bright_cyan)))
    
    Term.move(row + 2, col)
    IO.write(Term.bold(Term.color("╚" <> String.duplicate("═", width - 2) <> "╝", :bright_cyan)))
  end
  
  defp render_agents_panel(state) do
    {row, col, _width, height} = state.layout.agents_panel
    
    # Panel header
    Term.move(row, col)
    IO.write(Term.bold("🤖 AGENTS (BDI Architecture)"))
    
    # Draw agents
    state.agents
    |> Map.values()
    |> Enum.with_index()
    |> Enum.each(fn {agent, idx} ->
      if idx < height - 2 do
        render_agent(agent, row + idx + 2, col)
      end
    end)
  end
  
  defp render_agent(agent, row, col) do
    Term.move(row, col)
    
    # Agent icon and name
    icon = case agent.type do
      :director -> "👑"
      :manager -> "👔"
      :worker -> "👷"
      _ -> "🤖"
    end
    
    status_color = cond do
      agent[:thinking] -> :bright_yellow
      agent[:communicating] -> :bright_blue
      true -> :green
    end
    
    IO.write("#{icon} #{Term.color(agent.name, status_color)}")
    
    # Show current state
    Term.move(row, col + 25)
    beliefs = agent[:beliefs] || 0
    desires = agent[:desires] || 0
    intentions = agent[:intentions] || 0
    
    IO.write(Term.dim("B:#{beliefs} D:#{desires} I:#{intentions}"))
    
    # Activity indicator
    if agent[:thinking] do
      Term.move(row, col + 45)
      IO.write(Term.blink(Term.color("⚡", :yellow)))
    end
  end
  
  defp render_messages_panel(state) do
    {row, col, _width, height} = state.layout.messages_panel
    
    Term.move(row, col)
    IO.write(Term.bold("💬 FIPA ACL MESSAGES"))
    
    state.messages
    |> Enum.take(height - 2)
    |> Enum.with_index()
    |> Enum.each(fn {msg, idx} ->
      Term.move(row + idx + 2, col)
      IO.write(msg)
    end)
  end
  
  defp render_blackboard_panel(state) do
    {row, col, _width, height} = state.layout.blackboard_panel
    
    Term.move(row, col)
    IO.write(Term.bold("📋 BLACKBOARD"))
    
    state.blackboard_entries
    |> Enum.take(height - 2)
    |> Enum.with_index()
    |> Enum.each(fn {entry, idx} ->
      Term.move(row + idx + 2, col)
      
      category_color = case entry.category do
        :problems -> :red
        :partial_solutions -> :yellow
        :constraints -> :magenta
        :hypotheses -> :cyan
        _ -> :white
      end
      
      IO.write("#{Term.color(to_string(entry.category), category_color)}: #{entry.key} = #{entry.value}")
    end)
  end
  
  defp render_metrics_panel(state) do
    {row, col, _width, _height} = state.layout.metrics_panel
    
    Term.move(row, col)
    IO.write(Term.bold("📊 METRICS"))
    
    metrics = [
      {"Messages", state.metrics.total_messages, :bright_blue},
      {"Beliefs Updated", state.metrics.beliefs_updated, :bright_yellow},
      {"Goals Achieved", state.metrics.goals_achieved, :bright_green},
      {"Contracts", state.metrics.contracts_completed, :bright_magenta}
    ]
    
    metrics
    |> Enum.with_index()
    |> Enum.each(fn {{label, value, color}, idx} ->
      Term.move(row + 2 + idx, col)
      IO.write("#{label}: #{Term.color(to_string(value), color)}")
    end)
  end
  
  defp render_contracts_panel(state) do
    {row, col, _width, _height} = state.layout.contracts_panel
    
    Term.move(row, col)
    IO.write(Term.bold("📜 CONTRACT NET PROTOCOL"))
    
    if map_size(state.contracts) > 0 do
      state.contracts
      |> Enum.take(3)
      |> Enum.with_index()
      |> Enum.each(fn {{id, status}, idx} ->
        Term.move(row + idx + 2, col)
        
        status_icon = case status do
          :collecting_proposals -> "📢"
          :evaluating -> "🤔"
          :awarded -> "✅"
          :completed -> "🎉"
          _ -> "❓"
        end
        
        IO.write("#{status_icon} Contract #{String.slice(id, 0, 8)}: #{status}")
      end)
    else
      Term.move(row + 2, col)
      IO.write(Term.dim("No active contracts"))
    end
  end
  
  defp enrich_agent_info(agent_info) do
    Map.merge(agent_info, %{
      thinking: false,
      communicating: false,
      beliefs: :rand.uniform(5),
      desires: :rand.uniform(3),
      intentions: :rand.uniform(2)
    })
  end
  
  defp format_message(%ACL{} = msg) do
    time = Calendar.strftime(DateTime.utc_now(), "%H:%M:%S")
    
    perf_icon = case msg.performative do
      :inform -> "ℹ️"
      :request -> "❓"
      :cfp -> "📢"
      :propose -> "💡"
      :accept_proposal -> "✅"
      :reject_proposal -> "❌"
      :agree -> "🤝"
      :refuse -> "🚫"
      :failure -> "⚠️"
      _ -> "📨"
    end
    
    sender = format_agent_name(msg.sender)
    receiver = format_agent_name(msg.receiver)
    
    "#{Term.dim(time)} #{perf_icon} #{Term.color(sender, :bright_cyan)} → #{Term.color(receiver, :bright_green)}"
  end
  
  defp format_agent_name(id) when is_binary(id) do
    case String.split(id, "_") do
      [type, _num] -> String.capitalize(type)
      _ -> id
    end
  end
  defp format_agent_name(pid) when is_pid(pid), do: "PID"
  defp format_agent_name(other), do: to_string(other)
  
  defp summarize_value(value) when is_map(value) do
    "#{map_size(value)} items"
  end
  defp summarize_value(value) when is_list(value) do
    "#{length(value)} items"
  end
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

# Demo runner
defmodule VisualizationDemo do
  def run do
    IO.puts(MASVisualization.Term.bold(MASVisualization.Term.color("""
    
    🌟 MULTI-AGENT SYSTEM VISUALIZATION
    ==================================
    
    Watch a multi-agent system in action with:
    • BDI agents making decisions
    • FIPA ACL message exchanges  
    • Blackboard collaboration
    • Contract Net Protocol
    
    Starting visualization...
    """, :bright_cyan)))
    
    Process.sleep(2000)
    
    # Start visualization
    {:ok, _pid} = MASVisualization.start_link()
    
    # Keep running
    Process.sleep(:infinity)
  end
end

# Handle termination
Process.flag(:trap_exit, true)

# Run the visualization
VisualizationDemo.run()