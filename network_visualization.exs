#!/usr/bin/env elixir

# Network Topology Visualization and Real-time Monitoring
# Advanced visualization for the massive node simulation

defmodule NetworkVisualization do
  use GenServer
  require Logger

  defstruct [
    :topology_data,
    :metrics_history,
    :real_time_stats,
    :visualization_mode,
    :update_frequency
  ]

  @update_interval 2000
  @history_limit 100

  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  def init(_opts) do
    initial_state = %__MODULE__{
      topology_data: %{},
      metrics_history: [],
      real_time_stats: %{},
      visualization_mode: :network_graph,
      update_frequency: @update_interval
    }

    # Start monitoring
    schedule_update()
    
    Logger.info("🎨 Network Visualization system started")
    {:ok, initial_state}
  end

  def handle_info(:update_visualization, state) do
    new_state = update_network_data(state)
    display_visualization(new_state)
    schedule_update()
    {:noreply, new_state}
  end

  defp update_network_data(state) do
    # Get current network status from the simulation
    current_stats = case GenServer.whereis(MassiveNodeSimulation) do
      nil -> %{total_nodes: 0, total_connections: 0, active_chats: 0, network_density: 0.0}
      _pid -> 
        try do
          MassiveNodeSimulation.get_network_status()
        rescue
          _ -> %{total_nodes: 0, total_connections: 0, active_chats: 0, network_density: 0.0}
        end
    end

    # Add timestamp
    timestamped_stats = Map.put(current_stats, :timestamp, DateTime.utc_now())

    # Update history
    new_history = [timestamped_stats | state.metrics_history]
      |> Enum.take(@history_limit)

    %{state |
      real_time_stats: current_stats,
      metrics_history: new_history
    }
  end

  defp display_visualization(state) do
    stats = state.real_time_stats
    
    # Clear screen and display header
    IO.write("\e[2J\e[H")
    
    display_header()
    display_network_overview(stats)
    display_metrics_graph(state.metrics_history)
    display_node_distribution(stats)
    display_real_time_activity(stats)
    display_network_health(stats)
  end

  defp display_header do
    IO.puts """
    #{IO.ANSI.cyan()}╔══════════════════════════════════════════════════════════════════════════════╗
    ║                      🌐 MASSIVE NETWORK SIMULATION MONITOR                   ║
    ║                           Real-time Visualization System                     ║
    ╚══════════════════════════════════════════════════════════════════════════════╝#{IO.ANSI.reset()}
    """
  end

  defp display_network_overview(stats) do
    nodes = stats[:total_nodes] || 0
    connections = stats[:total_connections] || 0
    chats = stats[:active_chats] || 0
    density = stats[:network_density] || 0.0

    # Color coding based on network size
    {node_color, node_status} = case nodes do
      n when n >= 3000 -> {IO.ANSI.green(), "MASSIVE"}
      n when n >= 2000 -> {IO.ANSI.yellow(), "LARGE"}
      n when n >= 1000 -> {IO.ANSI.cyan(), "MEDIUM"}
      _ -> {IO.ANSI.red(), "SMALL"}
    end

    IO.puts """
    #{IO.ANSI.bright()}📊 NETWORK OVERVIEW#{IO.ANSI.reset()}
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │ #{node_color}● Nodes: #{String.pad_leading("#{nodes}", 8)} (#{node_status})#{IO.ANSI.reset()}      │ #{IO.ANSI.blue()}● Connections: #{String.pad_leading("#{connections}", 8)}#{IO.ANSI.reset()}     │
    │ #{IO.ANSI.magenta()}● Active Chats: #{String.pad_leading("#{chats}", 4)}#{IO.ANSI.reset()}          │ #{IO.ANSI.green()}● Density: #{String.pad_leading("#{Float.round(density * 100, 2)}%", 8)}#{IO.ANSI.reset()}        │
    └─────────────────────────────────────────────────────────────────────────────┘
    """
  end

  defp display_metrics_graph(history) do
    if length(history) > 1 do
      IO.puts "\n#{IO.ANSI.bright()}📈 NETWORK GROWTH TRENDS#{IO.ANSI.reset()}"
      
      # Extract data for graphing
      nodes_data = Enum.map(history, & &1[:total_nodes] || 0) |> Enum.reverse()
      connections_data = Enum.map(history, & &1[:total_connections] || 0) |> Enum.reverse()
      
      # Create simple ASCII graphs
      display_ascii_graph("Nodes", nodes_data, IO.ANSI.green())
      display_ascii_graph("Connections", connections_data, IO.ANSI.blue())
    end
  end

  defp display_ascii_graph(label, data, color) do
    if length(data) > 0 do
      max_val = Enum.max(data)
      min_val = Enum.min(data)
      range = max_val - min_val
      
      if range > 0 do
        # Normalize data to fit in display width
        width = 60
        normalized = Enum.map(data, fn val ->
          round((val - min_val) / range * (width - 1))
        end)
        
        # Take last 60 points for display
        display_data = Enum.take(normalized, -width)
        
        IO.puts "#{color}#{label}: #{min_val} ▁▂▃▄▅▆▇█ #{max_val}#{IO.ANSI.reset()}"
        
        # Create graph lines
        for height <- 10..1 do
          line = Enum.map(display_data, fn val ->
            if val * 10 / width >= height, do: "█", else: " "
          end) |> Enum.join("")
          IO.puts "#{color}│#{line}│#{IO.ANSI.reset()}"
        end
        
        IO.puts "#{color}└#{String.duplicate("─", width)}┘#{IO.ANSI.reset()}"
      end
    end
  end

  defp display_node_distribution(stats) do
    personalities = stats[:personalities] || %{}
    node_types = stats[:node_types] || %{}
    
    IO.puts "\n#{IO.ANSI.bright()}🎭 NODE DISTRIBUTION#{IO.ANSI.reset()}"
    IO.puts "┌─────────────────────────────────────────────────────────────────────────────┐"
    
    # Display personalities
    IO.puts "│ #{IO.ANSI.bright()}Personalities:#{IO.ANSI.reset()}"
    personality_colors = %{
      "analytical" => IO.ANSI.cyan(),
      "creative" => IO.ANSI.magenta(),
      "social" => IO.ANSI.green(),
      "guardian" => IO.ANSI.red(),
      "explorer" => IO.ANSI.yellow()
    }
    
    Enum.each(personalities, fn {personality, count} ->
      color = personality_colors[personality] || IO.ANSI.white()
      bar = create_progress_bar(count, (stats[:total_nodes] || 1), 20)
      IO.puts "│   #{color}#{String.pad_trailing(personality, 12)}: #{String.pad_leading("#{count}", 4)} #{bar}#{IO.ANSI.reset()}"
    end)
    
    # Display node types
    IO.puts "│ #{IO.ANSI.bright()}Node Types:#{IO.ANSI.reset()}"
    Enum.each(node_types, fn {type, count} ->
      bar = create_progress_bar(count, (stats[:total_nodes] || 1), 20)
      IO.puts "│   #{String.pad_trailing("#{type}", 12)}: #{String.pad_leading("#{count}", 4)} #{bar}"
    end)
    
    IO.puts "└─────────────────────────────────────────────────────────────────────────────┘"
  end

  defp create_progress_bar(value, max_value, width) do
    if max_value > 0 do
      filled = round(value / max_value * width)
      filled_chars = String.duplicate("█", filled)
      empty_chars = String.duplicate("░", width - filled)
      "#{IO.ANSI.green()}#{filled_chars}#{IO.ANSI.dark()}#{empty_chars}#{IO.ANSI.reset()}"
    else
      String.duplicate("░", width)
    end
  end

  defp display_real_time_activity(stats) do
    IO.puts "\n#{IO.ANSI.bright()}⚡ REAL-TIME ACTIVITY#{IO.ANSI.reset()}"
    IO.puts "┌─────────────────────────────────────────────────────────────────────────────┐"
    
    # Simulate real-time activity indicators
    activity_level = calculate_activity_level(stats)
    activity_color = case activity_level do
      level when level > 0.8 -> IO.ANSI.red()
      level when level > 0.6 -> IO.ANSI.yellow()
      level when level > 0.4 -> IO.ANSI.green()
      _ -> IO.ANSI.blue()
    end
    
    IO.puts "│ #{activity_color}● Activity Level: #{activity_indicator(activity_level)} #{Float.round(activity_level * 100, 1)}%#{IO.ANSI.reset()}"
    
    # Show recent events simulation
    recent_events = generate_recent_events(stats)
    IO.puts "│ #{IO.ANSI.bright()}Recent Events:#{IO.ANSI.reset()}"
    Enum.each(recent_events, fn event ->
      IO.puts "│   #{event.icon} #{event.description}"
    end)
    
    IO.puts "└─────────────────────────────────────────────────────────────────────────────┘"
  end

  defp calculate_activity_level(stats) do
    nodes = stats[:total_nodes] || 0
    chats = stats[:active_chats] || 0
    density = stats[:network_density] || 0.0
    
    # Calculate activity as combination of factors
    node_factor = min(nodes / 3500, 1.0)
    chat_factor = min(chats / 50, 1.0)
    density_factor = min(density * 2, 1.0)
    
    (node_factor + chat_factor + density_factor) / 3
  end

  defp activity_indicator(level) do
    case level do
      level when level > 0.9 -> "🔥🔥🔥"
      level when level > 0.7 -> "🔥🔥"
      level when level > 0.5 -> "🔥"
      level when level > 0.3 -> "⚡"
      _ -> "💤"
    end
  end

  defp generate_recent_events(stats) do
    base_events = [
      %{icon: "🔗", description: "New connections established"},
      %{icon: "💬", description: "Chat conversations active"},
      %{icon: "⚡", description: "Function calls executing"},
      %{icon: "🔍", description: "Network discovery in progress"},
      %{icon: "📊", description: "Metrics analysis running"}
    ]
    
    # Add dynamic events based on stats
    dynamic_events = []
    
    dynamic_events = if (stats[:total_nodes] || 0) > 3000 do
      [%{icon: "🚀", description: "MASSIVE scale achieved!"} | dynamic_events]
    else
      dynamic_events
    end
    
    dynamic_events = if (stats[:network_density] || 0) > 0.1 do
      [%{icon: "🌐", description: "High network density detected"} | dynamic_events]
    else
      dynamic_events
    end
    
    all_events = base_events ++ dynamic_events
    Enum.take_random(all_events, min(4, length(all_events)))
  end

  defp display_network_health(stats) do
    IO.puts "\n#{IO.ANSI.bright()}🏥 NETWORK HEALTH#{IO.ANSI.reset()}"
    IO.puts "┌─────────────────────────────────────────────────────────────────────────────┐"
    
    # Calculate health metrics
    health_metrics = calculate_health_metrics(stats)
    
    Enum.each(health_metrics, fn {metric, %{value: value, status: status, color: color}} ->
      status_icon = case status do
        :excellent -> "🟢"
        :good -> "🟡"
        :warning -> "🟠"
        :critical -> "🔴"
      end
      
      IO.puts "│ #{status_icon} #{color}#{String.pad_trailing(metric, 20)}: #{value}#{IO.ANSI.reset()}"
    end)
    
    IO.puts "└─────────────────────────────────────────────────────────────────────────────┘"
    
    # Display timestamp
    IO.puts "\n#{IO.ANSI.dark()}Last updated: #{DateTime.utc_now() |> DateTime.to_string()}#{IO.ANSI.reset()}"
  end

  defp calculate_health_metrics(stats) do
    nodes = stats[:total_nodes] || 0
    connections = stats[:total_connections] || 0
    chats = stats[:active_chats] || 0
    density = stats[:network_density] || 0.0
    
    %{
      "Node Count" => evaluate_metric(nodes, [
        {3000, :excellent},
        {2000, :good},
        {1000, :warning},
        {0, :critical}
      ], "#{nodes} nodes"),
      
      "Connectivity" => evaluate_metric(connections, [
        {5000, :excellent},
        {3000, :good},
        {1000, :warning},
        {0, :critical}
      ], "#{connections} connections"),
      
      "Communication" => evaluate_metric(chats, [
        {20, :excellent},
        {10, :good},
        {5, :warning},
        {0, :critical}
      ], "#{chats} active chats"),
      
      "Network Density" => evaluate_metric(density * 100, [
        {15, :excellent},
        {10, :good},
        {5, :warning},
        {0, :critical}
      ], "#{Float.round(density * 100, 2)}%"),
      
      "Avg Connections/Node" => if(nodes > 0, 
        do: evaluate_metric(connections / nodes, [
          {5, :excellent},
          {3, :good},
          {2, :warning},
          {0, :critical}
        ], "#{Float.round(connections / nodes, 1)}"),
        else: %{value: "N/A", status: :critical, color: IO.ANSI.red()}
      )
    }
  end

  defp evaluate_metric(value, thresholds, display_value) do
    {status, color} = Enum.find(thresholds, fn {threshold, _status} ->
      value >= threshold
    end) |> case do
      {_, :excellent} -> {:excellent, IO.ANSI.green()}
      {_, :good} -> {:good, IO.ANSI.yellow()}
      {_, :warning} -> {:warning, IO.ANSI.red()}
      {_, :critical} -> {:critical, IO.ANSI.red()}
      nil -> {:critical, IO.ANSI.red()}
    end
    
    %{value: display_value, status: status, color: color}
  end

  defp schedule_update do
    Process.send_after(self(), :update_visualization, @update_interval)
  end

  # Public API
  def set_update_frequency(frequency) do
    GenServer.cast(__MODULE__, {:set_frequency, frequency})
  end

  def get_current_stats do
    GenServer.call(__MODULE__, :get_stats)
  end

  def handle_cast({:set_frequency, frequency}, state) do
    {:noreply, %{state | update_frequency: frequency}}
  end

  def handle_call(:get_stats, _from, state) do
    {:reply, state.real_time_stats, state}
  end
end

# Auto-start visualization if simulation is running
case GenServer.whereis(MassiveNodeSimulation) do
  nil -> 
    IO.puts "⚠️  Massive Node Simulation not running. Start it first!"
  _pid -> 
    case NetworkVisualization.start_link() do
      {:ok, _pid} ->
        IO.puts "\n🎨 Network Visualization started!"
        IO.puts "📊 Real-time monitoring active"
        IO.puts "🔄 Updates every 2 seconds"
        Process.sleep(:infinity)
      {:error, reason} ->
        IO.puts "❌ Failed to start visualization: #{inspect(reason)}"
    end
end