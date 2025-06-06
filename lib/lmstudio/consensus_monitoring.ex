defmodule LMStudio.ConsensusMonitoring do
  @moduledoc """
  Real-time monitoring, analytics, and observability for consensus systems.
  Provides comprehensive metrics, alerts, and visualization capabilities.
  """

  use GenServer
  require Logger

  defstruct [
    :node_id,
    :monitored_systems,
    :metrics_buffer,
    :aggregated_metrics,
    :performance_history,
    :anomaly_detector,
    :alert_config,
    :visualization_server,
    :export_handlers,
    :config,
    :enable_visualization
  ]


  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  def init(opts) do
    state = %__MODULE__{
      node_id: Keyword.get(opts, :node_id, node()),
      monitored_systems: %{},
      metrics_buffer: [],
      aggregated_metrics: %{},
      performance_history: [],
      anomaly_detector: init_anomaly_detector(),
      alert_config: Keyword.get(opts, :alert_config, default_alert_config()),
      visualization_server: nil,
      export_handlers: [],
      config: opts,
      enable_visualization: Keyword.get(opts, :enable_visualization, false)
    }
    
    # Start visualization if enabled
    state = if state.enable_visualization do
      # In a real implementation, would start a Phoenix LiveView or similar
      Logger.info("📊 Visualization enabled for consensus monitoring")
      state
    else
      state
    end
    
    # Schedule periodic aggregation
    schedule_aggregation()
    
    {:ok, state}
  end

  # Public API

  def register_system(monitor \\ __MODULE__, system_id, system_info) do
    GenServer.call(monitor, {:register_system, system_id, system_info})
  end

  def unregister_system(monitor \\ __MODULE__, system_id) do
    GenServer.call(monitor, {:unregister_system, system_id})
  end

  def record_metric(monitor \\ __MODULE__, system_id, metric_type, value, metadata \\ %{}) do
    GenServer.cast(monitor, {:record_metric, system_id, metric_type, value, metadata})
  end

  def record_event(monitor \\ __MODULE__, system_id, event_type, event_data) do
    GenServer.cast(monitor, {:record_event, system_id, event_type, event_data})
  end

  def get_metrics(monitor \\ __MODULE__, system_id, opts \\ []) do
    GenServer.call(monitor, {:get_metrics, system_id, opts})
  end

  def get_system_health(monitor \\ __MODULE__, system_id) do
    GenServer.call(monitor, {:get_system_health, system_id})
  end

  def get_performance_report(monitor \\ __MODULE__, opts \\ []) do
    GenServer.call(monitor, {:get_performance_report, opts})
  end

  def set_alert_threshold(monitor \\ __MODULE__, metric_type, threshold) do
    GenServer.call(monitor, {:set_alert_threshold, metric_type, threshold})
  end

  def enable_export(monitor \\ __MODULE__, handler, config) do
    GenServer.call(monitor, {:enable_export, handler, config})
  end

  # Callbacks

  def handle_call({:register_system, system_id, system_info}, _from, state) do
    monitored_systems = Map.put(state.monitored_systems, system_id, %{
      info: system_info,
      metrics: %{},
      events: [],
      health_score: 1.0,
      registered_at: System.monotonic_time(:millisecond)
    })
    
    Logger.info("📊 Registered system for monitoring: #{system_id}")
    {:reply, :ok, %{state | monitored_systems: monitored_systems}}
  end

  def handle_call({:unregister_system, system_id}, _from, state) do
    monitored_systems = Map.delete(state.monitored_systems, system_id)
    {:reply, :ok, %{state | monitored_systems: monitored_systems}}
  end

  def handle_call({:get_metrics, system_id, opts}, _from, state) do
    case Map.get(state.monitored_systems, system_id) do
      nil -> 
        {:reply, {:error, :system_not_found}, state}
      system ->
        metrics = get_filtered_metrics(system.metrics, opts)
        {:reply, {:ok, metrics}, state}
    end
  end

  def handle_call({:get_system_health, system_id}, _from, state) do
    case Map.get(state.monitored_systems, system_id) do
      nil -> 
        {:reply, {:error, :system_not_found}, state}
      system ->
        health = calculate_health_status(system)
        {:reply, {:ok, health}, state}
    end
  end

  def handle_call({:get_performance_report, opts}, _from, state) do
    report = generate_performance_report(state, opts)
    {:reply, {:ok, report}, state}
  end

  def handle_call({:set_alert_threshold, metric_type, threshold}, _from, state) do
    alert_config = update_alert_threshold(state.alert_config, metric_type, threshold)
    {:reply, :ok, %{state | alert_config: alert_config}}
  end

  def handle_call({:enable_export, handler, config}, _from, state) do
    export_handlers = [{handler, config} | state.export_handlers]
    {:reply, :ok, %{state | export_handlers: export_handlers}}
  end

  def handle_call(:get_dashboard, _from, state) do
    dashboard = generate_dashboard(state)
    {:reply, dashboard, state}
  end

  def handle_cast({:record_metric, system_id, metric_type, value, metadata}, state) do
    state = case Map.get(state.monitored_systems, system_id) do
      nil -> 
        state
      _system ->
        # Record the metric
        timestamp = System.monotonic_time(:millisecond)
        metric = %{
          type: metric_type,
          value: value,
          timestamp: timestamp,
          metadata: metadata
        }
        
        # Update metrics
        state = update_system_metrics(state, system_id, metric)
        
        # Check for anomalies
        state = check_anomalies(state, system_id, metric)
        
        # Check alerts
        check_alerts(state, system_id, metric)
        
        state
    end
    
    {:noreply, state}
  end

  def handle_cast({:record_event, system_id, event_type, event_data}, state) do
    state = case Map.get(state.monitored_systems, system_id) do
      nil -> 
        state
      _system ->
        event = %{
          type: event_type,
          data: event_data,
          timestamp: System.monotonic_time(:millisecond)
        }
        
        update_system_events(state, system_id, event)
    end
    
    {:noreply, state}
  end

  def handle_info(:aggregate_metrics, state) do
    state = aggregate_all_metrics(state)
    schedule_aggregation()
    {:noreply, state}
  end

  # Private helpers

  defp init_anomaly_detector do
    %{
      thresholds: %{
        message_latency: {0, 1000},  # ms
        byzantine_nodes_detected: {0, 2},
        vote_participation_rate: {0.7, 1.0},
        consensus_rounds: {1, 10}
      },
      history_size: 100,
      detection_enabled: true
    }
  end

  defp default_alert_config do
    %{
      rules: [
        %{
          name: "high_latency",
          metric_type: :message_latency,
          threshold: 500,
          severity: :warning
        },
        %{
          name: "byzantine_threshold",
          metric_type: :byzantine_nodes_detected,
          threshold: 1,
          severity: :critical
        },
        %{
          name: "low_participation",
          metric_type: :vote_participation_rate,
          threshold: 0.66,
          severity: :warning
        }
      ],
      notification_handlers: []
    }
  end

  defp schedule_aggregation do
    Process.send_after(self(), :aggregate_metrics, 5_000)
  end

  defp update_system_metrics(state, system_id, metric) do
    update_in(state.monitored_systems[system_id].metrics[metric.type], fn metrics ->
      metrics = metrics || []
      # Keep last 1000 metrics per type
      metrics = [metric | metrics] |> Enum.take(1000)
      metrics
    end)
  end

  defp update_system_events(state, system_id, event) do
    update_in(state.monitored_systems[system_id].events, fn events ->
      # Keep last 500 events
      [event | events] |> Enum.take(500)
    end)
  end

  defp check_anomalies(state, _system_id, metric) do
    case Map.get(state.anomaly_detector.thresholds, metric.type) do
      {min, max} when metric.value < min or metric.value > max ->
        Logger.warning("⚠️  Anomaly detected: #{metric.type} = #{metric.value} (expected: #{min}-#{max})")
        state
      _ ->
        state
    end
  end

  defp check_alerts(state, system_id, metric) do
    Enum.each(state.alert_config.rules, fn rule ->
      if rule.metric_type == metric.type and violates_rule?(metric.value, rule) do
        trigger_alert(system_id, metric, rule, state)
      end
    end)
  end

  defp violates_rule?(value, %{threshold: threshold, metric_type: type}) do
    case type do
      :vote_participation_rate -> value < threshold
      _ -> value > threshold
    end
  end

  defp trigger_alert(system_id, metric, rule, state) do
    alert = %{
      system_id: system_id,
      rule_name: rule.name,
      severity: rule.severity,
      metric_type: metric.type,
      metric_value: metric.value,
      threshold: rule.threshold,
      timestamp: System.monotonic_time(:millisecond),
      message: format_alert_message(rule, metric)
    }
    
    Logger.warning("🚨 Alert: #{alert.message}")
    
    # Export alert if handlers configured
    Enum.each(state.export_handlers, fn {handler, _config} ->
      spawn(fn -> handler.export_alert(alert) end)
    end)
  end

  defp format_alert_message(rule, metric) do
    "#{rule.severity |> to_string() |> String.upcase()}: #{metric.type} = #{metric.value} (threshold: #{rule.threshold})"
  end

  defp aggregate_all_metrics(state) do
    # Aggregate metrics for each system
    Enum.reduce(state.monitored_systems, state, fn {system_id, _system}, acc_state ->
      analyze_system_performance(acc_state, system_id)
    end)
  end

  defp analyze_system_performance(state, system_id) do
    system = Map.get(state.monitored_systems, system_id)
    return = if system do
      # Calculate aggregated metrics
      aggregated = calculate_aggregated_metrics(system.metrics)
      
      # Update health score
      health_score = calculate_health_score(aggregated)
      
      # Update monitored systems with new health score
      updated_system = Map.put(system, :health_score, health_score)
      updated_systems = Map.put(state.monitored_systems, system_id, updated_system)
      
      # Update aggregated metrics
      updated_aggregated = Map.put(state.aggregated_metrics, system_id, aggregated)
      
      %{state | 
        monitored_systems: updated_systems,
        aggregated_metrics: updated_aggregated
      }
    else
      state
    end
    return
  end

  defp calculate_aggregated_metrics(metrics) do
    Enum.reduce(metrics, %{}, fn {metric_type, values}, acc ->
      if values && length(values) > 0 do
        # Check if values are numeric
        first_value = hd(values).value
        
        if is_number(first_value) do
          sorted_values = values
            |> Enum.map(& &1.value)
            |> Enum.filter(&is_number/1)
            |> Enum.sort()
          
          if length(sorted_values) > 0 do
            Map.put(acc, metric_type, %{
              min: Enum.min(sorted_values),
              max: Enum.max(sorted_values),
              avg: Enum.sum(sorted_values) / length(sorted_values),
              median: median(sorted_values),
              p95: percentile(sorted_values, 0.95),
              p99: percentile(sorted_values, 0.99),
              count: length(sorted_values)
            })
          else
            acc
          end
        else
          # For non-numeric values, just count occurrences
          value_counts = values
            |> Enum.map(& &1.value)
            |> Enum.frequencies()
          
          Map.put(acc, metric_type, %{
            values: value_counts,
            count: length(values),
            type: :categorical
          })
        end
      else
        acc
      end
    end)
  end

  defp median(sorted_list) do
    len = length(sorted_list)
    mid = div(len, 2)
    
    if rem(len, 2) == 0 do
      (Enum.at(sorted_list, mid - 1) + Enum.at(sorted_list, mid)) / 2
    else
      Enum.at(sorted_list, mid)
    end
  end

  defp percentile(sorted_list, p) do
    index = round(p * (length(sorted_list) - 1))
    Enum.at(sorted_list, index)
  end

  defp calculate_health_score(aggregated_metrics) do
    # Simple health score calculation based on key metrics
    scores = []
    
    # Latency score
    scores = case Map.get(aggregated_metrics, :message_latency) do
      %{avg: avg} when avg < 100 -> [1.0 | scores]
      %{avg: avg} when avg < 500 -> [0.7 | scores]
      %{avg: _avg} -> [0.3 | scores]
      _ -> scores
    end
    
    # Byzantine nodes score
    scores = case Map.get(aggregated_metrics, :byzantine_nodes_detected) do
      %{max: 0} -> [1.0 | scores]
      %{max: max} when max <= 1 -> [0.5 | scores]
      _ -> [0.0 | scores]
    end
    
    # Participation score
    scores = case Map.get(aggregated_metrics, :vote_participation_rate) do
      %{avg: avg} when avg > 0.9 -> [1.0 | scores]
      %{avg: avg} when avg > 0.7 -> [0.7 | scores]
      %{avg: _avg} -> [0.3 | scores]
      _ -> scores
    end
    
    # Calculate average score
    if length(scores) > 0 do
      Enum.sum(scores) / length(scores)
    else
      1.0
    end
  end

  defp calculate_health_status(system) do
    %{
      score: system.health_score,
      status: health_score_to_status(system.health_score),
      metrics_summary: summarize_metrics(system.metrics),
      recent_events: Enum.take(system.events, 10)
    }
  end

  defp health_score_to_status(score) do
    cond do
      score >= 0.9 -> :healthy
      score >= 0.7 -> :degraded
      score >= 0.5 -> :unhealthy
      true -> :critical
    end
  end

  defp summarize_metrics(metrics) do
    Enum.reduce(metrics, %{}, fn {type, values}, acc ->
      if values && length(values) > 0 do
        latest = hd(values)
        Map.put(acc, type, %{
          latest: latest.value,
          timestamp: latest.timestamp
        })
      else
        acc
      end
    end)
  end

  defp generate_performance_report(state, _opts) do
    %{
      timestamp: System.monotonic_time(:millisecond),
      total_systems: map_size(state.monitored_systems),
      systems: generate_systems_view(state),
      aggregated_metrics: state.aggregated_metrics,
      overall_health: calculate_overall_health(state)
    }
  end

  defp generate_systems_view(state) do
    Enum.map(state.monitored_systems, fn {system_id, _info} ->
      %{
        id: system_id,
        health_score: Map.get(state.monitored_systems, system_id).health_score,
        status: health_score_to_status(Map.get(state.monitored_systems, system_id).health_score)
      }
    end)
  end

  defp calculate_overall_health(state) do
    if map_size(state.monitored_systems) > 0 do
      total_score = Enum.reduce(state.monitored_systems, 0, fn {_id, system}, acc ->
        acc + system.health_score
      end)
      
      avg_score = total_score / map_size(state.monitored_systems)
      
      %{
        score: avg_score,
        status: health_score_to_status(avg_score)
      }
    else
      %{score: 1.0, status: :healthy}
    end
  end

  defp get_filtered_metrics(metrics, opts) do
    metric_type = Keyword.get(opts, :type)
    limit = Keyword.get(opts, :limit, 100)
    
    metrics = if metric_type do
      Map.take(metrics, [metric_type])
    else
      metrics
    end
    
    # Limit number of values per metric type
    Enum.reduce(metrics, %{}, fn {type, values}, acc ->
      Map.put(acc, type, Enum.take(values, limit))
    end)
  end

  defp update_alert_threshold(alert_config, metric_type, threshold) do
    rules = Enum.map(alert_config.rules, fn rule ->
      if rule.metric_type == metric_type do
        %{rule | threshold: threshold}
      else
        rule
      end
    end)
    
    %{alert_config | rules: rules}
  end

  def get_dashboard(monitor \\ __MODULE__) do
    GenServer.call(monitor, :get_dashboard)
  end

  defp generate_dashboard(state) do
    systems = generate_systems_dashboard(state)
    metrics_summary = generate_metrics_summary(state)
    
    %{
      timestamp: System.monotonic_time(:millisecond),
      systems: systems,
      alerts: generate_recent_alerts(state),
      overview: %{
        total_systems: map_size(state.monitored_systems),
        active_alerts: 0,  # Would track real alerts in production
        metrics_processed: metrics_summary.total_metrics
      },
      health_overview: calculate_overall_health(state),
      visualization_url: nil
    }
  end

  defp generate_systems_dashboard(state) do
    Enum.map(state.monitored_systems, fn {system_id, system} ->
      %{
        id: system_id,
        health_score: system.health_score,
        status: health_score_to_status(system.health_score),
        metrics_count: map_size(system.metrics),
        events_count: length(system.events),
        registered_at: system.registered_at
      }
    end)
  end

  defp generate_recent_alerts(_state) do
    # In a real implementation, would track alerts
    []
  end

  defp generate_metrics_summary(state) do
    total_metrics = Enum.reduce(state.monitored_systems, 0, fn {_id, system}, acc ->
      acc + Enum.reduce(system.metrics, 0, fn {_type, values}, inner_acc ->
        inner_acc + length(values)
      end)
    end)
    
    %{
      total_metrics: total_metrics,
      systems_monitored: map_size(state.monitored_systems),
      aggregation_enabled: true
    }
  end
end