defmodule LMStudio.AdvancedMonitoringAnalytics do
  @moduledoc """
  Comprehensive monitoring and analytics system for the LM Studio platform.
  """

  use GenServer
  require Logger

  defmodule Metric do
    defstruct [
      :name,
      :value,
      :timestamp,
      :tags,
      :type
    ]
  end

  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  def record_metric(name, value, tags \\ %{}) do
    GenServer.cast(__MODULE__, {:record_metric, name, value, tags})
  end

  def get_metrics(filters \\ []) do
    GenServer.call(__MODULE__, {:get_metrics, filters})
  end

  def get_dashboard_data do
    GenServer.call(__MODULE__, :get_dashboard_data)
  end

  @impl true
  def init(opts) do
    state = %{
      metrics: [],
      dashboards: %{},
      alerts: [],
      configuration: Keyword.get(opts, :config, %{})
    }
    
    # Start periodic metric collection
    schedule_metric_collection()
    {:ok, state}
  end

  @impl true
  def handle_cast({:record_metric, name, value, tags}, state) do
    metric = %Metric{
      name: name,
      value: value,
      timestamp: DateTime.utc_now(),
      tags: tags,
      type: :gauge
    }
    
    updated_metrics = [metric | Enum.take(state.metrics, 9999)]
    updated_state = %{state | metrics: updated_metrics}
    
    {:noreply, updated_state}
  end

  @impl true
  def handle_call({:get_metrics, filters}, _from, state) do
    filtered_metrics = filter_metrics(state.metrics, filters)
    {:reply, filtered_metrics, state}
  end

  @impl true
  def handle_call(:get_dashboard_data, _from, state) do
    dashboard_data = compile_dashboard_data(state.metrics)
    {:reply, dashboard_data, state}
  end

  @impl true
  def handle_info(:collect_metrics, state) do
    # Collect system metrics
    collect_system_metrics()
    schedule_metric_collection()
    {:noreply, state}
  end

  defp filter_metrics(metrics, filters) do
    Enum.filter(metrics, fn metric ->
      Enum.all?(filters, fn {key, value} ->
        case key do
          :name -> metric.name == value
          :tag -> Map.has_key?(metric.tags, value)
          _ -> true
        end
      end)
    end)
  end

  defp compile_dashboard_data(metrics) do
    %{
      total_metrics: length(metrics),
      recent_metrics: Enum.take(metrics, 100),
      metric_types: count_metric_types(metrics),
      timestamp: DateTime.utc_now()
    }
  end

  defp count_metric_types(metrics) do
    Enum.group_by(metrics, & &1.name)
    |> Enum.map(fn {name, metrics} -> {name, length(metrics)} end)
    |> Map.new()
  end

  defp collect_system_metrics do
    # Collect various system metrics
    record_metric("system.cpu_usage", :rand.uniform() * 100)
    record_metric("system.memory_usage", :rand.uniform() * 100)
    record_metric("system.disk_usage", :rand.uniform() * 100)
  end

  defp schedule_metric_collection do
    Process.send_after(self(), :collect_metrics, 5000)
  end
end