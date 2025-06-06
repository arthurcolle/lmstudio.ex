defmodule LMStudio.AdvancedStreaming do
  @moduledoc """
  Advanced streaming capabilities with backpressure control, flow management,
  and sophisticated token processing for LM Studio integration.
  
  Features:
  - Adaptive backpressure control
  - Multi-consumer streaming
  - Token-level flow control
  - Stream composition and transformation
  - Latency optimization
  - Error recovery and circuit breaker patterns
  """

  use GenServer
  require Logger

  defmodule StreamConfig do
    @moduledoc "Configuration for advanced streaming"
    
    defstruct [
      :stream_id,
      :max_buffer_size,
      :backpressure_threshold,
      :flow_control_strategy,
      :latency_target_ms,
      :error_recovery_strategy,
      :consumer_timeout_ms,
      :chunk_size_strategy,
      :compression_enabled,
      :priority_queue_enabled
    ]
    
    def new(opts \\ []) do
      %__MODULE__{
        stream_id: Keyword.get(opts, :stream_id, UUID.uuid4()),
        max_buffer_size: Keyword.get(opts, :max_buffer_size, 10_000),
        backpressure_threshold: Keyword.get(opts, :backpressure_threshold, 0.8),
        flow_control_strategy: Keyword.get(opts, :flow_control_strategy, :adaptive),
        latency_target_ms: Keyword.get(opts, :latency_target_ms, 50),
        error_recovery_strategy: Keyword.get(opts, :error_recovery_strategy, :retry_with_backoff),
        consumer_timeout_ms: Keyword.get(opts, :consumer_timeout_ms, 30_000),
        chunk_size_strategy: Keyword.get(opts, :chunk_size_strategy, :dynamic),
        compression_enabled: Keyword.get(opts, :compression_enabled, false),
        priority_queue_enabled: Keyword.get(opts, :priority_queue_enabled, true)
      }
    end
  end

  defmodule StreamState do
    @moduledoc "Internal state for stream management"
    
    defstruct [
      :config,
      :buffer,
      :consumers,
      :producer_pid,
      :metrics,
      :flow_control,
      :error_state,
      :circuit_breaker,
      :priority_queue
    ]
  end

  defmodule StreamMetrics do
    @moduledoc "Real-time metrics for stream performance"
    
    defstruct [
      :tokens_processed,
      :bytes_transferred,
      :latency_histogram,
      :throughput_samples,
      :error_count,
      :backpressure_events,
      :consumer_lag,
      :buffer_utilization,
      :timestamp_start,
      :timestamp_last_update
    ]
    
    def new do
      now = DateTime.utc_now()
      %__MODULE__{
        tokens_processed: 0,
        bytes_transferred: 0,
        latency_histogram: %{},
        throughput_samples: [],
        error_count: 0,
        backpressure_events: 0,
        consumer_lag: %{},
        buffer_utilization: 0.0,
        timestamp_start: now,
        timestamp_last_update: now
      }
    end
  end

  defmodule FlowControl do
    @moduledoc "Advanced flow control algorithms"
    
    defstruct [
      :strategy,
      :window_size,
      :rate_limit_tokens_per_sec,
      :burst_capacity,
      :current_rate,
      :last_adjustment,
      :congestion_detected,
      :adaptive_parameters
    ]
    
    def new(strategy \\ :adaptive) do
      %__MODULE__{
        strategy: strategy,
        window_size: 1000,
        rate_limit_tokens_per_sec: 5000,
        burst_capacity: 2000,
        current_rate: 0.0,
        last_adjustment: DateTime.utc_now(),
        congestion_detected: false,
        adaptive_parameters: %{
          slow_start_threshold: 1000,
          congestion_window: 1000,
          rtt_estimate: 50.0,
          rtt_variance: 25.0
        }
      }
    end
  end

  # Public API

  def start_link(config \\ %StreamConfig{}) do
    GenServer.start_link(__MODULE__, config, name: via_tuple(config.stream_id))
  end

  def create_advanced_stream(messages, opts \\ []) do
    config = StreamConfig.new(opts)
    
    case start_link(config) do
      {:ok, pid} ->
        GenServer.call(pid, {:initialize_stream, messages, opts})
      error ->
        error
    end
  end

  def subscribe_consumer(stream_id, consumer_pid, opts \\ []) do
    GenServer.call(via_tuple(stream_id), {:subscribe_consumer, consumer_pid, opts})
  end

  def unsubscribe_consumer(stream_id, consumer_pid) do
    GenServer.call(via_tuple(stream_id), {:unsubscribe_consumer, consumer_pid})
  end

  def get_stream_metrics(stream_id) do
    GenServer.call(via_tuple(stream_id), :get_metrics)
  end

  def adjust_flow_control(stream_id, parameters) do
    GenServer.call(via_tuple(stream_id), {:adjust_flow_control, parameters})
  end

  def pause_stream(stream_id) do
    GenServer.call(via_tuple(stream_id), :pause_stream)
  end

  def resume_stream(stream_id) do
    GenServer.call(via_tuple(stream_id), :resume_stream)
  end

  def terminate_stream(stream_id) do
    GenServer.call(via_tuple(stream_id), :terminate_stream)
  end

  # GenServer Implementation

  @impl true
  def init(config) do
    state = %StreamState{
      config: config,
      buffer: :queue.new(),
      consumers: %{},
      producer_pid: nil,
      metrics: StreamMetrics.new(),
      flow_control: FlowControl.new(config.flow_control_strategy),
      error_state: %{consecutive_errors: 0, last_error: nil},
      circuit_breaker: %{state: :closed, failure_count: 0, last_failure: nil},
      priority_queue: if(config.priority_queue_enabled, do: init_priority_queue(), else: nil)
    }
    
    Logger.info("Advanced streaming initialized: #{config.stream_id}")
    {:ok, state}
  end

  @impl true
  def handle_call({:initialize_stream, messages, opts}, _from, state) do
    # Start the streaming process
    producer_pid = spawn_link(fn -> 
      stream_producer(messages, opts, self(), state.config)
    end)
    
    updated_state = %{state | producer_pid: producer_pid}
    {:reply, {:ok, state.config.stream_id}, updated_state}
  end

  @impl true
  def handle_call({:subscribe_consumer, consumer_pid, opts}, _from, state) do
    consumer_config = %{
      pid: consumer_pid,
      priority: Keyword.get(opts, :priority, :normal),
      buffer_size: Keyword.get(opts, :buffer_size, 100),
      flow_control: Keyword.get(opts, :flow_control, true),
      last_ack: DateTime.utc_now(),
      lag: 0
    }
    
    updated_consumers = Map.put(state.consumers, consumer_pid, consumer_config)
    updated_state = %{state | consumers: updated_consumers}
    
    Logger.info("Consumer subscribed: #{inspect(consumer_pid)}")
    {:reply, :ok, updated_state}
  end

  @impl true
  def handle_call({:unsubscribe_consumer, consumer_pid}, _from, state) do
    updated_consumers = Map.delete(state.consumers, consumer_pid)
    updated_state = %{state | consumers: updated_consumers}
    
    Logger.info("Consumer unsubscribed: #{inspect(consumer_pid)}")
    {:reply, :ok, updated_state}
  end

  @impl true
  def handle_call(:get_metrics, _from, state) do
    enriched_metrics = enrich_metrics(state.metrics, state)
    {:reply, enriched_metrics, state}
  end

  @impl true
  def handle_call({:adjust_flow_control, parameters}, _from, state) do
    updated_flow_control = update_flow_control(state.flow_control, parameters)
    updated_state = %{state | flow_control: updated_flow_control}
    
    Logger.info("Flow control adjusted: #{inspect(parameters)}")
    {:reply, :ok, updated_state}
  end

  @impl true
  def handle_call(:pause_stream, _from, state) do
    if state.producer_pid do
      send(state.producer_pid, :pause)
    end
    {:reply, :ok, state}
  end

  @impl true
  def handle_call(:resume_stream, _from, state) do
    if state.producer_pid do
      send(state.producer_pid, :resume)
    end
    {:reply, :ok, state}
  end

  @impl true
  def handle_call(:terminate_stream, _from, state) do
    if state.producer_pid do
      send(state.producer_pid, :terminate)
    end
    {:stop, :normal, :ok, state}
  end

  @impl true
  def handle_info({:stream_token, token_data}, state) do
    # Process incoming token with advanced flow control
    case should_accept_token(state) do
      true ->
        updated_state = process_token(token_data, state)
        {:noreply, updated_state}
      
      false ->
        # Apply backpressure
        updated_state = apply_backpressure(state)
        {:noreply, updated_state}
    end
  end

  @impl true
  def handle_info({:consumer_ack, consumer_pid, ack_data}, state) do
    updated_state = process_consumer_ack(consumer_pid, ack_data, state)
    {:noreply, updated_state}
  end

  @impl true
  def handle_info(:flow_control_tick, state) do
    updated_state = perform_flow_control_adjustment(state)
    schedule_flow_control_tick()
    {:noreply, updated_state}
  end

  @impl true
  def handle_info(:metrics_update, state) do
    updated_metrics = update_metrics(state.metrics, state)
    updated_state = %{state | metrics: updated_metrics}
    schedule_metrics_update()
    {:noreply, updated_state}
  end

  @impl true
  def handle_info({:producer_error, error}, state) do
    updated_state = handle_producer_error(error, state)
    {:noreply, updated_state}
  end

  @impl true
  def handle_info(_msg, state) do
    {:noreply, state}
  end

  # Private Functions

  defp via_tuple(stream_id) do
    {:via, Registry, {LMStudio.StreamRegistry, stream_id}}
  end

  defp stream_producer(messages, opts, stream_manager_pid, config) do
    # Initialize LM Studio streaming request
    stream_opts = [
      stream: true,
      stream_callback: fn 
        {:chunk, content} -> 
          send(stream_manager_pid, {:stream_token, %{
            content: content,
            timestamp: DateTime.utc_now(),
            sequence: System.unique_integer([:positive])
          }})
        {:done, _} -> 
          send(stream_manager_pid, {:stream_complete})
        {:error, reason} -> 
          send(stream_manager_pid, {:producer_error, reason})
      end
    ]
    
    # Enhanced request with retry logic
    perform_streaming_request(messages, stream_opts, config, 0)
  end

  defp perform_streaming_request(messages, opts, config, retry_count) do
    case LMStudio.complete(messages, opts) do
      {:ok, _result} -> 
        :ok
      
      {:error, reason} when retry_count < 3 ->
        backoff_time = min(1000 * :math.pow(2, retry_count), 10_000)
        Process.sleep(round(backoff_time))
        perform_streaming_request(messages, opts, config, retry_count + 1)
      
      {:error, reason} ->
        Logger.error("Streaming failed after retries: #{inspect(reason)}")
        {:error, reason}
    end
  end

  defp should_accept_token(state) do
    buffer_utilization = calculate_buffer_utilization(state)
    threshold = state.config.backpressure_threshold
    
    cond do
      buffer_utilization < threshold -> true
      state.flow_control.congestion_detected -> false
      map_size(state.consumers) == 0 -> false
      true -> apply_adaptive_acceptance(state)
    end
  end

  defp apply_adaptive_acceptance(state) do
    # Implement TCP-like congestion control
    case state.flow_control.strategy do
      :adaptive ->
        adaptive_acceptance_decision(state)
      :fixed_rate ->
        fixed_rate_acceptance_decision(state)
      :token_bucket ->
        token_bucket_acceptance_decision(state)
      _ ->
        true
    end
  end

  defp adaptive_acceptance_decision(state) do
    current_rate = calculate_current_throughput(state)
    target_rate = state.flow_control.rate_limit_tokens_per_sec
    
    current_rate < target_rate * 0.9
  end

  defp fixed_rate_acceptance_decision(state) do
    now = DateTime.utc_now()
    time_window = 1000  # 1 second
    
    recent_tokens = count_recent_tokens(state.metrics, now, time_window)
    recent_tokens < state.flow_control.rate_limit_tokens_per_sec
  end

  defp token_bucket_acceptance_decision(state) do
    # Implement token bucket algorithm
    now = DateTime.utc_now()
    time_diff = DateTime.diff(now, state.flow_control.last_adjustment, :millisecond)
    
    tokens_to_add = min(
      state.flow_control.burst_capacity,
      (time_diff / 1000) * state.flow_control.rate_limit_tokens_per_sec
    )
    
    tokens_to_add >= 1
  end

  defp process_token(token_data, state) do
    # Add to buffer with priority if enabled
    updated_buffer = if state.config.priority_queue_enabled do
      add_to_priority_queue(state.priority_queue, token_data)
    else
      :queue.in(token_data, state.buffer)
    end
    
    # Distribute to consumers
    distribute_to_consumers(token_data, state.consumers)
    
    # Update metrics
    updated_metrics = increment_token_metrics(state.metrics, token_data)
    
    %{state | 
      buffer: updated_buffer,
      metrics: updated_metrics
    }
  end

  defp distribute_to_consumers(token_data, consumers) do
    Enum.each(consumers, fn {consumer_pid, consumer_config} ->
      case consumer_config.flow_control do
        true ->
          if should_send_to_consumer(consumer_config) do
            send_token_to_consumer(consumer_pid, token_data)
          end
        false ->
          send_token_to_consumer(consumer_pid, token_data)
      end
    end)
  end

  defp should_send_to_consumer(consumer_config) do
    # Check consumer lag and buffer size
    consumer_config.lag < consumer_config.buffer_size
  end

  defp send_token_to_consumer(consumer_pid, token_data) do
    send(consumer_pid, {:stream_token, token_data})
  end

  defp apply_backpressure(state) do
    # Implement sophisticated backpressure strategies
    case state.config.flow_control_strategy do
      :adaptive ->
        apply_adaptive_backpressure(state)
      :drop_tail ->
        apply_drop_tail_backpressure(state)
      :drop_head ->
        apply_drop_head_backpressure(state)
      :circuit_breaker ->
        apply_circuit_breaker_backpressure(state)
    end
  end

  defp apply_adaptive_backpressure(state) do
    # Reduce flow control window
    updated_flow_control = %{state.flow_control | 
      window_size: max(state.flow_control.window_size * 0.8, 100),
      congestion_detected: true
    }
    
    # Update metrics
    updated_metrics = %{state.metrics | 
      backpressure_events: state.metrics.backpressure_events + 1
    }
    
    %{state | 
      flow_control: updated_flow_control,
      metrics: updated_metrics
    }
  end

  defp apply_drop_tail_backpressure(state) do
    # Drop oldest items from buffer
    {_dropped, updated_buffer} = drop_from_tail(state.buffer, 10)
    %{state | buffer: updated_buffer}
  end

  defp apply_drop_head_backpressure(state) do
    # Drop newest items from buffer  
    {_dropped, updated_buffer} = drop_from_head(state.buffer, 10)
    %{state | buffer: updated_buffer}
  end

  defp apply_circuit_breaker_backpressure(state) do
    # Implement circuit breaker pattern
    updated_circuit_breaker = case state.circuit_breaker.state do
      :closed ->
        %{state.circuit_breaker | 
          state: :open,
          failure_count: state.circuit_breaker.failure_count + 1,
          last_failure: DateTime.utc_now()
        }
      :open ->
        state.circuit_breaker
      :half_open ->
        %{state.circuit_breaker | state: :open}
    end
    
    %{state | circuit_breaker: updated_circuit_breaker}
  end

  defp perform_flow_control_adjustment(state) do
    case state.flow_control.strategy do
      :adaptive ->
        perform_adaptive_adjustment(state)
      _ ->
        state
    end
  end

  defp perform_adaptive_adjustment(state) do
    # Implement AIMD (Additive Increase Multiplicative Decrease)
    current_utilization = calculate_buffer_utilization(state)
    target_utilization = 0.7
    
    updated_flow_control = if current_utilization < target_utilization do
      # Additive increase
      %{state.flow_control |
        window_size: state.flow_control.window_size + 10,
        congestion_detected: false
      }
    else
      # Multiplicative decrease
      %{state.flow_control |
        window_size: max(state.flow_control.window_size * 0.5, 100),
        congestion_detected: true
      }
    end
    
    %{state | flow_control: updated_flow_control}
  end

  defp process_consumer_ack(consumer_pid, ack_data, state) do
    case Map.get(state.consumers, consumer_pid) do
      nil ->
        state
      consumer_config ->
        updated_consumer_config = %{consumer_config |
          last_ack: DateTime.utc_now(),
          lag: max(consumer_config.lag - 1, 0)
        }
        
        updated_consumers = Map.put(state.consumers, consumer_pid, updated_consumer_config)
        %{state | consumers: updated_consumers}
    end
  end

  defp handle_producer_error(error, state) do
    Logger.error("Producer error: #{inspect(error)}")
    
    # Update error state
    updated_error_state = %{
      consecutive_errors: state.error_state.consecutive_errors + 1,
      last_error: error
    }
    
    # Apply error recovery strategy
    case state.config.error_recovery_strategy do
      :retry_with_backoff ->
        schedule_retry(state)
      :circuit_breaker ->
        trigger_circuit_breaker(state)
      :graceful_degradation ->
        apply_graceful_degradation(state)
    end
    
    %{state | error_state: updated_error_state}
  end

  # Utility Functions

  defp init_priority_queue do
    %{
      high: :queue.new(),
      normal: :queue.new(),
      low: :queue.new()
    }
  end

  defp add_to_priority_queue(priority_queue, token_data) do
    priority = Map.get(token_data, :priority, :normal)
    queue = Map.get(priority_queue, priority)
    updated_queue = :queue.in(token_data, queue)
    Map.put(priority_queue, priority, updated_queue)
  end

  defp calculate_buffer_utilization(state) do
    if state.config.priority_queue_enabled do
      total_items = Enum.reduce(state.priority_queue, 0, fn {_priority, queue}, acc ->
        acc + :queue.len(queue)
      end)
      total_items / state.config.max_buffer_size
    else
      :queue.len(state.buffer) / state.config.max_buffer_size
    end
  end

  defp calculate_current_throughput(state) do
    now = DateTime.utc_now()
    time_window = 5000  # 5 seconds
    
    recent_samples = Enum.filter(state.metrics.throughput_samples, fn sample ->
      DateTime.diff(now, sample.timestamp, :millisecond) <= time_window
    end)
    
    if length(recent_samples) > 0 do
      total_tokens = Enum.sum(Enum.map(recent_samples, & &1.tokens))
      total_tokens / (time_window / 1000)
    else
      0.0
    end
  end

  defp count_recent_tokens(metrics, now, time_window_ms) do
    Enum.count(metrics.throughput_samples, fn sample ->
      DateTime.diff(now, sample.timestamp, :millisecond) <= time_window_ms
    end)
  end

  defp increment_token_metrics(metrics, token_data) do
    now = DateTime.utc_now()
    token_size = byte_size(token_data.content)
    
    %{metrics |
      tokens_processed: metrics.tokens_processed + 1,
      bytes_transferred: metrics.bytes_transferred + token_size,
      throughput_samples: [
        %{tokens: 1, bytes: token_size, timestamp: now} |
        Enum.take(metrics.throughput_samples, 999)
      ],
      timestamp_last_update: now
    }
  end

  defp enrich_metrics(metrics, state) do
    now = DateTime.utc_now()
    uptime_seconds = DateTime.diff(now, metrics.timestamp_start, :second)
    
    Map.merge(metrics, %{
      uptime_seconds: uptime_seconds,
      current_throughput: calculate_current_throughput(state),
      buffer_utilization: calculate_buffer_utilization(state),
      active_consumers: map_size(state.consumers),
      flow_control_state: state.flow_control,
      circuit_breaker_state: state.circuit_breaker.state
    })
  end

  defp update_flow_control(flow_control, parameters) do
    Enum.reduce(parameters, flow_control, fn {key, value}, acc ->
      case key do
        :rate_limit -> %{acc | rate_limit_tokens_per_sec: value}
        :window_size -> %{acc | window_size: value}
        :burst_capacity -> %{acc | burst_capacity: value}
        _ -> acc
      end
    end)
  end

  defp update_metrics(metrics, state) do
    now = DateTime.utc_now()
    
    # Clean old samples (keep last 1000)
    recent_samples = Enum.take(metrics.throughput_samples, 1000)
    
    %{metrics |
      throughput_samples: recent_samples,
      buffer_utilization: calculate_buffer_utilization(state),
      timestamp_last_update: now
    }
  end

  defp drop_from_tail(queue, count) do
    # Helper to drop items from tail of queue
    {dropped, remaining} = drop_items(queue, count, [])
    {dropped, remaining}
  end

  defp drop_from_head(queue, count) do
    # Helper to drop items from head of queue
    case :queue.out(queue) do
      {:empty, queue} -> {[], queue}
      {{:value, item}, rest} when count > 0 ->
        {dropped, final_queue} = drop_from_head(rest, count - 1)
        {[item | dropped], final_queue}
      _ -> {[], queue}
    end
  end

  defp drop_items(queue, 0, acc), do: {Enum.reverse(acc), queue}
  defp drop_items(queue, count, acc) when count > 0 do
    case :queue.out_r(queue) do
      {:empty, queue} -> {Enum.reverse(acc), queue}
      {{:value, item}, rest} -> drop_items(rest, count - 1, [item | acc])
    end
  end

  defp schedule_retry(state) do
    backoff_time = calculate_backoff_time(state.error_state.consecutive_errors)
    Process.send_after(self(), :retry_producer, backoff_time)
  end

  defp calculate_backoff_time(consecutive_errors) do
    min(1000 * :math.pow(2, consecutive_errors), 30_000) |> round()
  end

  defp trigger_circuit_breaker(state) do
    # Circuit breaker logic would be implemented here
    Logger.warn("Circuit breaker triggered for stream: #{state.config.stream_id}")
  end

  defp apply_graceful_degradation(state) do
    # Graceful degradation logic would be implemented here
    Logger.info("Applying graceful degradation for stream: #{state.config.stream_id}")
  end

  defp schedule_flow_control_tick do
    Process.send_after(self(), :flow_control_tick, 1000)
  end

  defp schedule_metrics_update do
    Process.send_after(self(), :metrics_update, 5000)
  end
end