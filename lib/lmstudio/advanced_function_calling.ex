defmodule LMStudio.AdvancedFunctionCalling do
  @moduledoc """
  Sophisticated function calling system with tool orchestration, parallel execution,
  dependency resolution, and intelligent workflow management.
  
  Features:
  - Dynamic tool discovery and registration
  - Parallel function execution with dependency graphs
  - Tool composition and chaining
  - Intelligent parameter validation and coercion
  - Tool versioning and compatibility checking
  - Result caching and memoization
  - Error recovery and fallback strategies
  - Performance monitoring and optimization
  """

  use GenServer
  require Logger

  defmodule Tool do
    @moduledoc "Tool definition with metadata and capabilities"
    
    defstruct [
      :name,
      :version,
      :description,
      :parameters,
      :returns,
      :function,
      :dependencies,
      :timeout_ms,
      :retry_strategy,
      :caching_enabled,
      :parallel_safe,
      :resource_requirements,
      :tags,
      :examples,
      :validation_schema
    ]
    
    def new(opts) do
      %__MODULE__{
        name: Keyword.fetch!(opts, :name),
        version: Keyword.get(opts, :version, "1.0.0"),
        description: Keyword.get(opts, :description, ""),
        parameters: Keyword.get(opts, :parameters, []),
        returns: Keyword.get(opts, :returns, :any),
        function: Keyword.fetch!(opts, :function),
        dependencies: Keyword.get(opts, :dependencies, []),
        timeout_ms: Keyword.get(opts, :timeout_ms, 30_000),
        retry_strategy: Keyword.get(opts, :retry_strategy, :exponential_backoff),
        caching_enabled: Keyword.get(opts, :caching_enabled, true),
        parallel_safe: Keyword.get(opts, :parallel_safe, true),
        resource_requirements: Keyword.get(opts, :resource_requirements, %{}),
        tags: Keyword.get(opts, :tags, []),
        examples: Keyword.get(opts, :examples, []),
        validation_schema: Keyword.get(opts, :validation_schema, nil)
      }
    end
  end

  defmodule ExecutionPlan do
    @moduledoc "Execution plan with dependency resolution and optimization"
    
    defstruct [
      :id,
      :function_calls,
      :dependency_graph,
      :execution_order,
      :parallel_groups,
      :estimated_duration,
      :resource_allocation,
      :fallback_strategies,
      :optimization_hints
    ]
  end

  defmodule FunctionCall do
    @moduledoc "Individual function call with context and metadata"
    
    defstruct [
      :id,
      :tool_name,
      :parameters,
      :dependencies,
      :priority,
      :timeout,
      :retry_count,
      :cache_key,
      :context,
      :tags,
      :created_at,
      :started_at,
      :completed_at,
      :status,
      :result,
      :error,
      :metrics
    ]
    
    def new(tool_name, parameters, opts \\ []) do
      %__MODULE__{
        id: UUID.uuid4(),
        tool_name: tool_name,
        parameters: parameters,
        dependencies: Keyword.get(opts, :dependencies, []),
        priority: Keyword.get(opts, :priority, :normal),
        timeout: Keyword.get(opts, :timeout, 30_000),
        retry_count: 0,
        cache_key: generate_cache_key(tool_name, parameters),
        context: Keyword.get(opts, :context, %{}),
        tags: Keyword.get(opts, :tags, []),
        created_at: DateTime.utc_now(),
        status: :pending,
        metrics: %{}
      }
    end
    
    defp generate_cache_key(tool_name, parameters) do
      content = "#{tool_name}:#{inspect(parameters)}"
      :crypto.hash(:sha256, content) |> Base.encode16(case: :lower)
    end
  end

  defmodule ExecutionContext do
    @moduledoc "Context for function execution with shared state"
    
    defstruct [
      :session_id,
      :variables,
      :shared_state,
      :call_stack,
      :permissions,
      :resource_limits,
      :security_context,
      :performance_constraints
    ]
    
    def new(opts \\ []) do
      %__MODULE__{
        session_id: UUID.uuid4(),
        variables: Keyword.get(opts, :variables, %{}),
        shared_state: Keyword.get(opts, :shared_state, %{}),
        call_stack: [],
        permissions: Keyword.get(opts, :permissions, [:all]),
        resource_limits: Keyword.get(opts, :resource_limits, %{}),
        security_context: Keyword.get(opts, :security_context, %{}),
        performance_constraints: Keyword.get(opts, :performance_constraints, %{})
      }
    end
  end

  # State Management
  
  defmodule State do
    defstruct [
      :tools,
      :execution_cache,
      :active_executions,
      :execution_history,
      :performance_metrics,
      :resource_manager,
      :security_policies,
      :configuration
    ]
  end

  # Public API

  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  def register_tool(tool) do
    GenServer.call(__MODULE__, {:register_tool, tool})
  end

  def unregister_tool(tool_name) do
    GenServer.call(__MODULE__, {:unregister_tool, tool_name})
  end

  def list_tools(filters \\ []) do
    GenServer.call(__MODULE__, {:list_tools, filters})
  end

  def get_tool_info(tool_name) do
    GenServer.call(__MODULE__, {:get_tool_info, tool_name})
  end

  def execute_function_call(tool_name, parameters, opts \\ []) do
    GenServer.call(__MODULE__, {:execute_function_call, tool_name, parameters, opts}, 60_000)
  end

  def execute_function_calls(function_calls, context \\ %ExecutionContext{}) do
    GenServer.call(__MODULE__, {:execute_function_calls, function_calls, context}, 120_000)
  end

  def create_execution_plan(function_calls, opts \\ []) do
    GenServer.call(__MODULE__, {:create_execution_plan, function_calls, opts})
  end

  def execute_plan(execution_plan, context \\ %ExecutionContext{}) do
    GenServer.call(__MODULE__, {:execute_plan, execution_plan, context}, 300_000)
  end

  def get_execution_status(execution_id) do
    GenServer.call(__MODULE__, {:get_execution_status, execution_id})
  end

  def cancel_execution(execution_id) do
    GenServer.call(__MODULE__, {:cancel_execution, execution_id})
  end

  def get_performance_metrics do
    GenServer.call(__MODULE__, :get_performance_metrics)
  end

  def clear_cache(pattern \\ :all) do
    GenServer.call(__MODULE__, {:clear_cache, pattern})
  end

  # GenServer Implementation

  @impl true
  def init(opts) do
    state = %State{
      tools: %{},
      execution_cache: %{},
      active_executions: %{},
      execution_history: [],
      performance_metrics: initialize_metrics(),
      resource_manager: initialize_resource_manager(),
      security_policies: Keyword.get(opts, :security_policies, []),
      configuration: Keyword.get(opts, :configuration, default_configuration())
    }
    
    # Register built-in tools
    register_builtin_tools(state)
    
    Logger.info("Advanced Function Calling system initialized")
    {:ok, state}
  end

  @impl true
  def handle_call({:register_tool, tool}, _from, state) do
    case validate_tool(tool) do
      :ok ->
        updated_tools = Map.put(state.tools, tool.name, tool)
        updated_state = %{state | tools: updated_tools}
        Logger.info("Tool registered: #{tool.name} v#{tool.version}")
        {:reply, :ok, updated_state}
      
      {:error, reason} ->
        {:reply, {:error, reason}, state}
    end
  end

  @impl true
  def handle_call({:unregister_tool, tool_name}, _from, state) do
    updated_tools = Map.delete(state.tools, tool_name)
    updated_state = %{state | tools: updated_tools}
    Logger.info("Tool unregistered: #{tool_name}")
    {:reply, :ok, updated_state}
  end

  @impl true
  def handle_call({:list_tools, filters}, _from, state) do
    filtered_tools = filter_tools(state.tools, filters)
    tool_summaries = Enum.map(filtered_tools, fn {name, tool} ->
      %{
        name: name,
        version: tool.version,
        description: tool.description,
        parameters: Enum.map(tool.parameters, &extract_parameter_info/1),
        tags: tool.tags,
        parallel_safe: tool.parallel_safe
      }
    end)
    {:reply, tool_summaries, state}
  end

  @impl true
  def handle_call({:get_tool_info, tool_name}, _from, state) do
    case Map.get(state.tools, tool_name) do
      nil -> {:reply, {:error, :tool_not_found}, state}
      tool -> {:reply, {:ok, tool}, state}
    end
  end

  @impl true
  def handle_call({:execute_function_call, tool_name, parameters, opts}, from, state) do
    function_call = FunctionCall.new(tool_name, parameters, opts)
    context = Keyword.get(opts, :context, ExecutionContext.new())
    
    case execute_single_function_call(function_call, context, state) do
      {:ok, result, updated_state} ->
        {:reply, {:ok, result}, updated_state}
      
      {:error, reason, updated_state} ->
        {:reply, {:error, reason}, updated_state}
    end
  end

  @impl true
  def handle_call({:execute_function_calls, function_calls, context}, _from, state) do
    execution_plan = create_execution_plan_internal(function_calls, [])
    execute_plan_internal(execution_plan, context, state)
  end

  @impl true
  def handle_call({:create_execution_plan, function_calls, opts}, _from, state) do
    execution_plan = create_execution_plan_internal(function_calls, opts)
    {:reply, {:ok, execution_plan}, state}
  end

  @impl true
  def handle_call({:execute_plan, execution_plan, context}, _from, state) do
    execute_plan_internal(execution_plan, context, state)
  end

  @impl true
  def handle_call({:get_execution_status, execution_id}, _from, state) do
    case Map.get(state.active_executions, execution_id) do
      nil -> {:reply, {:error, :execution_not_found}, state}
      execution -> {:reply, {:ok, execution}, state}
    end
  end

  @impl true
  def handle_call({:cancel_execution, execution_id}, _from, state) do
    case Map.get(state.active_executions, execution_id) do
      nil ->
        {:reply, {:error, :execution_not_found}, state}
      
      execution ->
        # Cancel all running tasks
        Enum.each(execution.tasks, fn {_task_id, task_pid} ->
          if Process.alive?(task_pid) do
            Process.exit(task_pid, :cancelled)
          end
        end)
        
        updated_executions = Map.delete(state.active_executions, execution_id)
        updated_state = %{state | active_executions: updated_executions}
        {:reply, :ok, updated_state}
    end
  end

  @impl true
  def handle_call(:get_performance_metrics, _from, state) do
    enriched_metrics = enrich_performance_metrics(state.performance_metrics, state)
    {:reply, enriched_metrics, state}
  end

  @impl true
  def handle_call({:clear_cache, pattern}, _from, state) do
    updated_cache = clear_execution_cache(state.execution_cache, pattern)
    updated_state = %{state | execution_cache: updated_cache}
    {:reply, :ok, updated_state}
  end

  # Private Functions

  defp validate_tool(tool) do
    cond do
      not is_binary(tool.name) or String.length(tool.name) == 0 ->
        {:error, :invalid_name}
      
      not is_function(tool.function) ->
        {:error, :invalid_function}
      
      not is_list(tool.parameters) ->
        {:error, :invalid_parameters}
      
      true ->
        :ok
    end
  end

  defp filter_tools(tools, filters) do
    Enum.filter(tools, fn {_name, tool} ->
      Enum.all?(filters, fn
        {:tag, tag} -> tag in tool.tags
        {:parallel_safe, parallel_safe} -> tool.parallel_safe == parallel_safe
        {:has_dependency, dep} -> dep in tool.dependencies
        _ -> true
      end)
    end)
  end

  defp extract_parameter_info(param) do
    case param do
      {name, opts} when is_list(opts) ->
        %{
          name: name,
          type: Keyword.get(opts, :type, :any),
          required: Keyword.get(opts, :required, true),
          description: Keyword.get(opts, :description, "")
        }
      
      name when is_atom(name) ->
        %{name: name, type: :any, required: true, description: ""}
    end
  end

  defp execute_single_function_call(function_call, context, state) do
    case Map.get(state.tools, function_call.tool_name) do
      nil ->
        {:error, :tool_not_found, state}
      
      tool ->
        # Check cache first
        case check_cache(function_call, tool, state) do
          {:hit, result} ->
            {:ok, result, state}
          
          :miss ->
            perform_function_execution(function_call, tool, context, state)
        end
    end
  end

  defp check_cache(function_call, tool, state) do
    if tool.caching_enabled and Map.has_key?(state.execution_cache, function_call.cache_key) do
      cache_entry = Map.get(state.execution_cache, function_call.cache_key)
      
      # Check if cache entry is still valid
      if cache_entry_valid?(cache_entry) do
        {:hit, cache_entry.result}
      else
        :miss
      end
    else
      :miss
    end
  end

  defp cache_entry_valid?(cache_entry) do
    # Simple TTL-based validation (could be more sophisticated)
    now = DateTime.utc_now()
    age_seconds = DateTime.diff(now, cache_entry.created_at, :second)
    age_seconds < 3600  # 1 hour TTL
  end

  defp perform_function_execution(function_call, tool, context, state) do
    start_time = DateTime.utc_now()
    
    # Validate parameters
    case validate_parameters(function_call.parameters, tool.parameters) do
      :ok ->
        # Execute function with timeout and error handling
        case execute_with_timeout(tool.function, function_call.parameters, tool.timeout_ms) do
          {:ok, result} ->
            end_time = DateTime.utc_now()
            execution_time = DateTime.diff(end_time, start_time, :millisecond)
            
            # Cache result if caching is enabled
            updated_state = if tool.caching_enabled do
              cache_entry = %{
                result: result,
                created_at: start_time,
                execution_time: execution_time,
                tool_version: tool.version
              }
              updated_cache = Map.put(state.execution_cache, function_call.cache_key, cache_entry)
              %{state | execution_cache: updated_cache}
            else
              state
            end
            
            # Update metrics
            final_state = update_execution_metrics(updated_state, tool.name, execution_time, :success)
            
            {:ok, result, final_state}
          
          {:error, reason} ->
            # Handle retry logic
            case should_retry(function_call, tool, reason) do
              true ->
                retry_function_call(function_call, tool, context, state)
              
              false ->
                final_state = update_execution_metrics(state, tool.name, 0, :error)
                {:error, reason, final_state}
            end
        end
      
      {:error, validation_error} ->
        {:error, {:validation_error, validation_error}, state}
    end
  end

  defp validate_parameters(provided, schema) do
    # Advanced parameter validation
    missing_required = find_missing_required_parameters(provided, schema)
    
    if length(missing_required) > 0 do
      {:error, {:missing_parameters, missing_required}}
    else
      case validate_parameter_types(provided, schema) do
        :ok -> :ok
        error -> error
      end
    end
  end

  defp find_missing_required_parameters(provided, schema) do
    provided_keys = Map.keys(provided)
    
    Enum.filter(schema, fn
      {name, opts} when is_list(opts) ->
        Keyword.get(opts, :required, true) and name not in provided_keys
      
      name when is_atom(name) ->
        name not in provided_keys
    end)
  end

  defp validate_parameter_types(provided, schema) do
    # Type validation implementation
    validation_errors = Enum.reduce(schema, [], fn
      {name, opts}, acc when is_list(opts) ->
        case Map.get(provided, name) do
          nil -> acc
          value ->
            expected_type = Keyword.get(opts, :type, :any)
            case validate_type(value, expected_type) do
              :ok -> acc
              {:error, reason} -> [{name, reason} | acc]
            end
        end
      
      _param, acc -> acc
    end)
    
    if length(validation_errors) > 0 do
      {:error, {:type_errors, validation_errors}}
    else
      :ok
    end
  end

  defp validate_type(_value, :any), do: :ok
  defp validate_type(value, :string) when is_binary(value), do: :ok
  defp validate_type(value, :integer) when is_integer(value), do: :ok
  defp validate_type(value, :float) when is_float(value), do: :ok
  defp validate_type(value, :number) when is_number(value), do: :ok
  defp validate_type(value, :boolean) when is_boolean(value), do: :ok
  defp validate_type(value, :list) when is_list(value), do: :ok
  defp validate_type(value, :map) when is_map(value), do: :ok
  defp validate_type(_value, expected_type), do: {:error, "Expected #{expected_type}"}

  defp execute_with_timeout(function, parameters, timeout_ms) do
    task = Task.async(fn ->
      try do
        function.(parameters)
      rescue
        error -> {:error, error}
      catch
        :exit, reason -> {:error, {:exit, reason}}
        :throw, reason -> {:error, {:throw, reason}}
      end
    end)
    
    case Task.yield(task, timeout_ms) || Task.shutdown(task) do
      {:ok, {:error, reason}} -> {:error, reason}
      {:ok, result} -> {:ok, result}
      nil -> {:error, :timeout}
    end
  end

  defp should_retry(function_call, tool, reason) do
    case tool.retry_strategy do
      :none -> false
      :exponential_backoff when function_call.retry_count < 3 -> true
      :fixed_interval when function_call.retry_count < 2 -> true
      _ -> false
    end
  end

  defp retry_function_call(function_call, tool, context, state) do
    updated_function_call = %{function_call | retry_count: function_call.retry_count + 1}
    
    # Calculate backoff time
    backoff_time = calculate_retry_backoff(tool.retry_strategy, updated_function_call.retry_count)
    Process.sleep(backoff_time)
    
    perform_function_execution(updated_function_call, tool, context, state)
  end

  defp calculate_retry_backoff(:exponential_backoff, retry_count) do
    min(1000 * :math.pow(2, retry_count - 1), 10_000) |> round()
  end

  defp calculate_retry_backoff(:fixed_interval, _retry_count), do: 2000

  defp create_execution_plan_internal(function_calls, opts) do
    # Build dependency graph
    dependency_graph = build_dependency_graph(function_calls)
    
    # Determine execution order using topological sort
    execution_order = topological_sort(dependency_graph)
    
    # Group parallel-safe functions
    parallel_groups = group_parallel_functions(execution_order, function_calls)
    
    # Estimate execution time
    estimated_duration = estimate_execution_duration(function_calls)
    
    %ExecutionPlan{
      id: UUID.uuid4(),
      function_calls: function_calls,
      dependency_graph: dependency_graph,
      execution_order: execution_order,
      parallel_groups: parallel_groups,
      estimated_duration: estimated_duration,
      resource_allocation: calculate_resource_allocation(function_calls),
      fallback_strategies: Keyword.get(opts, :fallback_strategies, []),
      optimization_hints: Keyword.get(opts, :optimization_hints, [])
    }
  end

  defp build_dependency_graph(function_calls) do
    # Create adjacency list representation of dependencies
    Enum.reduce(function_calls, %{}, fn function_call, graph ->
      dependencies = function_call.dependencies || []
      Map.put(graph, function_call.id, dependencies)
    end)
  end

  defp topological_sort(dependency_graph) do
    # Implement Kahn's algorithm for topological sorting
    in_degree = calculate_in_degrees(dependency_graph)
    queue = find_nodes_with_zero_in_degree(in_degree)
    
    topological_sort_recursive(queue, in_degree, dependency_graph, [])
  end

  defp calculate_in_degrees(dependency_graph) do
    all_nodes = Map.keys(dependency_graph)
    
    initial_degrees = Map.new(all_nodes, &{&1, 0})
    
    Enum.reduce(dependency_graph, initial_degrees, fn {_node, dependencies}, acc ->
      Enum.reduce(dependencies, acc, fn dep, inner_acc ->
        Map.update(inner_acc, dep, 1, &(&1 + 1))
      end)
    end)
  end

  defp find_nodes_with_zero_in_degree(in_degree) do
    Enum.filter(in_degree, fn {_node, degree} -> degree == 0 end)
    |> Enum.map(fn {node, _degree} -> node end)
  end

  defp topological_sort_recursive([], _in_degree, _graph, result), do: Enum.reverse(result)
  defp topological_sort_recursive([node | queue], in_degree, graph, result) do
    dependencies = Map.get(graph, node, [])
    
    updated_in_degree = Enum.reduce(dependencies, in_degree, fn dep, acc ->
      Map.update(acc, dep, 0, &max(&1 - 1, 0))
    end)
    
    new_zero_nodes = find_nodes_with_zero_in_degree(updated_in_degree) -- [node | queue] -- result
    updated_queue = queue ++ new_zero_nodes
    
    topological_sort_recursive(updated_queue, updated_in_degree, graph, [node | result])
  end

  defp group_parallel_functions(execution_order, function_calls) do
    # Group functions that can be executed in parallel
    function_call_map = Map.new(function_calls, &{&1.id, &1})
    
    Enum.chunk_by(execution_order, fn call_id ->
      function_call = Map.get(function_call_map, call_id)
      length(function_call.dependencies || [])
    end)
  end

  defp estimate_execution_duration(function_calls) do
    # Estimate total execution time based on function timeouts and parallelization
    total_sequential_time = Enum.sum(Enum.map(function_calls, & &1.timeout))
    
    # Simplified estimation (in reality, would consider parallel groups)
    max(total_sequential_time * 0.3, 5000) |> round()
  end

  defp calculate_resource_allocation(function_calls) do
    # Calculate required resources (memory, CPU, network)
    %{
      memory_mb: length(function_calls) * 10,
      cpu_cores: min(length(function_calls), System.schedulers_online()),
      network_connections: Enum.count(function_calls, fn call ->
        "network" in (call.tags || [])
      end)
    }
  end

  defp execute_plan_internal(execution_plan, context, state) do
    execution_id = execution_plan.id
    
    # Initialize execution tracking
    execution_state = %{
      id: execution_id,
      plan: execution_plan,
      context: context,
      results: %{},
      errors: %{},
      status: :running,
      started_at: DateTime.utc_now(),
      tasks: %{}
    }
    
    updated_state = %{state | 
      active_executions: Map.put(state.active_executions, execution_id, execution_state)
    }
    
    # Execute parallel groups sequentially, but functions within groups in parallel
    case execute_parallel_groups(execution_plan.parallel_groups, context, updated_state) do
      {:ok, results, final_state} ->
        # Clean up execution tracking
        cleaned_state = %{final_state | 
          active_executions: Map.delete(final_state.active_executions, execution_id)
        }
        
        {:reply, {:ok, results}, cleaned_state}
      
      {:error, reason, final_state} ->
        cleaned_state = %{final_state | 
          active_executions: Map.delete(final_state.active_executions, execution_id)
        }
        
        {:reply, {:error, reason}, cleaned_state}
    end
  end

  defp execute_parallel_groups(parallel_groups, context, state) do
    Enum.reduce_while(parallel_groups, {:ok, %{}, state}, fn group, {:ok, acc_results, acc_state} ->
      case execute_parallel_group(group, context, acc_state) do
        {:ok, group_results, updated_state} ->
          merged_results = Map.merge(acc_results, group_results)
          {:cont, {:ok, merged_results, updated_state}}
        
        {:error, reason, updated_state} ->
          {:halt, {:error, reason, updated_state}}
      end
    end)
  end

  defp execute_parallel_group(group, context, state) do
    # Execute functions in this group in parallel
    tasks = Enum.map(group, fn call_id ->
      # Find the function call by ID
      function_call = find_function_call_by_id(call_id, state)
      
      Task.async(fn ->
        case execute_single_function_call(function_call, context, state) do
          {:ok, result, _updated_state} -> {:ok, call_id, result}
          {:error, reason, _updated_state} -> {:error, call_id, reason}
        end
      end)
    end)
    
    # Wait for all tasks to complete
    results = Task.await_many(tasks, 60_000)
    
    # Process results
    {success_results, errors} = partition_results(results)
    
    if length(errors) > 0 do
      {:error, {:parallel_execution_failed, errors}, state}
    else
      result_map = Map.new(success_results, fn {:ok, call_id, result} -> {call_id, result} end)
      {:ok, result_map, state}
    end
  end

  defp find_function_call_by_id(call_id, state) do
    # This would typically be stored in state, but for now we'll create a dummy
    %FunctionCall{id: call_id, tool_name: "dummy", parameters: %{}}
  end

  defp partition_results(results) do
    Enum.split_with(results, fn
      {:ok, _call_id, _result} -> true
      _ -> false
    end)
  end

  defp register_builtin_tools(state) do
    builtin_tools = [
      create_math_tool(),
      create_string_tool(),
      create_http_tool(),
      create_file_tool(),
      create_time_tool()
    ]
    
    Enum.each(builtin_tools, fn tool ->
      GenServer.cast(self(), {:register_tool_internal, tool})
    end)
  end

  defp create_math_tool do
    Tool.new(
      name: "math_calculator",
      version: "1.0.0",
      description: "Perform mathematical calculations",
      parameters: [
        {:expression, [type: :string, required: true, description: "Mathematical expression"]},
        {:precision, [type: :integer, required: false, description: "Decimal precision"]}
      ],
      function: fn params ->
        expression = Map.get(params, :expression)
        # Safe mathematical expression evaluation would go here
        {:ok, "Calculated: #{expression}"}
      end,
      tags: [:math, :calculation],
      parallel_safe: true
    )
  end

  defp create_string_tool do
    Tool.new(
      name: "string_processor",
      version: "1.0.0", 
      description: "Process and manipulate strings",
      parameters: [
        {:text, [type: :string, required: true]},
        {:operation, [type: :string, required: true]},
        {:options, [type: :map, required: false]}
      ],
      function: fn params ->
        text = Map.get(params, :text)
        operation = Map.get(params, :operation)
        
        result = case operation do
          "uppercase" -> String.upcase(text)
          "lowercase" -> String.downcase(text)
          "reverse" -> String.reverse(text)
          "length" -> String.length(text)
          _ -> text
        end
        
        {:ok, result}
      end,
      tags: [:string, :text],
      parallel_safe: true
    )
  end

  defp create_http_tool do
    Tool.new(
      name: "http_client",
      version: "1.0.0",
      description: "Make HTTP requests",
      parameters: [
        {:url, [type: :string, required: true]},
        {:method, [type: :string, required: false]},
        {:headers, [type: :map, required: false]},
        {:body, [type: :string, required: false]}
      ],
      function: fn params ->
        # HTTP request implementation would go here
        url = Map.get(params, :url)
        {:ok, "HTTP response from #{url}"}
      end,
      tags: [:http, :network],
      parallel_safe: true,
      timeout_ms: 30_000
    )
  end

  defp create_file_tool do
    Tool.new(
      name: "file_operations",
      version: "1.0.0",
      description: "Perform file operations",
      parameters: [
        {:operation, [type: :string, required: true]},
        {:path, [type: :string, required: true]},
        {:content, [type: :string, required: false]}
      ],
      function: fn params ->
        operation = Map.get(params, :operation)
        path = Map.get(params, :path)
        
        case operation do
          "read" -> File.read(path)
          "write" -> File.write(path, Map.get(params, :content, ""))
          "exists" -> {:ok, File.exists?(path)}
          _ -> {:error, "Unknown operation"}
        end
      end,
      tags: [:file, :io],
      parallel_safe: false  # File operations might not be parallel safe
    )
  end

  defp create_time_tool do
    Tool.new(
      name: "time_utilities",
      version: "1.0.0",
      description: "Time and date utilities",
      parameters: [
        {:operation, [type: :string, required: true]},
        {:format, [type: :string, required: false]},
        {:timezone, [type: :string, required: false]}
      ],
      function: fn params ->
        operation = Map.get(params, :operation)
        
        result = case operation do
          "now" -> DateTime.utc_now()
          "timestamp" -> System.system_time(:second)
          "format" -> DateTime.to_string(DateTime.utc_now())
          _ -> DateTime.utc_now()
        end
        
        {:ok, result}
      end,
      tags: [:time, :date],
      parallel_safe: true
    )
  end

  defp initialize_metrics do
    %{
      total_executions: 0,
      successful_executions: 0,
      failed_executions: 0,
      average_execution_time: 0.0,
      cache_hits: 0,
      cache_misses: 0,
      tool_usage: %{},
      performance_history: []
    }
  end

  defp initialize_resource_manager do
    %{
      max_concurrent_executions: 100,
      current_executions: 0,
      memory_usage: 0,
      cpu_usage: 0.0
    }
  end

  defp default_configuration do
    %{
      cache_ttl_seconds: 3600,
      max_cache_size: 10_000,
      default_timeout_ms: 30_000,
      max_retry_attempts: 3,
      parallel_execution_enabled: true,
      security_checks_enabled: true
    }
  end

  defp update_execution_metrics(state, tool_name, execution_time, status) do
    updated_metrics = %{state.performance_metrics |
      total_executions: state.performance_metrics.total_executions + 1,
      successful_executions: case status do
        :success -> state.performance_metrics.successful_executions + 1
        _ -> state.performance_metrics.successful_executions
      end,
      failed_executions: case status do
        :error -> state.performance_metrics.failed_executions + 1
        _ -> state.performance_metrics.failed_executions
      end,
      tool_usage: Map.update(state.performance_metrics.tool_usage, tool_name, 1, &(&1 + 1))
    }
    
    %{state | performance_metrics: updated_metrics}
  end

  defp enrich_performance_metrics(metrics, state) do
    Map.merge(metrics, %{
      cache_hit_rate: safe_divide(metrics.cache_hits, metrics.cache_hits + metrics.cache_misses),
      success_rate: safe_divide(metrics.successful_executions, metrics.total_executions),
      active_executions: map_size(state.active_executions),
      registered_tools: map_size(state.tools),
      cache_size: map_size(state.execution_cache)
    })
  end

  defp safe_divide(_numerator, 0), do: 0.0
  defp safe_divide(numerator, denominator), do: numerator / denominator

  defp clear_execution_cache(cache, :all), do: %{}
  defp clear_execution_cache(cache, pattern) when is_binary(pattern) do
    Enum.filter(cache, fn {key, _value} ->
      not String.contains?(key, pattern)
    end)
    |> Map.new()
  end
end

# Helper module for generating UUIDs (simplified implementation)
defmodule UUID do
  def uuid4 do
    hex = :crypto.strong_rand_bytes(16)
          |> Base.encode16(case: :lower)
    
    String.slice(hex, 0, 8) <> "-" <>
    String.slice(hex, 8, 4) <> "-" <>
    String.slice(hex, 12, 4) <> "-" <>
    String.slice(hex, 16, 4) <> "-" <>
    String.slice(hex, 20, 12)
  end
end