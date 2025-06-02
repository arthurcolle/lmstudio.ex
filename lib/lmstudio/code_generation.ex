defmodule LMStudio.CodeGeneration do
  @moduledoc """
  Dynamic code generation system that leverages deep Erlang/OTP knowledge.
  
  This module can generate, modify, and evolve Erlang/Elixir code based on:
  - OTP behaviors (GenServer, Supervisor, Application, GenStateMachine, etc.)
  - Functional programming patterns
  - Concurrent programming patterns
  - Fault tolerance patterns
  - Performance optimization patterns
  """
  
  alias LMStudio.Persistence.Helpers, as: PersistenceHelpers
  require Logger
  
  # ========== Code Generation API ==========
  
  def generate_genserver(name, state_structure, opts \\ []) do
    template = load_genserver_template()
    
    generated_code = template
    |> replace_placeholder("MODULE_NAME", name)
    |> replace_placeholder("INITIAL_STATE", format_state(state_structure))
    |> apply_genserver_options(opts)
    
    save_generated_code("genserver_#{String.downcase(name)}", generated_code)
    generated_code
  end
  
  def generate_supervisor(name, children_specs, strategy \\ :one_for_one) do
    template = load_supervisor_template()
    
    generated_code = template
    |> replace_placeholder("MODULE_NAME", name)
    |> replace_placeholder("STRATEGY", inspect(strategy))
    |> replace_placeholder("CHILDREN_SPECS", format_children_specs(children_specs))
    
    save_generated_code("supervisor_#{String.downcase(name)}", generated_code)
    generated_code
  end
  
  def generate_application(name, modules, opts \\ []) do
    template = load_application_template()
    
    generated_code = template
    |> replace_placeholder("APP_NAME", name)
    |> replace_placeholder("MODULES", format_module_list(modules))
    |> apply_application_options(opts)
    
    save_generated_code("application_#{String.downcase(name)}", generated_code)
    generated_code
  end
  
  def generate_agent_pattern(name, initial_value \\ nil) do
    template = load_agent_template()
    
    generated_code = template
    |> replace_placeholder("MODULE_NAME", name)
    |> replace_placeholder("INITIAL_VALUE", inspect(initial_value || %{}))
    
    save_generated_code("agent_#{String.downcase(name)}", generated_code)
    generated_code
  end
  
  def generate_task_supervisor_worker(name, work_function) do
    template = load_task_worker_template()
    
    generated_code = template
    |> replace_placeholder("MODULE_NAME", name)
    |> replace_placeholder("WORK_FUNCTION", work_function)
    
    save_generated_code("task_worker_#{String.downcase(name)}", generated_code)
    generated_code
  end
  
  def generate_registry_worker(name, registry_name) do
    template = load_registry_worker_template()
    
    generated_code = template
    |> replace_placeholder("MODULE_NAME", name)
    |> replace_placeholder("REGISTRY_NAME", registry_name)
    
    save_generated_code("registry_worker_#{String.downcase(name)}", generated_code)
    generated_code
  end
  
  # ========== Advanced Patterns ==========
  
  def generate_state_machine(name, states, transitions) do
    template = load_state_machine_template()
    
    generated_code = template
    |> replace_placeholder("MODULE_NAME", name)
    |> replace_placeholder("STATES", format_states(states))
    |> replace_placeholder("TRANSITIONS", format_transitions(transitions))
    
    save_generated_code("state_machine_#{String.downcase(name)}", generated_code)
    generated_code
  end
  
  def generate_pool_manager(name, worker_module, pool_size \\ 5) do
    template = load_pool_manager_template()
    
    generated_code = template
    |> replace_placeholder("MODULE_NAME", name)
    |> replace_placeholder("WORKER_MODULE", worker_module)
    |> replace_placeholder("POOL_SIZE", inspect(pool_size))
    
    save_generated_code("pool_manager_#{String.downcase(name)}", generated_code)
    generated_code
  end
  
  def generate_distributed_worker(name, cluster_nodes \\ []) do
    template = load_distributed_worker_template()
    
    generated_code = template
    |> replace_placeholder("MODULE_NAME", name)
    |> replace_placeholder("CLUSTER_NODES", inspect(cluster_nodes))
    
    save_generated_code("distributed_worker_#{String.downcase(name)}", generated_code)
    generated_code
  end
  
  # ========== Code Evolution and Optimization ==========
  
  def evolve_code(code_id, evolution_hints \\ []) do
    case load_generated_code(code_id) do
      nil -> {:error, :code_not_found}
      existing_code ->
        evolved_code = apply_evolution_patterns(existing_code, evolution_hints)
        save_generated_code("#{code_id}_evolved_#{System.system_time()}", evolved_code)
        {:ok, evolved_code}
    end
  end
  
  def optimize_for_performance(code) do
    code
    |> add_performance_monitoring()
    |> add_caching_patterns()
    |> add_concurrent_processing()
    |> add_memory_optimization()
  end
  
  def optimize_for_fault_tolerance(code) do
    code
    |> add_error_handling()
    |> add_restart_strategies()
    |> add_circuit_breaker()
    |> add_health_monitoring()
  end
  
  # ========== Templates (Embedded Erlang/OTP Knowledge) ==========
  
  defp load_genserver_template do
    """
    defmodule MODULE_NAME do
      @moduledoc \"\"\"
      Generated GenServer with self-monitoring and evolution capabilities.
      
      Includes:
      - Comprehensive error handling
      - Performance monitoring
      - State persistence
      - Graceful shutdown
      \"\"\"
      
      use GenServer
      require Logger
      alias LMStudio.Persistence.Helpers
      
      @timeout 30_000
      @hibernate_after 60_000
      
      # Client API
      
      def start_link(opts \\\\ []) do
        name = Keyword.get(opts, :name, __MODULE__)
        GenServer.start_link(__MODULE__, opts, name: name, timeout: @timeout)
      end
      
      def get_state(pid) do
        GenServer.call(pid, :get_state)
      end
      
      def update_state(pid, update_fun) when is_function(update_fun) do
        GenServer.call(pid, {:update_state, update_fun})
      end
      
      def async_operation(pid, operation) do
        GenServer.cast(pid, {:async_operation, operation})
      end
      
      def stop(pid) do
        GenServer.stop(pid, :normal, @timeout)
      end
      
      # Server Callbacks
      
      @impl true
      def init(opts) do
        Process.flag(:trap_exit, true)
        
        initial_state = INITIAL_STATE
        
        # Load persisted state if available
        name = Keyword.get(opts, :name, __MODULE__)
        persisted_state = Helpers.load_agent_state(inspect(name))
        
        state = if persisted_state do
          Logger.info("\#{__MODULE__} restored persisted state")
          Map.merge(initial_state, persisted_state)
        else
          initial_state
        end
        
        # Schedule periodic cleanup and persistence
        :timer.send_interval(30_000, :periodic_maintenance)
        :timer.send_interval(5_000, :persist_state)
        
        Logger.info("\#{__MODULE__} initialized successfully")
        {:ok, state, @hibernate_after}
      end
      
      @impl true
      def handle_call(:get_state, _from, state) do
        {:reply, state, state, @hibernate_after}
      end
      
      @impl true
      def handle_call({:update_state, update_fun}, _from, state) do
        try do
          new_state = update_fun.(state)
          {:reply, {:ok, new_state}, new_state, @hibernate_after}
        rescue
          error ->
            Logger.error("State update failed: \#{inspect(error)}")
            {:reply, {:error, error}, state, @hibernate_after}
        end
      end
      
      @impl true
      def handle_cast({:async_operation, operation}, state) do
        try do
          new_state = perform_operation(operation, state)
          {:noreply, new_state, @hibernate_after}
        rescue
          error ->
            Logger.error("Async operation failed: \#{inspect(error)}")
            {:noreply, state, @hibernate_after}
        end
      end
      
      @impl true
      def handle_info(:periodic_maintenance, state) do
        Logger.debug("\#{__MODULE__} performing periodic maintenance")
        
        # Garbage collection
        :erlang.garbage_collect()
        
        # State cleanup if needed
        cleaned_state = cleanup_state(state)
        
        {:noreply, cleaned_state, @hibernate_after}
      end
      
      @impl true
      def handle_info(:persist_state, state) do
        Helpers.save_agent_state(inspect(__MODULE__), state)
        {:noreply, state, @hibernate_after}
      end
      
      @impl true
      def handle_info({:EXIT, _pid, reason}, state) do
        Logger.warning("\#{__MODULE__} received EXIT signal: \#{inspect(reason)}")
        {:noreply, state, @hibernate_after}
      end
      
      @impl true
      def handle_info(msg, state) do
        Logger.debug("\#{__MODULE__} received unexpected message: \#{inspect(msg)}")
        {:noreply, state, @hibernate_after}
      end
      
      @impl true
      def terminate(reason, state) do
        Logger.info("\#{__MODULE__} terminating: \#{inspect(reason)}")
        Helpers.save_agent_state(inspect(__MODULE__), state)
        :ok
      end
      
      # Private Functions
      
      defp perform_operation(operation, state) do
        # Override this function in generated modules
        Logger.debug("Performing operation: \#{inspect(operation)}")
        state
      end
      
      defp cleanup_state(state) do
        # Override this function for state-specific cleanup
        state
      end
    end
    """
  end
  
  defp load_supervisor_template do
    """
    defmodule MODULE_NAME do
      @moduledoc \"\"\"
      Generated Supervisor with advanced fault tolerance patterns.
      
      Features:
      - Configurable restart strategies
      - Child health monitoring
      - Dynamic child management
      - Performance tracking
      \"\"\"
      
      use Supervisor
      require Logger
      
      @restart_intensity 10
      @restart_period 60
      
      def start_link(opts \\\\ []) do
        Supervisor.start_link(__MODULE__, opts, name: __MODULE__)
      end
      
      def add_child(child_spec) do
        Supervisor.start_child(__MODULE__, child_spec)
      end
      
      def remove_child(child_id) do
        case Supervisor.terminate_child(__MODULE__, child_id) do
          :ok -> Supervisor.delete_child(__MODULE__, child_id)
          error -> error
        end
      end
      
      def get_children do
        Supervisor.which_children(__MODULE__)
      end
      
      def get_child_pids do
        Supervisor.which_children(__MODULE__)
        |> Enum.map(fn {_id, pid, _type, _modules} -> pid end)
      end
      
      def restart_child(child_id) do
        case Supervisor.restart_child(__MODULE__, child_id) do
          {:ok, pid} -> 
            Logger.info("Successfully restarted child \#{child_id}")
            {:ok, pid}
          error -> 
            Logger.error("Failed to restart child \#{child_id}: \#{inspect(error)}")
            error
        end
      end
      
      @impl true
      def init(opts) do
        Logger.info("\#{__MODULE__} supervisor starting")
        
        children = CHILDREN_SPECS
        
        supervisor_opts = [
          strategy: STRATEGY,
          max_restarts: Keyword.get(opts, :max_restarts, @restart_intensity),
          max_seconds: Keyword.get(opts, :max_seconds, @restart_period)
        ]
        
        # Schedule health checks
        :timer.send_interval(10_000, self(), :health_check)
        
        Supervisor.init(children, supervisor_opts)
      end
      
      @impl true
      def handle_info(:health_check, state) do
        children = Supervisor.which_children(__MODULE__)
        
        down_children = Enum.filter(children, fn
          {_id, :undefined, _type, _modules} -> true
          {_id, pid, _type, _modules} when is_pid(pid) -> not Process.alive?(pid)
          _ -> false
        end)
        
        if length(down_children) > 0 do
          Logger.warning("\#{__MODULE__} found \#{length(down_children)} unhealthy children")
        end
        
        {:noreply, state}
      end
      
      @impl true
      def handle_info(msg, state) do
        Logger.debug("\#{__MODULE__} received message: \#{inspect(msg)}")
        {:noreply, state}
      end
    end
    """
  end
  
  defp load_application_template do
    """
    defmodule APP_NAME do
      @moduledoc \"\"\"
      Generated OTP Application with comprehensive startup and monitoring.
      
      Features:
      - Environment-based configuration
      - Graceful startup and shutdown
      - Resource monitoring
      - Health endpoints
      \"\"\"
      
      use Application
      require Logger
      
      @app_name :APP_NAME_ATOM
      
      def start(_type, _args) do
        Logger.info("Starting \#{@app_name} application")
        
        # Load configuration
        config = load_configuration()
        Logger.info("Application configuration: \#{inspect(config)}")
        
        children = build_supervision_tree(config)
        
        opts = [strategy: :one_for_one, name: \#{@app_name}.Supervisor]
        
        case Supervisor.start_link(children, opts) do
          {:ok, pid} ->
            Logger.info("\#{@app_name} application started successfully")
            post_startup_tasks()
            {:ok, pid}
          error ->
            Logger.error("Failed to start \#{@app_name} application: \#{inspect(error)}")
            error
        end
      end
      
      def stop(_state) do
        Logger.info("Stopping \#{@app_name} application")
        pre_shutdown_tasks()
        :ok
      end
      
      def health_check do
        %{
          status: :healthy,
          uptime: System.system_time(:second) - get_startup_time(),
          memory_usage: :erlang.memory(),
          process_count: Process.list() |> length(),
          modules: MODULES
        }
      end
      
      defp load_configuration do
        Application.get_all_env(@app_name)
      end
      
      defp build_supervision_tree(config) do
        base_children = [
          # Core services
          {\#{@app_name}.Registry, [keys: :unique, name: \#{@app_name}.Registry]},
          {\#{@app_name}.Telemetry, []},
          
          # Application-specific children
          # Add your children here
        ]
        
        # Add conditional children based on configuration
        children = if Keyword.get(config, :enable_persistence, true) do
          [LMStudio.Persistence | base_children]
        else
          base_children
        end
        
        children
      end
      
      defp post_startup_tasks do
        # Register health check endpoint
        :timer.apply_interval(30_000, __MODULE__, :log_health, [])
        
        # Initialize metrics
        initialize_telemetry()
      end
      
      defp pre_shutdown_tasks do
        # Cleanup resources
        Logger.info("Performing cleanup before shutdown")
      end
      
      defp get_startup_time do
        case Process.get(:startup_time) do
          nil ->
            startup_time = System.system_time(:second)
            Process.put(:startup_time, startup_time)
            startup_time
          time -> time
        end
      end
      
      defp log_health do
        health = health_check()
        Logger.debug("Health check: \#{inspect(health)}")
      end
      
      defp initialize_telemetry do
        # Initialize telemetry events and metrics
        :telemetry.execute([\#{@app_name}, :application, :start], %{}, %{})
      end
    end
    """
  end
  
  defp load_agent_template do
    """
    defmodule MODULE_NAME do
      @moduledoc \"\"\"
      Generated Agent with persistence and monitoring capabilities.
      
      Features:
      - Automatic state persistence
      - Performance monitoring
      - Concurrent access patterns
      - Error recovery
      \"\"\"
      
      use Agent
      require Logger
      alias LMStudio.Persistence.Helpers
      
      @agent_name __MODULE__
      
      def start_link(opts \\\\ []) do
        initial_state = Keyword.get(opts, :initial_state, INITIAL_VALUE)
        
        # Try to load persisted state
        persisted_state = Helpers.load_agent_state(inspect(@agent_name))
        final_initial_state = persisted_state || initial_state
        
        Agent.start_link(fn -> final_initial_state end, name: @agent_name)
      end
      
      def get(key \\\\ nil) do
        Agent.get(@agent_name, fn state ->
          if key do
            Map.get(state, key)
          else
            state
          end
        end)
      end
      
      def put(key, value) do
        result = Agent.update(@agent_name, fn state ->
          Map.put(state, key, value)
        end)
        
        # Persist state change
        :timer.apply_after(1000, __MODULE__, :persist_state, [])
        result
      end
      
      def update(key, update_fun) when is_function(update_fun) do
        result = Agent.update(@agent_name, fn state ->
          current_value = Map.get(state, key)
          new_value = update_fun.(current_value)
          Map.put(state, key, new_value)
        end)
        
        :timer.apply_after(1000, __MODULE__, :persist_state, [])
        result
      end
      
      def delete(key) do
        result = Agent.update(@agent_name, fn state ->
          Map.delete(state, key)
        end)
        
        :timer.apply_after(1000, __MODULE__, :persist_state, [])
        result
      end
      
      def reset(new_state \\\\ INITIAL_VALUE) do
        Agent.update(@agent_name, fn _state -> new_state end)
        persist_state()
      end
      
      def get_and_update(key, update_fun) when is_function(update_fun) do
        result = Agent.get_and_update(@agent_name, fn state ->
          current_value = Map.get(state, key)
          {return_value, new_value} = update_fun.(current_value)
          new_state = Map.put(state, key, new_value)
          {return_value, new_state}
        end)
        
        :timer.apply_after(1000, __MODULE__, :persist_state, [])
        result
      end
      
      def persist_state do
        current_state = Agent.get(@agent_name, & &1)
        Helpers.save_agent_state(inspect(@agent_name), current_state)
      end
      
      def stop do
        persist_state()
        Agent.stop(@agent_name)
      end
    end
    """
  end
  
  defp load_task_worker_template do
    """
    defmodule MODULE_NAME do
      @moduledoc \"\"\"
      Generated Task-based worker with supervision and error handling.
      
      Features:
      - Supervised task execution
      - Retry mechanisms
      - Performance monitoring
      - Result persistence
      \"\"\"
      
      require Logger
      
      def start_link(opts \\\\ []) do
        Task.start_link(__MODULE__, :run, [opts])
      end
      
      def run(opts \\\\ []) do
        Logger.info("\#{__MODULE__} worker starting")
        
        # Setup monitoring
        Process.flag(:trap_exit, true)
        
        # Main work loop
        work_loop(opts)
      end
      
      def async_work(work_data, opts \\\\ []) do
        Task.Supervisor.async_nolink(LMStudio.TaskSupervisor, fn ->
          perform_work(work_data, opts)
        end)
      end
      
      def await_work(task, timeout \\\\ 30_000) do
        try do
          Task.await(task, timeout)
        catch
          :exit, {:timeout, _} ->
            Logger.warning("\#{__MODULE__} work timed out")
            {:error, :timeout}
        end
      end
      
      defp work_loop(opts) do
        work_interval = Keyword.get(opts, :work_interval, 5_000)
        
        try do
          # Perform work
          result = WORK_FUNCTION
          
          # Log result
          Logger.debug("\#{__MODULE__} work completed: \#{inspect(result)}")
          
          # Sleep before next iteration
          Process.sleep(work_interval)
          
          # Continue loop
          work_loop(opts)
        rescue
          error ->
            Logger.error("\#{__MODULE__} work failed: \#{inspect(error)}")
            
            # Retry after delay
            retry_delay = Keyword.get(opts, :retry_delay, 10_000)
            Process.sleep(retry_delay)
            work_loop(opts)
        end
      end
      
      defp perform_work(work_data, opts) do
        start_time = System.monotonic_time()
        
        try do
          result = do_work(work_data, opts)
          
          # Record performance
          duration = System.monotonic_time() - start_time
          Logger.debug("\#{__MODULE__} work completed in \#{duration}ns")
          
          result
        rescue
          error ->
            duration = System.monotonic_time() - start_time
            Logger.error("\#{__MODULE__} work failed after \#{duration}ns: \#{inspect(error)}")
            reraise error, __STACKTRACE__
        end
      end
      
      defp do_work(work_data, _opts) do
        # Override this function with actual work logic
        Logger.info("Processing work: \#{inspect(work_data)}")
        {:ok, work_data}
      end
    end
    """
  end
  
  defp load_registry_worker_template do
    """
    defmodule MODULE_NAME do
      @moduledoc \"\"\"
      Generated Registry-based worker with dynamic registration and discovery.
      
      Features:
      - Dynamic process registration
      - Load balancing across workers
      - Health monitoring
      - Graceful scaling
      \"\"\"
      
      use GenServer
      require Logger
      
      @registry REGISTRY_NAME
      
      def start_link(opts \\\\ []) do
        worker_id = Keyword.get(opts, :worker_id, make_ref())
        GenServer.start_link(__MODULE__, opts, name: via_tuple(worker_id))
      end
      
      def find_worker(worker_id) do
        case Registry.lookup(@registry, worker_id) do
          [{pid, _}] -> {:ok, pid}
          [] -> {:error, :not_found}
        end
      end
      
      def list_workers do
        Registry.select(@registry, [{{:"$1", :"$2", :"$3"}, [], [{{:"$1", :"$2"}}]}])
      end
      
      def send_work(worker_id, work) do
        case find_worker(worker_id) do
          {:ok, pid} -> GenServer.call(pid, {:work, work})
          error -> error
        end
      end
      
      def broadcast_work(work) do
        list_workers()
        |> Enum.map(fn {_id, pid} ->
          Task.async(fn -> GenServer.call(pid, {:work, work}) end)
        end)
        |> Task.await_many(30_000)
      end
      
      def get_least_loaded_worker do
        workers_with_load = list_workers()
        |> Enum.map(fn {id, pid} ->
          load = GenServer.call(pid, :get_load)
          {id, pid, load}
        end)
        
        case Enum.min_by(workers_with_load, fn {_id, _pid, load} -> load end, fn -> nil end) do
          {id, pid, _load} -> {:ok, id, pid}
          nil -> {:error, :no_workers}
        end
      end
      
      defp via_tuple(worker_id) do
        {:via, Registry, {@registry, worker_id}}
      end
      
      @impl true
      def init(opts) do
        worker_id = Keyword.get(opts, :worker_id, make_ref())
        
        state = %{
          worker_id: worker_id,
          work_count: 0,
          current_load: 0,
          max_load: Keyword.get(opts, :max_load, 100)
        }
        
        Logger.info("\#{__MODULE__} worker \#{worker_id} started")
        {:ok, state}
      end
      
      @impl true
      def handle_call({:work, work}, _from, state) do
        if state.current_load < state.max_load do
          result = process_work(work, state)
          new_state = %{
            state | 
            work_count: state.work_count + 1,
            current_load: state.current_load + 1
          }
          
          # Simulate work completion
          :timer.apply_after(1000, __MODULE__, :work_completed, [self()])
          
          {:reply, {:ok, result}, new_state}
        else
          {:reply, {:error, :overloaded}, state}
        end
      end
      
      @impl true
      def handle_call(:get_load, _from, state) do
        {:reply, state.current_load, state}
      end
      
      @impl true
      def handle_call(:get_stats, _from, state) do
        stats = %{
          worker_id: state.worker_id,
          work_count: state.work_count,
          current_load: state.current_load,
          max_load: state.max_load
        }
        {:reply, stats, state}
      end
      
      @impl true
      def handle_cast(:work_completed, state) do
        new_state = %{state | current_load: max(0, state.current_load - 1)}
        {:noreply, new_state}
      end
      
      @impl true
      def handle_info(msg, state) do
        Logger.debug("\#{__MODULE__} received unexpected message: \#{inspect(msg)}")
        {:noreply, state}
      end
      
      defp process_work(work, _state) do
        # Override this function with actual work processing logic
        Logger.debug("Processing work: \#{inspect(work)}")
        work
      end
      
      def work_completed(pid) do
        GenServer.cast(pid, :work_completed)
      end
    end
    """
  end
  
  defp load_state_machine_template do
    """
    defmodule MODULE_NAME do
      @moduledoc \"\"\"
      Generated State Machine using GenStateMachine behavior.
      
      Features:
      - Type-safe state transitions
      - Event logging and monitoring
      - State persistence
      - Timeout handling
      \"\"\"
      
      use GenStateMachine
      require Logger
      alias LMStudio.Persistence.Helpers
      
      # Client API
      
      def start_link(opts \\\\ []) do
        GenStateMachine.start_link(__MODULE__, opts, name: __MODULE__)
      end
      
      def current_state do
        GenStateMachine.call(__MODULE__, :current_state)
      end
      
      def transition(event, data \\\\ %{}) do
        GenStateMachine.call(__MODULE__, {:transition, event, data})
      end
      
      def get_history do
        GenStateMachine.call(__MODULE__, :get_history)
      end
      
      # Server Callbacks
      
      @impl true
      def init(opts) do
        initial_state = Keyword.get(opts, :initial_state, :initial)
        
        # Load persisted state if available
        case Helpers.load_agent_state(inspect(__MODULE__)) do
          nil -> 
            data = %{
              history: [],
              transitions: TRANSITIONS,
              metadata: %{}
            }
            {:ok, initial_state, data}
          
          persisted_data ->
            Logger.info("\#{__MODULE__} restored from persistence")
            {:ok, persisted_data.current_state, persisted_data}
        end
      end
      
      @impl true
      def handle_event({:call, from}, :current_state, state, data) do
        {:next_state, state, data, [{:reply, from, state}]}
      end
      
      @impl true
      def handle_event({:call, from}, {:transition, event, event_data}, state, data) do
        case get_valid_transitions(state, data.transitions) do
          transitions when is_list(transitions) ->
            case find_transition(transitions, event) do
              {:ok, next_state} ->
                Logger.info("\#{__MODULE__} transitioning from \#{state} to \#{next_state} via \#{event}")
                
                new_history = [
                  %{
                    from: state,
                    to: next_state,
                    event: event,
                    data: event_data,
                    timestamp: DateTime.utc_now()
                  } | data.history
                ] |> Enum.take(100)  # Keep last 100 transitions
                
                new_data = %{data | history: new_history, current_state: next_state}
                
                # Persist state
                Helpers.save_agent_state(inspect(__MODULE__), new_data)
                
                {:next_state, next_state, new_data, [{:reply, from, {:ok, next_state}}]}
              
              {:error, reason} ->
                Logger.warning("\#{__MODULE__} invalid transition from \#{state} via \#{event}: \#{reason}")
                {:next_state, state, data, [{:reply, from, {:error, reason}}]}
            end
          
          _ ->
            {:next_state, state, data, [{:reply, from, {:error, :no_transitions_defined}}]}
        end
      end
      
      @impl true
      def handle_event({:call, from}, :get_history, state, data) do
        {:next_state, state, data, [{:reply, from, data.history}]}
      end
      
      @impl true
      def handle_event(:info, msg, state, data) do
        Logger.debug("\#{__MODULE__} received info: \#{inspect(msg)}")
        {:next_state, state, data}
      end
      
      # State-specific handlers (to be overridden)
      STATES
      
      # Private Functions
      
      defp get_valid_transitions(state, transitions) do
        Map.get(transitions, state, [])
      end
      
      defp find_transition(transitions, event) do
        case Enum.find(transitions, fn {transition_event, _next_state} -> 
          transition_event == event 
        end) do
          {_event, next_state} -> {:ok, next_state}
          nil -> {:error, :invalid_transition}
        end
      end
    end
    """
  end
  
  defp load_pool_manager_template do
    """
    defmodule MODULE_NAME do
      @moduledoc \"\"\"
      Generated Pool Manager for managing worker processes.
      
      Features:
      - Dynamic pool sizing
      - Worker health monitoring
      - Load balancing
      - Graceful scaling
      \"\"\"
      
      use GenServer
      require Logger
      
      @default_pool_size POOL_SIZE
      @worker_module WORKER_MODULE
      
      def start_link(opts \\\\ []) do
        GenServer.start_link(__MODULE__, opts, name: __MODULE__)
      end
      
      def get_worker do
        GenServer.call(__MODULE__, :get_worker)
      end
      
      def execute(work, timeout \\\\ 30_000) do
        with {:ok, worker} <- get_worker(),
             {:ok, result} <- GenServer.call(worker, {:work, work}, timeout) do
          {:ok, result}
        else
          error -> error
        end
      end
      
      def pool_status do
        GenServer.call(__MODULE__, :pool_status)
      end
      
      def resize_pool(new_size) do
        GenServer.call(__MODULE__, {:resize_pool, new_size})
      end
      
      @impl true
      def init(opts) do
        pool_size = Keyword.get(opts, :pool_size, @default_pool_size)
        
        state = %{
          workers: [],
          available_workers: [],
          busy_workers: [],
          pool_size: pool_size,
          worker_module: @worker_module,
          round_robin_index: 0
        }
        
        # Start initial workers
        {:ok, new_state} = start_workers(state, pool_size)
        
        # Schedule health checks
        :timer.send_interval(10_000, :health_check)
        
        Logger.info("\#{__MODULE__} initialized with \#{pool_size} workers")
        {:ok, new_state}
      end
      
      @impl true
      def handle_call(:get_worker, _from, state) do
        case get_available_worker(state) do
          {:ok, worker, new_state} ->
            {:reply, {:ok, worker}, new_state}
          {:error, reason} ->
            {:reply, {:error, reason}, state}
        end
      end
      
      @impl true
      def handle_call(:pool_status, _from, state) do
        status = %{
          total_workers: length(state.workers),
          available_workers: length(state.available_workers),
          busy_workers: length(state.busy_workers),
          pool_size: state.pool_size
        }
        {:reply, status, state}
      end
      
      @impl true
      def handle_call({:resize_pool, new_size}, _from, state) do
        current_size = length(state.workers)
        
        cond do
          new_size > current_size ->
            # Add workers
            workers_to_add = new_size - current_size
            {:ok, new_state} = start_workers(state, workers_to_add)
            updated_state = %{new_state | pool_size: new_size}
            Logger.info("Pool resized to \#{new_size} workers")
            {:reply, :ok, updated_state}
          
          new_size < current_size ->
            # Remove workers
            workers_to_remove = current_size - new_size
            new_state = stop_workers(state, workers_to_remove)
            updated_state = %{new_state | pool_size: new_size}
            Logger.info("Pool resized to \#{new_size} workers")
            {:reply, :ok, updated_state}
          
          true ->
            # No change needed
            {:reply, :ok, state}
        end
      end
      
      @impl true
      def handle_info(:health_check, state) do
        # Check worker health and restart if needed
        new_state = check_and_restart_workers(state)
        {:noreply, new_state}
      end
      
      @impl true
      def handle_info({:DOWN, _ref, :process, pid, reason}, state) do
        Logger.warning("Worker \#{inspect(pid)} died: \#{inspect(reason)}")
        
        # Remove dead worker and start a replacement
        new_state = remove_worker(state, pid)
        {:ok, final_state} = start_workers(new_state, 1)
        
        {:noreply, final_state}
      end
      
      defp get_available_worker(state) do
        case state.available_workers do
          [worker | rest] ->
            new_state = %{
              state |
              available_workers: rest,
              busy_workers: [worker | state.busy_workers]
            }
            {:ok, worker, new_state}
          
          [] ->
            # Use round-robin if no dedicated available workers
            if length(state.workers) > 0 do
              index = rem(state.round_robin_index, length(state.workers))
              worker = Enum.at(state.workers, index)
              new_state = %{state | round_robin_index: index + 1}
              {:ok, worker, new_state}
            else
              {:error, :no_workers_available}
            end
        end
      end
      
      defp start_workers(state, count) do
        new_workers = Enum.map(1..count, fn _ ->
          {:ok, pid} = @worker_module.start_link()
          Process.monitor(pid)
          pid
        end)
        
        new_state = %{
          state |
          workers: state.workers ++ new_workers,
          available_workers: state.available_workers ++ new_workers
        }
        
        {:ok, new_state}
      end
      
      defp stop_workers(state, count) do
        {workers_to_stop, remaining_workers} = Enum.split(state.workers, count)
        
        Enum.each(workers_to_stop, fn worker ->
          GenServer.stop(worker, :normal)
        end)
        
        %{
          state |
          workers: remaining_workers,
          available_workers: state.available_workers -- workers_to_stop,
          busy_workers: state.busy_workers -- workers_to_stop
        }
      end
      
      defp remove_worker(state, pid) do
        %{
          state |
          workers: List.delete(state.workers, pid),
          available_workers: List.delete(state.available_workers, pid),
          busy_workers: List.delete(state.busy_workers, pid)
        }
      end
      
      defp check_and_restart_workers(state) do
        dead_workers = Enum.filter(state.workers, fn pid ->
          not Process.alive?(pid)
        end)
        
        if length(dead_workers) > 0 do
          Logger.warning("Found \#{length(dead_workers)} dead workers, restarting...")
          
          # Remove dead workers
          cleaned_state = Enum.reduce(dead_workers, state, &remove_worker(&2, &1))
          
          # Start replacements
          {:ok, new_state} = start_workers(cleaned_state, length(dead_workers))
          new_state
        else
          state
        end
      end
    end
    """
  end
  
  defp load_distributed_worker_template do
    """
    defmodule MODULE_NAME do
      @moduledoc \"\"\"
      Generated Distributed Worker with cluster communication capabilities.
      
      Features:
      - Node discovery and monitoring
      - Work distribution across cluster
      - Fault tolerance for node failures
      - Load balancing
      \"\"\"
      
      use GenServer
      require Logger
      
      @cluster_nodes CLUSTER_NODES
      
      def start_link(opts \\\\ []) do
        GenServer.start_link(__MODULE__, opts, name: __MODULE__)
      end
      
      def submit_work(work, opts \\\\ []) do
        GenServer.call(__MODULE__, {:submit_work, work, opts})
      end
      
      def get_cluster_status do
        GenServer.call(__MODULE__, :cluster_status)
      end
      
      def discover_nodes do
        GenServer.cast(__MODULE__, :discover_nodes)
      end
      
      @impl true
      def init(opts) do
        # Connect to cluster nodes
        cluster_nodes = Keyword.get(opts, :cluster_nodes, @cluster_nodes)
        
        state = %{
          cluster_nodes: cluster_nodes,
          connected_nodes: [],
          work_queue: :queue.new(),
          node_loads: %{},
          node_monitors: %{}
        }
        
        # Initial node discovery
        send(self(), :discover_nodes)
        
        # Schedule periodic node health checks
        :timer.send_interval(30_000, :health_check_nodes)
        
        Logger.info("\#{__MODULE__} initialized for distributed work")
        {:ok, state}
      end
      
      @impl true
      def handle_call({:submit_work, work, opts}, from, state) do
        case select_best_node(state) do
          {:ok, node} ->
            # Submit work to selected node
            case submit_work_to_node(node, work, from) do
              :ok ->
                {:reply, {:ok, :submitted}, state}
              {:error, reason} ->
                # Fall back to local execution
                Logger.warning("Failed to submit to node \#{node}: \#{reason}, executing locally")
                result = execute_work_locally(work)
                {:reply, result, state}
            end
          
          {:error, :no_nodes_available} ->
            # Execute locally if no nodes available
            result = execute_work_locally(work)
            {:reply, result, state}
        end
      end
      
      @impl true
      def handle_call(:cluster_status, _from, state) do
        status = %{
          configured_nodes: state.cluster_nodes,
          connected_nodes: state.connected_nodes,
          node_loads: state.node_loads,
          queue_size: :queue.len(state.work_queue)
        }
        {:reply, status, state}
      end
      
      @impl true
      def handle_cast(:discover_nodes, state) do
        new_connected_nodes = discover_and_connect_nodes(state.cluster_nodes)
        
        # Update node monitors
        new_monitors = setup_node_monitors(new_connected_nodes, state.node_monitors)
        
        new_state = %{
          state |
          connected_nodes: new_connected_nodes,
          node_monitors: new_monitors
        }
        
        Logger.info("Connected to \#{length(new_connected_nodes)} cluster nodes")
        {:noreply, new_state}
      end
      
      @impl true
      def handle_info(:discover_nodes, state) do
        send(self(), {:cast, :discover_nodes})
        {:noreply, state}
      end
      
      @impl true
      def handle_info(:health_check_nodes, state) do
        # Check health of connected nodes
        healthy_nodes = Enum.filter(state.connected_nodes, fn node ->
          Node.ping(node) == :pong
        end)
        
        if length(healthy_nodes) != length(state.connected_nodes) do
          Logger.warning("Some cluster nodes are unreachable")
          new_state = %{state | connected_nodes: healthy_nodes}
          {:noreply, new_state}
        else
          {:noreply, state}
        end
      end
      
      @impl true
      def handle_info({:nodedown, node}, state) do
        Logger.warning("Node \#{node} went down")
        
        new_connected_nodes = List.delete(state.connected_nodes, node)
        new_node_loads = Map.delete(state.node_loads, node)
        
        new_state = %{
          state |
          connected_nodes: new_connected_nodes,
          node_loads: new_node_loads
        }
        
        {:noreply, new_state}
      end
      
      @impl true
      def handle_info({:nodeup, node}, state) do
        Logger.info("Node \#{node} came up")
        
        if node in state.cluster_nodes and node not in state.connected_nodes do
          new_connected_nodes = [node | state.connected_nodes]
          new_state = %{state | connected_nodes: new_connected_nodes}
          {:noreply, new_state}
        else
          {:noreply, state}
        end
      end
      
      defp discover_and_connect_nodes(cluster_nodes) do
        Enum.filter(cluster_nodes, fn node ->
          case Node.ping(node) do
            :pong -> 
              Logger.debug("Successfully connected to node \#{node}")
              true
            :pang -> 
              Logger.debug("Failed to connect to node \#{node}")
              false
          end
        end)
      end
      
      defp setup_node_monitors(nodes, existing_monitors) do
        # Remove old monitors
        Enum.each(existing_monitors, fn {_node, monitor_ref} ->
          Node.monitor(monitor_ref, false)
        end)
        
        # Setup new monitors
        Enum.reduce(nodes, %{}, fn node, acc ->
          monitor_ref = Node.monitor(node, true)
          Map.put(acc, node, monitor_ref)
        end)
      end
      
      defp select_best_node(state) do
        case state.connected_nodes do
          [] -> {:error, :no_nodes_available}
          nodes ->
            # Simple round-robin selection
            # In production, you might want more sophisticated load balancing
            selected_node = Enum.random(nodes)
            {:ok, selected_node}
        end
      end
      
      defp submit_work_to_node(node, work, from) do
        try do
          # This would call a remote worker on the selected node
          # For now, we'll simulate the call
          :rpc.call(node, __MODULE__, :execute_work_remotely, [work, from])
          :ok
        rescue
          error ->
            {:error, error}
        end
      end
      
      defp execute_work_locally(work) do
        # Override this function with actual work execution logic
        Logger.info("Executing work locally: \#{inspect(work)}")
        {:ok, work}
      end
      
      def execute_work_remotely(work, _from) do
        # This function would be called on remote nodes
        Logger.info("Executing work remotely: \#{inspect(work)}")
        {:ok, work}
      end
    end
    """
  end
  
  # ========== Helper Functions ==========
  
  defp replace_placeholder(template, placeholder, replacement) do
    String.replace(template, placeholder, replacement)
  end
  
  defp format_state(state) when is_map(state), do: inspect(state, pretty: true)
  defp format_state(state), do: inspect(state)
  
  defp format_children_specs(children) when is_list(children) do
    children
    |> Enum.map(&inspect/1)
    |> Enum.join(",\n      ")
    |> then(&"[\n      #{&1}\n    ]")
  end
  
  defp format_module_list(modules) when is_list(modules) do
    inspect(modules)
  end
  
  defp format_states(states) when is_list(states) do
    Enum.map_join(states, "\n", fn state ->
      """
      @impl true
      def handle_event(event_type, event, :#{state}, data) do
        # Handle events in #{state} state
        Logger.debug("Handling event \#{inspect(event)} in state :#{state}")
        {:next_state, :#{state}, data}
      end
      """
    end)
  end
  
  defp format_transitions(transitions) when is_map(transitions) do
    inspect(transitions, pretty: true)
  end
  
  defp apply_genserver_options(code, opts) do
    # Apply additional GenServer options
    timeout = Keyword.get(opts, :timeout, 30_000)
    hibernate_after = Keyword.get(opts, :hibernate_after, 60_000)
    
    code
    |> String.replace("@timeout 30_000", "@timeout #{timeout}")
    |> String.replace("@hibernate_after 60_000", "@hibernate_after #{hibernate_after}")
  end
  
  defp apply_application_options(code, _opts) do
    # Apply additional application options
    code
  end
  
  # ========== Evolution and Optimization ==========
  
  defp apply_evolution_patterns(code, hints) do
    code
    |> maybe_add_telemetry(hints)
    |> maybe_add_circuit_breaker(hints)
    |> maybe_add_rate_limiting(hints)
    |> maybe_add_caching(hints)
  end
  
  defp maybe_add_telemetry(code, hints) do
    if :telemetry in hints do
      add_telemetry_instrumentation(code)
    else
      code
    end
  end
  
  defp maybe_add_circuit_breaker(code, hints) do
    if :circuit_breaker in hints do
      add_circuit_breaker_pattern(code)
    else
      code
    end
  end
  
  defp maybe_add_rate_limiting(code, hints) do
    if :rate_limiting in hints do
      add_rate_limiting_pattern(code)
    else
      code
    end
  end
  
  defp maybe_add_caching(code, hints) do
    if :caching in hints do
      add_caching_pattern(code)
    else
      code
    end
  end
  
  defp add_performance_monitoring(code) do
    monitoring_code = """
    
    # Performance monitoring additions
    defp with_timing(operation_name, fun) do
      start_time = System.monotonic_time()
      result = fun.()
      duration = System.monotonic_time() - start_time
      Logger.debug("Operation \#{operation_name} took \#{duration}ns")
      result
    end
    """
    
    code <> monitoring_code
  end
  
  defp add_caching_patterns(code) do
    caching_code = """
    
    # Caching additions
    defp with_cache(cache_key, compute_fun, ttl \\\\ 300_000) do
      case :ets.lookup(:cache_table, cache_key) do
        [{^cache_key, value, expiry}] when expiry > System.system_time(:millisecond) ->
          value
        _ ->
          value = compute_fun.()
          expiry = System.system_time(:millisecond) + ttl
          :ets.insert(:cache_table, {cache_key, value, expiry})
          value
      end
    end
    """
    
    code <> caching_code
  end
  
  defp add_concurrent_processing(code) do
    concurrency_code = """
    
    # Concurrent processing additions
    defp parallel_map(enumerable, fun, opts \\\\ []) do
      max_concurrency = Keyword.get(opts, :max_concurrency, System.schedulers_online())
      timeout = Keyword.get(opts, :timeout, 30_000)
      
      enumerable
      |> Task.async_stream(fun, max_concurrency: max_concurrency, timeout: timeout)
      |> Enum.map(fn {:ok, result} -> result end)
    end
    """
    
    code <> concurrency_code
  end
  
  defp add_memory_optimization(code) do
    optimization_code = """
    
    # Memory optimization additions
    defp maybe_garbage_collect(threshold \\\\ 1000) do
      if rem(System.system_time(:second), threshold) == 0 do
        :erlang.garbage_collect()
      end
    end
    """
    
    code <> optimization_code
  end
  
  defp add_error_handling(code) do
    error_handling_code = """
    
    # Enhanced error handling
    defp safe_call(fun, default \\\\ nil, max_retries \\\\ 3) do
      do_safe_call(fun, default, max_retries, 0)
    end
    
    defp do_safe_call(fun, default, max_retries, attempt) when attempt < max_retries do
      try do
        fun.()
      rescue
        error ->
          Logger.warning("Attempt \#{attempt + 1} failed: \#{inspect(error)}")
          :timer.sleep(trunc(:math.pow(2, attempt) * 1000))  # Exponential backoff
          do_safe_call(fun, default, max_retries, attempt + 1)
      end
    end
    defp do_safe_call(_fun, default, _max_retries, _attempt), do: default
    """
    
    code <> error_handling_code
  end
  
  defp add_restart_strategies(code) do
    restart_code = """
    
    # Restart strategy enhancements
    defp handle_restart(reason, state) do
      Logger.info("Restarting due to: \#{inspect(reason)}")
      
      # Implement restart logic based on reason
      case reason do
        :normal -> {:ok, state}
        :shutdown -> {:ok, state}
        {:shutdown, _} -> {:ok, state}
        _ -> 
          # Abnormal restart - maybe reset some state
          {:ok, reset_transient_state(state)}
      end
    end
    
    defp reset_transient_state(state) do
      # Reset any transient state that shouldn't survive restarts
      Map.drop(state, [:temporary_data, :cached_values])
    end
    """
    
    code <> restart_code
  end
  
  defp add_circuit_breaker(code) do
    circuit_breaker_code = """
    
    # Circuit breaker pattern
    defp with_circuit_breaker(operation_name, fun, opts \\\\ []) do
      failure_threshold = Keyword.get(opts, :failure_threshold, 5)
      timeout = Keyword.get(opts, :timeout, 60_000)
      
      case get_circuit_state(operation_name) do
        :closed -> 
          try do
            result = fun.()
            reset_circuit_failure_count(operation_name)
            result
          rescue
            error ->
              increment_circuit_failure_count(operation_name)
              if get_circuit_failure_count(operation_name) >= failure_threshold do
                open_circuit(operation_name, timeout)
              end
              reraise error, __STACKTRACE__
          end
        
        :open ->
          if circuit_timeout_expired?(operation_name) do
            set_circuit_state(operation_name, :half_open)
            with_circuit_breaker(operation_name, fun, opts)
          else
            {:error, :circuit_open}
          end
        
        :half_open ->
          try do
            result = fun.()
            close_circuit(operation_name)
            result
          rescue
            error ->
              open_circuit(operation_name, timeout)
              reraise error, __STACKTRACE__
          end
      end
    end
    """
    
    code <> circuit_breaker_code
  end
  
  defp add_health_monitoring(code) do
    health_code = """
    
    # Health monitoring
    defp health_check do
      %{
        status: :healthy,
        uptime: get_uptime(),
        memory: :erlang.memory(),
        message_queue_len: Process.info(self(), :message_queue_len),
        reductions: Process.info(self(), :reductions)
      }
    end
    
    defp get_uptime do
      case Process.get(:start_time) do
        nil ->
          start_time = System.system_time(:second)
          Process.put(:start_time, start_time)
          0
        start_time ->
          System.system_time(:second) - start_time
      end
    end
    """
    
    code <> health_code
  end
  
  defp add_telemetry_instrumentation(code) do
    telemetry_code = """
    
    # Telemetry instrumentation
    defp emit_telemetry(event_name, measurements \\\\ %{}, metadata \\\\ %{}) do
      :telemetry.execute([\#{__MODULE__} | event_name], measurements, metadata)
    end
    
    defp with_telemetry(event_name, fun, metadata \\\\ %{}) do
      start_time = System.monotonic_time()
      
      try do
        result = fun.()
        duration = System.monotonic_time() - start_time
        
        emit_telemetry([event_name, :success], %{duration: duration}, metadata)
        result
      rescue
        error ->
          duration = System.monotonic_time() - start_time
          
          emit_telemetry([event_name, :error], %{duration: duration}, 
                        Map.put(metadata, :error, error))
          reraise error, __STACKTRACE__
      end
    end
    """
    
    code <> telemetry_code
  end
  
  defp add_circuit_breaker_pattern(code) do
    add_circuit_breaker(code)
  end
  
  defp add_rate_limiting_pattern(code) do
    rate_limiting_code = """
    
    # Rate limiting pattern
    defp with_rate_limit(operation_name, fun, opts \\\\ []) do
      max_requests = Keyword.get(opts, :max_requests, 100)
      time_window = Keyword.get(opts, :time_window, 60_000)  # 1 minute
      
      current_time = System.system_time(:millisecond)
      window_start = current_time - time_window
      
      # Clean old requests
      clean_old_requests(operation_name, window_start)
      
      # Check current request count
      current_count = get_request_count(operation_name, window_start)
      
      if current_count < max_requests do
        record_request(operation_name, current_time)
        fun.()
      else
        {:error, :rate_limited}
      end
    end
    """
    
    code <> rate_limiting_code
  end
  
  defp add_caching_pattern(code) do
    add_caching_patterns(code)
  end
  
  # ========== Persistence Integration ==========
  
  def save_generated_code(code_id, code) do
    PersistenceHelpers.save_generated_code(code_id, %{
      code: code,
      generated_at: DateTime.utc_now(),
      module_name: extract_module_name(code)
    })
  end
  
  defp load_generated_code(code_id) do
    case PersistenceHelpers.load_generated_code(code_id) do
      nil -> nil
      data -> data.code
    end
  end
  
  defp extract_module_name(code) do
    case Regex.run(~r/defmodule\s+([A-Za-z0-9_.]+)/, code) do
      [_, module_name] -> module_name
      _ -> "Unknown"
    end
  end
end