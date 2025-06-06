#!/usr/bin/env elixir

Mix.install([
  {:phoenix_pubsub, "~> 2.1"},
  {:jason, "~> 1.4"}
])

defmodule CleanRecursiveAgent do
  @moduledoc """
  Clean recursive agent with better message handling and visualization
  """
  
  use GenServer
  require Logger
  
  defstruct [
    :id,
    :parent_id,
    :depth,
    :max_depth,
    :children,
    :results,
    :status,
    :task,
    :monitor_refs,
    :pubsub,
    :start_time
  ]
  
  # Client API
  
  def start_link(opts) do
    GenServer.start_link(__MODULE__, opts)
  end
  
  def execute_task(agent, task, split_factor \\ 2) do
    GenServer.call(agent, {:execute_task, task, split_factor}, 30000)
  end
  
  def get_tree_view(agent) do
    GenServer.call(agent, :get_tree_view)
  end
  
  # Server callbacks
  
  def init(opts) do
    id = opts[:id] || generate_id()
    
    # Initialize PubSub system once
    pubsub = init_pubsub()
    
    state = %__MODULE__{
      id: id,
      parent_id: opts[:parent_id],
      depth: opts[:depth] || 0,
      max_depth: opts[:max_depth] || 3,
      children: [],
      results: %{},
      status: :idle,
      task: opts[:task],
      monitor_refs: %{},
      pubsub: pubsub,
      start_time: System.monotonic_time()
    }
    
    {:ok, state}
  end
  
  def handle_call({:execute_task, task, split_factor}, from, state) do
    state = %{state | task: task, status: :processing, start_time: System.monotonic_time()}
    
    if state.depth >= state.max_depth do
      # Leaf node - process directly
      result = process_leaf_task(task, state)
      duration = System.monotonic_time() - state.start_time
      
      final_result = %{
        id: state.id,
        depth: state.depth,
        result: result,
        duration_ms: System.convert_time_unit(duration, :native, :millisecond)
      }
      
      {:reply, {:ok, final_result}, %{state | status: :completed}}
    else
      # Branch node - split and delegate
      subtasks = split_task(task, split_factor)
      
      # Spawn children asynchronously
      Task.start(fn ->
        children = spawn_children(subtasks, state, split_factor)
        GenServer.cast(self(), {:children_spawned, children, from})
      end)
      
      {:noreply, %{state | status: :splitting}}
    end
  end
  
  def handle_call(:get_tree_view, _from, state) do
    tree = build_tree_view(state)
    {:reply, tree, state}
  end
  
  def handle_cast({:children_spawned, children, from}, state) do
    # Monitor all children
    monitor_refs = Enum.reduce(children, %{}, fn {child_id, child_pid}, acc ->
      ref = Process.monitor(child_pid)
      Map.put(acc, ref, child_id)
    end)
    
    state = %{state | 
      children: Map.keys(children),
      monitor_refs: monitor_refs,
      status: :waiting_for_children
    }
    
    # Store the reply destination
    Process.put({:reply_to, state.id}, from)
    
    {:noreply, state}
  end
  
  def handle_cast({:child_result, child_id, result}, state) do
    state = %{state | results: Map.put(state.results, child_id, result)}
    
    if all_children_completed?(state) do
      # All children done - aggregate and reply
      duration = System.monotonic_time() - state.start_time
      
      aggregated = aggregate_results(state.results, state.task)
      
      final_result = %{
        id: state.id,
        depth: state.depth,
        children: state.children,
        result: aggregated,
        duration_ms: System.convert_time_unit(duration, :native, :millisecond)
      }
      
      # Reply to original caller
      case Process.get({:reply_to, state.id}) do
        nil -> :ok
        from -> GenServer.reply(from, {:ok, final_result})
      end
      
      # Notify parent if exists
      if state.parent_id do
        case Process.whereis(String.to_atom("agent_#{state.parent_id}")) do
          nil -> :ok
          parent -> GenServer.cast(parent, {:child_result, state.id, final_result})
        end
      end
      
      {:noreply, %{state | status: :completed}}
    else
      {:noreply, state}
    end
  end
  
  def handle_info({:DOWN, ref, :process, _pid, reason}, state) do
    case Map.get(state.monitor_refs, ref) do
      nil -> 
        {:noreply, state}
      child_id ->
        # Handle child failure
        state = %{state | results: Map.put(state.results, child_id, {:error, reason})}
        
        if all_children_completed?(state) do
          # Reply with partial results
          handle_cast({:child_result, child_id, {:error, reason}}, state)
        else
          {:noreply, state}
        end
    end
  end
  
  # Private functions
  
  defp init_pubsub do
    # Use a singleton approach for PubSub
    case Process.whereis(:recursive_pubsub_supervisor) do
      nil ->
        children = [{Phoenix.PubSub, name: :recursive_agent_pubsub}]
        {:ok, pid} = Supervisor.start_link(children, strategy: :one_for_one)
        Process.register(pid, :recursive_pubsub_supervisor)
        :recursive_agent_pubsub
      _pid ->
        :recursive_agent_pubsub
    end
  end
  
  defp generate_id do
    :crypto.strong_rand_bytes(4) |> Base.encode16()
  end
  
  defp spawn_children(subtasks, state, split_factor) do
    Enum.with_index(subtasks)
    |> Enum.map(fn {subtask, index} ->
      child_id = "#{state.id}_#{index}"
      
      {:ok, pid} = CleanRecursiveAgent.start_link(
        id: child_id,
        parent_id: state.id,
        depth: state.depth + 1,
        max_depth: state.max_depth,
        task: subtask
      )
      
      # Register with a name for easy lookup
      Process.register(pid, String.to_atom("agent_#{child_id}"))
      
      # Start child processing
      Task.start(fn ->
        {:ok, result} = CleanRecursiveAgent.execute_task(pid, subtask, split_factor)
        
        # Send result to parent
        case Process.whereis(String.to_atom("agent_#{state.id}")) do
          nil -> :ok
          parent -> GenServer.cast(parent, {:child_result, child_id, result})
        end
      end)
      
      {child_id, pid}
    end)
    |> Enum.into(%{})
  end
  
  defp split_task(task, split_factor) do
    case task do
      {:sum, range} ->
        {start, finish} = range
        total = finish - start + 1
        chunk_size = div(total, split_factor)
        
        Enum.map(0..(split_factor - 1), fn i ->
          chunk_start = start + (i * chunk_size)
          chunk_end = if i == split_factor - 1 do
            finish
          else
            chunk_start + chunk_size - 1
          end
          {:sum, {chunk_start, chunk_end}}
        end)
        
      {:analyze_text, text} ->
        words = String.split(text, " ")
        chunks = Enum.chunk_every(words, ceil(length(words) / split_factor))
        Enum.map(chunks, fn chunk ->
          {:analyze_text, Enum.join(chunk, " ")}
        end)
        
      {:process_list, items} ->
        chunks = Enum.chunk_every(items, ceil(length(items) / split_factor))
        Enum.map(chunks, fn chunk ->
          {:process_list, chunk}
        end)
        
      _ ->
        # Generic splitting
        Enum.map(1..split_factor, fn i ->
          {:subtask, task, i, split_factor}
        end)
    end
  end
  
  defp process_leaf_task(task, _state) do
    Process.sleep(50) # Simulate work
    
    case task do
      {:sum, {start, finish}} ->
        Enum.sum(start..finish)
        
      {:analyze_text, text} ->
        %{
          words: length(String.split(text, " ")),
          chars: String.length(text)
        }
        
      {:process_list, items} ->
        %{
          count: length(items),
          processed: Enum.map(items, &(&1 * 2))
        }
        
      _ ->
        {:processed, task}
    end
  end
  
  defp aggregate_results(results, original_task) do
    values = Map.values(results)
    
    case original_task do
      {:sum, _} ->
        Enum.reduce(values, 0, fn
          %{result: sum}, acc -> acc + sum
          _, acc -> acc
        end)
        
      {:analyze_text, _} ->
        Enum.reduce(values, %{words: 0, chars: 0}, fn
          %{result: %{words: w, chars: c}}, acc ->
            %{words: acc.words + w, chars: acc.chars + c}
          _, acc -> acc
        end)
        
      {:process_list, _} ->
        Enum.reduce(values, %{count: 0, processed: []}, fn
          %{result: %{count: c, processed: p}}, acc ->
            %{count: acc.count + c, processed: acc.processed ++ p}
          _, acc -> acc
        end)
        
      _ ->
        values
    end
  end
  
  defp all_children_completed?(state) do
    length(Map.keys(state.results)) == length(state.children)
  end
  
  defp build_tree_view(state) do
    %{
      id: state.id,
      depth: state.depth,
      status: state.status,
      task: inspect(state.task),
      children_count: length(state.children),
      results_count: map_size(state.results)
    }
  end
end

defmodule RecursiveAgentDemo do
  @moduledoc """
  Demo showing recursive agent splitting and monitoring
  """
  
  def run do
    IO.puts("\n🚀 Clean Recursive Agent Demonstration\n")
    
    # Demo 1: Recursive sum calculation
    IO.puts("📊 Demo 1: Recursive Sum Calculation")
    demo_sum_calculation()
    
    # Demo 2: Recursive text analysis
    IO.puts("\n📝 Demo 2: Recursive Text Analysis")
    demo_text_analysis()
    
    # Demo 3: Deep recursion with visualization
    IO.puts("\n🌳 Demo 3: Deep Recursion with List Processing")
    demo_deep_recursion()
    
    IO.puts("\n✅ All demonstrations complete!")
  end
  
  defp demo_sum_calculation do
    {:ok, agent} = CleanRecursiveAgent.start_link(
      id: "sum_root",
      max_depth: 3
    )
    
    IO.puts("Calculating sum of 1 to 1000 recursively...")
    {:ok, result} = CleanRecursiveAgent.execute_task(agent, {:sum, {1, 1000}}, 4)
    
    IO.puts("Result: #{inspect(result.result)}")
    IO.puts("Total time: #{result.duration_ms}ms")
    IO.puts("Tree structure:")
    visualize_result(result, 0)
    IO.puts("")
  end
  
  defp demo_text_analysis do
    text = """
    The recursive agent system demonstrates powerful distributed processing
    capabilities. Each agent can split tasks, spawn children, monitor their
    progress, and aggregate results. This creates a tree of computation that
    efficiently processes complex tasks in parallel.
    """
    
    {:ok, agent} = CleanRecursiveAgent.start_link(
      id: "text_root",
      max_depth: 2
    )
    
    IO.puts("Analyzing text recursively...")
    {:ok, result} = CleanRecursiveAgent.execute_task(agent, {:analyze_text, text}, 3)
    
    IO.puts("Result: #{inspect(result.result)}")
    IO.puts("Total time: #{result.duration_ms}ms")
    IO.puts("")
  end
  
  defp demo_deep_recursion do
    items = Enum.to_list(1..64)
    
    {:ok, agent} = CleanRecursiveAgent.start_link(
      id: "list_root",
      max_depth: 3
    )
    
    IO.puts("Processing list of #{length(items)} items recursively...")
    {:ok, result} = CleanRecursiveAgent.execute_task(agent, {:process_list, items}, 2)
    
    IO.puts("Items processed: #{result.result.count}")
    IO.puts("Sample output: #{inspect(Enum.take(result.result.processed, 10))}...")
    IO.puts("Total time: #{result.duration_ms}ms")
    
    # Show execution tree
    IO.puts("\nExecution tree:")
    visualize_result(result, 0)
  end
  
  defp visualize_result(result, depth) do
    indent = String.duplicate("  ", depth)
    
    node_icon = if result[:children] && length(result.children) > 0, do: "📦", else: "📄"
    
    IO.puts("#{indent}#{node_icon} #{result.id}")
    IO.puts("#{indent}├─ Depth: #{result.depth}")
    IO.puts("#{indent}├─ Duration: #{result.duration_ms}ms")
    
    if is_map(result.result) && Map.has_key?(result.result, :count) do
      IO.puts("#{indent}└─ Processed: #{result.result.count} items")
    else
      IO.puts("#{indent}└─ Result: #{inspect(result.result)}")
    end
    
    # Note: In this demo, we don't have access to child details
    # In a real system, you'd store and visualize the full tree
  end
end

# Message flow visualization
defmodule MessageFlowVisualizer do
  @moduledoc """
  Shows the message flow in the recursive agent system
  """
  
  def demonstrate_flow do
    IO.puts("\n📨 Message Flow Demonstration\n")
    
    IO.puts("1. Parent agent receives task")
    IO.puts("   └─> Splits task into subtasks")
    IO.puts("")
    
    IO.puts("2. Parent spawns child agents")
    IO.puts("   ├─> Child 1 (monitors)")
    IO.puts("   ├─> Child 2 (monitors)")
    IO.puts("   └─> Child N (monitors)")
    IO.puts("")
    
    IO.puts("3. Children process recursively")
    IO.puts("   ├─> May spawn their own children")
    IO.puts("   └─> Or process directly (leaf nodes)")
    IO.puts("")
    
    IO.puts("4. Results bubble up")
    IO.puts("   ├─> Leaf sends result to parent")
    IO.puts("   ├─> Parent aggregates child results")
    IO.puts("   └─> Parent sends to its parent")
    IO.puts("")
    
    IO.puts("5. Root returns final result")
    IO.puts("   └─> All results aggregated")
    IO.puts("")
  end
end

# Run everything
RecursiveAgentDemo.run()
MessageFlowVisualizer.demonstrate_flow()