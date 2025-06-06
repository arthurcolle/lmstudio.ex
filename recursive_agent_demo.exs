#!/usr/bin/env elixir

Mix.install([
  {:phoenix_pubsub, "~> 2.1"},
  {:jason, "~> 1.4"}
])

defmodule RecursiveAgent do
  @moduledoc """
  Recursive agent that can split into children, monitor them,
  collect messages, and generate outputs
  """
  
  use GenServer
  require Logger
  
  defstruct [
    :id,
    :parent_id,
    :depth,
    :max_depth,
    :children,
    :collected_messages,
    :status,
    :task,
    :monitor_refs,
    :pubsub
  ]
  
  # Public API
  
  def start_link(opts) do
    GenServer.start_link(__MODULE__, opts)
  end
  
  def split_task(agent, task, num_children \\ 2) do
    GenServer.call(agent, {:split_task, task, num_children})
  end
  
  def get_status(agent) do
    GenServer.call(agent, :get_status)
  end
  
  def get_collected_output(agent) do
    GenServer.call(agent, :get_collected_output)
  end
  
  # GenServer callbacks
  
  def init(opts) do
    id = opts[:id] || generate_id()
    parent_id = opts[:parent_id]
    depth = opts[:depth] || 0
    max_depth = opts[:max_depth] || 3
    task = opts[:task]
    
    # Start PubSub if not already started
    pubsub = opts[:pubsub] || start_pubsub()
    
    # Subscribe to child messages
    if parent_id do
      Phoenix.PubSub.subscribe(pubsub, "agent:#{parent_id}:children")
    end
    
    # Subscribe to own messages
    Phoenix.PubSub.subscribe(pubsub, "agent:#{id}")
    
    state = %__MODULE__{
      id: id,
      parent_id: parent_id,
      depth: depth,
      max_depth: max_depth,
      children: [],
      collected_messages: [],
      status: :idle,
      task: task,
      monitor_refs: %{},
      pubsub: pubsub
    }
    
    Logger.info("Agent #{id} started at depth #{depth}")
    
    # Notify parent of creation
    if parent_id do
      broadcast(state, "agent:#{parent_id}", {:child_created, id, self()})
    end
    
    {:ok, state}
  end
  
  def handle_call({:split_task, task, num_children}, _from, state) do
    Logger.info("Agent #{state.id} splitting task into #{num_children} subtasks")
    
    if state.depth >= state.max_depth do
      # At max depth, process task directly
      result = process_task_directly(task, state)
      {:reply, {:ok, result}, %{state | status: :completed}}
    else
      # Split into children
      state = %{state | task: task, status: :splitting}
      subtasks = split_into_subtasks(task, num_children)
      
      # Create child agents
      children = Enum.map(Enum.with_index(subtasks), fn {subtask, index} ->
        child_id = "#{state.id}_child_#{index}"
        {:ok, child_pid} = start_child(child_id, state.id, state.depth + 1, 
                                       state.max_depth, subtask, state.pubsub)
        
        # Monitor the child
        ref = Process.monitor(child_pid)
        
        %{
          id: child_id,
          pid: child_pid,
          task: subtask,
          status: :running,
          monitor_ref: ref
        }
      end)
      
      # Store monitor refs
      monitor_refs = Enum.reduce(children, %{}, fn child, acc ->
        Map.put(acc, child.monitor_ref, child.id)
      end)
      
      state = %{state | 
        children: children,
        monitor_refs: monitor_refs,
        status: :monitoring
      }
      
      # Start children processing
      Enum.each(children, fn child ->
        Task.start(fn ->
          Process.sleep(100) # Small delay to ensure setup
          RecursiveAgent.split_task(child.pid, child.task)
        end)
      end)
      
      {:reply, {:ok, :splitting}, state}
    end
  end
  
  def handle_call(:get_status, _from, state) do
    status = %{
      id: state.id,
      depth: state.depth,
      status: state.status,
      num_children: length(state.children),
      num_messages: length(state.collected_messages),
      children_status: Enum.map(state.children, & &1.status)
    }
    {:reply, status, state}
  end
  
  def handle_call(:get_collected_output, _from, state) do
    output = generate_output(state)
    {:reply, output, state}
  end
  
  # Handle messages from children
  def handle_info({:child_created, child_id, child_pid}, state) do
    Logger.info("Agent #{state.id} received child creation notification: #{child_id}")
    {:noreply, state}
  end
  
  def handle_info({:child_message, child_id, message}, state) do
    Logger.info("Agent #{state.id} received message from #{child_id}: #{inspect(message)}")
    
    state = %{state | 
      collected_messages: [{child_id, message} | state.collected_messages]
    }
    
    # Check if all children completed
    state = update_child_status(state, child_id, :completed)
    
    if all_children_completed?(state) do
      state = %{state | status: :aggregating}
      
      # Generate final output
      output = generate_output(state)
      
      # Notify parent if exists
      if state.parent_id do
        broadcast(state, "agent:#{state.parent_id}:children", 
                 {:child_message, state.id, output})
      end
      
      state = %{state | status: :completed}
    end
    
    {:noreply, state}
  end
  
  # Handle child process termination
  def handle_info({:DOWN, ref, :process, pid, reason}, state) do
    case Map.get(state.monitor_refs, ref) do
      nil -> 
        {:noreply, state}
      child_id ->
        Logger.warn("Child #{child_id} terminated: #{inspect(reason)}")
        state = update_child_status(state, child_id, :failed)
        
        # Handle failure
        if all_children_completed?(state) do
          output = generate_output(state)
          if state.parent_id do
            broadcast(state, "agent:#{state.parent_id}:children", 
                     {:child_message, state.id, output})
          end
          state = %{state | status: :completed}
        end
        
        {:noreply, state}
    end
  end
  
  def handle_info(msg, state) do
    Logger.debug("Agent #{state.id} received unknown message: #{inspect(msg)}")
    {:noreply, state}
  end
  
  # Private functions
  
  defp start_pubsub do
    children = [
      {Phoenix.PubSub, name: :recursive_agent_pubsub}
    ]
    
    case Supervisor.start_link(children, strategy: :one_for_one) do
      {:ok, _pid} -> :recursive_agent_pubsub
      {:error, {:already_started, _pid}} -> :recursive_agent_pubsub
    end
  end
  
  defp generate_id do
    :crypto.strong_rand_bytes(8) |> Base.encode16()
  end
  
  defp start_child(id, parent_id, depth, max_depth, task, pubsub) do
    RecursiveAgent.start_link(
      id: id,
      parent_id: parent_id,
      depth: depth,
      max_depth: max_depth,
      task: task,
      pubsub: pubsub
    )
  end
  
  defp split_into_subtasks(task, num_children) do
    case task do
      {:compute, range} ->
        # Split numeric range
        {start, finish} = range
        chunk_size = div(finish - start + 1, num_children)
        
        Enum.map(0..(num_children - 1), fn i ->
          chunk_start = start + (i * chunk_size)
          chunk_end = if i == num_children - 1 do
            finish
          else
            chunk_start + chunk_size - 1
          end
          {:compute, {chunk_start, chunk_end}}
        end)
        
      {:analyze, text} ->
        # Split text analysis
        words = String.split(text, " ")
        chunks = Enum.chunk_every(words, ceil(length(words) / num_children))
        Enum.map(chunks, fn chunk ->
          {:analyze, Enum.join(chunk, " ")}
        end)
        
      {:process, items} when is_list(items) ->
        # Split list processing
        chunks = Enum.chunk_every(items, ceil(length(items) / num_children))
        Enum.map(chunks, fn chunk ->
          {:process, chunk}
        end)
        
      _ ->
        # Generic task splitting
        Enum.map(1..num_children, fn i ->
          {:subtask, task, i, num_children}
        end)
    end
  end
  
  defp process_task_directly(task, state) do
    Logger.info("Agent #{state.id} processing task directly: #{inspect(task)}")
    
    result = case task do
      {:compute, {start, finish}} ->
        # Simulate computation
        sum = Enum.reduce(start..finish, 0, &+/2)
        Process.sleep(100) # Simulate work
        %{type: :computation, result: sum, range: {start, finish}}
        
      {:analyze, text} ->
        # Simulate text analysis
        word_count = length(String.split(text, " "))
        Process.sleep(50)
        %{type: :analysis, word_count: word_count, sample: String.slice(text, 0, 20)}
        
      {:process, items} ->
        # Simulate item processing
        processed = Enum.map(items, &process_item/1)
        Process.sleep(80)
        %{type: :processing, count: length(items), results: processed}
        
      _ ->
        # Generic processing
        Process.sleep(200)
        %{type: :generic, task: task, completed: true}
    end
    
    # Notify parent
    if state.parent_id do
      broadcast(state, "agent:#{state.parent_id}:children", 
               {:child_message, state.id, result})
    end
    
    result
  end
  
  defp process_item(item) do
    # Simulate item processing
    %{item: item, processed: true, timestamp: System.system_time(:millisecond)}
  end
  
  defp broadcast(state, topic, message) do
    Phoenix.PubSub.broadcast(state.pubsub, topic, message)
  end
  
  defp update_child_status(state, child_id, new_status) do
    children = Enum.map(state.children, fn child ->
      if child.id == child_id do
        %{child | status: new_status}
      else
        child
      end
    end)
    %{state | children: children}
  end
  
  defp all_children_completed?(state) do
    Enum.all?(state.children, fn child ->
      child.status in [:completed, :failed]
    end)
  end
  
  defp generate_output(state) do
    Logger.info("Agent #{state.id} generating output from #{length(state.collected_messages)} messages")
    
    %{
      agent_id: state.id,
      depth: state.depth,
      task: state.task,
      status: state.status,
      children_count: length(state.children),
      messages: state.collected_messages,
      aggregated_result: aggregate_results(state.collected_messages),
      timestamp: System.system_time(:millisecond)
    }
  end
  
  defp aggregate_results(messages) do
    results = Enum.map(messages, fn {_child_id, msg} -> msg end)
    
    # Aggregate based on result types
    cond do
      Enum.all?(results, &match?(%{type: :computation}, &1)) ->
        total = Enum.reduce(results, 0, fn %{result: sum}, acc -> acc + sum end)
        %{type: :computation_aggregate, total: total}
        
      Enum.all?(results, &match?(%{type: :analysis}, &1)) ->
        total_words = Enum.reduce(results, 0, fn %{word_count: wc}, acc -> acc + wc end)
        %{type: :analysis_aggregate, total_word_count: total_words}
        
      Enum.all?(results, &match?(%{type: :processing}, &1)) ->
        total_items = Enum.reduce(results, 0, fn %{count: c}, acc -> acc + c end)
        %{type: :processing_aggregate, total_items_processed: total_items}
        
      true ->
        %{type: :mixed_aggregate, results: results}
    end
  end
end

defmodule RecursiveAgentDemo do
  @moduledoc """
  Demonstration of recursive agent splitting and monitoring
  """
  
  def run do
    IO.puts("\n🚀 Starting Recursive Agent Demonstration\n")
    
    # Demo 1: Computation splitting
    IO.puts("📊 Demo 1: Recursive Computation")
    demo_computation()
    Process.sleep(3000)
    
    # Demo 2: Text analysis splitting
    IO.puts("\n📝 Demo 2: Recursive Text Analysis")
    demo_text_analysis()
    Process.sleep(3000)
    
    # Demo 3: Multi-level recursion
    IO.puts("\n🌳 Demo 3: Deep Recursive Processing")
    demo_deep_recursion()
    Process.sleep(5000)
    
    IO.puts("\n✅ Demonstration complete!")
  end
  
  defp demo_computation do
    {:ok, agent} = RecursiveAgent.start_link(
      id: "root_compute",
      max_depth: 2
    )
    
    # Split computation across range
    task = {:compute, {1, 1000}}
    {:ok, :splitting} = RecursiveAgent.split_task(agent, task, 4)
    
    # Monitor progress
    monitor_agent(agent, "Computation")
  end
  
  defp demo_text_analysis do
    text = """
    The recursive agent system demonstrates powerful capabilities for distributed
    processing. It can split tasks, monitor children, collect results, and
    aggregate outputs. This enables complex hierarchical processing patterns.
    Each agent operates independently while maintaining communication with its
    parent and children through a publish-subscribe system.
    """
    
    {:ok, agent} = RecursiveAgent.start_link(
      id: "root_analyze",
      max_depth: 2
    )
    
    task = {:analyze, text}
    {:ok, :splitting} = RecursiveAgent.split_task(agent, task, 3)
    
    monitor_agent(agent, "Text Analysis")
  end
  
  defp demo_deep_recursion do
    items = Enum.to_list(1..100)
    
    {:ok, agent} = RecursiveAgent.start_link(
      id: "root_deep",
      max_depth: 3
    )
    
    task = {:process, items}
    {:ok, :splitting} = RecursiveAgent.split_task(agent, task, 2)
    
    monitor_agent(agent, "Deep Recursion", 4000)
  end
  
  defp monitor_agent(agent, label, timeout \\ 2000) do
    # Poll for status updates
    Task.start(fn ->
      Enum.each(1..20, fn i ->
        Process.sleep(100)
        status = RecursiveAgent.get_status(agent)
        
        if rem(i, 5) == 0 do
          IO.puts("#{label} Status: #{inspect(status)}")
        end
        
        if status.status == :completed do
          output = RecursiveAgent.get_collected_output(agent)
          IO.puts("\n#{label} Final Output:")
          IO.inspect(output, pretty: true, limit: :infinity)
          Process.exit(self(), :normal)
        end
      end)
    end)
    
    Process.sleep(timeout)
  end
end

# Visualization module
defmodule RecursiveAgentVisualizer do
  @moduledoc """
  ASCII visualization of recursive agent hierarchy
  """
  
  def visualize_tree(agent_data, depth \\ 0) do
    indent = String.duplicate("  ", depth)
    
    IO.puts("#{indent}📦 Agent: #{agent_data.agent_id}")
    IO.puts("#{indent}├─ Depth: #{agent_data.depth}")
    IO.puts("#{indent}├─ Status: #{agent_data.status}")
    IO.puts("#{indent}├─ Children: #{agent_data.children_count}")
    
    if agent_data[:aggregated_result] do
      IO.puts("#{indent}└─ Result: #{inspect(agent_data.aggregated_result)}")
    end
    
    # Recursively show children data if available
    if agent_data[:messages] && length(agent_data.messages) > 0 do
      IO.puts("#{indent}└─ Messages:")
      Enum.each(agent_data.messages, fn {child_id, msg} ->
        if is_map(msg) && Map.has_key?(msg, :agent_id) do
          visualize_tree(msg, depth + 1)
        else
          IO.puts("#{indent}    └─ #{child_id}: #{inspect(msg)}")
        end
      end)
    end
  end
end

# Run the demonstration
RecursiveAgentDemo.run()