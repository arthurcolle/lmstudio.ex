#!/usr/bin/env elixir

defmodule SimpleRecursiveAgent do
  @moduledoc """
  Simple demonstration of recursive agent splitting, monitoring, and message collection
  """
  
  def demonstrate do
    IO.puts("\n🚀 Recursive Agent Demonstration: Splitting, Monitoring & Message Collection\n")
    
    # Start root agent
    root_pid = spawn_agent("root", nil, 0)
    
    # Execute recursive task
    task = {:compute_sum, 1, 100}
    send(root_pid, {:execute, task, self()})
    
    # Wait for result
    receive do
      {:result, result} ->
        IO.puts("\n✅ Final Result: #{inspect(result)}")
    after
      5000 -> IO.puts("❌ Timeout waiting for result")
    end
  end
  
  def spawn_agent(id, parent_pid, depth) do
    spawn(fn -> agent_loop(%{
      id: id,
      parent: parent_pid,
      depth: depth,
      max_depth: 2,
      children: [],
      collected_messages: [],
      status: :idle
    }) end)
  end
  
  def agent_loop(state) do
    receive do
      {:execute, task, reply_to} ->
        IO.puts("#{indent(state.depth)}📦 Agent #{state.id} received task: #{inspect(task)}")
        
        if state.depth >= state.max_depth do
          # Leaf node - process directly
          result = process_task(task, state)
          IO.puts("#{indent(state.depth)}✅ Agent #{state.id} completed with result: #{result}")
          
          # Send result to parent or original requester
          target = state.parent || reply_to
          send(target, {:child_result, state.id, result})
          
          agent_loop(%{state | status: :completed})
        else
          # Branch node - split and delegate
          subtasks = split_task(task)
          IO.puts("#{indent(state.depth)}🔀 Agent #{state.id} splitting into #{length(subtasks)} children")
          
          # Spawn and monitor children
          children = Enum.with_index(subtasks) |> Enum.map(fn {subtask, idx} ->
            child_id = "#{state.id}_child#{idx}"
            child_pid = spawn_agent(child_id, self(), state.depth + 1)
            
            # Monitor the child
            Process.monitor(child_pid)
            
            # Send task to child
            send(child_pid, {:execute, subtask, nil})
            
            {child_id, child_pid}
          end)
          
          agent_loop(Map.merge(state, %{
            status: :waiting_for_children,
            children: children,
            reply_to: reply_to
          }))
        end
        
      {:child_result, child_id, result} ->
        IO.puts("#{indent(state.depth)}📨 Agent #{state.id} received from #{child_id}: #{result}")
        
        # Collect the message
        new_messages = [{child_id, result} | state.collected_messages]
        new_state = %{state | collected_messages: new_messages}
        
        # Check if all children reported
        if length(new_messages) == length(state.children) do
          # Aggregate results
          aggregated = aggregate_results(new_messages)
          IO.puts("#{indent(state.depth)}📊 Agent #{state.id} aggregated result: #{aggregated}")
          
          # Report to parent or original requester
          if state.parent do
            send(state.parent, {:child_result, state.id, aggregated})
          else
            # Root agent - send final result
            send(state.reply_to, {:result, aggregated})
          end
          
          agent_loop(%{new_state | status: :completed})
        else
          agent_loop(new_state)
        end
        
      {:DOWN, _ref, :process, pid, reason} ->
        # Handle child process failure
        child_id = Enum.find_value(state.children, fn {id, p} -> 
          if p == pid, do: id
        end)
        
        IO.puts("#{indent(state.depth)}⚠️  Agent #{state.id}: child #{child_id} failed (#{inspect(reason)})")
        
        # Could implement retry logic or partial result handling here
        agent_loop(state)
        
      {:get_status, reply_to} ->
        send(reply_to, {:status, %{
          id: state.id,
          depth: state.depth,
          status: state.status,
          children_count: length(state.children),
          messages_collected: length(state.collected_messages)
        }})
        agent_loop(state)
    end
  end
  
  defp split_task({:compute_sum, start, finish}) do
    mid = div(start + finish, 2)
    [
      {:compute_sum, start, mid},
      {:compute_sum, mid + 1, finish}
    ]
  end
  
  defp process_task({:compute_sum, start, finish}, _state) do
    # Simulate some work
    Process.sleep(100)
    
    # Compute the sum
    Enum.sum(start..finish)
  end
  
  defp aggregate_results(messages) do
    messages
    |> Enum.map(fn {_child_id, result} -> result end)
    |> Enum.sum()
  end
  
  defp indent(depth), do: String.duplicate("  ", depth)
end

defmodule RecursiveAgentVisualization do
  @moduledoc """
  Visualization of the recursive agent execution flow
  """
  
  def show_execution_flow do
    IO.puts("\n📊 Execution Flow Visualization:\n")
    
    IO.puts("""
    🌳 Agent Tree Structure:
    
    root (depth: 0)
    ├─ root_child0 (depth: 1)
    │  ├─ root_child0_child0 (depth: 2) [leaf]
    │  └─ root_child0_child1 (depth: 2) [leaf]
    └─ root_child1 (depth: 1)
       ├─ root_child1_child0 (depth: 2) [leaf]
       └─ root_child1_child1 (depth: 2) [leaf]
    
    📨 Message Flow:
    
    1. Root receives task: sum(1, 100)
    2. Root splits → [sum(1, 50), sum(51, 100)]
    3. Children split → [[sum(1, 25), sum(26, 50)], [sum(51, 75), sum(76, 100)]]
    4. Leaves compute: [325, 950, 1575, 2200]
    5. Parents aggregate: [1275, 3775]
    6. Root aggregates: 5050
    
    🔄 Monitoring:
    
    • Each parent monitors its children
    • Failures are detected via Process.monitor
    • Parents can retry or handle partial results
    """)
  end
end

defmodule AdvancedFeatures do
  @moduledoc """
  Description of advanced features in a recursive agent system
  """
  
  def describe_features do
    IO.puts("\n🎯 Advanced Recursive Agent Features:\n")
    
    IO.puts("""
    1. **Dynamic Task Splitting**
       - Adaptive split factor based on task complexity
       - Load balancing across children
       - Different splitting strategies per task type
    
    2. **Fault Tolerance**
       - Child failure detection and handling
       - Retry mechanisms with backoff
       - Partial result aggregation
       - Circuit breaker patterns
    
    3. **Message Collection Strategies**
       - Streaming results as they arrive
       - Buffering with timeout
       - Priority-based collection
       - Deduplication
    
    4. **Resource Management**
       - Process pool limits
       - Memory usage monitoring
       - CPU throttling
       - Backpressure handling
    
    5. **Observability**
       - Distributed tracing
       - Metrics collection
       - Real-time visualization
       - Performance profiling
    
    6. **Advanced Patterns**
       - MapReduce implementation
       - Fork-join parallelism
       - Pipeline processing
       - Scatter-gather
    """)
  end
end

# Run the demonstration
SimpleRecursiveAgent.demonstrate()
RecursiveAgentVisualization.show_execution_flow()
AdvancedFeatures.describe_features()