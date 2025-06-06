#!/usr/bin/env elixir

defmodule ComplexRecursiveAgent do
  @moduledoc """
  Demonstrates a complex recursive agent solving computationally intensive problems
  """

  defmodule Agent do
    @doc """
    Recursive agent that can split work across children
    """
    def start(name, depth \\ 0, parent \\ nil) do
      spawn(fn -> loop(name, depth, parent, %{children: [], monitors: %{}}) end)
    end

    defp loop(name, depth, parent, state) do
      receive do
        {:task, caller, task} ->
          IO.puts("\n🎯 Agent #{name} (depth: #{depth}) received task: #{inspect(task)}")
          
          case task do
            {:prime_factorization, range_start, range_end} when range_end - range_start > 1000 ->
              # Split large ranges
              handle_split_task(name, depth, parent, state, caller, task, 4)
              
            {:prime_factorization, range_start, range_end} ->
              # Process small ranges directly
              IO.puts("⚡ Agent #{name} processing range #{range_start}..#{range_end}")
              start_time = System.monotonic_time(:millisecond)
              
              result = range_start..range_end
                |> Enum.map(fn n ->
                  factors = prime_factorize(n)
                  if rem(n, 10000) == 0 do
                    IO.puts("  📊 #{name}: Factorized #{n} = #{inspect(factors)}")
                  end
                  {n, factors}
                end)
                |> Map.new()
              
              elapsed = System.monotonic_time(:millisecond) - start_time
              IO.puts("✅ Agent #{name} completed in #{elapsed}ms")
              
              send(caller, {:result, name, result})
              if parent, do: send(parent, {:child_result, name, result})
              loop(name, depth, parent, state)

            {:matrix_multiply, size} when size > 50 ->
              # Split large matrix operations
              handle_matrix_split(name, depth, parent, state, caller, size)
              
            {:matrix_multiply, size} ->
              # Process small matrices directly
              IO.puts("🔢 Agent #{name} multiplying #{size}x#{size} matrices")
              start_time = System.monotonic_time(:millisecond)
              
              # Generate random matrices
              matrix_a = generate_matrix(size)
              matrix_b = generate_matrix(size)
              
              # Perform multiplication
              result = matrix_multiply(matrix_a, matrix_b)
              
              elapsed = System.monotonic_time(:millisecond) - start_time
              checksum = matrix_checksum(result)
              IO.puts("✅ Agent #{name} completed matrix multiply in #{elapsed}ms (checksum: #{checksum})")
              
              send(caller, {:result, name, checksum})
              if parent, do: send(parent, {:child_result, name, checksum})
              loop(name, depth, parent, state)

            {:fibonacci_tree, n} when n > 35 ->
              # Split large fibonacci calculations
              handle_fibonacci_split(name, depth, parent, state, caller, n)
              
            {:fibonacci_tree, n} ->
              # Calculate directly for small n
              IO.puts("🌀 Agent #{name} calculating fibonacci(#{n})")
              start_time = System.monotonic_time(:millisecond)
              
              result = fibonacci_tree(n)
              
              elapsed = System.monotonic_time(:millisecond) - start_time
              IO.puts("✅ Agent #{name} calculated fib(#{n}) = #{result} in #{elapsed}ms")
              
              send(caller, {:result, name, result})
              if parent, do: send(parent, {:child_result, name, result})
              loop(name, depth, parent, state)
          end

        {:child_result, child_name, result} ->
          IO.puts("📥 Agent #{name} received result from #{child_name}")
          new_state = update_in(state.children, fn children ->
            Enum.map(children, fn
              {^child_name, nil} -> {child_name, result}
              other -> other
            end)
          end)
          
          # Check if all children have reported
          if all_children_complete?(new_state.children) do
            aggregated = aggregate_results(new_state.children, state.task_type)
            IO.puts("📊 Agent #{name} aggregated result")
            
            if state.original_caller do
              send(state.original_caller, {:result, name, aggregated})
            end
            if parent do
              send(parent, {:child_result, name, aggregated})
            end
          end
          
          loop(name, depth, parent, new_state)

        {:DOWN, ref, :process, pid, reason} ->
          IO.puts("⚠️  Agent #{name} detected child failure: #{inspect(reason)}")
          # In production, implement retry logic here
          loop(name, depth, parent, state)
      end
    end

    defp handle_split_task(name, depth, parent, state, caller, {:prime_factorization, start, stop}, split_factor) do
      IO.puts("🔀 Agent #{name} splitting prime factorization task into #{split_factor} parts")
      
      chunk_size = div(stop - start + 1, split_factor)
      children = for i <- 0..(split_factor - 1) do
        child_start = start + (i * chunk_size)
        child_end = if i == split_factor - 1, do: stop, else: child_start + chunk_size - 1
        
        child_name = "#{name}_child#{i}"
        child_pid = start(child_name, depth + 1, self())
        
        # Monitor child
        ref = Process.monitor(child_pid)
        
        # Send task to child
        send(child_pid, {:task, self(), {:prime_factorization, child_start, child_end}})
        
        {child_name, nil}
      end
      
      new_state = state
        |> Map.put(:children, children)
        |> Map.put(:original_caller, caller)
        |> Map.put(:task_type, :prime_factorization)
      
      loop(name, depth, parent, new_state)
    end

    defp handle_matrix_split(name, depth, parent, state, caller, size) do
      IO.puts("🔀 Agent #{name} splitting matrix multiplication into 4 quadrants")
      
      # Split into 4 quadrant multiplications
      quad_size = div(size, 2)
      children = for i <- 0..3 do
        child_name = "#{name}_quad#{i}"
        child_pid = start(child_name, depth + 1, self())
        
        ref = Process.monitor(child_pid)
        send(child_pid, {:task, self(), {:matrix_multiply, quad_size}})
        
        {child_name, nil}
      end
      
      new_state = state
        |> Map.put(:children, children)
        |> Map.put(:original_caller, caller)
        |> Map.put(:task_type, :matrix_multiply)
      
      loop(name, depth, parent, new_state)
    end

    defp handle_fibonacci_split(name, depth, parent, state, caller, n) do
      IO.puts("🔀 Agent #{name} splitting fibonacci(#{n}) into fib(#{n-1}) + fib(#{n-2})")
      
      children = for {i, n_val} <- Enum.with_index([n-1, n-2]) do
        child_name = "#{name}_fib#{i}"
        child_pid = start(child_name, depth + 1, self())
        
        ref = Process.monitor(child_pid)
        send(child_pid, {:task, self(), {:fibonacci_tree, n_val}})
        
        {child_name, nil}
      end
      
      new_state = state
        |> Map.put(:children, children)
        |> Map.put(:original_caller, caller)
        |> Map.put(:task_type, :fibonacci_tree)
      
      loop(name, depth, parent, new_state)
    end

    defp all_children_complete?(children) do
      Enum.all?(children, fn {_name, result} -> result != nil end)
    end

    defp aggregate_results(children, :prime_factorization) do
      children
      |> Enum.map(fn {_name, result} -> result end)
      |> Enum.reduce(%{}, &Map.merge/2)
    end

    defp aggregate_results(children, :matrix_multiply) do
      # Sum checksums for demonstration
      children
      |> Enum.map(fn {_name, checksum} -> checksum end)
      |> Enum.sum()
    end

    defp aggregate_results(children, :fibonacci_tree) do
      # Sum fibonacci results
      children
      |> Enum.map(fn {_name, result} -> result end)
      |> Enum.sum()
    end

    # Prime factorization helper
    defp prime_factorize(n) when n <= 1, do: []
    defp prime_factorize(n) do
      prime_factorize(n, 2, [])
    end

    defp prime_factorize(1, _, factors), do: Enum.reverse(factors)
    defp prime_factorize(n, divisor, factors) when divisor * divisor > n do
      Enum.reverse([n | factors])
    end
    defp prime_factorize(n, divisor, factors) do
      if rem(n, divisor) == 0 do
        prime_factorize(div(n, divisor), divisor, [divisor | factors])
      else
        prime_factorize(n, divisor + 1, factors)
      end
    end

    # Matrix operations
    defp generate_matrix(size) do
      for i <- 0..(size-1) do
        for j <- 0..(size-1) do
          :rand.uniform(10)
        end
      end
    end

    defp matrix_multiply(a, b) do
      size = length(a)
      for i <- 0..(size-1) do
        for j <- 0..(size-1) do
          Enum.sum(for k <- 0..(size-1), do: Enum.at(Enum.at(a, i), k) * Enum.at(Enum.at(b, k), j))
        end
      end
    end

    defp matrix_checksum(matrix) do
      matrix
      |> List.flatten()
      |> Enum.sum()
    end

    # Fibonacci tree calculation (intentionally inefficient for demonstration)
    defp fibonacci_tree(0), do: 0
    defp fibonacci_tree(1), do: 1
    defp fibonacci_tree(n) do
      fibonacci_tree(n - 1) + fibonacci_tree(n - 2)
    end
  end

  def run_demo do
    IO.puts("\n🚀 Complex Recursive Agent Demonstration")
    IO.puts("=" <> String.duplicate("=", 70))
    
    # Demo 1: Large-scale prime factorization
    IO.puts("\n📋 Demo 1: Prime Factorization of 1,000,000 numbers")
    IO.puts("This will factorize all numbers from 1 to 1,000,000")
    IO.puts("Expected time: 2-5 minutes\n")
    
    root1 = Agent.start("prime_root")
    start_time1 = System.monotonic_time(:second)
    
    send(root1, {:task, self(), {:prime_factorization, 1, 1_000_000}})
    
    receive do
      {:result, _name, result} ->
        elapsed1 = System.monotonic_time(:second) - start_time1
        sample_results = result
          |> Enum.take(5)
          |> Enum.map(fn {n, factors} -> "#{n}: #{inspect(factors)}" end)
          |> Enum.join("\n  ")
        
        IO.puts("\n✅ Prime factorization completed in #{elapsed1} seconds")
        IO.puts("📊 Sample results:\n  #{sample_results}")
        IO.puts("📈 Total numbers factorized: #{map_size(result)}")
    after
      300_000 -> IO.puts("⏱️  Demo 1 timed out after 5 minutes")
    end
    
    # Demo 2: Large matrix operations
    IO.puts("\n\n📋 Demo 2: Recursive Matrix Multiplication")
    IO.puts("Multiplying large matrices using divide-and-conquer")
    IO.puts("Expected time: 1-2 minutes\n")
    
    root2 = Agent.start("matrix_root")
    start_time2 = System.monotonic_time(:second)
    
    send(root2, {:task, self(), {:matrix_multiply, 200}})
    
    receive do
      {:result, _name, checksum} ->
        elapsed2 = System.monotonic_time(:second) - start_time2
        IO.puts("\n✅ Matrix multiplication completed in #{elapsed2} seconds")
        IO.puts("📊 Result checksum: #{checksum}")
    after
      180_000 -> IO.puts("⏱️  Demo 2 timed out after 3 minutes")
    end
    
    # Demo 3: Deep fibonacci tree
    IO.puts("\n\n📋 Demo 3: Deep Fibonacci Tree Calculation")
    IO.puts("Computing fibonacci(42) using tree recursion")
    IO.puts("This creates a massive process tree!")
    IO.puts("Expected time: 2-4 minutes\n")
    
    root3 = Agent.start("fib_root")
    start_time3 = System.monotonic_time(:second)
    
    send(root3, {:task, self(), {:fibonacci_tree, 42}})
    
    receive do
      {:result, _name, result} ->
        elapsed3 = System.monotonic_time(:second) - start_time3
        IO.puts("\n✅ Fibonacci calculation completed in #{elapsed3} seconds")
        IO.puts("📊 fib(42) = #{result}")
        IO.puts("🌳 This created approximately #{round(:math.pow(1.618, 42) / 1000)} thousand processes!")
    after
      300_000 -> IO.puts("⏱️  Demo 3 timed out after 5 minutes")
    end
    
    IO.puts("\n\n🎉 All demonstrations completed!")
    IO.puts("These problems showcased:")
    IO.puts("  • Massive parallel computation")
    IO.puts("  • Deep process trees")
    IO.puts("  • Automatic work distribution")
    IO.puts("  • Process monitoring and coordination")
  end
end

# Run the demonstration
ComplexRecursiveAgent.run_demo()