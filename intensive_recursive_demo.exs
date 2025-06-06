#!/usr/bin/env elixir

defmodule IntensiveRecursiveDemo do
  @moduledoc """
  Demonstrates computationally intensive recursive agent problems
  """

  defmodule CryptoMining do
    @doc """
    Simulates cryptocurrency mining by finding nonces that produce hashes with specific prefixes
    """
    def mine_blocks(start_nonce, end_nonce, difficulty) do
      prefix = String.duplicate("0", difficulty)
      
      start_nonce..end_nonce
      |> Enum.reduce(%{}, fn nonce, acc ->
        data = "block_#{nonce}"
        hash = :crypto.hash(:sha256, data) |> Base.encode16()
        
        if String.starts_with?(hash, prefix) do
          if rem(nonce, 100_000) == 0 do
            IO.puts("⛏️  Found golden nonce at #{nonce}: #{hash}")
          end
          Map.put(acc, nonce, hash)
        else
          if rem(nonce, 1_000_000) == 0 do
            IO.puts("🔍 Checked #{nonce} nonces...")
          end
          acc
        end
      end)
    end
  end

  defmodule MandelbrotSet do
    @doc """
    Computes the Mandelbrot set for a given region
    """
    def compute_region(x_min, x_max, y_min, y_max, resolution) do
      x_step = (x_max - x_min) / resolution
      y_step = (y_max - y_min) / resolution
      
      points = for i <- 0..resolution-1,
                   j <- 0..resolution-1 do
        x = x_min + i * x_step
        y = y_min + j * y_step
        iterations = mandelbrot_iterations(x, y, 1000)
        {{x, y}, iterations}
      end
      
      total_iterations = points |> Enum.map(&elem(&1, 1)) |> Enum.sum()
      
      %{
        points: length(points),
        total_iterations: total_iterations,
        average_iterations: total_iterations / length(points)
      }
    end
    
    defp mandelbrot_iterations(x0, y0, max_iter) do
      mandelbrot_iter(0, 0, x0, y0, 0, max_iter)
    end
    
    defp mandelbrot_iter(_x, _y, _x0, _y0, iter, max_iter) when iter >= max_iter, do: max_iter
    defp mandelbrot_iter(x, y, _x0, _y0, iter, _max_iter) when x*x + y*y > 4, do: iter
    defp mandelbrot_iter(x, y, x0, y0, iter, max_iter) do
      x_new = x*x - y*y + x0
      y_new = 2*x*y + y0
      mandelbrot_iter(x_new, y_new, x0, y0, iter + 1, max_iter)
    end
  end

  defmodule PrimeSearch do
    @doc """
    Searches for large prime numbers using trial division
    """
    def find_primes_in_range(start, stop) do
      start..stop
      |> Enum.filter(&is_prime?/1)
      |> Enum.map(fn prime ->
        if rem(prime - start, 10000) == 0 do
          IO.puts("🔢 Found prime: #{prime}")
        end
        prime
      end)
    end
    
    defp is_prime?(n) when n < 2, do: false
    defp is_prime?(2), do: true
    defp is_prime?(n) when rem(n, 2) == 0, do: false
    defp is_prime?(n) do
      limit = :math.sqrt(n) |> ceil()
      not Enum.any?(3..limit//2, fn i -> rem(n, i) == 0 end)
    end
  end

  defmodule RecursiveAgent do
    def start(name, depth \\ 0, parent \\ nil) do
      spawn(fn -> loop(name, depth, parent, %{children: [], start_time: nil}) end)
    end

    defp loop(name, depth, parent, state) do
      receive do
        {:task, caller, task} ->
          IO.puts("\n🚀 Agent #{name} (depth: #{depth}) starting #{elem(task, 0)}")
          start_time = System.monotonic_time(:millisecond)
          new_state = Map.put(state, :start_time, start_time)
          
          case task do
            {:crypto_mining, start_nonce, end_nonce, _difficulty} when end_nonce - start_nonce > 10_000_000 ->
              # Split large mining tasks
              handle_split_mining(name, depth, parent, new_state, caller, task)
              
            {:crypto_mining, start_nonce, end_nonce, difficulty} ->
              # Mine directly
              IO.puts("⛏️  Agent #{name} mining nonces #{start_nonce}..#{end_nonce} (difficulty: #{difficulty})")
              
              result = CryptoMining.mine_blocks(start_nonce, end_nonce, difficulty)
              
              elapsed = System.monotonic_time(:millisecond) - start_time
              IO.puts("✅ Agent #{name} found #{map_size(result)} valid blocks in #{elapsed/1000}s")
              
              send(caller, {:result, name, result})
              if parent, do: send(parent, {:child_result, name, result})
              loop(name, depth, parent, new_state)

            {:mandelbrot, region, resolution} when resolution > 500 ->
              # Split large mandelbrot computations
              handle_split_mandelbrot(name, depth, parent, new_state, caller, region, resolution)
              
            {:mandelbrot, {x_min, x_max, y_min, y_max}, resolution} ->
              # Compute directly
              IO.puts("🎨 Agent #{name} computing Mandelbrot region (#{resolution}x#{resolution})")
              
              result = MandelbrotSet.compute_region(x_min, x_max, y_min, y_max, resolution)
              
              elapsed = System.monotonic_time(:millisecond) - start_time
              IO.puts("✅ Agent #{name} computed #{result.points} points in #{elapsed/1000}s")
              
              send(caller, {:result, name, result})
              if parent, do: send(parent, {:child_result, name, result})
              loop(name, depth, parent, new_state)

            {:prime_search, start, stop} when stop - start > 100_000 ->
              # Split large prime searches
              handle_split_prime_search(name, depth, parent, new_state, caller, start, stop)
              
            {:prime_search, start, stop} ->
              # Search directly
              IO.puts("🔍 Agent #{name} searching for primes in #{start}..#{stop}")
              
              primes = PrimeSearch.find_primes_in_range(start, stop)
              
              elapsed = System.monotonic_time(:millisecond) - start_time
              IO.puts("✅ Agent #{name} found #{length(primes)} primes in #{elapsed/1000}s")
              
              send(caller, {:result, name, primes})
              if parent, do: send(parent, {:child_result, name, primes})
              loop(name, depth, parent, new_state)
          end

        {:child_result, child_name, result} ->
          IO.puts("📥 Agent #{name} received result from #{child_name}")
          new_state = update_in(state.children, fn children ->
            Enum.map(children, fn
              {^child_name, nil} -> {child_name, result}
              other -> other
            end)
          end)
          
          if all_children_complete?(new_state.children) do
            aggregated = aggregate_results(new_state.children, state.task_type)
            elapsed = System.monotonic_time(:millisecond) - state.start_time
            
            IO.puts("📊 Agent #{name} aggregated results in #{elapsed/1000}s")
            
            if state.original_caller do
              send(state.original_caller, {:result, name, aggregated})
            end
            if parent do
              send(parent, {:child_result, name, aggregated})
            end
          end
          
          loop(name, depth, parent, new_state)
      end
    end

    defp handle_split_mining(name, depth, parent, state, caller, {:crypto_mining, start, stop, difficulty}) do
      IO.puts("🔀 Splitting mining task into 4 parts")
      
      chunk_size = div(stop - start, 4)
      children = for i <- 0..3 do
        child_start = start + (i * chunk_size)
        child_end = if i == 3, do: stop, else: child_start + chunk_size
        
        child_name = "#{name}_m#{i}"
        child_pid = start(child_name, depth + 1, self())
        
        send(child_pid, {:task, self(), {:crypto_mining, child_start, child_end, difficulty}})
        
        {child_name, nil}
      end
      
      new_state = state
        |> Map.put(:children, children)
        |> Map.put(:original_caller, caller)
        |> Map.put(:task_type, :crypto_mining)
      
      loop(name, depth, parent, new_state)
    end

    defp handle_split_mandelbrot(name, depth, parent, state, caller, {x_min, x_max, y_min, y_max}, resolution) do
      IO.puts("🔀 Splitting Mandelbrot into 4 quadrants")
      
      x_mid = (x_min + x_max) / 2
      y_mid = (y_min + y_max) / 2
      new_res = div(resolution, 2)
      
      quadrants = [
        {x_min, x_mid, y_min, y_mid},
        {x_mid, x_max, y_min, y_mid},
        {x_min, x_mid, y_mid, y_max},
        {x_mid, x_max, y_mid, y_max}
      ]
      
      children = for {i, {x1, x2, y1, y2}} <- Enum.with_index(quadrants) do
        child_name = "#{name}_q#{i}"
        child_pid = start(child_name, depth + 1, self())
        
        send(child_pid, {:task, self(), {:mandelbrot, {x1, x2, y1, y2}, new_res}})
        
        {child_name, nil}
      end
      
      new_state = state
        |> Map.put(:children, children)
        |> Map.put(:original_caller, caller)
        |> Map.put(:task_type, :mandelbrot)
      
      loop(name, depth, parent, new_state)
    end

    defp handle_split_prime_search(name, depth, parent, state, caller, start, stop) do
      IO.puts("🔀 Splitting prime search into 4 parts")
      
      chunk_size = div(stop - start, 4)
      children = for i <- 0..3 do
        child_start = start + (i * chunk_size)
        child_end = if i == 3, do: stop, else: child_start + chunk_size
        
        child_name = "#{name}_p#{i}"
        child_pid = start(child_name, depth + 1, self())
        
        send(child_pid, {:task, self(), {:prime_search, child_start, child_end}})
        
        {child_name, nil}
      end
      
      new_state = state
        |> Map.put(:children, children)
        |> Map.put(:original_caller, caller)
        |> Map.put(:task_type, :prime_search)
      
      loop(name, depth, parent, new_state)
    end

    defp all_children_complete?(children) do
      Enum.all?(children, fn {_name, result} -> result != nil end)
    end

    defp aggregate_results(children, :crypto_mining) do
      children
      |> Enum.map(fn {_name, result} -> result end)
      |> Enum.reduce(%{}, &Map.merge/2)
    end

    defp aggregate_results(children, :mandelbrot) do
      stats = children
      |> Enum.map(fn {_name, result} -> result end)
      
      %{
        points: stats |> Enum.map(& &1.points) |> Enum.sum(),
        total_iterations: stats |> Enum.map(& &1.total_iterations) |> Enum.sum(),
        average_iterations: 
          Enum.sum(Enum.map(stats, & &1.total_iterations)) / 
          Enum.sum(Enum.map(stats, & &1.points))
      }
    end

    defp aggregate_results(children, :prime_search) do
      children
      |> Enum.map(fn {_name, primes} -> primes end)
      |> List.flatten()
      |> Enum.sort()
    end
  end

  def run_intensive_demo do
    IO.puts("\n🔥 INTENSIVE RECURSIVE AGENT DEMONSTRATION 🔥")
    IO.puts("=" <> String.duplicate("=", 70))
    IO.puts("This will run computationally intensive tasks that take several minutes")
    
    # Demo 1: Cryptocurrency Mining Simulation
    IO.puts("\n\n📋 Demo 1: Distributed Cryptocurrency Mining")
    IO.puts("Mining blocks with SHA-256 hashes starting with '00000'")
    IO.puts("Searching through 50 million nonces")
    IO.puts("Expected time: 3-5 minutes\n")
    
    root1 = RecursiveAgent.start("mining_root")
    start_time1 = System.monotonic_time(:second)
    
    send(root1, {:task, self(), {:crypto_mining, 1, 50_000_000, 5}})
    
    receive do
      {:result, _name, blocks} ->
        elapsed1 = System.monotonic_time(:second) - start_time1
        IO.puts("\n\n✅ Mining completed in #{elapsed1} seconds!")
        IO.puts("📊 Found #{map_size(blocks)} valid blocks")
        
        sample = blocks |> Enum.take(3) |> Enum.map(fn {nonce, hash} ->
          "  Nonce #{nonce}: #{hash}"
        end) |> Enum.join("\n")
        
        IO.puts("🏆 Sample blocks:\n#{sample}")
    after
      600_000 -> IO.puts("⏱️  Mining timed out after 10 minutes")
    end
    
    # Demo 2: Mandelbrot Set Computation
    IO.puts("\n\n📋 Demo 2: Parallel Mandelbrot Set Computation")
    IO.puts("Computing a 2000x2000 region of the Mandelbrot set")
    IO.puts("Region: [-2, 1] x [-1.5, 1.5]")
    IO.puts("Expected time: 2-4 minutes\n")
    
    root2 = RecursiveAgent.start("mandelbrot_root")
    start_time2 = System.monotonic_time(:second)
    
    send(root2, {:task, self(), {:mandelbrot, {-2.0, 1.0, -1.5, 1.5}, 2000}})
    
    receive do
      {:result, _name, result} ->
        elapsed2 = System.monotonic_time(:second) - start_time2
        IO.puts("\n\n✅ Mandelbrot computation completed in #{elapsed2} seconds!")
        IO.puts("📊 Computed #{result.points} points")
        IO.puts("🎨 Total iterations: #{result.total_iterations}")
        IO.puts("📈 Average iterations per point: #{Float.round(result.average_iterations, 2)}")
    after
      600_000 -> IO.puts("⏱️  Mandelbrot timed out after 10 minutes")
    end
    
    # Demo 3: Large Prime Number Search
    IO.puts("\n\n📋 Demo 3: Distributed Prime Number Search")
    IO.puts("Finding all prime numbers between 10,000,000 and 11,000,000")
    IO.puts("Using trial division algorithm")
    IO.puts("Expected time: 2-3 minutes\n")
    
    root3 = RecursiveAgent.start("prime_root")
    start_time3 = System.monotonic_time(:second)
    
    send(root3, {:task, self(), {:prime_search, 10_000_000, 11_000_000}})
    
    receive do
      {:result, _name, primes} ->
        elapsed3 = System.monotonic_time(:second) - start_time3
        IO.puts("\n\n✅ Prime search completed in #{elapsed3} seconds!")
        IO.puts("📊 Found #{length(primes)} prime numbers")
        
        sample = primes |> Enum.take(10) |> Enum.join(", ")
        IO.puts("🔢 First 10 primes: #{sample}")
        
        largest = primes |> Enum.take(-5) |> Enum.join(", ")
        IO.puts("🏆 Last 5 primes: #{largest}")
    after
      600_000 -> IO.puts("⏱️  Prime search timed out after 10 minutes")
    end
    
    IO.puts("\n\n🎉 Demonstration Complete!")
    IO.puts("This showcased:")
    IO.puts("  • CPU-intensive parallel computations")
    IO.puts("  • Dynamic work distribution")
    IO.puts("  • Hierarchical aggregation of results")
    IO.puts("  • Real-world algorithm parallelization")
  end
end

# Run the intensive demonstration
IntensiveRecursiveDemo.run_intensive_demo()