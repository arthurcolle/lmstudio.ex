defmodule MinimalTest do
  def start do
    IO.puts("🚀 Starting minimal evolution test...")
    
    # Simple mutation counter
    spawn(fn -> mutation_loop(0) end)
    
    # Monitor the mutations
    monitor_loop()
  end

  defp mutation_loop(count) do
    new_count = count + 1
    
    # Simulate mutation with occasional errors
    result = try do
      if :rand.uniform() < 0.3 do
        raise "Test error #{new_count}"
      else
        "✅ Mutation #{new_count} successful"
      end
    rescue
      error ->
        "🔧 Error corrected: #{inspect(error)}"
    end
    
    IO.puts(result)
    
    Process.sleep(2000)
    mutation_loop(new_count)
  end

  defp monitor_loop do
    IO.puts("📊 System running and evolving...")
    
    receive after
      10_000 ->
        IO.puts("🔄 Still running - mutations continue...")
        monitor_loop()
    end
  end
end

MinimalTest.start()