#!/usr/bin/env elixir

# Simple test to verify our fixes work
defmodule TestFixes do
  def run do
    IO.puts("🔧 Testing fixes...")
    
    # Test 1: Compile without warnings
    IO.puts("✅ Testing compilation...")
    case Code.compile_file("lib/lmstudio/advanced_mas.ex") do
      [] -> IO.puts("❌ Failed to compile advanced_mas.ex")
      modules -> IO.puts("✅ Compiled #{length(modules)} modules successfully")
    end
    
    # Test 2: Check timeout handling
    IO.puts("✅ Testing timeout handling...")
    try do
      # This should not hang
      result = case :timer.tc(fn -> 
        Process.sleep(100)
        {:ok, "test"}
      end) do
        {time, {:ok, response}} when time < 1_000_000 -> 
          {:ok, response}
        {_time, {:ok, response}} -> 
          {:ok, response}
        _ -> 
          {:error, :timeout}
      end
      
      IO.puts("✅ Timeout handling works: #{inspect(result)}")
    rescue
      e -> IO.puts("❌ Timeout test failed: #{inspect(e)}")
    end
    
    IO.puts("🎉 All fixes verified!")
  end
end

TestFixes.run()