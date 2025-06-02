#!/usr/bin/env elixir

# Quick test to verify the timeout fix works
defmodule TestFix do
  def run do
    IO.puts("Testing timeout fix...")
    
    # Start registry
    {:ok, _} = Registry.start_link(keys: :unique, name: LMStudio.AgentRegistry)
    
    # Start agent
    case LMStudio.CognitiveAgent.start_link([name: "TestAgent", thinking_enabled: true]) do
      {:ok, _} ->
        IO.puts("✓ Agent started successfully")
        
        # Test query processing
        start_time = System.monotonic_time(:millisecond)
        case LMStudio.CognitiveAgent.process_query("TestAgent", "Hello") do
          {:ok, result} ->
            end_time = System.monotonic_time(:millisecond)
            IO.puts("✓ Query processed successfully in #{end_time - start_time}ms")
            IO.puts("  Response length: #{byte_size(result.response)}")
            IO.puts("  Mutations applied: #{result.mutations_applied}")
            
          {:error, reason} ->
            IO.puts("✗ Query failed: #{inspect(reason)}")
        end
        
      {:error, reason} ->
        IO.puts("✗ Failed to start agent: #{inspect(reason)}")
    end
  end
end

TestFix.run()