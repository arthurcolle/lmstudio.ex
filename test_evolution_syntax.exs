#!/usr/bin/env elixir

# Simple syntax test
defmodule TestSyntax do
  def test do
    IO.puts("Testing basic functionality...")
    
    try do
      # Test the LMStudio modules exist
      case Code.ensure_loaded(LMStudio.EvolutionSystem) do
        {:module, _} -> IO.puts("✅ EvolutionSystem module available")
        {:error, _} -> IO.puts("❌ EvolutionSystem module not found")
      end
      
      case Code.ensure_loaded(LMStudio.CognitiveAgent) do
        {:module, _} -> IO.puts("✅ CognitiveAgent module available")
        {:error, _} -> IO.puts("❌ CognitiveAgent module not found")
      end
      
      IO.puts("✅ Syntax test completed successfully")
    rescue
      error ->
        IO.puts("❌ Error during test: #{inspect(error)}")
    end
  end
end

TestSyntax.test()