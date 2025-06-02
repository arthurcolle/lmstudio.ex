#!/usr/bin/env elixir

# Simple test script to verify MetaDSL modules are working

IO.puts("Testing MetaDSL modules...")

# Test 1: Basic Grid
IO.puts("\n1. Testing SelfModifyingGrid:")
try do
  {:ok, grid} = LMStudio.MetaDSL.SelfModifyingGrid.start_link(
    initial_data: %{"test" => "data"}
  )
  IO.puts("✓ Grid started successfully")
  
  data = LMStudio.MetaDSL.SelfModifyingGrid.get_data(grid)
  IO.puts("✓ Grid data: #{inspect(data)}")
  
  GenServer.stop(grid)
  IO.puts("✓ Grid stopped successfully")
rescue
  e -> IO.puts("✗ Error: #{inspect(e)}")
end

# Test 2: Mutation
IO.puts("\n2. Testing Mutation:")
try do
  mutation = LMStudio.MetaDSL.Mutation.new(:append, "test", content: " appended")
  IO.puts("✓ Mutation created: #{inspect(mutation.type)} on #{mutation.target}")
rescue
  e -> IO.puts("✗ Error: #{inspect(e)}")
end

IO.puts("\nAll tests completed!")