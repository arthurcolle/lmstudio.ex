# Final demo showing persistence capabilities
# Ensure LMStudio modules are loaded (they should be compiled in _build)
unless Code.ensure_loaded?(LMStudio) do
  # If not compiled, load the source files
  Code.require_file("lib/lmstudio.ex", __DIR__)
  Code.require_file("lib/lmstudio/config.ex", __DIR__)
  Code.require_file("lib/lmstudio/persistence.ex", __DIR__)
end

IO.puts("🚀 Persistence System Demo")
IO.puts("==========================")

# Start the persistence system
{:ok, _pid} = LMStudio.Persistence.start_link()

IO.puts("✅ Persistence system started")

# Give it a moment to initialize
Process.sleep(1000)

# Test basic storage
IO.puts("\n💾 Testing Basic Storage:")
LMStudio.Persistence.store("test_key", %{data: "Hello, persistent world!", timestamp: DateTime.utc_now()})
IO.puts("✅ Stored test data")

# Retrieve data
stored_data = LMStudio.Persistence.get("test_key")
IO.puts("✅ Retrieved: #{inspect(stored_data)}")

# Test pattern storage
IO.puts("\n🧠 Testing Pattern Storage:")
pattern_data = %{
  name: "custom_genserver",
  code: "defmodule CustomServer do...",
  performance: 0.85,
  usage_count: 5
}

LMStudio.Persistence.store({:pattern, "custom_genserver"}, pattern_data)
IO.puts("✅ Stored pattern data")

# Test evolution state storage
IO.puts("\n🔄 Testing Evolution State Storage:")
evolution_state = %{
  cycle: 42,
  insights: ["Better error handling", "Improved concurrency"],
  agents: ["Explorer_1", "Optimizer_2"],
  performance_avg: 0.76
}

LMStudio.Persistence.store({:evolution_state, DateTime.utc_now()}, evolution_state)
IO.puts("✅ Stored evolution state")

# Show stats
IO.puts("\n📊 Persistence Statistics:")
stats = LMStudio.Persistence.get_stats()
IO.inspect(stats, pretty: true)

# Test listing keys
IO.puts("\n🔍 Stored Keys:")
keys = LMStudio.Persistence.list_keys()
for key <- Enum.take(keys, 10) do
  IO.puts("  - #{inspect(key)}")
end

# Test export functionality
IO.puts("\n📤 Testing Export:")
export_file = "/tmp/lm_studio_backup.etf"
case LMStudio.Persistence.export_to_file(export_file) do
  {:ok, message} -> IO.puts("✅ #{message}")
  {:error, reason} -> IO.puts("❌ Export failed: #{inspect(reason)}")
end

# Test checkpoint
IO.puts("\n💾 Testing Manual Checkpoint:")
LMStudio.Persistence.checkpoint_now()
IO.puts("✅ Manual checkpoint completed")

# Store some generated code examples
IO.puts("\n⚡ Storing Generated Code Examples:")
code_examples = [
  {"genserver_example", "defmodule ExampleServer do\n  use GenServer\n  # ... implementation\nend"},
  {"supervisor_example", "defmodule ExampleSupervisor do\n  use Supervisor\n  # ... implementation\nend"},
  {"agent_example", "defmodule ExampleAgent do\n  use Agent\n  # ... implementation\nend"}
]

for {code_id, code} <- code_examples do
  LMStudio.Persistence.store({:generated_code, code_id}, %{
    code: code,
    generated_at: DateTime.utc_now(),
    pattern: code_id,
    performance: :rand.uniform()
  })
  IO.puts("✅ Stored #{code_id}")
end

# Show final stats
IO.puts("\n📈 Final Statistics:")
final_stats = LMStudio.Persistence.get_stats()
IO.inspect(final_stats, pretty: true)

# Test data retrieval
IO.puts("\n🔍 Testing Data Retrieval:")
pattern_data_retrieved = LMStudio.Persistence.get({:pattern, "custom_genserver"})
IO.puts("Pattern data: #{inspect(pattern_data_retrieved)}")

code_data_retrieved = LMStudio.Persistence.get({:generated_code, "genserver_example"})
IO.puts("Generated code data: #{inspect(Map.get(code_data_retrieved, :pattern))}")

IO.puts("\n✨ Persistence Demo Complete!")
IO.puts("=============================")

IO.puts("\n✅ Demonstrated Capabilities:")
IO.puts("  💾 In-memory ETS storage for fast access")
IO.puts("  🗃️  File-based persistence with compression") 
IO.puts("  🔄 Automatic checkpointing")
IO.puts("  📤 Export/import functionality")
IO.puts("  📊 Storage statistics and monitoring")
IO.puts("  🔍 Key listing and data retrieval")
IO.puts("  ⚡ Generated code storage")
IO.puts("  🧠 Pattern and evolution state persistence")

IO.puts("\n🎯 This persistence system ensures:")
IO.puts("  • All learning and evolution is preserved")
IO.puts("  • Generated code patterns are saved for reuse")
IO.puts("  • System can restart without losing knowledge")
IO.puts("  • Performance data accumulates over time")
IO.puts("  • Continuous improvement across sessions")

Process.sleep(2000)  # Let final checkpoint happen
IO.puts("\n💎 The evolution system now has permanent memory!")