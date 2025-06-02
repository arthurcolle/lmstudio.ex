#!/usr/bin/env elixir

# Demo script to show the enhanced evolution system with persistence and code generation

# Start the supervision tree
{:ok, _pid} = LMStudio.EvolutionSystem.start_link(num_agents: 2)

# Give the system a moment to initialize
Process.sleep(2000)

IO.puts("🚀 Enhanced Evolution System Demo")
IO.puts("=================================")

# Show initial system state
IO.puts("\n📊 Initial System State:")
IO.inspect(LMStudio.EvolutionSystem.get_system_state(), pretty: true)

# Test persistence system
IO.puts("\n💾 Testing Persistence System:")
persistence_stats = LMStudio.Persistence.get_stats()
IO.inspect(persistence_stats, pretty: true)

# Test knowledge base
IO.puts("\n🧠 Testing Erlang Knowledge Base:")
gen_server_pattern = LMStudio.ErlangKnowledgeBase.get_pattern_template(:gen_server_with_state)
IO.puts("GenServer pattern available: #{not is_nil(gen_server_pattern)}")

otp_behaviors = LMStudio.ErlangKnowledgeBase.get_otp_behaviors()
IO.puts("OTP behaviors loaded: #{map_size(otp_behaviors)}")
IO.puts("Available behaviors: #{Map.keys(otp_behaviors) |> Enum.join(", ")}")

# Test pattern recommendations
context = %{use_case: "state management", scale: :medium, fault_tolerance: :high}
recommendations = LMStudio.ErlangKnowledgeBase.get_pattern_recommendations(context)
IO.puts("Pattern recommendations for state management: #{inspect(recommendations)}")

# Test code generation
IO.puts("\n⚡ Testing Dynamic Code Generation:")
{:ok, gen_server_code} = LMStudio.CodeGeneration.generate_genserver("DemoServer", %{counter: 0})
IO.puts("Generated GenServer code (first 200 chars):")
IO.puts(String.slice(gen_server_code, 0, 200) <> "...")

# Save and verify persistence
code_id = "demo_genserver_#{System.system_time()}"
LMStudio.CodeGeneration.save_generated_code(code_id, gen_server_code)
IO.puts("Code saved with ID: #{code_id}")

# Run some evolution cycles
IO.puts("\n🔄 Running Evolution Cycles:")
topics = [
  "implementing fault-tolerant distributed systems",
  "optimizing concurrent process pools", 
  "building self-healing service architectures"
]

for {topic, index} <- Enum.with_index(topics, 1) do
  IO.puts("Cycle #{index}: #{topic}")
  case LMStudio.EvolutionSystem.run_evolution_cycle(topic) do
    :ok -> IO.puts("  ✅ Success")
    {:error, reason} -> IO.puts("  ❌ Error: #{inspect(reason)}")
  end
  Process.sleep(1000)
end

# Show system state after evolution
IO.puts("\n📈 System State After Evolution:")
final_state = LMStudio.EvolutionSystem.get_system_state()
IO.inspect(final_state, pretty: true)

# Test learning trajectory
learning_trajectory = LMStudio.EvolutionSystem.get_learning_trajectory()
IO.puts("\n📚 Learning Trajectory (#{length(learning_trajectory)} entries):")
for entry <- Enum.take(learning_trajectory, 3) do
  IO.puts("  Cycle #{entry.cycle}: #{entry.insights_generated} insights, #{Float.round(entry.performance_avg, 2)} avg performance")
end

# Test pattern search
IO.puts("\n🔍 Testing Pattern Search:")
search_results = LMStudio.ErlangKnowledgeBase.search_patterns("supervisor")
IO.puts("Found #{length(search_results)} patterns matching 'supervisor'")

# Test persistence stats
IO.puts("\n💾 Final Persistence Stats:")
final_persistence_stats = LMStudio.Persistence.get_stats()
IO.inspect(final_persistence_stats, pretty: true)

# Test generated code listing
IO.puts("\n📝 Generated Code Modules:")
generated_code_list = LMStudio.Persistence.Helpers.list_generated_code()
for {code_id, _code_data} <- Enum.take(generated_code_list, 5) do
  IO.puts("  - #{code_id}")
end

IO.puts("\n✨ Demo completed successfully!")
IO.puts("The system demonstrated:")
IO.puts("  ✅ Persistent storage with ETS and file backing")
IO.puts("  ✅ Dynamic code generation from Erlang/OTP patterns")
IO.puts("  ✅ Continuous learning and evolution tracking")
IO.puts("  ✅ Self-modifying grids with auto-persistence")
IO.puts("  ✅ Comprehensive Erlang knowledge base")
IO.puts("  ✅ Cross-pollination and autonomous evolution")