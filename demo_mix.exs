# Demo script for the enhanced evolution system
Mix.install([
  {:jason, "~> 1.0"}
])

# Start our application modules manually since we're in a script
Application.ensure_all_started(:logger)

# Ensure LMStudio modules are loaded (they should be compiled in _build)
unless Code.ensure_loaded?(LMStudio) do
  # If not compiled, load the source files
  Code.require_file("lib/lmstudio.ex", __DIR__)
  Code.require_file("lib/lmstudio/config.ex", __DIR__)
  Code.require_file("lib/lmstudio/persistence.ex", __DIR__)
  Code.require_file("lib/lmstudio/meta_dsl.ex", __DIR__)
  Code.require_file("lib/lmstudio/cognitive_agent.ex", __DIR__)
  Code.require_file("lib/lmstudio/erlang_knowledge_base.ex", __DIR__)
  Code.require_file("lib/lmstudio/code_generation.ex", __DIR__)
  Code.require_file("lib/lmstudio/evolution_system.ex", __DIR__)
end

IO.puts("🚀 Enhanced Evolution System Demo")
IO.puts("=================================")

# Test the knowledge base first
IO.puts("\n🧠 Testing Erlang Knowledge Base:")
otp_behaviors = LMStudio.ErlangKnowledgeBase.get_otp_behaviors()
IO.puts("OTP behaviors loaded: #{map_size(otp_behaviors)}")
IO.puts("Available behaviors: #{Map.keys(otp_behaviors) |> Enum.join(", ")}")

# Test pattern recommendations
context = %{use_case: "state management", scale: :medium, fault_tolerance: :high}
recommendations = LMStudio.ErlangKnowledgeBase.get_pattern_recommendations(context)
IO.puts("Pattern recommendations for state management: #{inspect(recommendations)}")

# Test code generation
IO.puts("\n⚡ Testing Dynamic Code Generation:")

# Generate a GenServer
{:ok, gen_server_code} = LMStudio.CodeGeneration.generate_genserver("DemoServer", %{counter: 0})
IO.puts("✅ Generated GenServer code (first 300 chars):")
IO.puts(String.slice(gen_server_code, 0, 300) <> "...")

# Generate a Supervisor
supervisor_code = LMStudio.CodeGeneration.generate_supervisor("DemoSupervisor", [
  {DemoWorker1, []},
  {DemoWorker2, []}
])
IO.puts("\n✅ Generated Supervisor code (first 300 chars):")
IO.puts(String.slice(supervisor_code, 0, 300) <> "...")

# Generate an Agent
agent_code = LMStudio.CodeGeneration.generate_agent_pattern("DemoAgent", %{data: []})
IO.puts("\n✅ Generated Agent code (first 300 chars):")
IO.puts(String.slice(agent_code, 0, 300) <> "...")

# Test pattern templates
IO.puts("\n📋 Testing Pattern Templates:")
gen_server_template = LMStudio.ErlangKnowledgeBase.get_pattern_template(:gen_server_with_state)
IO.puts("GenServer template available: #{not is_nil(gen_server_template)}")

supervisor_template = LMStudio.ErlangKnowledgeBase.get_pattern_template(:supervisor_one_for_one)
IO.puts("Supervisor template available: #{not is_nil(supervisor_template)}")

# Test knowledge search
IO.puts("\n🔍 Testing Pattern Search:")
search_results = LMStudio.ErlangKnowledgeBase.search_patterns("fault tolerance")
IO.puts("Found #{length(search_results)} patterns matching 'fault tolerance':")
for {category, pattern_name, _pattern_data} <- Enum.take(search_results, 3) do
  IO.puts("  - #{category}: #{pattern_name}")
end

# Test concurrency patterns
IO.puts("\n🔄 Testing Concurrency Knowledge:")
concurrency_patterns = LMStudio.ErlangKnowledgeBase.get_concurrency_patterns()
IO.puts("Concurrency patterns available: #{Map.keys(concurrency_patterns) |> Enum.join(", ")}")

# Test fault tolerance patterns
fault_patterns = LMStudio.ErlangKnowledgeBase.get_fault_tolerance_patterns()
IO.puts("Fault tolerance patterns: #{Map.keys(fault_patterns) |> Enum.join(", ")}")

# Test performance patterns
perf_patterns = LMStudio.ErlangKnowledgeBase.get_performance_patterns()
IO.puts("Performance patterns: #{Map.keys(perf_patterns) |> Enum.join(", ")}")

# Test best practices
IO.puts("\n📚 Testing Best Practices:")
gen_server_practices = LMStudio.ErlangKnowledgeBase.get_best_practices_for_behavior(:gen_server)
IO.puts("GenServer best practices (#{length(gen_server_practices)} total):")
for practice <- Enum.take(gen_server_practices, 3) do
  IO.puts("  - #{practice}")
end

# Test anti-patterns
IO.puts("\n⚠️  Testing Anti-pattern Knowledge:")
antipatterns = LMStudio.ErlangKnowledgeBase.get_common_antipatterns()
IO.puts("Known anti-patterns: #{Map.keys(antipatterns) |> Enum.join(", ")}")

# Test code optimization
IO.puts("\n🚀 Testing Code Optimization:")
sample_code = """
defmodule SampleGenServer do
  use GenServer
  
  def handle_call(:get_data, _from, state) do
    {:reply, state, state}
  end
end
"""

optimized_code = LMStudio.CodeGeneration.optimize_for_performance(sample_code)
IO.puts("Code optimization added #{String.length(optimized_code) - String.length(sample_code)} characters of enhancements")

fault_tolerant_code = LMStudio.CodeGeneration.optimize_for_fault_tolerance(sample_code)
IO.puts("Fault tolerance optimization added #{String.length(fault_tolerant_code) - String.length(sample_code)} characters of enhancements")

IO.puts("\n✨ Knowledge Base Demo completed successfully!")
IO.puts("The system demonstrated:")
IO.puts("  ✅ Comprehensive Erlang/OTP knowledge base")
IO.puts("  ✅ Dynamic code generation for all major OTP behaviors") 
IO.puts("  ✅ Pattern-based recommendations")
IO.puts("  ✅ Best practices and anti-pattern detection")
IO.puts("  ✅ Code optimization for performance and fault tolerance")
IO.puts("  ✅ Searchable pattern database")
IO.puts("  ✅ Context-aware pattern selection")

IO.puts("\n🎯 This demonstrates the core intelligence that will be used by the evolution system")
IO.puts("   to dynamically generate and evolve sophisticated Erlang/OTP applications!")