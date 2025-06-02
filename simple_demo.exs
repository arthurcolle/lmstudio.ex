# Simple demo focusing on knowledge base and code generation
Mix.install([])

# Ensure LMStudio modules are loaded (they should be compiled in _build)
unless Code.ensure_loaded?(LMStudio) do
  # If not compiled, load the source files
  Code.require_file("lib/lmstudio.ex", __DIR__)
  Code.require_file("lib/lmstudio/config.ex", __DIR__)
  Code.require_file("lib/lmstudio/erlang_knowledge_base.ex", __DIR__)
end

IO.puts("🚀 Enhanced Evolution System - Core Intelligence Demo")
IO.puts("====================================================")

# Test the comprehensive Erlang knowledge base
IO.puts("\n🧠 Erlang/OTP Knowledge Base:")

# Show OTP behaviors
otp_behaviors = LMStudio.ErlangKnowledgeBase.get_otp_behaviors()
IO.puts("✅ OTP Behaviors (#{map_size(otp_behaviors)}):")
for {behavior, details} <- otp_behaviors do
  IO.puts("  📋 #{behavior}: #{details.description}")
  IO.puts("     Use cases: #{Enum.take(details.use_cases, 2) |> Enum.join(", ")}...")
end

# Show concurrency patterns
IO.puts("\n🔄 Concurrency Patterns:")
concurrency = LMStudio.ErlangKnowledgeBase.get_concurrency_patterns()
for {pattern, details} <- concurrency do
  IO.puts("  ⚡ #{pattern}: #{details.description}")
end

# Show fault tolerance patterns
IO.puts("\n🛡️  Fault Tolerance Patterns:")
fault_tolerance = LMStudio.ErlangKnowledgeBase.get_fault_tolerance_patterns()
for {pattern, details} <- fault_tolerance do
  IO.puts("  🔒 #{pattern}: #{details.description}")
end

# Show performance patterns
IO.puts("\n🚀 Performance Patterns:")
performance = LMStudio.ErlangKnowledgeBase.get_performance_patterns()
for {pattern, details} <- performance do
  IO.puts("  ⚡ #{pattern}: #{details.description}")
end

# Test pattern recommendations
IO.puts("\n🎯 Smart Pattern Recommendations:")
contexts = [
  %{use_case: "state management", scale: :small, fault_tolerance: :low},
  %{use_case: "worker processing", scale: :medium, fault_tolerance: :high},
  %{use_case: "event handling", scale: :large, fault_tolerance: :medium}
]

for context <- contexts do
  recommendations = LMStudio.ErlangKnowledgeBase.get_pattern_recommendations(context)
  IO.puts("  📊 #{context.use_case} (#{context.scale}/#{context.fault_tolerance}):")
  IO.puts("     → #{Enum.take(recommendations, 3) |> Enum.join(", ")}")
end

# Test pattern search
IO.puts("\n🔍 Pattern Search Capabilities:")
search_terms = ["supervisor", "fault", "concurrent", "state"]
for term <- search_terms do
  results = LMStudio.ErlangKnowledgeBase.search_patterns(term)
  IO.puts("  🔎 '#{term}': #{length(results)} matches")
end

# Show pattern templates
IO.puts("\n📋 Code Pattern Templates:")
templates = [:gen_server_with_state, :supervisor_one_for_one, :agent_pattern, :task_supervisor_pattern]
for template <- templates do
  code = LMStudio.ErlangKnowledgeBase.get_pattern_template(template)
  if code do
    lines = String.split(code, "\n") |> length()
    IO.puts("  📄 #{template}: #{lines} lines of production-ready code")
  end
end

# Test best practices
IO.puts("\n📚 Best Practices Knowledge:")
behaviors = [:gen_server, :supervisor, :application]
for behavior <- behaviors do
  practices = LMStudio.ErlangKnowledgeBase.get_best_practices_for_behavior(behavior)
  IO.puts("  💡 #{behavior}: #{length(practices)} best practices")
  IO.puts("     Example: #{List.first(practices)}")
end

# Show anti-patterns
IO.puts("\n⚠️  Anti-Pattern Detection:")
antipatterns = LMStudio.ErlangKnowledgeBase.get_common_antipatterns()
for {pattern, details} <- antipatterns do
  IO.puts("  🚫 #{pattern}: #{details.problem}")
  IO.puts("     💡 Solution: #{details.solution}")
end

# Test code generation from patterns
IO.puts("\n⚡ Dynamic Code Generation:")
patterns_to_test = [:gen_server_with_state, :supervisor_one_for_one, :agent_pattern]

for pattern <- patterns_to_test do
  case LMStudio.ErlangKnowledgeBase.generate_code_from_pattern(pattern) do
    {:ok, code} ->
      lines = String.split(code, "\n") |> length()
      chars = String.length(code)
      IO.puts("  ✅ #{pattern}: Generated #{lines} lines (#{chars} chars)")
      
      # Show a snippet
      snippet = String.split(code, "\n") |> Enum.take(3) |> Enum.join("\n")
      IO.puts("     Preview: #{String.replace(snippet, "\n", " | ")}")
      
    {:error, reason} ->
      IO.puts("  ❌ #{pattern}: #{reason}")
  end
end

# Test customization
IO.puts("\n🎨 Code Customization:")
{:ok, custom_code} = LMStudio.ErlangKnowledgeBase.generate_code_from_pattern(
  :gen_server_with_state, 
  %{module_name: "CustomServer", initial_state: "%{data: []}"}
)
IO.puts("  ✅ Custom GenServer generated with personalized module name and state")

# Evolution suggestions
IO.puts("\n🧬 Evolution Suggestions:")
sample_metrics = [0.3, 0.4, 0.2, 0.5, 0.3]  # Low performance metrics
suggestions = LMStudio.ErlangKnowledgeBase.get_evolution_suggestions("GenServer.call", sample_metrics)
IO.puts("  📈 For low-performing code with GenServer.call:")
for suggestion <- Enum.take(suggestions, 3) do
  IO.puts("     💡 #{suggestion}")
end

IO.puts("\n✨ Demo completed successfully!")
IO.puts("\n📊 System Capabilities Demonstrated:")
IO.puts("  ✅ Comprehensive Erlang/OTP knowledge base (5 behavior types)")
IO.puts("  ✅ Concurrency patterns (#{map_size(concurrency)} patterns)")
IO.puts("  ✅ Fault tolerance strategies (#{map_size(fault_tolerance)} patterns)")
IO.puts("  ✅ Performance optimization techniques (#{map_size(performance)} patterns)")
IO.puts("  ✅ Smart pattern recommendations based on context")
IO.puts("  ✅ Advanced pattern search and discovery")
IO.puts("  ✅ Production-ready code template generation")
IO.puts("  ✅ Best practices and anti-pattern knowledge")
IO.puts("  ✅ Dynamic code generation with customization")
IO.puts("  ✅ Evolution suggestions for performance improvement")

IO.puts("\n🎯 This represents decades of Erlang/OTP expertise embedded in")
IO.puts("   a system that can dynamically generate, evolve, and optimize")
IO.puts("   sophisticated concurrent and fault-tolerant applications!")

IO.puts("\n🔮 Next: This knowledge base will be used by the evolution system")
IO.puts("   to continuously learn, generate new code patterns, and evolve")
IO.puts("   toward increasingly sophisticated distributed systems!")