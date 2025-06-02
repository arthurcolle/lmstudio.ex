# Demo to show the actual generated code quality
# Ensure LMStudio modules are loaded (they should be compiled in _build)
unless Code.ensure_loaded?(LMStudio) do
  # If not compiled, load the source files
  Code.require_file("lib/lmstudio.ex", __DIR__)
  Code.require_file("lib/lmstudio/config.ex", __DIR__)
  Code.require_file("lib/lmstudio/erlang_knowledge_base.ex", __DIR__)
end

IO.puts("🚀 Generated Code Quality Demo")
IO.puts("==============================")

# Generate and display a full GenServer
{:ok, genserver_code} = LMStudio.ErlangKnowledgeBase.generate_code_from_pattern(:gen_server_with_state)
IO.puts("\n📄 Generated GenServer Pattern:")
IO.puts("================================")
IO.puts(genserver_code)

# Generate a Supervisor
{:ok, supervisor_code} = LMStudio.ErlangKnowledgeBase.generate_code_from_pattern(:supervisor_one_for_one)
IO.puts("\n📄 Generated Supervisor Pattern:")
IO.puts("=================================")
IO.puts(supervisor_code)

# Generate an Agent
{:ok, agent_code} = LMStudio.ErlangKnowledgeBase.generate_code_from_pattern(:agent_pattern)
IO.puts("\n📄 Generated Agent Pattern:")
IO.puts("============================")
IO.puts(agent_code)

# Show pattern details
IO.puts("\n📋 Pattern Analysis:")
IO.puts("===================")

gen_server_info = LMStudio.ErlangKnowledgeBase.get_otp_behaviors()[:gen_server]
IO.puts("GenServer capabilities:")
for callback <- gen_server_info.callbacks do
  IO.puts("  ✅ #{callback}")
end

IO.puts("\nGenServer best practices included:")
practices = LMStudio.ErlangKnowledgeBase.get_best_practices_for_behavior(:gen_server)
for practice <- Enum.take(practices, 5) do
  IO.puts("  💡 #{practice}")
end

IO.puts("\n🎯 This demonstrates production-ready, comprehensive code generation")
IO.puts("   with deep Erlang/OTP expertise embedded in every template!")