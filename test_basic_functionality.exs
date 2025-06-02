# Test basic functionality without LM Studio calls

IO.puts "Testing basic module loading..."

# Test LMStudio module
try do
  Application.ensure_all_started(:lmstudio)
rescue
  _ -> :ok
end

IO.puts "✓ LMStudio module loaded"

# Test Config module
base_url = LMStudio.Config.base_url()
IO.puts "✓ Config module works - Base URL: #{base_url}"

# Test Persistence
{:ok, _pid} = LMStudio.Persistence.start_link()
LMStudio.Persistence.store(:test_key, "test_value")
value = LMStudio.Persistence.get(:test_key)
IO.puts "✓ Persistence module works - Stored and retrieved: #{value}"

# Test MetaDSL
alias LMStudio.MetaDSL.Mutation
mutation = Mutation.new(:append, "test", content: "test content", metadata: %{})
IO.puts "✓ MetaDSL modules work - Created mutation: #{inspect(mutation.type)}"

IO.puts "\n✅ All basic modules loaded successfully!"
IO.puts "\nNote: The agent demos require a running LM Studio instance."
IO.puts "make sure LM Studio is running on #{base_url}"