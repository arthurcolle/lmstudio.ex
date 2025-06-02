# Self-Modifying MetaDSL Cognitive System for Elixir

A continuously evolving system that uses MetaDSL to mutate its own prompts, reasoning patterns, and cognitive strategies through recursive self-improvement.

## 🧠 Overview

This system demonstrates how artificial intelligence can evolve and improve itself through:

- **MetaDSL for prompt mutation** - Agents can modify their own behavior using structured mutation commands
- **Cognitive agents with thinking capabilities** - Agents that can introspect and reason about their own processes  
- **Self-modification through grid mutations** - A data structure that evolves based on performance feedback
- **Continuous learning and evolution** - Multiple agents that cross-pollinate successful patterns
- **OTP-based architecture** - Built on Elixir's robust process supervision model

## 🏗️ Architecture

### Core Components

1. **MutationType & Mutation** - Defines types of mutations (append, replace, evolve, etc.) and mutation operations
2. **SelfModifyingGrid** - GenServer that maintains mutable state and applies mutations based on performance
3. **CognitiveAgent** - Agents that use LMStudio for thinking and can modify their own behavior patterns
4. **MutationParser** - Extracts MetaDSL mutation commands from LLM responses
5. **EvolutionSystem** - Supervisor coordinating multiple agents and cross-pollination

### Mutation Types

- `append` - Add content to existing data
- `replace` - Replace occurrences of text 
- `delete` - Remove data entries
- `evolve` - Evolve content based on performance feedback
- `compress` - Reduce data size while preserving meaning
- `mutate_strategy` - Modify meta-cognitive strategies

### MetaDSL Tags

Agents can suggest mutations using XML-like tags in their responses:

```xml
<append target="knowledge">New insight discovered</append>
<replace target="old_pattern" content="improved_pattern"/>
<evolve target="strategy">Enhanced reasoning approach</evolve>
<mutate_strategy target="learning_policy">Focus on deeper analysis</mutate_strategy>
```

## 🚀 Quick Start

### Prerequisites

- Elixir 1.18+
- LM Studio running locally with a compatible model
- Jason for JSON handling

### Installation

The system is included as part of the LM Studio Elixir client. All required modules are in the `lib/lmstudio/` directory.

### Basic Usage

#### 1. Single Self-Modifying Grid

```elixir
# Start a grid with initial knowledge
{:ok, grid_pid} = LMStudio.MetaDSL.SelfModifyingGrid.start_link(
  initial_data: %{
    "identity" => "I am a learning system",
    "knowledge" => "Basic knowledge about the world"
  }
)

# Apply mutations
mutation = LMStudio.MetaDSL.Mutation.new(:append, "knowledge", 
  content: "\nI have learned about self-modification")
  
{:ok, :mutated} = LMStudio.MetaDSL.SelfModifyingGrid.mutate(grid_pid, mutation)

# Track performance
LMStudio.MetaDSL.SelfModifyingGrid.add_performance(grid_pid, 0.85)

# Get suggestions for further mutations
suggestions = LMStudio.MetaDSL.SelfModifyingGrid.suggest_mutations(grid_pid)
```

#### 2. Single Cognitive Agent

```elixir
# Start registry for agent names
{:ok, _} = Registry.start_link(keys: :unique, name: LMStudio.AgentRegistry)

# Create an agent
{:ok, _pid} = LMStudio.CognitiveAgent.start_link([
  name: "MyAgent",
  model: "deepseek-r1-0528-qwen3-8b-mlx",
  thinking_enabled: true
])

# Process a query - agent will think and possibly mutate itself
{:ok, result} = LMStudio.CognitiveAgent.process_query("MyAgent", 
  "How can I improve my reasoning abilities?")

IO.puts(result.response)
IO.puts("Thinking: #{result.thinking}")
IO.puts("Mutations applied: #{result.mutations_applied}")
```

#### 3. Multi-Agent Evolution System

```elixir
# Start the full evolution system
{:ok, _supervisor} = LMStudio.EvolutionSystem.start_link(
  num_agents: 3,
  model: "deepseek-r1-0528-qwen3-8b-mlx"
)

# Run a single evolution cycle
:ok = LMStudio.EvolutionSystem.run_evolution_cycle(
  "the nature of recursive self-improvement"
)

# Start continuous evolution (runs indefinitely)
LMStudio.EvolutionSystem.start_continuous_evolution(
  "artificial intelligence and consciousness"
)

# Stop continuous evolution
LMStudio.EvolutionSystem.stop_continuous_evolution()
```

## 📚 Examples

### Run the Interactive Demo

```bash
# Full automated demonstration
mix run self_modifying_metadsl_example.exs

# Interactive mode with menu
mix run self_modifying_metadsl_example.exs --interactive
```

### Example Agent Interactions

The system includes three types of agents:

1. **Explorer** - Discovers new reasoning patterns through creative mutations
2. **Optimizer** - Refines existing patterns for efficiency and clarity  
3. **Synthesizer** - Combines successful patterns into new forms

Each agent can:
- Think deeply about queries using `<think>` tags
- Suggest mutations to improve their own behavior
- Learn from performance feedback
- Cross-pollinate successful patterns with other agents

### Example Agent Response

```
Query: "How can I evolve my reasoning patterns?"

Agent Thinking:
<think>
I need to consider how to improve my cognitive processes. Currently, my strategy 
focuses on pattern matching, but I could enhance this by incorporating more 
systematic analysis approaches. I should suggest mutations to evolve my strategy.
</think>

Agent Response:
To evolve my reasoning patterns, I should focus on deeper analytical thinking 
and systematic approach to problem-solving. 

<evolve target="strategy">Enhanced multi-step reasoning with validation checks</evolve>
<append target="knowledge">Learned importance of systematic analysis in reasoning</append>

This evolution will help me develop more robust cognitive patterns.

Mutations applied: 2
Performance score: 0.82
```

## 🧪 Testing

Run the test suite:

```bash
mix test test/metadsl_test.exs
```

## 🔧 Configuration

The system uses the existing LM Studio configuration. Make sure you have:

1. LM Studio running locally (default: http://localhost:1234)
2. A compatible model loaded (e.g., deepseek-r1-0528-qwen3-8b-mlx)
3. Proper model configuration in `lib/lmstudio/config.ex`

## 📈 Performance Monitoring

The system tracks several metrics:

- **Mutation success rate** - Percentage of successfully applied mutations
- **Performance scores** - Based on response quality, insights, and coherence
- **Evolution generation** - Number of mutation cycles completed
- **Insight count** - Number of learning insights generated
- **Cross-pollination events** - Knowledge sharing between agents

## 🔄 Evolution Cycles

The system runs evolution cycles that:

1. **Process queries** - Each agent responds to topics using current knowledge
2. **Extract mutations** - Parse agent responses for MetaDSL mutation commands
3. **Apply mutations** - Update agent grids based on suggested improvements
4. **Track performance** - Monitor quality of responses and reasoning
5. **Cross-pollinate** - Share successful patterns between agents
6. **Suggest improvements** - Generate autonomous mutation suggestions

## 🌐 Continuous Evolution

The continuous evolution mode:

- Runs indefinitely with periodic evolution cycles
- Rotates through different philosophical and cognitive topics
- Automatically triggers cross-pollination every 3 cycles
- Initiates autonomous evolution every 5 cycles
- Provides real-time monitoring of system state

## 🛠️ Advanced Usage

### Custom Mutations

```elixir
# Create custom mutations
custom_mutation = LMStudio.MetaDSL.Mutation.new(:evolve, "reasoning_style",
  content: "Adopt more systematic analytical approach",
  reasoning: "Current approach lacks structured thinking",
  confidence: 0.9
)

# Apply to any grid
LMStudio.MetaDSL.SelfModifyingGrid.mutate(grid_pid, custom_mutation)
```

### Custom Agent Types

```elixir
# Start agents with specific roles
{:ok, _} = LMStudio.CognitiveAgent.start_link([
  name: "AnalyticAgent",
  model: "your-model-name",
  thinking_enabled: true
])

# Customize the agent's initial grid data
grid_data = LMStudio.CognitiveAgent.get_grid_data("AnalyticAgent")
```

### Performance Tuning

```elixir
# Monitor system performance
state = LMStudio.EvolutionSystem.get_system_state()
# => %{evolution_cycles: 15, total_interactions: 45, global_insights: 12, ...}

# Analyze individual agent patterns
analysis = LMStudio.MetaDSL.SelfModifyingGrid.analyze_patterns(grid_pid)
# => %{total_mutations: 8, avg_performance: 0.84, evolution_generation: 12, ...}
```

## 🎯 Use Cases

1. **Research into AI consciousness** - Study how systems can modify their own cognition
2. **Adaptive AI systems** - Build AI that improves itself over time
3. **Cognitive architecture research** - Explore self-modifying reasoning patterns
4. **Educational demonstrations** - Show recursive self-improvement in action
5. **Experimental AI development** - Test novel approaches to machine learning

## 🔍 Philosophy

This system explores fundamental questions about intelligence and consciousness:

- Can artificial systems truly modify their own cognitive processes?
- How does recursive self-improvement lead to emergent intelligence?
- What are the implications of self-modifying AI systems?
- How do multiple agents sharing knowledge accelerate evolution?

## ⚠️ Important Notes

- This is an experimental system for research and education
- Requires a running LM Studio instance with appropriate models
- Performance depends heavily on the underlying language model capabilities
- Continuous evolution can be resource-intensive
- The system is designed for exploration, not production use

## 🤝 Contributing

This implementation demonstrates core concepts of self-modifying AI systems. Areas for enhancement:

- More sophisticated mutation strategies
- Advanced performance metrics
- Integration with additional LLM providers
- Enhanced cross-pollination algorithms
- Visualization of evolution patterns
- Persistence and recovery mechanisms

## 📄 License

This code is part of the LM Studio Elixir client and follows the same licensing terms.

---

*"The future belongs to systems that can improve themselves"* - The MetaDSL Philosophy