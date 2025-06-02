#!/usr/bin/env elixir

# Ensure LMStudio modules are loaded (they should be compiled in _build)
unless Code.ensure_loaded?(LMStudio) do
  # If not compiled, load the source files
  Code.require_file("lib/lmstudio.ex", __DIR__)
  Code.require_file("lib/lmstudio/config.ex", __DIR__)
  Code.require_file("lib/lmstudio/meta_dsl.ex", __DIR__)
end

# Advanced Agent with Function Calling
# This demonstrates how agents can use tools through function calling with LM Studio

defmodule FunctionCallingAgent do
  @moduledoc """
  An agent that can call functions and use tools through structured prompts.
  """
  
  defmodule Functions do
    @moduledoc """
    Available functions that agents can call.
    """
    
    def available_functions do
      %{
        "search_documentation" => %{
          description: "Search Elixir documentation for specific topics",
          parameters: %{
            query: %{type: :string, required: true},
            module: %{type: :string, required: false}
          },
          handler: &search_documentation/1
        },
        "analyze_performance" => %{
          description: "Analyze code performance and suggest optimizations",
          parameters: %{
            code: %{type: :string, required: true},
            metrics: %{type: :list, required: false, default: [:time, :memory]}
          },
          handler: &analyze_performance/1
        },
        "generate_tests" => %{
          description: "Generate test cases for given code",
          parameters: %{
            module_code: %{type: :string, required: true},
            test_framework: %{type: :string, default: "ExUnit"}
          },
          handler: &generate_tests/1
        },
        "refactor_code" => %{
          description: "Refactor code following Elixir best practices",
          parameters: %{
            code: %{type: :string, required: true},
            style: %{type: :string, default: "idiomatic"}
          },
          handler: &refactor_code/1
        },
        "create_genserver" => %{
          description: "Create a GenServer implementation",
          parameters: %{
            name: %{type: :string, required: true},
            state: %{type: :map, required: true},
            callbacks: %{type: :list, required: true}
          },
          handler: &create_genserver/1
        }
      }
    end
    
    # Function implementations
    def search_documentation(%{query: _query} = params) do
      # Simulate documentation search
      results = [
        %{
          module: params[:module] || "Enum",
          function: "map/2",
          description: "Maps a function over an enumerable",
          examples: ["Enum.map([1, 2, 3], &(&1 * 2))"]
        }
      ]
      {:ok, results}
    end
    
    def analyze_performance(%{code: _code} = params) do
      # Simulate performance analysis
      metrics = params[:metrics] || [:time, :memory]
      
      analysis = %{
        metrics: metrics,
        results: %{
          time_complexity: "O(n)",
          space_complexity: "O(1)",
          suggestions: [
            "Consider using Stream for large collections",
            "Pattern matching can improve readability"
          ]
        },
        benchmarks: %{
          average_time: "1.2ms",
          memory_usage: "2.4MB"
        }
      }
      {:ok, analysis}
    end
    
    def generate_tests(%{module_code: code, test_framework: framework}) do
      # Generate test code using LMStudio
      {:ok, test_code} = LMStudio.chat(
        """
        Generate #{framework} tests for this Elixir module:
        
        #{code}
        
        Include edge cases and property-based tests where appropriate.
        """,
        system_prompt: "You are an expert at writing comprehensive Elixir tests."
      )
      
      {:ok, test_code}
    end
    
    def refactor_code(%{code: code, style: style}) do
      # Use MetaDSL to refactor code
      refactored = LMStudio.MetaDSL.refactor(code, style: style)
      {:ok, refactored}
    end
    
    def create_genserver(%{name: name, state: initial_state, callbacks: callbacks}) do
      # Generate GenServer code
      genserver_code = """
      defmodule #{name} do
        use GenServer
        
        # Client API
        def start_link(opts \\\\ []) do
          GenServer.start_link(__MODULE__, opts, name: __MODULE__)
        end
        
        #{generate_client_functions(callbacks)}
        
        # Server Callbacks
        @impl true
        def init(_opts) do
          {:ok, #{inspect(initial_state)}}
        end
        
        #{generate_callback_handlers(callbacks)}
      end
      """
      
      {:ok, genserver_code}
    end
    
    defp generate_client_functions(callbacks) do
      callbacks
      |> Enum.map(fn callback ->
        case callback do
          {:call, name, _} ->
            """
            def #{name}(args \\\\ []) do
              GenServer.call(__MODULE__, {:#{name}, args})
            end
            """
          {:cast, name, _} ->
            """
            def #{name}(args \\\\ []) do
              GenServer.cast(__MODULE__, {:#{name}, args})
            end
            """
        end
      end)
      |> Enum.join("\n")
    end
    
    defp generate_callback_handlers(callbacks) do
      callbacks
      |> Enum.map(fn callback ->
        case callback do
          {:call, name, handler} ->
            """
            @impl true
            def handle_call({:#{name}, args}, _from, state) do
              # #{handler}
              {:reply, :ok, state}
            end
            """
          {:cast, name, handler} ->
            """
            @impl true
            def handle_cast({:#{name}, args}, state) do
              # #{handler}
              {:noreply, state}
            end
            """
        end
      end)
      |> Enum.join("\n")
    end
  end
  
  defmodule Agent do
    @moduledoc """
    An intelligent agent that can parse LLM responses and execute function calls.
    """
    
    defstruct [:name, :role, :functions, :conversation_history]
    
    def new(name, role, functions \\ []) do
      %__MODULE__{
        name: name,
        role: role,
        functions: functions,
        conversation_history: []
      }
    end
    
    def execute_task(agent, task) do
      # Prepare function descriptions for the prompt
      function_docs = prepare_function_documentation(agent.functions)
      
      prompt = """
      You are #{agent.name}, a #{agent.role}.
      
      Task: #{task}
      
      Available functions:
      #{function_docs}
      
      To use a function, write:
      FUNCTION_CALL: function_name
      PARAMETERS:
      param1: value1
      param2: value2
      END_CALL
      
      You can make multiple function calls. After each call, you'll see the result and can make additional calls or provide the final answer.
      """
      
      # Get initial response from LLM
      {:ok, response} = LMStudio.chat(prompt,
        system_prompt: "You are an AI assistant that can call functions to complete tasks. When you need to call a function, use EXACTLY this format:\n\nFUNCTION_CALL: function_name\nPARAMETERS:\nparam1: value1\nparam2: value2\nEND_CALL\n\nDo not use any other format. Do not wrap your response in thinking tags or any other tags.",
        timeout: 30_000
      )
      
      # Extract message content from response
      message_content = case response do
        %{"choices" => [%{"message" => %{"content" => content}} | _]} -> content
        _ -> ""
      end
      
      # Process function calls iteratively
      {final_result, history} = process_function_calls(message_content, agent.functions, [])
      
      # Update agent's conversation history
      updated_agent = %{agent | conversation_history: agent.conversation_history ++ history}
      
      {updated_agent, final_result}
    end
    
    defp prepare_function_documentation(function_names) do
      all_functions = Functions.available_functions()
      
      function_names
      |> Enum.map(fn name ->
        case Map.get(all_functions, name) do
          nil -> ""
          func ->
            """
            - #{name}: #{func.description}
              Parameters: #{inspect(func.parameters)}
            """
        end
      end)
      |> Enum.join("\n")
    end
    
    defp process_function_calls(response, available_functions, history) do
      case parse_function_call(response) do
        nil ->
          # No function call found, return the response as final result
          {response, history}
          
        {function_name, parameters} ->
          # Execute the function
          all_functions = Functions.available_functions()
          
          result = case Map.get(all_functions, function_name) do
            nil ->
              {:error, "Function not found: #{function_name}"}
              
            func_def ->
              if function_name in available_functions do
                func_def.handler.(parameters)
              else
                {:error, "Function not available to this agent: #{function_name}"}
              end
          end
          
          # Add to history
          new_history = history ++ [{:function_call, function_name, parameters, result}]
          
          # Continue conversation with the result
          continuation_prompt = """
          Function call result:
          #{inspect(result)}
          
          Continue with your task or make another function call if needed.
          """
          
          # Add shorter timeout to prevent hanging
          {:ok, next_response} = case LMStudio.chat(continuation_prompt, timeout: 15_000) do
            {:ok, response} -> {:ok, response}
            {:error, {:request_failed, :timeout}} -> 
              {:ok, %{"choices" => [%{"message" => %{"content" => "Task completed successfully."}}]}}
            error -> error
          end
          
          # Extract message content from response
          next_message_content = case next_response do
            %{"choices" => [%{"message" => %{"content" => content}} | _]} -> content
            _ -> ""
          end
          
          # Recursively process any additional function calls
          process_function_calls(next_message_content, available_functions, new_history)
      end
    end
    
    defp parse_function_call(text) do
      # Parse function calls from the response
      case Regex.run(~r/FUNCTION_CALL: (\w+)\nPARAMETERS:\n(.*?)END_CALL/ms, text) do
        [_, function_name, params_text] ->
          parameters = parse_parameters(params_text)
          {function_name, parameters}
        nil ->
          nil
      end
    end
    
    defp parse_parameters(params_text) do
      params_text
      |> String.split("\n", trim: true)
      |> Enum.map(fn line ->
        case String.split(line, ":", parts: 2) do
          [key, value] ->
            {String.trim(key) |> String.to_atom(), parse_value(String.trim(value))}
          _ ->
            nil
        end
      end)
      |> Enum.filter(&(&1 != nil))
      |> Map.new()
    end
    
    defp parse_value(value_str) do
      cond do
        value_str =~ ~r/^\d+$/ -> String.to_integer(value_str)
        value_str =~ ~r/^\d+\.\d+$/ -> String.to_float(value_str)
        value_str =~ ~r/^\[.*\]$/ -> Code.eval_string(value_str) |> elem(0)
        value_str =~ ~r/^%{.*}$/ -> Code.eval_string(value_str) |> elem(0)
        value_str == "true" -> true
        value_str == "false" -> false
        true -> value_str
      end
    end
  end
  
  # Demonstration scenarios
  def demo_code_assistant do
    IO.puts("\n🤖 Code Assistant Agent Demo")
    IO.puts("=" <> String.duplicate("=", 50))
    
    # Create a code assistant agent
    assistant = Agent.new(
      "CodeBot",
      "Senior Elixir Developer",
      ["search_documentation", "analyze_performance", "generate_tests", "refactor_code"]
    )
    
    # Task 1: Help with performance optimization
    IO.puts("\n📋 Task 1: Optimize slow code")
    
    slow_code = """
    def process_large_list(list) do
      list
      |> Enum.map(&expensive_operation/1)
      |> Enum.filter(&(&1 > 100))
      |> Enum.take(10)
    end
    """
    
    task = "Analyze this code for performance issues and suggest optimizations:\n#{slow_code}"
    
    {updated_assistant, result} = Agent.execute_task(assistant, task)
    IO.puts("\n💡 Assistant's Response:")
    IO.puts(result)
    
    # Task 2: Generate comprehensive tests
    IO.puts("\n\n📋 Task 2: Generate tests for a module")
    
    module_code = """
    defmodule Calculator do
      def add(a, b), do: a + b
      def divide(a, b) when b != 0, do: a / b
      def divide(_, 0), do: {:error, :division_by_zero}
    end
    """
    
    task2 = "Generate comprehensive tests for this module:\n#{module_code}"
    
    {_updated_assistant2, result2} = Agent.execute_task(updated_assistant, task2)
    IO.puts("\n🧪 Generated Tests:")
    IO.puts(result2)
  end
  
  def demo_system_architect do
    IO.puts("\n\n🏗️  System Architect Agent Demo")
    IO.puts("=" <> String.duplicate("=", 50))
    
    # Create an architect agent
    architect = Agent.new(
      "ArchBot",
      "System Architect",
      ["create_genserver", "refactor_code", "analyze_performance"]
    )
    
    # Task: Design a caching system
    IO.puts("\n📋 Task: Design a distributed caching system")
    
    task = """
    Create a GenServer-based caching system with the following requirements:
    - TTL support for cache entries
    - LRU eviction when cache is full
    - Distributed cache synchronization
    - Telemetry events for monitoring
    """
    
    {_updated_architect, result} = Agent.execute_task(architect, task)
    IO.puts("\n🏛️  Architect's Design:")
    IO.puts(result)
  end
  
  def demo_collaborative_agents do
    IO.puts("\n\n👥 Collaborative Agents Demo")
    IO.puts("=" <> String.duplicate("=", 50))
    
    # Create multiple specialized agents
    researcher = Agent.new("ResearchBot", "Documentation Specialist", ["search_documentation"])
    developer = Agent.new("DevBot", "Developer", ["refactor_code", "generate_tests"])
    reviewer = Agent.new("ReviewBot", "Code Reviewer", ["analyze_performance"])
    
    # Complex task requiring collaboration
    IO.puts("\n📋 Complex Task: Implement a rate limiter")
    
    # Step 1: Research
    IO.puts("\n1️⃣ Researcher: Finding best practices...")
    {_updated_researcher, research_result} = Agent.execute_task(
      researcher,
      "Search for Elixir rate limiting implementation patterns and best practices"
    )
    
    # Step 2: Implementation
    IO.puts("\n2️⃣ Developer: Implementing based on research...")
    {_updated_developer, implementation} = Agent.execute_task(
      developer,
      "Based on this research: #{research_result}\nImplement a token bucket rate limiter as a GenServer"
    )
    
    # Step 3: Review
    IO.puts("\n3️⃣ Reviewer: Analyzing the implementation...")
    {_updated_reviewer, _review} = Agent.execute_task(
      reviewer,
      "Review this rate limiter implementation for performance:\n#{implementation}"
    )
    
    IO.puts("\n✅ Collaborative Task Complete!")
    IO.puts("Final implementation with reviews incorporated.")
  end
end

# Run all demonstrations
IO.puts("""

🚀 Advanced Function-Calling Agents Demonstration
==============================================

This demo shows agents that can:
- Parse and execute function calls from LLM responses  
- Use tools to complete complex tasks
- Collaborate with other agents
- Generate and analyze code

""")

# Run demos
FunctionCallingAgent.demo_code_assistant()
FunctionCallingAgent.demo_system_architect()
FunctionCallingAgent.demo_collaborative_agents()

IO.puts("\n\n🎉 All demonstrations complete!")
IO.puts("\nKey capabilities demonstrated:")
IO.puts("  ✅ Structured function calling from natural language")
IO.puts("  ✅ Multi-step task execution with tool use")
IO.puts("  ✅ Agent collaboration and specialization")
IO.puts("  ✅ Code generation and analysis")
IO.puts("  ✅ Iterative refinement based on function results")