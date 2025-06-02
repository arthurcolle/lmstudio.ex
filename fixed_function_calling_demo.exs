#!/usr/bin/env elixir

defmodule FixedFunctionCallingDemo do
  defmodule Functions do
    def available_functions do
      %{
        "analyze_performance" => %{
          description: "Analyze code performance and suggest optimizations",
          parameters: %{code: %{type: :string, required: true}, metrics: %{type: :list, default: [:time, :memory]}},
          handler: &handle_analyze_performance/1
        },
        "generate_tests" => %{
          description: "Generate test cases for given code", 
          parameters: %{module_code: %{type: :string, required: true}, test_framework: %{type: :string, default: "ExUnit"}},
          handler: &handle_generate_tests/1
        }
      }
    end
    
    defp handle_analyze_performance(%{code: code} = params) do
      metrics = Map.get(params, :metrics, [:time, :memory])
      
      # Simulate performance analysis
      {:ok, %{
        analysis: "Code uses Enum operations which are eager. For large lists, consider using Stream for lazy evaluation.",
        metrics: %{
          time_complexity: "O(n)",
          space_complexity: "O(1)",
          memory_usage: "2.4MB"
        },
        suggestions: [
          "Use Stream.map instead of Enum.map for large datasets",
          "Consider pattern matching for better readability"
        ]
      }}
    end
    
    defp handle_generate_tests(%{module_code: code, test_framework: framework}) do
      # Simulate test generation
      test_code = """
      defmodule CalculatorTest do
        use #{framework}.Case
        
        describe "add/2" do
          test "adds two positive numbers" do
            assert Calculator.add(2, 3) == 5
          end
          
          test "adds negative numbers" do
            assert Calculator.add(-1, -1) == -2
          end
        end
        
        describe "divide/2" do
          test "divides two numbers" do
            assert Calculator.divide(10, 2) == 5
          end
          
          test "returns error for division by zero" do
            assert Calculator.divide(10, 0) == {:error, :division_by_zero}
          end
        end
      end
      """
      
      {:ok, %{test_code: test_code, test_count: 4}}
    end
  end
  
  def run do
    IO.puts """
    
    🚀 Fixed Function-Calling Demo
    ==============================
    
    This demo simulates function calling without relying on specific LLM output formats.
    """
    
    # Demo 1: Analyze performance
    IO.puts "\n📋 Demo 1: Performance Analysis"
    IO.puts "--------------------------------"
    
    code_to_analyze = """
    def process_large_list(list) do
      list
      |> Enum.map(&expensive_operation/1)
      |> Enum.filter(&(&1 > 100))
      |> Enum.take(10)
    end
    """
    
    IO.puts "Analyzing code:\n#{code_to_analyze}"
    
    # Simulate function call
    functions = Functions.available_functions()
    analyze_func = functions["analyze_performance"]
    
    {:ok, result} = analyze_func.handler.(%{code: code_to_analyze})
    
    IO.puts "\n🔍 Analysis Results:"
    IO.puts "  #{result.analysis}"
    IO.puts "\n📊 Metrics:"
    Enum.each(result.metrics, fn {k, v} ->
      IO.puts "  - #{k}: #{v}"
    end)
    IO.puts "\n💡 Suggestions:"
    Enum.each(result.suggestions, fn suggestion ->
      IO.puts "  - #{suggestion}"
    end)
    
    # Demo 2: Generate tests
    IO.puts "\n\n📋 Demo 2: Test Generation"
    IO.puts "--------------------------------"
    
    module_code = """
    defmodule Calculator do
      def add(a, b), do: a + b
      def divide(a, b) when b != 0, do: a / b
      def divide(_, 0), do: {:error, :division_by_zero}
    end
    """
    
    IO.puts "Generating tests for:\n#{module_code}"
    
    generate_func = functions["generate_tests"]
    {:ok, test_result} = generate_func.handler.(%{module_code: module_code, test_framework: "ExUnit"})
    
    IO.puts "\n🧪 Generated Tests:"
    IO.puts test_result.test_code
    IO.puts "\n✅ Generated #{test_result.test_count} test cases"
    
    # Demo 3: Interactive function selection
    IO.puts "\n\n📋 Demo 3: Interactive Function Selection"
    IO.puts "-----------------------------------------"
    
    IO.puts "Available functions:"
    Enum.each(functions, fn {name, func} ->
      IO.puts "  - #{name}: #{func.description}"
    end)
    
    # Simulate user asking for help
    user_request = "I need to analyze this code for performance issues"
    IO.puts "\nUser: #{user_request}"
    
    # Simple keyword matching to determine function
    selected_function = cond do
      String.contains?(user_request, "analyze") && String.contains?(user_request, "performance") ->
        "analyze_performance"
      String.contains?(user_request, "test") || String.contains?(user_request, "generate") ->
        "generate_tests"
      true ->
        nil
    end
    
    if selected_function do
      IO.puts "Assistant: I'll use the '#{selected_function}' function to help you."
    else
      IO.puts "Assistant: I'm not sure which function to use. Please be more specific."
    end
    
    IO.puts "\n\n✅ Demo completed successfully!"
  end
end

FixedFunctionCallingDemo.run()