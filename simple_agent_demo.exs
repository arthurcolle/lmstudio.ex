#!/usr/bin/env elixir

defmodule SimpleAgentDemo do
  def run do
    IO.puts """
    
    🤖 Simple LM Studio Agent Demo
    ===================================
    
    This demo shows basic interaction with LM Studio.
    """
    
    # Test 1: Simple code analysis
    IO.puts "\n📋 Task 1: Analyze Elixir code"
    IO.puts "----------------------------------------"
    
    code_prompt = """
    Analyze this Elixir code and suggest improvements:
    
    ```elixir
    def calculate_total(items) do
      total = 0
      for item <- items do
        total = total + item.price * item.quantity
      end
      total
    end
    ```
    
    Please provide:
    1. What's wrong with this code
    2. How to fix it
    3. The corrected version
    """
    
    case LMStudio.chat(code_prompt, temperature: 0.7, max_tokens: 500) do
      {:ok, response} ->
        content = get_in(response, ["choices", Access.at(0), "message", "content"])
        IO.puts "Assistant: #{content}"
        
      {:error, reason} ->
        IO.puts "Error: #{inspect(reason)}"
    end
    
    # Test 2: Generate a function
    IO.puts "\n\n📋 Task 2: Generate a function"
    IO.puts "----------------------------------------"
    
    generate_prompt = """
    Generate an Elixir function that:
    - Takes a list of numbers
    - Filters out even numbers
    - Squares the remaining odd numbers
    - Returns the sum
    
    Include a docstring and examples.
    """
    
    case LMStudio.chat(generate_prompt, temperature: 0.7, max_tokens: 400) do
      {:ok, response} ->
        content = get_in(response, ["choices", Access.at(0), "message", "content"])
        IO.puts "Generated function:\n#{content}"
        
      {:error, reason} ->
        IO.puts "Error: #{inspect(reason)}"
    end
    
    # Test 3: Interactive conversation
    IO.puts "\n\n📋 Task 3: Multi-turn conversation"
    IO.puts "----------------------------------------"
    
    messages = [
      %{role: "user", content: "What are the benefits of using GenServer in Elixir?"},
    ]
    
    case LMStudio.complete(messages, temperature: 0.7, max_tokens: 300) do
      {:ok, response} ->
        content = get_in(response, ["choices", Access.at(0), "message", "content"])
        IO.puts "Assistant: #{content}"
        
        # Follow-up question
        messages = messages ++ [
          %{role: "assistant", content: content},
          %{role: "user", content: "Can you show me a simple example?"}
        ]
        
        IO.puts "\nUser: Can you show me a simple example?"
        
        case LMStudio.complete(messages, temperature: 0.7, max_tokens: 400) do
          {:ok, response2} ->
            content2 = get_in(response2, ["choices", Access.at(0), "message", "content"])
            IO.puts "\nAssistant: #{content2}"
            
          {:error, reason} ->
            IO.puts "Error in follow-up: #{inspect(reason)}"
        end
        
      {:error, reason} ->
        IO.puts "Error: #{inspect(reason)}"
    end
    
    IO.puts "\n\n✅ Demo completed!"
  end
end

SimpleAgentDemo.run()