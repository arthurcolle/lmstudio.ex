#!/usr/bin/env elixir

defmodule WorkingAgentDemo do
  def run do
    IO.puts """
    
    🤖 Working LM Studio Agent Demo
    ===================================
    
    This demo shows how to work with models that use thinking tags.
    """
    
    # Test with system prompt to get direct responses
    system_prompt = """
    You are a helpful AI assistant. Please provide direct, clear responses without using any thinking tags or internal monologue. 
    When asked to analyze code or perform tasks, provide your answer immediately and concisely.
    """
    
    # Test 1: Simple code analysis
    IO.puts "\n📋 Task 1: Analyze Elixir code"
    IO.puts "----------------------------------------"
    
    code_prompt = """
    Analyze this Elixir code briefly and suggest the main improvement needed:
    
    ```elixir
    def calculate_total(items) do
      total = 0
      for item <- items do
        total = total + item.price * item.quantity
      end
      total
    end
    ```
    """
    
    case LMStudio.chat(code_prompt, 
      system_prompt: system_prompt,
      temperature: 0.3, 
      max_tokens: 200
    ) do
      {:ok, response} ->
        content = extract_content(response)
        IO.puts "Analysis: #{content}"
        
      {:error, reason} ->
        IO.puts "Error: #{inspect(reason)}"
    end
    
    # Test 2: Generate a simple function
    IO.puts "\n\n📋 Task 2: Generate a function"
    IO.puts "----------------------------------------"
    
    generate_prompt = """
    Write a simple Elixir function called `sum_odd_squares` that takes a list of numbers,
    filters odd numbers, squares them, and returns the sum. Just the function code, no explanation.
    """
    
    case LMStudio.chat(generate_prompt,
      system_prompt: system_prompt,
      temperature: 0.3,
      max_tokens: 150
    ) do
      {:ok, response} ->
        content = extract_content(response)
        IO.puts "Generated function:\n#{content}"
        
      {:error, reason} ->
        IO.puts "Error: #{inspect(reason)}"
    end
    
    # Test 3: Function calling simulation
    IO.puts "\n\n📋 Task 3: Function calling simulation"
    IO.puts "----------------------------------------"
    
    function_system_prompt = """
    You are an AI assistant that can call functions. When you need to use a function, 
    respond ONLY with this exact format:
    
    FUNCTION: function_name
    PARAMS: param1=value1, param2=value2
    
    No other text before or after. Available functions:
    - search_docs(query) - Search Elixir documentation
    - analyze_code(code) - Analyze code for issues
    """
    
    function_prompt = "Search the documentation for GenServer"
    
    case LMStudio.chat(function_prompt,
      system_prompt: function_system_prompt,
      temperature: 0.1,
      max_tokens: 50
    ) do
      {:ok, response} ->
        content = extract_content(response)
        IO.puts "Response: #{content}"
        
        # Parse function call
        if String.contains?(content, "FUNCTION:") do
          [_, function_part] = String.split(content, "FUNCTION:", parts: 2)
          [function_line | _] = String.split(function_part, "\n", trim: true)
          function_name = String.trim(function_line)
          
          IO.puts "\nDetected function call: #{function_name}"
          
          # Extract params if present
          if String.contains?(content, "PARAMS:") do
            [_, params_part] = String.split(content, "PARAMS:", parts: 2)
            [params_line | _] = String.split(params_part, "\n", trim: true)
            params = String.trim(params_line)
            IO.puts "Parameters: #{params}"
          end
        end
        
      {:error, reason} ->
        IO.puts "Error: #{inspect(reason)}"
    end
    
    IO.puts "\n\n✅ Demo completed!"
  end
  
  defp extract_content(response) do
    case response do
      %{"choices" => [%{"message" => %{"content" => content}} | _]} -> 
        # Remove thinking tags if present
        content
        |> String.replace(~r/<think>.*?<\/think>/s, "")
        |> String.trim()
      _ -> 
        ""
    end
  end
end

WorkingAgentDemo.run()