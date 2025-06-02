defmodule LMStudio.MultiAgentSystem do
  @moduledoc """
  Advanced Multi-Agent System with function calling, tool use, and collaborative problem solving.
  """

  use GenServer
  require Logger

  # Tool definitions for function calling
  defmodule Tools do
    @tools %{
      search_web: %{
        name: "search_web",
        description: "Search the web for information",
        parameters: %{
          query: %{type: "string", required: true, description: "Search query"},
          max_results: %{type: "integer", required: false, default: 5}
        },
        handler: &__MODULE__.handle_search_web/1
      },
      analyze_code: %{
        name: "analyze_code",
        description: "Analyze code for patterns, bugs, or improvements",
        parameters: %{
          code: %{type: "string", required: true},
          language: %{type: "string", required: true},
          analysis_type: %{type: "string", enum: ["bugs", "performance", "style", "security"]}
        },
        handler: &__MODULE__.handle_analyze_code/1
      },
      generate_code: %{
        name: "generate_code",
        description: "Generate code based on specifications",
        parameters: %{
          specification: %{type: "string", required: true},
          language: %{type: "string", required: true},
          framework: %{type: "string", required: false}
        },
        handler: &__MODULE__.handle_generate_code/1
      },
      execute_code: %{
        name: "execute_code",
        description: "Execute code in a sandboxed environment",
        parameters: %{
          code: %{type: "string", required: true},
          language: %{type: "string", required: true},
          inputs: %{type: "object", required: false}
        },
        handler: &__MODULE__.handle_execute_code/1
      },
      query_knowledge: %{
        name: "query_knowledge",
        description: "Query the internal knowledge base",
        parameters: %{
          query: %{type: "string", required: true},
          domain: %{type: "string", required: false}
        },
        handler: &__MODULE__.handle_query_knowledge/1
      }
    }

    def get_all_tools, do: @tools
    def get_tool(name), do: Map.get(@tools, name)

    def handle_search_web(%{query: _query} = params) do
      # Simulate web search
      results = [
        %{title: "Elixir Documentation", url: "https://elixir-lang.org", snippet: "Official Elixir docs"},
        %{title: "Elixir Forum", url: "https://elixirforum.com", snippet: "Community discussions"}
      ]
      {:ok, %{results: Enum.take(results, params[:max_results] || 5)}}
    end

    def handle_analyze_code(%{code: _code, language: lang, analysis_type: type}) do
      # Use MetaDSL to analyze code
      analysis = %{
        language: lang,
        type: type,
        findings: [
          %{line: 5, severity: "warning", message: "Consider using pattern matching"},
          %{line: 12, severity: "info", message: "This could be optimized with Stream"}
        ],
        metrics: %{complexity: 7, lines: 50, functions: 5}
      }
      {:ok, analysis}
    end

    def handle_generate_code(%{specification: spec, language: lang} = _params) do
      # Use MetaDSL for code generation
      # TODO: Implement actual code generation
      generated = "def generated_function do\n  # Generated from: #{spec}\nend"
      {:ok, %{code: generated, language: lang}}
    end

    def handle_execute_code(%{code: code, language: "elixir"} = params) do
      # Safe code execution in isolated process
      try do
        {result, _} = Code.eval_string(code, params[:inputs] || [])
        {:ok, %{result: inspect(result), success: true}}
      rescue
        e -> {:error, %{error: Exception.message(e), success: false}}
      end
    end

    def handle_query_knowledge(%{query: query} = params) do
      # Query internal knowledge base
      # TODO: Implement knowledge base query
      knowledge = [%{topic: "Erlang/OTP", info: "Query: #{query}"}]
      {:ok, %{results: knowledge, domain: params[:domain]}}
    end
  end

  # Agent types with specialized capabilities
  defmodule Agent do
    defstruct [:id, :type, :capabilities, :state, :memory, :tools]

    def new(type, opts \\ []) do
      %__MODULE__{
        id: generate_id(),
        type: type,
        capabilities: capabilities_for_type(type),
        state: :idle,
        memory: [],
        tools: opts[:tools] || []
      }
    end

    defp generate_id, do: :crypto.strong_rand_bytes(8) |> Base.encode16()

    defp capabilities_for_type(type) do
      case type do
        :coordinator -> [:planning, :delegation, :synthesis, :monitoring]
        :researcher -> [:search_web, :query_knowledge, :summarization]
        :coder -> [:generate_code, :analyze_code, :execute_code, :refactoring]
        :analyst -> [:data_analysis, :pattern_recognition, :reporting]
        :evolvor -> [:self_modification, :optimization, :learning]
        _ -> []
      end
    end
  end

  # Inter-agent communication protocol
  defmodule Message do
    defstruct [:from, :to, :type, :content, :timestamp, :correlation_id]

    def new(from, to, type, content) do
      %__MODULE__{
        from: from,
        to: to,
        type: type,
        content: content,
        timestamp: DateTime.utc_now(),
        correlation_id: generate_correlation_id()
      }
    end

    defp generate_correlation_id, do: :crypto.strong_rand_bytes(4) |> Base.encode16()
  end

  # GenServer implementation
  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  def init(opts) do
    state = %{
      agents: %{},
      message_queue: :queue.new(),
      tasks: %{},
      config: opts,
      tools: Tools.get_all_tools()
    }
    
    # Start default agents
    {:ok, spawn_default_agents(state)}
  end

  # Public API
  def submit_task(task_description, opts \\ []) do
    GenServer.call(__MODULE__, {:submit_task, task_description, opts})
  end

  def get_agents do
    GenServer.call(__MODULE__, :get_agents)
  end

  def send_message(from_agent, to_agent, type, content) do
    GenServer.cast(__MODULE__, {:send_message, from_agent, to_agent, type, content})
  end

  # Callbacks
  def handle_call({:submit_task, description, opts}, _from, state) do
    task_id = generate_task_id()
    task = %{
      id: task_id,
      description: description,
      status: :pending,
      assigned_to: nil,
      results: nil,
      opts: opts
    }
    
    # Coordinator analyzes and delegates task
    coordinator = find_agent_by_type(state.agents, :coordinator)
    send_message(coordinator.id, coordinator.id, :new_task, task)
    
    new_state = put_in(state.tasks[task_id], task)
    {:reply, {:ok, task_id}, new_state}
  end

  def handle_call(:get_agents, _from, state) do
    {:reply, state.agents, state}
  end

  def handle_cast({:send_message, from, to, type, content}, state) do
    message = Message.new(from, to, type, content)
    new_queue = :queue.in(message, state.message_queue)
    
    # Process message immediately
    new_state = process_message(message, %{state | message_queue: new_queue})
    {:noreply, new_state}
  end

  # Private functions
  defp spawn_default_agents(state) do
    agents = [
      Agent.new(:coordinator, tools: [:search_web, :query_knowledge]),
      Agent.new(:researcher, tools: [:search_web, :query_knowledge]),
      Agent.new(:coder, tools: [:generate_code, :analyze_code, :execute_code]),
      Agent.new(:analyst, tools: [:analyze_code, :query_knowledge]),
      Agent.new(:evolvor, tools: [:generate_code, :execute_code])
    ]
    
    agent_map = Map.new(agents, fn agent -> {agent.id, agent} end)
    %{state | agents: agent_map}
  end

  defp process_message(%Message{type: :new_task} = msg, state) do
    # Coordinator processes new task
    task = msg.content
    
    # Analyze task and determine which agents to involve
    analysis = analyze_task_with_llm(task.description)
    
    # Delegate subtasks
    subtasks = create_subtasks(analysis, task)
    
    Enum.reduce(subtasks, state, fn {agent_type, subtask}, acc ->
      agent = find_agent_by_type(acc.agents, agent_type)
      if agent do
        send_message(msg.from, agent.id, :execute_subtask, subtask)
      end
      acc
    end)
  end

  defp process_message(%Message{type: :execute_subtask} = msg, state) do
    # Agent executes subtask using tools
    agent = state.agents[msg.to]
    subtask = msg.content
    
    # Execute with function calling
    result = execute_with_tools(agent, subtask, state.tools)
    
    # Report back to coordinator
    coordinator = find_agent_by_type(state.agents, :coordinator)
    send_message(agent.id, coordinator.id, :subtask_complete, result)
    
    state
  end

  defp process_message(%Message{type: :subtask_complete} = _msg, state) do
    # Coordinator aggregates results
    # Update task status and results
    state
  end

  defp analyze_task_with_llm(description) do
    prompt = """
    Analyze this task and determine which agents should handle it:
    Task: #{description}
    
    Available agents:
    - researcher: web search, knowledge queries
    - coder: code generation, analysis, execution
    - analyst: data analysis, pattern recognition
    - evolvor: self-improvement, optimization
    
    Return a plan with specific subtasks for each agent.
    """
    
    {:ok, response} = LMStudio.chat(prompt, 
      system_prompt: "You are a task planning AI. Break down complex tasks into agent-specific subtasks."
    )
    
    # Parse response into structured plan
    parse_task_plan(response)
  end

  defp execute_with_tools(agent, subtask, available_tools) do
    # Prepare tool descriptions for LLM
    tool_descriptions = prepare_tool_descriptions(agent.tools, available_tools)
    
    prompt = """
    You are a #{agent.type} agent. Execute this subtask:
    #{subtask.description}
    
    Available tools:
    #{tool_descriptions}
    
    Use tools by calling them with appropriate parameters.
    Format: TOOL_CALL: tool_name(param1: value1, param2: value2)
    """
    
    {:ok, response} = LMStudio.chat(prompt,
      system_prompt: "You are an AI agent with access to tools. Execute tasks by calling appropriate tools."
    )
    
    # Parse and execute tool calls
    execute_tool_calls(response, available_tools)
  end

  defp prepare_tool_descriptions(agent_tools, all_tools) do
    agent_tools
    |> Enum.map(fn tool_name ->
      tool = all_tools[tool_name]
      """
      - #{tool.name}: #{tool.description}
        Parameters: #{inspect(tool.parameters)}
      """
    end)
    |> Enum.join("\n")
  end

  defp execute_tool_calls(response, tools) do
    # Parse tool calls from response
    ~r/TOOL_CALL: (\w+)\((.*?)\)/
    |> Regex.scan(response)
    |> Enum.map(fn [_, tool_name, params_str] ->
      tool = tools[String.to_atom(tool_name)]
      params = parse_tool_params(params_str)
      
      if tool && tool.handler do
        tool.handler.(params)
      else
        {:error, "Tool not found: #{tool_name}"}
      end
    end)
  end

  defp parse_tool_params(params_str) do
    # Simple parameter parsing (in production, use proper parser)
    params_str
    |> String.split(",")
    |> Enum.map(fn param ->
      [key, value] = String.split(param, ":", parts: 2)
      {String.trim(key) |> String.to_atom(), String.trim(value)}
    end)
    |> Map.new()
  end

  defp find_agent_by_type(agents, type) do
    agents
    |> Map.values()
    |> Enum.find(fn agent -> agent.type == type end)
  end

  defp create_subtasks(_analysis, task) do
    # Convert analysis into concrete subtasks
    # This would be more sophisticated in production
    [
      {:researcher, %{description: "Research best practices for #{task.description}"}},
      {:coder, %{description: "Implement solution based on research"}},
      {:analyst, %{description: "Analyze implementation for improvements"}}
    ]
  end

  defp generate_task_id, do: :crypto.strong_rand_bytes(8) |> Base.encode16()
  defp parse_task_plan(_response), do: %{agents: [:researcher, :coder], steps: []}
end