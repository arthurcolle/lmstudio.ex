defmodule LMStudio.CognitiveAgent do
  @moduledoc """
  Self-modifying cognitive agent that can think and evolve its own reasoning patterns.
  
  This agent uses the LMStudio client to generate responses while continuously
  modifying its own prompts and behavioral patterns through MetaDSL mutations.
  """
  
  use GenServer
  require Logger
  alias LMStudio.MetaDSL.{Mutation, SelfModifyingGrid}
  
  @type state :: %{
    name: String.t(),
    grid_pid: pid(),
    model: String.t(),
    interaction_count: integer(),
    insight_count: integer(),
    thinking_enabled: boolean(),
    conversation_history: [map()],
    performance_tracker: pid()
  }
  
  # Client API
  
  def start_link(opts) do
    name = Keyword.fetch!(opts, :name)
    model = Keyword.get(opts, :model, "deepseek-r1-0528-qwen3-8b-mlx")
    thinking_enabled = Keyword.get(opts, :thinking_enabled, true)
    
    GenServer.start_link(__MODULE__, %{
      name: name,
      model: model,
      thinking_enabled: thinking_enabled
    }, name: via_tuple(name))
  end
  
  def process_query(agent_name, query, conversation_context \\ "") do
    GenServer.call(via_tuple(agent_name), {:process_query, query, conversation_context}, 120_000)
  end
  
  def get_state(agent_name) do
    GenServer.call(via_tuple(agent_name), :get_state)
  end
  
  def evolve_autonomously(agent_name, iterations \\ 5) do
    GenServer.cast(via_tuple(agent_name), {:evolve_autonomously, iterations})
  end
  
  def get_grid_data(agent_name) do
    GenServer.call(via_tuple(agent_name), :get_grid_data)
  end
  
  def add_performance_score(agent_name, score) do
    GenServer.cast(via_tuple(agent_name), {:add_performance, score})
  end
  
  defp via_tuple(name), do: {:via, Registry, {LMStudio.AgentRegistry, name}}
  
  # Server Callbacks
  
  @impl true
  def init(opts) do
    # Start the agent's personal self-modifying grid
    initial_grid_data = %{
      "identity" => "I am #{opts.name}, a self-modifying cognitive agent",
      "purpose" => "To evolve and improve through recursive self-modification",
      "strategy" => "Use <think> tags for deep reasoning and MetaDSL for self-mutation",
      "knowledge" => "I can modify my own prompts, learn from interactions, and evolve"
    }
    
    {:ok, grid_pid} = SelfModifyingGrid.start_link(initial_data: initial_grid_data)
    
    state = %{
      name: opts.name,
      grid_pid: grid_pid,
      model: opts.model,
      interaction_count: 0,
      insight_count: 0,
      thinking_enabled: opts.thinking_enabled,
      conversation_history: [],
      performance_tracker: nil
    }
    
    Logger.info("CognitiveAgent #{opts.name} initialized with thinking: #{opts.thinking_enabled}")
    {:ok, state}
  end
  
  @impl true
  def handle_call({:process_query, query, conversation_context}, _from, state) do
    case process_query_with_thinking(state, query, conversation_context) do
      {:ok, result, new_state} ->
        {:reply, {:ok, result}, new_state}
      {:error, reason} ->
        {:reply, {:error, reason}, state}
    end
  end
  
  @impl true
  def handle_call(:get_state, _from, state) do
    {:reply, state, state}
  end
  
  @impl true
  def handle_call(:get_grid_data, _from, state) do
    grid_data = SelfModifyingGrid.get_data(state.grid_pid)
    {:reply, grid_data, state}
  end
  
  @impl true
  def handle_cast({:add_performance, score}, state) do
    SelfModifyingGrid.add_performance(state.grid_pid, score)
    {:noreply, state}
  end
  
  @impl true
  def handle_cast({:evolve_autonomously, iterations}, state) do
    spawn(fn -> perform_autonomous_evolution(state, iterations) end)
    {:noreply, state}
  end
  
  # Private Functions
  
  defp process_query_with_thinking(state, query, conversation_context) do
    # Build context from grid
    grid_context = SelfModifyingGrid.render_grid(state.grid_pid)
    
    # Create meta-prompt with thinking capability
    meta_prompt = build_meta_prompt(state, grid_context, conversation_context, query)
    
    # Generate response using LMStudio
    messages = [
      %{role: "system", content: meta_prompt},
      %{role: "user", content: query}
    ]
    
    case LMStudio.complete(messages, model: state.model, temperature: 0.8, max_tokens: 2048) do
      {:ok, response} ->
        # Extract content from response
        content = extract_response_content(response)
        
        # Parse thinking and response parts
        {thinking_content, response_content} = parse_thinking_response(content)
        
        # Extract and apply mutations
        mutations = LMStudio.MutationParser.parse(response_content <> thinking_content)
        apply_mutations(state.grid_pid, mutations)
        
        # Update performance metrics
        performance_score = calculate_performance_score(thinking_content, response_content)
        SelfModifyingGrid.add_performance(state.grid_pid, performance_score)
        
        # Update state
        new_state = %{
          state |
          interaction_count: state.interaction_count + 1,
          insight_count: state.insight_count + count_insights(thinking_content),
          conversation_history: add_to_history(state.conversation_history, query, response_content)
        }
        
        result = %{
          thinking: thinking_content,
          response: response_content,
          mutations_applied: length(mutations),
          performance_score: performance_score
        }
        
        {:ok, result, new_state}
        
      {:error, {:request_failed, :timeout}} ->
        Logger.warning("LMStudio request timed out, using simulated response")
        simulate_response(state, query, "timeout")
        
      {:error, {:request_failed, :econnrefused}} ->
        Logger.warning("LMStudio server not available (connection refused), using simulated response")
        simulate_response(state, query, "connection_refused")
        
      {:error, {:http_error, status, _body}} when status >= 500 ->
        Logger.warning("LMStudio server error (status #{status}), using simulated response")
        simulate_response(state, query, "server_error")
        
      {:error, reason} ->
        Logger.warning("LMStudio connection failed: #{inspect(reason)}, using simulated response")
        simulate_response(state, query, reason)
    end
  rescue
    exception ->
      Logger.error("Exception in process_query_with_thinking: #{inspect(exception)}")
      {:error, {:exception, exception}}
  end
  
  defp build_meta_prompt(state, grid_context, conversation_context, query) do
    thinking_instruction = if state.thinking_enabled do
      """
      <think>
      Current grid state:
      #{grid_context}
      
      Conversation context:
      #{conversation_context}
      
      Query: #{query}
      
      I must:
      1. Think deeply about the query
      2. Consider how to evolve my reasoning
      3. Suggest mutations to improve myself using MetaDSL tags
      4. Generate insights that persist across interactions
      </think>
      """
    else
      ""
    end
    
    """
    #{thinking_instruction}
    
    #{grid_context}
    
    As #{state.name}, I will now respond while considering self-improvement opportunities.
    
    I can suggest mutations using these MetaDSL tags:
    - <append target="key">content</append>
    - <replace target="old_text" content="new_text"/>
    - <evolve target="key">evolution_content</evolve>
    - <mutate_strategy target="strategy_name">new_strategy</mutate_strategy>
    
    I should focus on continuous self-improvement and learning.
    """
  end
  
  defp extract_response_content(%{"choices" => [%{"message" => %{"content" => content}} | _]}) do
    content
  end
  defp extract_response_content(_), do: ""
  
  defp parse_thinking_response(content) do
    case Regex.run(~r/<think>(.*?)<\/think>(.*)/s, content) do
      [_, thinking, response] ->
        {String.trim(thinking), String.trim(response)}
      _ ->
        # No thinking tags found
        {"", content}
    end
  end
  
  defp apply_mutations(grid_pid, mutations) do
    Enum.each(mutations, fn mutation ->
      case SelfModifyingGrid.mutate(grid_pid, mutation) do
        :ok ->
          Logger.debug("Applied mutation: #{mutation.type} on #{mutation.target}")
        {:error, reason} ->
          Logger.warning("Failed to apply mutation: #{inspect(reason)}")
      end
    end)
  end
  
  defp calculate_performance_score(thinking, response) do
    score = 0.5
    
    # Longer thinking indicates deeper processing
    score = if byte_size(thinking) > 200, do: score + 0.1, else: score
    
    # Mentions of insights or realizations
    insight_keywords = ["insight", "realize", "understand", "discover", "learn"]
    has_insights = Enum.any?(insight_keywords, &String.contains?(String.downcase(thinking), &1))
    score = if has_insights, do: score + 0.2, else: score
    
    # Contains MetaDSL mutation tags
    has_mutations = String.contains?(response, "<append>") or 
                   String.contains?(response, "<evolve>") or
                   String.contains?(response, "<replace>")
    score = if has_mutations, do: score + 0.2, else: score
    
    # Response length and coherence (simple heuristic)
    score = if byte_size(response) > 100, do: score + 0.1, else: score
    
    min(score, 1.0)
  end
  
  defp count_insights(thinking) do
    insight_keywords = ["insight", "realize", "understand", "discover"]
    Enum.count(insight_keywords, &String.contains?(String.downcase(thinking), &1))
  end
  
  defp add_to_history(history, query, response) do
    entry = %{
      timestamp: DateTime.utc_now(),
      query: query,
      response: response
    }
    
    [entry | history]
    |> Enum.take(50)  # Keep last 50 interactions
  end
  
  defp simulate_response(state, query, _reason) do
    # Generate a simulated thinking process and response for demo purposes
    thinking_content = """
    I need to think about this query: "#{query}".
    
    Looking at my current grid state, I can see my identity, purpose, strategy, and knowledge.
    This query relates to self-improvement and evolution.
    
    I should suggest some mutations to evolve my capabilities:
    - Append new knowledge based on this interaction
    - Evolve my strategy to incorporate lessons learned
    
    This is a simulated response since LMStudio is not available.
    """
    
    response_content = """
    Thank you for asking about #{String.slice(query, 0, 50)}. As a self-modifying cognitive agent, I'm constantly evolving my understanding.
    
    <append target="knowledge">
    Learned from query about #{query}: Self-improvement requires continuous reflection and adaptation.
    </append>
    
    <evolve target="strategy">
    Enhanced approach: Integrate lessons from each interaction to build cumulative wisdom.
    </evolve>
    
    This response demonstrates my ability to self-modify and learn, even in simulated mode.
    """
    
    # Extract and apply mutations
    mutations = LMStudio.MutationParser.parse(response_content)
    apply_mutations(state.grid_pid, mutations)
    
    # Update performance metrics
    performance_score = calculate_performance_score(thinking_content, response_content)
    SelfModifyingGrid.add_performance(state.grid_pid, performance_score)
    
    # Update state
    new_state = %{
      state |
      interaction_count: state.interaction_count + 1,
      insight_count: state.insight_count + count_insights(thinking_content),
      conversation_history: add_to_history(state.conversation_history, query, response_content)
    }
    
    result = %{
      thinking: thinking_content,
      response: response_content,
      mutations_applied: length(mutations),
      performance_score: performance_score
    }
    
    {:ok, result, new_state}
  end

  defp perform_autonomous_evolution(state, iterations) do
    Logger.info("#{state.name} starting autonomous evolution for #{iterations} iterations")
    
    Enum.each(1..iterations, fn i ->
      Logger.debug("#{state.name} evolution iteration #{i}/#{iterations}")
      
      # Analyze current state
      analysis = SelfModifyingGrid.analyze_patterns(state.grid_pid)
      
      # Generate self-improvement query
      query = "Based on my analysis showing #{inspect(analysis)}, how should I evolve to improve my reasoning and effectiveness?"
      
      # Process self-directed evolution
      {:ok, _result, _new_state} = process_query_with_thinking(state, query, "Evolution iteration #{i}")
      Logger.debug("#{state.name} completed evolution iteration #{i}")
      
      # Apply suggested mutations from the grid
      suggestions = SelfModifyingGrid.suggest_mutations(state.grid_pid)
      top_suggestions = Enum.take(suggestions, 2)
      
      Enum.each(top_suggestions, fn suggestion ->
        SelfModifyingGrid.mutate(state.grid_pid, suggestion)
      end)
      
      # Brief pause between iterations
      Process.sleep(1000)
    end)
    
    Logger.info("#{state.name} completed autonomous evolution")
  end
end

defmodule LMStudio.MutationParser do
  @moduledoc """
  Parser for extracting MetaDSL mutation commands from LLM responses.
  """
  
  alias LMStudio.MetaDSL.Mutation
  
  def parse(text) when is_binary(text) do
    []
    |> parse_append_mutations(text)
    |> parse_replace_mutations(text)
    |> parse_delete_mutations(text)
    |> parse_evolve_mutations(text)
    |> parse_strategy_mutations(text)
  end
  
  defp parse_append_mutations(mutations, text) do
    pattern = ~r/<append target="([^"]+)">([^<]*)<\/append>/
    
    Regex.scan(pattern, text)
    |> Enum.map(fn [_, target, content] ->
      Mutation.new(:append, target, content: String.trim(content))
    end)
    |> Kernel.++(mutations)
  end
  
  defp parse_replace_mutations(mutations, text) do
    pattern = ~r/<replace target="([^"]+)" content="([^"]*)"\s*\/>/
    
    Regex.scan(pattern, text)
    |> Enum.map(fn [_, target, content] ->
      Mutation.new(:replace, target, content: content)
    end)
    |> Kernel.++(mutations)
  end
  
  defp parse_delete_mutations(mutations, text) do
    pattern = ~r/<delete target="([^"]+)"\s*\/>/
    
    Regex.scan(pattern, text)
    |> Enum.map(fn [_, target] ->
      Mutation.new(:delete, target)
    end)
    |> Kernel.++(mutations)
  end
  
  defp parse_evolve_mutations(mutations, text) do
    pattern = ~r/<evolve target="([^"]+)">([^<]*)<\/evolve>/
    
    Regex.scan(pattern, text)
    |> Enum.map(fn [_, target, content] ->
      Mutation.new(:evolve, target, content: String.trim(content))
    end)
    |> Kernel.++(mutations)
  end
  
  defp parse_strategy_mutations(mutations, text) do
    pattern = ~r/<mutate_strategy target="([^"]+)">([^<]*)<\/mutate_strategy>/
    
    Regex.scan(pattern, text)
    |> Enum.map(fn [_, target, content] ->
      Mutation.new(:mutate_strategy, target, content: String.trim(content))
    end)
    |> Kernel.++(mutations)
  end
end