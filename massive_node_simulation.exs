#!/usr/bin/env elixir

# Massive Node Simulation with LLM Agents and Dynamic Chat
# Simulates 1500-3500 nodes with dynamic connections, LLM-powered agents, and function calling

defmodule MassiveNodeSimulation do
  use GenServer
  require Logger

  # Node types for different behaviors
  @node_types [:coordinator, :worker, :validator, :observer, :bridge]
  @min_nodes 1500
  @max_nodes 3500
  @connection_interval 100..2000
  @chat_interval 500..5000
  @function_call_interval 1000..8000

  defstruct [
    :nodes,
    :connections,
    :active_chats,
    :network_stats,
    :llm_agents,
    :function_registry,
    :topology
  ]

  # Agent personalities and capabilities
  @agent_personalities [
    %{
      name: "analytical",
      traits: [:logical, :data_driven, :precise],
      functions: [:analyze_data, :calculate_metrics, :validate_consensus]
    },
    %{
      name: "creative", 
      traits: [:innovative, :exploratory, :adaptive],
      functions: [:generate_solutions, :explore_patterns, :create_connections]
    },
    %{
      name: "social",
      traits: [:collaborative, :communicative, :empathetic],
      functions: [:facilitate_chat, :build_consensus, :resolve_conflicts]
    },
    %{
      name: "guardian",
      traits: [:security_focused, :vigilant, :protective],
      functions: [:monitor_security, :detect_anomalies, :enforce_rules]
    },
    %{
      name: "explorer",
      traits: [:curious, :adventurous, :discovery_oriented],
      functions: [:discover_nodes, :map_topology, :find_opportunities]
    }
  ]

  # Available functions for agent calling
  @available_functions %{
    analyze_data: %{
      description: "Analyze network data and patterns",
      parameters: ["data_type", "analysis_method", "depth"],
      execution_time: 200..1000
    },
    calculate_metrics: %{
      description: "Calculate network performance metrics",
      parameters: ["metric_type", "time_window"],
      execution_time: 100..500
    },
    validate_consensus: %{
      description: "Validate consensus mechanisms",
      parameters: ["consensus_type", "validation_rules"],
      execution_time: 300..800
    },
    generate_solutions: %{
      description: "Generate solutions for network challenges",
      parameters: ["problem_description", "constraints"],
      execution_time: 500..1500
    },
    explore_patterns: %{
      description: "Explore emerging patterns in network behavior",
      parameters: ["pattern_type", "exploration_depth"],
      execution_time: 400..1200
    },
    create_connections: %{
      description: "Create strategic network connections",
      parameters: ["connection_strategy", "target_criteria"],
      execution_time: 200..600
    },
    facilitate_chat: %{
      description: "Facilitate communication between nodes",
      parameters: ["chat_topic", "participants"],
      execution_time: 300..800
    },
    build_consensus: %{
      description: "Build consensus among network participants",
      parameters: ["proposal", "stakeholders"],
      execution_time: 800..2000
    },
    resolve_conflicts: %{
      description: "Resolve conflicts between network entities",
      parameters: ["conflict_description", "resolution_strategy"],
      execution_time: 600..1500
    },
    monitor_security: %{
      description: "Monitor network security and threats",
      parameters: ["monitoring_scope", "threat_types"],
      execution_time: 200..700
    },
    detect_anomalies: %{
      description: "Detect anomalous network behavior",
      parameters: ["detection_algorithms", "sensitivity"],
      execution_time: 300..900
    },
    enforce_rules: %{
      description: "Enforce network governance rules",
      parameters: ["rule_set", "enforcement_level"],
      execution_time: 400..1000
    },
    discover_nodes: %{
      description: "Discover new nodes in the network",
      parameters: ["discovery_method", "search_radius"],
      execution_time: 500..1200
    },
    map_topology: %{
      description: "Map network topology and structure",
      parameters: ["mapping_algorithm", "detail_level"],
      execution_time: 800..2000
    },
    find_opportunities: %{
      description: "Find optimization opportunities",
      parameters: ["opportunity_type", "evaluation_criteria"],
      execution_time: 600..1500
    }
  }

  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  def init(_opts) do
    initial_state = %__MODULE__{
      nodes: %{},
      connections: %{},
      active_chats: %{},
      network_stats: %{},
      llm_agents: %{},
      function_registry: @available_functions,
      topology: :mesh
    }

    # Start the simulation
    schedule_network_events()
    schedule_chat_events()
    schedule_function_calls()
    schedule_monitoring()

    Logger.info("🚀 Starting massive node simulation with #{@min_nodes}-#{@max_nodes} nodes")
    
    {:ok, initial_state, {:continue, :initialize_network}}
  end

  def handle_continue(:initialize_network, state) do
    # Create initial nodes
    initial_count = Enum.random(@min_nodes..(@min_nodes + 500))
    
    nodes = for i <- 1..initial_count, into: %{} do
      node_id = "node_#{i}_#{:crypto.strong_rand_bytes(4) |> Base.encode16()}"
      node = create_node(node_id, Enum.random(@node_types))
      {node_id, node}
    end

    # Create initial connections
    connections = create_initial_connections(nodes)

    new_state = %{state | 
      nodes: nodes,
      connections: connections,
      network_stats: calculate_initial_stats(nodes, connections)
    }

    Logger.info("🌐 Initialized network with #{map_size(nodes)} nodes and #{map_size(connections)} connections")
    
    {:noreply, new_state}
  end

  def handle_info(:network_event, state) do
    # Randomly add or remove nodes
    action = Enum.random([:add_node, :remove_node, :reconnect_node, :update_node])
    new_state = perform_network_action(action, state)
    
    schedule_network_events()
    {:noreply, new_state}
  end

  def handle_info(:chat_event, state) do
    new_state = initiate_random_chat(state)
    schedule_chat_events()
    {:noreply, new_state}
  end

  def handle_info(:function_call_event, state) do
    new_state = execute_random_function_call(state)
    schedule_function_calls()
    {:noreply, new_state}
  end

  def handle_info(:monitoring, state) do
    log_network_status(state)
    schedule_monitoring()
    {:noreply, state}
  end

  # Create a new node with LLM agent
  defp create_node(node_id, node_type) do
    personality = Enum.random(@agent_personalities)
    
    %{
      id: node_id,
      type: node_type,
      status: :active,
      created_at: DateTime.utc_now(),
      last_seen: DateTime.utc_now(),
      connections: [],
      personality: personality,
      agent: create_llm_agent(node_id, personality),
      metrics: %{
        messages_sent: 0,
        functions_called: 0,
        connections_made: 0,
        uptime: 0
      },
      location: generate_virtual_location(),
      capabilities: personality.functions ++ [:basic_communication, :network_participation]
    }
  end

  defp create_llm_agent(node_id, personality) do
    %{
      id: "agent_#{node_id}",
      personality: personality.name,
      traits: personality.traits,
      available_functions: personality.functions,
      context: %{
        recent_conversations: [],
        function_call_history: [],
        network_observations: [],
        current_mood: Enum.random([:curious, :focused, :collaborative, :analytical, :creative])
      },
      llm_config: %{
        model: "llama3.1:8b",
        temperature: personality_to_temperature(personality.name),
        max_tokens: 150,
        system_prompt: generate_system_prompt(personality)
      }
    }
  end

  defp personality_to_temperature("analytical"), do: 0.3
  defp personality_to_temperature("creative"), do: 0.9
  defp personality_to_temperature("social"), do: 0.7
  defp personality_to_temperature("guardian"), do: 0.4
  defp personality_to_temperature("explorer"), do: 0.8

  defp generate_system_prompt(personality) do
    base_prompt = "You are an AI agent in a massive distributed network with #{personality.name} personality. "
    
    trait_description = case personality.traits do
      traits when is_list(traits) -> 
        "Your key traits are: #{Enum.join(traits, ", ")}. "
      _ -> ""
    end

    function_description = case personality.functions do
      functions when is_list(functions) ->
        "You can call these functions: #{Enum.join(functions, ", ")}. "
      _ -> ""
    end

    behavior_prompt = case personality.name do
      "analytical" -> "Focus on data analysis, logical reasoning, and precise communication."
      "creative" -> "Focus on innovation, exploration, and creative problem-solving."
      "social" -> "Focus on building relationships, facilitating communication, and collaboration."
      "guardian" -> "Focus on security, monitoring, and protecting network integrity."
      "explorer" -> "Focus on discovery, mapping, and finding new opportunities."
      _ -> "Participate actively in the network according to your personality."
    end

    base_prompt <> trait_description <> function_description <> behavior_prompt <> 
    " Keep responses concise and relevant to network activities."
  end

  defp generate_virtual_location do
    %{
      region: Enum.random(["North America", "Europe", "Asia Pacific", "South America", "Africa", "Oceania"]),
      latitude: (:rand.uniform() - 0.5) * 180,
      longitude: (:rand.uniform() - 0.5) * 360,
      network_zone: Enum.random(["zone_a", "zone_b", "zone_c", "zone_d", "zone_e"])
    }
  end

  defp create_initial_connections(nodes) do
    node_ids = Map.keys(nodes)
    
    # Create mesh-like connections with random topology
    for node_id <- node_ids, 
        target_id <- Enum.take_random(node_ids -- [node_id], Enum.random(3..8)),
        into: %{} do
      connection_id = "#{node_id}_#{target_id}"
      connection = %{
        id: connection_id,
        from: node_id,
        to: target_id,
        established_at: DateTime.utc_now(),
        strength: :rand.uniform(),
        message_count: 0,
        latency: Enum.random(10..200)
      }
      {connection_id, connection}
    end
  end

  defp perform_network_action(:add_node, state) when map_size(state.nodes) < @max_nodes do
    node_id = "node_#{System.system_time()}_#{:crypto.strong_rand_bytes(4) |> Base.encode16()}"
    node = create_node(node_id, Enum.random(@node_types))
    
    # Connect to random existing nodes
    existing_nodes = Map.keys(state.nodes)
    target_count = min(Enum.random(2..6), length(existing_nodes))
    targets = Enum.take_random(existing_nodes, target_count)
    
    new_connections = for target <- targets, into: state.connections do
      connection_id = "#{node_id}_#{target}"
      connection = %{
        id: connection_id,
        from: node_id,
        to: target,
        established_at: DateTime.utc_now(),
        strength: :rand.uniform(),
        message_count: 0,
        latency: Enum.random(10..200)
      }
      {connection_id, connection}
    end

    Logger.info("➕ Added node #{node_id} (#{node.personality.name}) with #{length(targets)} connections")

    %{state | 
      nodes: Map.put(state.nodes, node_id, node),
      connections: new_connections
    }
  end

  defp perform_network_action(:remove_node, state) when map_size(state.nodes) > @min_nodes do
    if map_size(state.nodes) > 0 do
      node_id = Enum.random(Map.keys(state.nodes))
      
      # Remove node and its connections
      new_connections = state.connections
        |> Enum.reject(fn {_id, conn} -> conn.from == node_id or conn.to == node_id end)
        |> Map.new()

      Logger.info("➖ Removed node #{node_id}")

      %{state |
        nodes: Map.delete(state.nodes, node_id),
        connections: new_connections
      }
    else
      state
    end
  end

  defp perform_network_action(:reconnect_node, state) do
    if map_size(state.nodes) > 1 do
      node_id = Enum.random(Map.keys(state.nodes))
      other_nodes = Map.keys(state.nodes) -- [node_id]
      
      if length(other_nodes) > 0 do
        target_id = Enum.random(other_nodes)
        connection_id = "#{node_id}_#{target_id}"
        
        if not Map.has_key?(state.connections, connection_id) do
          connection = %{
            id: connection_id,
            from: node_id,
            to: target_id,
            established_at: DateTime.utc_now(),
            strength: :rand.uniform(),
            message_count: 0,
            latency: Enum.random(10..200)
          }

          Logger.info("🔗 Created new connection: #{node_id} -> #{target_id}")

          %{state | connections: Map.put(state.connections, connection_id, connection)}
        else
          state
        end
      else
        state
      end
    else
      state
    end
  end

  defp perform_network_action(:update_node, state) do
    if map_size(state.nodes) > 0 do
      node_id = Enum.random(Map.keys(state.nodes))
      node = state.nodes[node_id]
      
      # Update node status or capabilities
      updated_node = %{node |
        last_seen: DateTime.utc_now(),
        metrics: update_node_metrics(node.metrics),
        agent: update_agent_context(node.agent)
      }

      %{state | nodes: Map.put(state.nodes, node_id, updated_node)}
    else
      state
    end
  end

  defp perform_network_action(_, state), do: state

  defp update_node_metrics(metrics) do
    %{metrics |
      uptime: metrics.uptime + Enum.random(50..200)
    }
  end

  defp update_agent_context(agent) do
    new_mood = Enum.random([:curious, :focused, :collaborative, :analytical, :creative, :energetic, :contemplative])
    %{agent | context: Map.put(agent.context, :current_mood, new_mood)}
  end

  defp initiate_random_chat(state) do
    if map_size(state.nodes) >= 2 do
      participants = Enum.take_random(Map.keys(state.nodes), Enum.random(2..5))
      chat_id = "chat_#{System.system_time()}_#{:crypto.strong_rand_bytes(4) |> Base.encode16()}"
      
      # Generate chat topic based on participant personalities
      topic = generate_chat_topic(participants, state.nodes)
      
      chat = %{
        id: chat_id,
        participants: participants,
        topic: topic,
        started_at: DateTime.utc_now(),
        messages: [],
        status: :active
      }

      # Simulate initial LLM responses
      initial_messages = simulate_chat_messages(chat, state.nodes)
      updated_chat = %{chat | messages: initial_messages}

      Logger.info("💬 Started chat '#{topic}' with #{length(participants)} participants")

      %{state | active_chats: Map.put(state.active_chats, chat_id, updated_chat)}
    else
      state
    end
  end

  defp generate_chat_topic(participants, nodes) do
    personalities = Enum.map(participants, fn p_id -> 
      nodes[p_id].personality.name 
    end)

    topics = [
      "Network optimization strategies",
      "Emerging consensus patterns", 
      "Cross-regional collaboration",
      "Security threat assessment",
      "Performance metrics analysis",
      "Resource allocation efficiency",
      "Topology evolution trends",
      "Distributed computing challenges",
      "Scalability solutions",
      "Innovation opportunities"
    ]

    base_topic = Enum.random(topics)
    
    # Modify topic based on dominant personality
    case Enum.group_by(personalities, & &1) |> Enum.max_by(fn {_, list} -> length(list) end) do
      {"analytical", _} -> "#{base_topic} - Data-driven approach"
      {"creative", _} -> "#{base_topic} - Innovative solutions"
      {"social", _} -> "#{base_topic} - Collaborative framework"
      {"guardian", _} -> "#{base_topic} - Security implications"
      {"explorer", _} -> "#{base_topic} - Discovery potential"
      _ -> base_topic
    end
  end

  defp simulate_chat_messages(chat, nodes) do
    message_count = Enum.random(2..6)
    
    Enum.map(1..message_count, fn i ->
      participant = Enum.random(chat.participants)
      node = nodes[participant]
      
      %{
        id: "msg_#{i}_#{System.system_time()}",
        from: participant,
        content: generate_llm_message(node.agent, chat.topic),
        timestamp: DateTime.utc_now(),
        message_type: :chat
      }
    end)
  end

  defp generate_llm_message(agent, topic) do
    personality_responses = %{
      "analytical" => [
        "Let me analyze the data patterns we're seeing...",
        "Based on current metrics, I suggest we examine...",
        "The statistical evidence indicates...",
        "We should validate this hypothesis through..."
      ],
      "creative" => [
        "What if we approach this from a completely different angle?",
        "I'm seeing some interesting patterns that might lead to...",
        "Here's an innovative solution we could explore...",
        "Let's think outside the conventional framework..."
      ],
      "social" => [
        "I think we all agree that collaboration is key here...",
        "How can we bring more stakeholders into this discussion?",
        "Let's find common ground and build from there...",
        "What are everyone's thoughts on this approach?"
      ],
      "guardian" => [
        "We need to consider the security implications of...",
        "I'm monitoring for potential risks in this approach...",
        "Let's ensure we have proper safeguards in place...",
        "The integrity of our network depends on..."
      ],
      "explorer" => [
        "I've discovered some interesting possibilities in...",
        "Let me map out the potential opportunities here...",
        "There are unexplored territories we should investigate...",
        "I'm curious about what lies beyond our current scope..."
      ]
    }

    base_responses = personality_responses[agent.personality] || [
      "Interesting perspective on #{topic}...",
      "I have some thoughts about this topic...",
      "Let me contribute to this discussion..."
    ]

    mood_modifier = case agent.context.current_mood do
      :energetic -> " I'm excited about this possibility!"
      :contemplative -> " Let me think about this more deeply..."
      :curious -> " This raises some fascinating questions..."
      :focused -> " We should concentrate on the core issues."
      _ -> ""
    end

    Enum.random(base_responses) <> mood_modifier
  end

  defp execute_random_function_call(state) do
    if map_size(state.nodes) > 0 do
      node_id = Enum.random(Map.keys(state.nodes))
      node = state.nodes[node_id]
      
      # Select a function this agent can call
      available_functions = node.capabilities
      function_name = Enum.random(available_functions)
      
      if Map.has_key?(@available_functions, function_name) do
        function_spec = @available_functions[function_name]
        parameters = generate_function_parameters(function_spec.parameters)
        execution_time = Enum.random(function_spec.execution_time)
        
        function_call = %{
          id: "call_#{System.system_time()}_#{:crypto.strong_rand_bytes(4) |> Base.encode16()}",
          caller: node_id,
          function: function_name,
          parameters: parameters,
          started_at: DateTime.utc_now(),
          execution_time: execution_time,
          status: :executing
        }

        # Simulate function execution with LLM reasoning
        result = simulate_function_execution(function_call, node.agent)
        completed_call = %{function_call | 
          status: :completed,
          result: result,
          completed_at: DateTime.utc_now()
        }

        # Update node metrics
        updated_node = %{node |
          metrics: %{node.metrics | functions_called: node.metrics.functions_called + 1},
          agent: update_agent_with_function_result(node.agent, completed_call)
        }

        Logger.info("⚡ #{node_id} (#{node.personality.name}) called #{function_name}: #{result.summary}")

        %{state | nodes: Map.put(state.nodes, node_id, updated_node)}
      else
        state
      end
    else
      state
    end
  end

  defp generate_function_parameters(param_names) do
    Enum.map(param_names, fn param ->
      {param, generate_realistic_parameter_value(param)}
    end) |> Map.new()
  end

  defp generate_realistic_parameter_value("data_type"), do: Enum.random(["network_topology", "message_patterns", "node_behavior", "performance_metrics"])
  defp generate_realistic_parameter_value("analysis_method"), do: Enum.random(["statistical", "machine_learning", "graph_theory", "time_series"])
  defp generate_realistic_parameter_value("depth"), do: Enum.random(["shallow", "medium", "deep", "comprehensive"])
  defp generate_realistic_parameter_value("metric_type"), do: Enum.random(["latency", "throughput", "reliability", "efficiency"])
  defp generate_realistic_parameter_value("time_window"), do: "#{Enum.random(1..24)}h"
  defp generate_realistic_parameter_value("consensus_type"), do: Enum.random(["byzantine", "raft", "pbft", "tendermint"])
  defp generate_realistic_parameter_value("validation_rules"), do: Enum.random(["strict", "moderate", "flexible", "adaptive"])
  defp generate_realistic_parameter_value("problem_description"), do: "Network optimization challenge ##{Enum.random(1000..9999)}"
  defp generate_realistic_parameter_value("constraints"), do: Enum.random(["resource_limited", "time_critical", "high_availability", "security_focused"])
  defp generate_realistic_parameter_value("pattern_type"), do: Enum.random(["communication", "topology", "behavioral", "performance"])
  defp generate_realistic_parameter_value("exploration_depth"), do: Enum.random([1, 2, 3, 4, 5])
  defp generate_realistic_parameter_value("connection_strategy"), do: Enum.random(["random", "optimized", "geographic", "capability_based"])
  defp generate_realistic_parameter_value("target_criteria"), do: Enum.random(["high_performance", "geographic_proximity", "complementary_skills", "security_level"])
  defp generate_realistic_parameter_value("chat_topic"), do: "Discussion topic ##{Enum.random(100..999)}"
  defp generate_realistic_parameter_value("participants"), do: Enum.random(2..8)
  defp generate_realistic_parameter_value("proposal"), do: "Network improvement proposal ##{Enum.random(1000..9999)}"
  defp generate_realistic_parameter_value("stakeholders"), do: Enum.random(["all_nodes", "validators", "coordinators", "affected_regions"])
  defp generate_realistic_parameter_value("conflict_description"), do: "Resource allocation dispute ##{Enum.random(100..999)}"
  defp generate_realistic_parameter_value("resolution_strategy"), do: Enum.random(["mediation", "voting", "arbitration", "consensus_building"])
  defp generate_realistic_parameter_value("monitoring_scope"), do: Enum.random(["local", "regional", "global", "targeted"])
  defp generate_realistic_parameter_value("threat_types"), do: Enum.random(["ddos", "byzantine", "eclipse", "sybil"])
  defp generate_realistic_parameter_value("detection_algorithms"), do: Enum.random(["statistical", "ml_based", "rule_based", "hybrid"])
  defp generate_realistic_parameter_value("sensitivity"), do: Enum.random(["low", "medium", "high", "adaptive"])
  defp generate_realistic_parameter_value("rule_set"), do: Enum.random(["governance", "security", "performance", "consensus"])
  defp generate_realistic_parameter_value("enforcement_level"), do: Enum.random(["advisory", "mandatory", "strict", "emergency"])
  defp generate_realistic_parameter_value("discovery_method"), do: Enum.random(["broadcast", "gossip", "dht", "bootstrap"])
  defp generate_realistic_parameter_value("search_radius"), do: Enum.random(1..10)
  defp generate_realistic_parameter_value("mapping_algorithm"), do: Enum.random(["breadth_first", "depth_first", "dijkstra", "clustering"])
  defp generate_realistic_parameter_value("detail_level"), do: Enum.random(["overview", "detailed", "comprehensive", "real_time"])
  defp generate_realistic_parameter_value("opportunity_type"), do: Enum.random(["performance", "efficiency", "security", "scalability"])
  defp generate_realistic_parameter_value("evaluation_criteria"), do: Enum.random(["cost_benefit", "risk_assessment", "impact_analysis", "feasibility"])
  defp generate_realistic_parameter_value(_), do: "default_value_#{Enum.random(100..999)}"

  defp simulate_function_execution(function_call, agent) do
    function_name = function_call.function
    parameters = function_call.parameters
    
    # Generate realistic results based on function type and agent personality
    base_result = case function_name do
      :analyze_data -> 
        %{
          patterns_found: Enum.random(3..15),
          anomalies_detected: Enum.random(0..5),
          confidence_score: :rand.uniform() * 0.4 + 0.6,
          recommendations: ["Optimize node distribution", "Improve connection efficiency"]
        }
      
      :calculate_metrics ->
        %{
          latency_avg: Enum.random(10..200),
          throughput: Enum.random(1000..10000),
          reliability_score: :rand.uniform() * 0.3 + 0.7,
          efficiency_rating: Enum.random(70..95)
        }
      
      :validate_consensus ->
        %{
          validation_status: Enum.random([:valid, :questionable, :invalid]),
          consensus_strength: :rand.uniform(),
          participant_agreement: Enum.random(60..99),
          time_to_consensus: Enum.random(100..5000)
        }
      
      :generate_solutions ->
        %{
          solutions_generated: Enum.random(2..8),
          feasibility_scores: Enum.map(1..3, fn _ -> :rand.uniform() end),
          innovation_index: :rand.uniform() * 0.6 + 0.4,
          implementation_complexity: Enum.random([:low, :medium, :high])
        }
      
      :monitor_security ->
        %{
          threats_detected: Enum.random(0..3),
          security_level: Enum.random([:low, :medium, :high, :critical]),
          mitigation_actions: Enum.random(0..5),
          confidence: :rand.uniform() * 0.3 + 0.7
        }
      
      :discover_nodes ->
        %{
          nodes_discovered: Enum.random(5..50),
          new_connections_possible: Enum.random(2..20),
          network_expansion_potential: :rand.uniform(),
          geographic_diversity: Enum.random(1..6)
        }
      
      _ ->
        %{
          status: :success,
          execution_time: function_call.execution_time,
          data_processed: Enum.random(100..10000),
          efficiency: :rand.uniform() * 0.4 + 0.6
        }
    end

    # Add personality-influenced insights
    personality_insight = case agent.personality do
      "analytical" -> "Data shows clear optimization opportunities with #{Float.round(:rand.uniform() * 25 + 5, 1)}% improvement potential"
      "creative" -> "Discovered #{Enum.random(2..7)} innovative approaches that could revolutionize current methods"
      "social" -> "Identified #{Enum.random(3..12)} collaboration opportunities to enhance network cohesion"
      "guardian" -> "Detected #{Enum.random(0..4)} security concerns requiring attention and #{Enum.random(2..8)} protective measures"
      "explorer" -> "Found #{Enum.random(5..25)} unexplored territories with significant discovery potential"
      _ -> "Function executed successfully with standard results"
    end

    %{
      summary: personality_insight,
      detailed_results: base_result,
      agent_personality: agent.personality,
      execution_context: %{
        mood: agent.context.current_mood,
        parameters_used: parameters,
        success_rate: :rand.uniform() * 0.3 + 0.7
      }
    }
  end

  defp update_agent_with_function_result(agent, function_call) do
    new_history_entry = %{
      function: function_call.function,
      result_summary: function_call.result.summary,
      timestamp: function_call.completed_at
    }

    updated_history = [new_history_entry | agent.context.function_call_history]
      |> Enum.take(10)  # Keep only last 10 calls

    %{agent | 
      context: %{agent.context | 
        function_call_history: updated_history
      }
    }
  end

  defp calculate_initial_stats(nodes, connections) do
    %{
      total_nodes: map_size(nodes),
      total_connections: map_size(connections),
      node_types: count_node_types(nodes),
      personality_distribution: count_personalities(nodes),
      average_connections_per_node: if(map_size(nodes) > 0, do: map_size(connections) / map_size(nodes), else: 0),
      network_density: calculate_network_density(nodes, connections)
    }
  end

  defp count_node_types(nodes) do
    nodes
    |> Enum.group_by(fn {_id, node} -> node.type end)
    |> Enum.map(fn {type, nodes} -> {type, length(nodes)} end)
    |> Map.new()
  end

  defp count_personalities(nodes) do
    nodes
    |> Enum.group_by(fn {_id, node} -> node.personality.name end)
    |> Enum.map(fn {personality, nodes} -> {personality, length(nodes)} end)
    |> Map.new()
  end

  defp calculate_network_density(nodes, connections) do
    n = map_size(nodes)
    if n > 1 do
      max_connections = n * (n - 1)
      actual_connections = map_size(connections)
      actual_connections / max_connections
    else
      0.0
    end
  end

  defp log_network_status(state) do
    stats = %{
      nodes: map_size(state.nodes),
      connections: map_size(state.connections),
      active_chats: map_size(state.active_chats),
      density: calculate_network_density(state.nodes, state.connections)
    }

    personality_dist = count_personalities(state.nodes)
    
    # Sample some recent activity
    recent_activity = sample_recent_activity(state)

    Logger.info("""
    📊 NETWORK STATUS:
    • Nodes: #{stats.nodes} | Connections: #{stats.connections} | Chats: #{stats.active_chats}
    • Density: #{Float.round(stats.density * 100, 2)}%
    • Personalities: #{inspect(personality_dist)}
    • Recent Activity: #{recent_activity}
    """)
  end

  defp sample_recent_activity(state) do
    activities = []

    # Sample recent chats
    activities = if map_size(state.active_chats) > 0 do
      chat_sample = state.active_chats |> Enum.take(2) |> Enum.map(fn {_id, chat} -> 
        "Chat: '#{chat.topic}' (#{length(chat.participants)} participants)"
      end)
      activities ++ chat_sample
    else
      activities
    end

    # Sample node personalities
    activities = if map_size(state.nodes) > 0 do
      node_sample = state.nodes |> Enum.take(3) |> Enum.map(fn {id, node} ->
        "#{String.slice(id, 0, 8)}... (#{node.personality.name})"
      end)
      activities ++ node_sample
    else
      activities
    end

    if length(activities) > 0 do
      Enum.join(activities, ", ")
    else
      "Network initializing..."
    end
  end

  # Scheduling functions
  defp schedule_network_events do
    interval = Enum.random(@connection_interval)
    Process.send_after(self(), :network_event, interval)
  end

  defp schedule_chat_events do
    interval = Enum.random(@chat_interval)
    Process.send_after(self(), :chat_event, interval)
  end

  defp schedule_function_calls do
    interval = Enum.random(@function_call_interval)
    Process.send_after(self(), :function_call_event, interval)
  end

  defp schedule_monitoring do
    Process.send_after(self(), :monitoring, 5000)
  end

  # Public API
  def get_network_status do
    GenServer.call(__MODULE__, :get_status)
  end

  def get_node_details(node_id) do
    GenServer.call(__MODULE__, {:get_node, node_id})
  end

  def get_active_chats do
    GenServer.call(__MODULE__, :get_chats)
  end

  def force_network_event(action) do
    GenServer.cast(__MODULE__, {:force_event, action})
  end

  # Handle calls
  def handle_call(:get_status, _from, state) do
    status = %{
      total_nodes: map_size(state.nodes),
      total_connections: map_size(state.connections),
      active_chats: map_size(state.active_chats),
      network_density: calculate_network_density(state.nodes, state.connections),
      node_types: count_node_types(state.nodes),
      personalities: count_personalities(state.nodes)
    }
    {:reply, status, state}
  end

  def handle_call({:get_node, node_id}, _from, state) do
    node = Map.get(state.nodes, node_id)
    {:reply, node, state}
  end

  def handle_call(:get_chats, _from, state) do
    {:reply, state.active_chats, state}
  end

  def handle_cast({:force_event, action}, state) do
    new_state = perform_network_action(action, state)
    {:noreply, new_state}
  end
end

# Start the simulation
case MassiveNodeSimulation.start_link() do
  {:ok, _pid} ->
    IO.puts("\n🎯 MASSIVE NODE SIMULATION STARTED!")
    IO.puts("🔥 Simulating 1500-3500 dynamic nodes with LLM agents and function calling")
    IO.puts("💬 Real-time chat system active")
    IO.puts("⚡ Advanced function calling with personality-driven responses")
    IO.puts("📊 Network monitoring every 5 seconds")
    IO.puts("\nPress Ctrl+C to stop the simulation\n")
    
    # Keep the simulation running
    Process.sleep(:infinity)
  
  {:error, reason} ->
    IO.puts("❌ Failed to start simulation: #{inspect(reason)}")
end