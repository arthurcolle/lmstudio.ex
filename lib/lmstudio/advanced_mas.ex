defmodule LMStudio.AdvancedMAS do
  @moduledoc """
  Advanced Multi-Agent System with collaborative learning,
  genetic evolution, and emergent behaviors.
  """
  
  use GenServer
  require Logger
  
  alias LMStudio.CognitiveAgent
  # Removed unused aliases - SelfModifyingGrid, Mutation
  
  @type agent_id :: String.t()
  @type agent_role :: :explorer | :analyzer | :synthesizer | :optimizer | :coordinator
  
  @type agent_spec :: %{
    id: agent_id(),
    role: agent_role(),
    pid: pid(),
    genome: genome(),
    performance: float(),
    connections: [agent_id()]
  }
  
  @type genome :: %{
    traits: map(),
    strategies: map(),
    fitness: float()
  }
  
  @type knowledge_atom :: %{
    content: String.t(),
    source: agent_id(),
    confidence: float(),
    timestamp: DateTime.t()
  }
  
  @type state :: %{
    agents: %{agent_id() => agent_spec()},
    knowledge_base: [knowledge_atom()],
    evolution_params: map(),
    interaction_log: list(),
    emergent_patterns: map(),
    generation: integer()
  }
  
  # Evolution parameters
  @default_evolution_params %{
    mutation_rate: 0.3,
    crossover_rate: 0.7,
    selection_pressure: 0.8,
    knowledge_sharing_rate: 0.5,
    emergence_threshold: 0.75
  }
  
  # Agent role definitions with enhanced capabilities
  @role_definitions %{
    explorer: %{
      traits: %{
        curiosity: 0.9,
        risk_tolerance: 0.8,
        pattern_seeking: 0.6,
        collaboration: 0.5
      },
      strategies: %{
        search: "breadth_first",
        learning: "trial_and_error",
        mutation: "aggressive"
      }
    },
    analyzer: %{
      traits: %{
        curiosity: 0.5,
        risk_tolerance: 0.3,
        pattern_seeking: 0.95,
        collaboration: 0.7
      },
      strategies: %{
        search: "depth_first",
        learning: "analytical",
        mutation: "conservative"
      }
    },
    synthesizer: %{
      traits: %{
        curiosity: 0.7,
        risk_tolerance: 0.5,
        pattern_seeking: 0.8,
        collaboration: 0.9
      },
      strategies: %{
        search: "hybrid",
        learning: "integrative",
        mutation: "balanced"
      }
    },
    optimizer: %{
      traits: %{
        curiosity: 0.4,
        risk_tolerance: 0.2,
        pattern_seeking: 0.7,
        collaboration: 0.6
      },
      strategies: %{
        search: "greedy",
        learning: "refinement",
        mutation: "targeted"
      }
    },
    coordinator: %{
      traits: %{
        curiosity: 0.6,
        risk_tolerance: 0.4,
        pattern_seeking: 0.85,
        collaboration: 0.95
      },
      strategies: %{
        search: "meta_analysis",
        learning: "distributed",
        mutation: "coordinated"
      }
    }
  }
  
  # Client API
  
  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end
  
  def create_swarm(size, opts \\ []) do
    GenServer.call(__MODULE__, {:create_swarm, size, opts})
  end
  
  def evolve_generation do
    GenServer.call(__MODULE__, :evolve_generation, 60_000)
  end
  
  def share_knowledge(from_agent, to_agent, knowledge) do
    GenServer.cast(__MODULE__, {:share_knowledge, from_agent, to_agent, knowledge})
  end
  
  def detect_emergent_patterns do
    GenServer.call(__MODULE__, :detect_emergent_patterns)
  end
  
  def get_system_state do
    GenServer.call(__MODULE__, :get_state)
  end
  
  def visualize_network do
    GenServer.call(__MODULE__, :visualize_network)
  end
  
  # Server callbacks
  
  @impl true
  def init(opts) do
    evolution_params = Map.merge(@default_evolution_params, Map.new(opts[:evolution_params] || []))
    
    state = %{
      agents: %{},
      knowledge_base: [],
      evolution_params: evolution_params,
      interaction_log: [],
      emergent_patterns: %{},
      generation: 0
    }
    
    {:ok, state}
  end
  
  @impl true
  def handle_call({:create_swarm, size, opts}, _from, state) do
    roles = opts[:roles] || [:explorer, :analyzer, :synthesizer, :optimizer, :coordinator]
    
    # Create agents with diverse roles
    agents = for i <- 1..size do
      role = Enum.at(roles, rem(i - 1, length(roles)))
      agent_id = "agent_#{role}_#{i}"
      
      # Start cognitive agent
      {:ok, pid} = CognitiveAgent.start_link([
        name: agent_id,
        thinking_enabled: opts[:thinking_enabled] || true
      ])
      
      # Create genome based on role
      genome = create_genome(role)
      
      # Initialize connections (small-world network)
      connections = create_connections(i, size, opts[:connectivity] || 0.3)
      
      {agent_id, %{
        id: agent_id,
        role: role,
        pid: pid,
        genome: genome,
        performance: 0.5,
        connections: connections
      }}
    end
    |> Map.new()
    
    new_state = %{state | agents: agents}
    
    # Initialize agent grids with role-specific data
    initialize_agent_grids(agents)
    
    {:reply, {:ok, map_size(agents)}, new_state}
  end
  
  @impl true
  def handle_call(:evolve_generation, _from, state) do
    Logger.info("Starting evolution generation #{state.generation + 1}")
    
    # 1. Evaluate agent fitness
    evaluated_agents = evaluate_fitness(state.agents, state.knowledge_base)
    
    # 2. Selection
    selected_agents = tournament_selection(evaluated_agents, state.evolution_params)
    
    # 3. Crossover and mutation
    new_agents = create_next_generation(selected_agents, state.evolution_params)
    
    # 4. Knowledge transfer
    new_agents = transfer_knowledge(new_agents, state.knowledge_base, state.evolution_params)
    
    # 5. Detect emergent patterns
    patterns = detect_patterns(new_agents, state.interaction_log)
    
    new_state = %{state | 
      agents: new_agents,
      generation: state.generation + 1,
      emergent_patterns: patterns
    }
    
    {:reply, {:ok, state.generation + 1}, new_state}
  end
  
  @impl true
  def handle_call(:detect_emergent_patterns, _from, state) do
    patterns = analyze_emergence(state)
    {:reply, patterns, state}
  end
  
  @impl true
  def handle_call(:get_state, _from, state) do
    summary = %{
      generation: state.generation,
      agent_count: map_size(state.agents),
      knowledge_atoms: length(state.knowledge_base),
      emergent_patterns: Map.keys(state.emergent_patterns),
      avg_fitness: calculate_average_fitness(state.agents)
    }
    {:reply, summary, state}
  end
  
  @impl true
  def handle_call(:visualize_network, _from, state) do
    visualization = generate_network_visualization(state)
    {:reply, visualization, state}
  end
  
  @impl true
  def handle_cast({:share_knowledge, from_id, to_id, knowledge}, state) do
    with %{} = from_agent <- Map.get(state.agents, from_id),
         %{} = to_agent <- Map.get(state.agents, to_id) do
      
      # Create knowledge atom
      knowledge_atom = %{
        content: knowledge,
        source: from_id,
        confidence: from_agent.performance,
        timestamp: DateTime.utc_now()
      }
      
      # Add to knowledge base
      new_knowledge_base = [knowledge_atom | state.knowledge_base]
      
      # Log interaction
      interaction = %{
        from: from_id,
        to: to_id,
        type: :knowledge_share,
        timestamp: DateTime.utc_now()
      }
      
      new_state = %{state |
        knowledge_base: new_knowledge_base,
        interaction_log: [interaction | state.interaction_log]
      }
      
      # Apply knowledge to target agent
      apply_knowledge_to_agent(to_agent, knowledge_atom)
      
      {:noreply, new_state}
    else
      _ ->
        Logger.warning("Invalid agent IDs for knowledge sharing: #{from_id} -> #{to_id}")
        {:noreply, state}
    end
  end
  
  # Private functions
  
  defp create_genome(role) do
    base_definition = Map.get(@role_definitions, role)
    
    # Add random variations
    traits = Map.new(base_definition.traits, fn {k, v} ->
      # Add up to 10% variation
      variation = (0.9 + :rand.uniform() * 0.2)
      {k, min(1.0, v * variation)}
    end)
    
    %{
      traits: traits,
      strategies: base_definition.strategies,
      fitness: 0.5
    }
  end
  
  defp create_connections(agent_index, total_agents, connectivity) do
    # Create small-world network connections
    _connections = []
    
    # Connect to neighbors
    neighbors = [
      agent_index - 1,
      agent_index + 1
    ]
    |> Enum.filter(& &1 > 0 && &1 <= total_agents)
    
    # Add random long-range connections
    num_random = round(total_agents * connectivity)
    random_connections = 
      1..total_agents
      |> Enum.to_list()
      |> Enum.reject(& &1 == agent_index)
      |> Enum.take_random(num_random)
    
    (neighbors ++ random_connections)
    |> Enum.uniq()
    |> Enum.map(& "agent_#{Enum.random([:explorer, :analyzer, :synthesizer, :optimizer, :coordinator])}_#{&1}")
  end
  
  defp initialize_agent_grids(agents) do
    Enum.each(agents, fn {_id, agent} ->
      # Get current grid data
      _grid_data = CognitiveAgent.get_grid_data(agent.id)
      
      # Enhance with role-specific data
      _role_data = %{
        "role" => to_string(agent.role),
        "genome" => inspect(agent.genome),
        "connections" => Enum.join(agent.connections, ", ")
      }
      
      # Apply mutations based on genome
      apply_genome_mutations(agent)
    end)
  end
  
  defp apply_genome_mutations(agent) do
    # Convert genome traits to mutations
    _mutations = Enum.flat_map(agent.genome.traits, fn {trait, value} ->
      case trait do
        :curiosity when value > 0.7 ->
          [%{type: :append, target: "strategy", content: "\nActively explore unknown territories"}]
        :pattern_seeking when value > 0.8 ->
          [%{type: :evolve, target: "knowledge", content: "pattern recognition"}]
        :collaboration when value > 0.8 ->
          [%{type: :append, target: "strategy", content: "\nShare insights with other agents"}]
        _ ->
          []
      end
    end)
    
    # Apply mutations would go through agent's API
    # This is a simplified representation
  end
  
  defp evaluate_fitness(agents, knowledge_base) do
    Map.new(agents, fn {id, agent} ->
      # Calculate fitness based on multiple factors
      knowledge_contribution = count_knowledge_contributions(id, knowledge_base)
      connection_quality = evaluate_connections(agent, agents)
      performance = agent.performance
      
      fitness = 
        knowledge_contribution * 0.4 +
        connection_quality * 0.3 +
        performance * 0.3
      
      updated_genome = %{agent.genome | fitness: fitness}
      {id, %{agent | genome: updated_genome}}
    end)
  end
  
  defp count_knowledge_contributions(agent_id, knowledge_base) do
    contributions = Enum.count(knowledge_base, & &1.source == agent_id)
    # Normalize to 0-1 range
    min(1.0, contributions / 10.0)
  end
  
  defp evaluate_connections(agent, all_agents) do
    connected_agents = Enum.filter(all_agents, fn {id, _} ->
      id in agent.connections
    end)
    
    if length(connected_agents) == 0 do
      0.0
    else
      # Average fitness of connected agents
      total_fitness = Enum.reduce(connected_agents, 0, fn {_, a}, acc ->
        acc + a.genome.fitness
      end)
      total_fitness / length(connected_agents)
    end
  end
  
  defp tournament_selection(agents, _params) do
    agent_list = Map.values(agents)
    tournament_size = 3
    
    selected = for _ <- 1..map_size(agents) do
      tournament = Enum.take_random(agent_list, tournament_size)
      Enum.max_by(tournament, & &1.genome.fitness)
    end
    
    selected
  end
  
  defp create_next_generation(selected_agents, params) do
    # Group by role to maintain diversity
    by_role = Enum.group_by(selected_agents, & &1.role)
    
    new_agents = Enum.flat_map(by_role, fn {_role, role_agents} ->
      # Crossover within role
      pairs = Enum.chunk_every(role_agents, 2, 2, :discard)
      
      Enum.flat_map(pairs, fn [parent1, parent2] ->
        if :rand.uniform() < params.crossover_rate do
          [
            crossover_agents(parent1, parent2, params),
            crossover_agents(parent2, parent1, params)
          ]
        else
          [parent1, parent2]
        end
      end)
    end)
    
    # Convert back to map
    Map.new(new_agents, & {&1.id, &1})
  end
  
  defp crossover_agents(parent1, parent2, params) do
    # Crossover genomes
    new_traits = Map.new(parent1.genome.traits, fn {trait, value1} ->
      value2 = Map.get(parent2.genome.traits, trait, value1)
      new_value = if :rand.uniform() > 0.5, do: value1, else: value2
      
      # Apply mutation
      if :rand.uniform() < params.mutation_rate do
        mutated = new_value + (:rand.normal() * 0.1)
        {trait, max(0.0, min(1.0, mutated))}
      else
        {trait, new_value}
      end
    end)
    
    # Create new agent
    new_id = "agent_#{parent1.role}_#{:rand.uniform(10000)}"
    
    %{parent1 |
      id: new_id,
      genome: %{parent1.genome | traits: new_traits},
      performance: (parent1.performance + parent2.performance) / 2
    }
  end
  
  defp transfer_knowledge(agents, knowledge_base, params) do
    # Select high-value knowledge atoms
    valuable_knowledge = knowledge_base
    |> Enum.filter(& &1.confidence > 0.7)
    |> Enum.take(10)
    
    # Transfer to random agents
    Map.new(agents, fn {id, agent} ->
      updated_agent = if :rand.uniform() < params.knowledge_sharing_rate do
        # Apply random knowledge atom
        if not Enum.empty?(valuable_knowledge) do
          knowledge = Enum.random(valuable_knowledge)
          apply_knowledge_to_agent(agent, knowledge)
        else
          agent
        end
      else
        agent
      end
      
      {id, updated_agent}
    end)
  end
  
  defp apply_knowledge_to_agent(agent, knowledge_atom) do
    # This would integrate with the agent's grid
    # Simplified for demonstration
    Logger.debug("Applying knowledge '#{knowledge_atom.content}' to agent #{agent.id}")
  end
  
  defp detect_patterns(agents, interaction_log) do
    # Analyze interaction patterns
    interaction_frequency = analyze_interaction_frequency(interaction_log)
    
    # Detect clustering
    clusters = detect_agent_clusters(agents)
    
    # Identify emergent strategies
    emergent_strategies = identify_emergent_strategies(agents)
    
    %{
      interaction_patterns: interaction_frequency,
      agent_clusters: clusters,
      emergent_strategies: emergent_strategies,
      timestamp: DateTime.utc_now()
    }
  end
  
  defp analyze_interaction_frequency(log) do
    log
    |> Enum.take(100)  # Recent interactions
    |> Enum.group_by(& &1.type)
    |> Map.new(fn {type, interactions} ->
      {type, length(interactions)}
    end)
  end
  
  defp detect_agent_clusters(agents) do
    # Simple clustering based on trait similarity
    agent_list = Map.values(agents)
    
    clusters = Enum.group_by(agent_list, fn agent ->
      # Cluster by dominant trait
      {trait, _value} = Enum.max_by(agent.genome.traits, fn {_k, v} -> v end)
      trait
    end)
    
    Map.new(clusters, fn {trait, cluster_agents} ->
      {trait, length(cluster_agents)}
    end)
  end
  
  defp identify_emergent_strategies(agents) do
    # Look for common strategy patterns
    strategies = agents
    |> Map.values()
    |> Enum.flat_map(& Map.values(&1.genome.strategies))
    |> Enum.frequencies()
    |> Enum.filter(fn {_, count} -> count > map_size(agents) * 0.3 end)
    |> Map.new()
    
    strategies
  end
  
  defp analyze_emergence(state) do
    %{
      generation: state.generation,
      patterns: state.emergent_patterns,
      collective_intelligence: measure_collective_intelligence(state),
      diversity_index: calculate_diversity(state.agents),
      convergence_rate: calculate_convergence(state)
    }
  end
  
  defp measure_collective_intelligence(state) do
    # Measure based on knowledge base growth and quality
    knowledge_quality = state.knowledge_base
    |> Enum.map(& &1.confidence)
    |> Enum.sum()
    
    knowledge_quality / max(1, length(state.knowledge_base))
  end
  
  defp calculate_diversity(agents) do
    # Shannon diversity index on traits
    all_traits = agents
    |> Map.values()
    |> Enum.flat_map(& Map.to_list(&1.genome.traits))
    
    if length(all_traits) == 0 do
      0.0
    else
      trait_frequencies = Enum.frequencies_by(all_traits, fn {trait, _} -> trait end)
      total = map_size(agents)
      
      trait_frequencies
      |> Map.values()
      |> Enum.map(fn count ->
        p = count / total
        -p * :math.log(p)
      end)
      |> Enum.sum()
    end
  end
  
  defp calculate_convergence(state) do
    # Measure how similar agents are becoming
    if map_size(state.agents) < 2 do
      0.0
    else
      fitness_values = state.agents
      |> Map.values()
      |> Enum.map(& &1.genome.fitness)
      
      mean = Enum.sum(fitness_values) / length(fitness_values)
      variance = fitness_values
      |> Enum.map(fn f -> :math.pow(f - mean, 2) end)
      |> Enum.sum()
      |> Kernel./(length(fitness_values))
      
      # Lower variance means higher convergence
      1 - min(1.0, variance)
    end
  end
  
  defp calculate_average_fitness(agents) do
    if map_size(agents) == 0 do
      0.0
    else
      total = agents
      |> Map.values()
      |> Enum.map(& &1.genome.fitness)
      |> Enum.sum()
      
      total / map_size(agents)
    end
  end
  
  defp generate_network_visualization(state) do
    nodes = Map.values(state.agents) |> Enum.map(fn agent ->
      %{
        id: agent.id,
        role: agent.role,
        fitness: agent.genome.fitness,
        connections: agent.connections
      }
    end)
    
    edges = Enum.flat_map(state.agents, fn {id, agent} ->
      Enum.map(agent.connections, fn conn_id ->
        %{from: id, to: conn_id}
      end)
    end)
    
    %{
      nodes: nodes,
      edges: edges,
      metrics: %{
        total_nodes: length(nodes),
        total_edges: length(edges),
        avg_degree: if(length(nodes) > 0, do: length(edges) * 2 / length(nodes), else: 0)
      }
    }
  end
end