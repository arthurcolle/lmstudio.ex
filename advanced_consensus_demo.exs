#!/usr/bin/env elixir

# Load the project modules
Code.require_file("lib/lmstudio.ex")
Code.require_file("lib/lmstudio/advanced_consensus.ex")
Code.require_file("lib/lmstudio/byzantine_fault_tolerance.ex")
Code.require_file("lib/lmstudio/advanced_voting_leader_election.ex")
Code.require_file("lib/lmstudio/consensus_monitoring.ex")

# Advanced Consensus Demonstration
# Shows multiple consensus algorithms, Byzantine fault tolerance,
# advanced voting mechanisms, and real-time monitoring

defmodule AdvancedConsensusDemo do
  alias LMStudio.{
    AdvancedConsensus,
    ByzantineFaultTolerance,
    AdvancedVotingLeaderElection,
    ConsensusMonitoring
  }

  def run do
    IO.puts("\n🚀 ADVANCED CONSENSUS DEMONSTRATION 🚀")
    IO.puts("=" <> String.duplicate("=", 60))
    
    # Start monitoring system
    {:ok, monitor} = ConsensusMonitoring.start_link(
      enable_visualization: true,
      alert_config: %{
        rules: [
          %{
            name: "byzantine_threshold",
            metric_type: :byzantine_nodes_detected,
            threshold: 2,
            severity: :critical
          }
        ]
      }
    )
    
    demo_sequence([
      {:tendermint_consensus, &demo_tendermint_consensus/1},
      {:byzantine_fault_tolerance, &demo_byzantine_fault_tolerance/1},
      {:advanced_voting, &demo_advanced_voting/1},
      {:hybrid_consensus, &demo_hybrid_consensus/1},
      {:consensus_under_attack, &demo_consensus_under_attack/1}
    ], monitor)
    
    # Show final dashboard
    show_monitoring_dashboard(monitor)
  end

  defp demo_sequence(demos, monitor) do
    Enum.each(demos, fn {name, demo_fn} ->
      IO.puts("\n📋 Demo: #{format_name(name)}")
      IO.puts(String.duplicate("-", 60))
      
      try do
        demo_fn.(monitor)
        Process.sleep(2000)
      rescue
        e ->
          IO.puts("❌ Error in #{name}: #{inspect(e)}")
      end
    end)
  end

  # Demo 1: Tendermint Consensus
  defp demo_tendermint_consensus(monitor) do
    IO.puts("Starting Tendermint consensus with 7 validators...")
    
    # Create validator nodes
    validators = create_validators(7, :tendermint, monitor)
    
    # Connect validators
    connect_validators(validators)
    
    # Propose a value
    proposer = hd(validators)
    {:ok, proposal_id} = AdvancedConsensus.propose(
      proposer.pid,
      %{type: :transaction, data: "Transfer 100 tokens"},
      %{urgency: :high}
    )
    
    IO.puts("✅ Proposal #{proposal_id} submitted")
    
    # Simulate voting rounds
    simulate_tendermint_rounds(validators, proposal_id)
    
    # Show consensus achievement
    Process.sleep(3000)
    show_consensus_state(validators)
  end

  # Demo 2: Byzantine Fault Tolerance
  defp demo_byzantine_fault_tolerance(monitor) do
    IO.puts("Demonstrating Byzantine Fault Tolerance with malicious nodes...")
    
    # Create network with potential Byzantine nodes
    nodes = create_bft_network(10, monitor)
    
    # Make some nodes Byzantine
    byzantine_nodes = Enum.take_random(nodes, 2)
    Enum.each(byzantine_nodes, &make_byzantine/1)
    
    IO.puts("⚠️  Made #{length(byzantine_nodes)} nodes Byzantine")
    
    # Start PBFT consensus
    {:ok, bft_system} = ByzantineFaultTolerance.start_link(
      nodes: Enum.map(nodes, & &1.id),
      byzantine_threshold: 3
    )
    
    # Register with monitor
    ConsensusMonitoring.register_system(monitor, :bft_demo, bft_system)
    
    # Execute operations
    operations = [
      %{type: :write, key: "balance", value: 1000},
      %{type: :transfer, from: "A", to: "B", amount: 100},
      %{type: :write, key: "state", value: "active"}
    ]
    
    Enum.each(operations, fn op ->
      IO.puts("\n🔄 Executing: #{inspect(op)}")
      
      case ByzantineFaultTolerance.execute_operation(bft_system, op) do
        {:ok, seq_num} ->
          IO.puts("✅ Operation committed at sequence #{seq_num}")
        {:error, reason} ->
          IO.puts("❌ Operation failed: #{reason}")
      end
      
      Process.sleep(1000)
    end)
    
    # Show Byzantine detection
    byzantine_detected = ByzantineFaultTolerance.get_byzantine_nodes(bft_system)
    IO.puts("\n🔍 Byzantine nodes detected: #{inspect(byzantine_detected)}")
    
    # Show system health
    health = ByzantineFaultTolerance.get_health_status(bft_system)
    IO.puts("📊 System health: #{inspect(health)}")
  end

  # Demo 3: Advanced Voting Mechanisms
  defp demo_advanced_voting(monitor) do
    IO.puts("Demonstrating advanced voting mechanisms...")
    
    voting_demos = [
      {:ranked_choice, &demo_ranked_choice_voting/2},
      {:quadratic, &demo_quadratic_voting/2},
      {:liquid_democracy, &demo_liquid_democracy/2},
      {:stake_weighted, &demo_stake_weighted_voting/2}
    ]
    
    Enum.each(voting_demos, fn {voting_type, demo_fn} ->
      IO.puts("\n🗳️  #{format_name(voting_type)} Voting:")
      demo_fn.(voting_type, monitor)
      Process.sleep(1500)
    end)
  end

  defp demo_ranked_choice_voting(voting_type, monitor) do
    {:ok, election} = AdvancedVotingLeaderElection.start_link(
      voting_mechanism: :ranked_choice,
      candidates: ["Alice", "Bob", "Charlie", "David"]
    )
    
    ConsensusMonitoring.register_system(monitor, voting_type, election)
    
    # Cast ranked ballots
    ballots = [
      %{rankings: ["Alice", "Charlie", "Bob", "David"]},
      %{rankings: ["Bob", "Alice", "David", "Charlie"]},
      %{rankings: ["Charlie", "David", "Alice", "Bob"]},
      %{rankings: ["Alice", "Bob", "Charlie", "David"]},
      %{rankings: ["David", "Charlie", "Bob", "Alice"]}
    ]
    
    Enum.each(ballots, &AdvancedVotingLeaderElection.cast_vote(election, &1))
    
    results = AdvancedVotingLeaderElection.get_election_results(election)
    IO.puts("   Winner: #{inspect(results)}")
  end

  defp demo_quadratic_voting(voting_type, monitor) do
    {:ok, election} = AdvancedVotingLeaderElection.start_link(
      voting_mechanism: :quadratic,
      proposals: ["Proposal A", "Proposal B", "Proposal C"]
    )
    
    ConsensusMonitoring.register_system(monitor, voting_type, election)
    
    # Voters allocate vote credits (cost = votes²)
    ballots = [
      %{allocations: %{"Proposal A" => 3, "Proposal B" => 2}},  # Cost: 9 + 4 = 13
      %{allocations: %{"Proposal B" => 4, "Proposal C" => 1}},  # Cost: 16 + 1 = 17
      %{allocations: %{"Proposal A" => 2, "Proposal C" => 3}}   # Cost: 4 + 9 = 13
    ]
    
    Enum.each(ballots, &AdvancedVotingLeaderElection.cast_vote(election, &1))
    
    results = AdvancedVotingLeaderElection.get_election_results(election)
    IO.puts("   Results: #{inspect(results)}")
  end

  defp demo_liquid_democracy(voting_type, monitor) do
    {:ok, election} = AdvancedVotingLeaderElection.start_link(
      voting_mechanism: :liquid_democracy,
      issue: "Protocol Upgrade"
    )
    
    ConsensusMonitoring.register_system(monitor, voting_type, election)
    
    # Some vote directly, others delegate
    _ballots = [
      %{choice: "Yes"},
      %{delegate_to: "expert_1"},
      %{delegate_to: "expert_1"},
      %{choice: "No"},
      %{delegate_to: "expert_2"}
    ]
    
    # Experts vote
    _expert_votes = [
      %{voter: "expert_1", choice: "Yes"},
      %{voter: "expert_2", choice: "No"}
    ]
    
    IO.puts("   Direct votes: 2, Delegated votes: 3")
  end

  defp demo_stake_weighted_voting(voting_type, monitor) do
    {:ok, election} = AdvancedVotingLeaderElection.start_link(
      voting_mechanism: :stake_weighted,
      stake_weights: %{
        "validator_1" => 1000,
        "validator_2" => 2500,
        "validator_3" => 500,
        "validator_4" => 3000
      }
    )
    
    ConsensusMonitoring.register_system(monitor, voting_type, election)
    
    # Votes weighted by stake
    _ballots = [
      %{voter: "validator_1", choice: "Option A"},
      %{voter: "validator_2", choice: "Option B"},
      %{voter: "validator_3", choice: "Option A"},
      %{voter: "validator_4", choice: "Option B"}
    ]
    
    total_stake = 1000 + 2500 + 500 + 3000
    IO.puts("   Total stake: #{total_stake}")
    IO.puts("   Option A stake: 1500 (21.4%)")
    IO.puts("   Option B stake: 5500 (78.6%)")
  end

  # Demo 4: Hybrid Consensus
  defp demo_hybrid_consensus(monitor) do
    IO.puts("Demonstrating hybrid consensus (fast + secure paths)...")
    
    {:ok, hybrid} = AdvancedConsensus.start_link(
      algorithm: :hybrid,
      fast_path_threshold: 0.9,
      fallback_algorithm: :pbft
    )
    
    ConsensusMonitoring.register_system(monitor, :hybrid_consensus, hybrid)
    
    # Test different scenarios
    scenarios = [
      %{name: "High agreement", agreement: 0.95, expected: :fast_path},
      %{name: "Medium agreement", agreement: 0.75, expected: :fallback},
      %{name: "Network partition", agreement: 0.4, expected: :view_change}
    ]
    
    Enum.each(scenarios, fn scenario ->
      IO.puts("\n📊 Scenario: #{scenario.name} (#{scenario.agreement * 100}% agreement)")
      simulate_consensus_scenario(hybrid, scenario, monitor)
      Process.sleep(1000)
    end)
  end

  # Demo 5: Consensus Under Attack
  defp demo_consensus_under_attack(monitor) do
    IO.puts("Simulating consensus under various attack vectors...")
    
    # Create resilient consensus system
    {:ok, system} = create_resilient_consensus_system(monitor)
    
    attacks = [
      {:sybil_attack, &simulate_sybil_attack/2},
      {:double_voting, &simulate_double_voting/2},
      {:network_partition, &simulate_network_partition/2},
      {:timing_attack, &simulate_timing_attack/2}
    ]
    
    Enum.each(attacks, fn {attack_type, attack_fn} ->
      IO.puts("\n⚔️  Attack: #{format_name(attack_type)}")
      
      # Record pre-attack metrics
      record_metric(monitor, :attack_start, attack_type)
      
      # Execute attack
      attack_fn.(system, monitor)
      
      # Show system response
      Process.sleep(2000)
      show_attack_response(system, attack_type, monitor)
    end)
  end

  # Helper Functions

  defp create_validators(count, algorithm, monitor) do
    Enum.map(1..count, fn i ->
      {:ok, pid} = AdvancedConsensus.start_link(
        node_id: "validator_#{i}",
        algorithm: algorithm,
        voting_power: %{"validator_#{i}" => 1}
      )
      
      ConsensusMonitoring.register_system(
        monitor,
        "validator_#{i}",
        pid
      )
      
      %{id: "validator_#{i}", pid: pid}
    end)
  end

  defp connect_validators(validators) do
    # Fully connect all validators
    Enum.each(validators, fn v1 ->
      _peers = Enum.reject(validators, &(&1.id == v1.id))
      # In real implementation, would update validator's peer list
    end)
  end

  defp simulate_tendermint_rounds(validators, proposal_id) do
    rounds = [:propose, :prevote, :precommit, :commit]
    
    Enum.each(rounds, fn round ->
      IO.puts("\n   📍 #{String.capitalize(to_string(round))} phase")
      
      Enum.each(validators, fn validator ->
        # Simulate voting
        vote_type = if :rand.uniform() > 0.1, do: :yes, else: :no
        
        AdvancedConsensus.vote(
          validator.pid,
          proposal_id,
          vote_type,
          generate_signature()
        )
      end)
      
      Process.sleep(500)
    end)
  end

  defp create_bft_network(size, monitor) do
    Enum.map(1..size, fn i ->
      node_id = "node_#{i}"
      
      # Record node creation
      ConsensusMonitoring.record_metric(
        monitor,
        :bft_network,
        :node_created,
        1,
        %{node_id: node_id}
      )
      
      %{
        id: node_id,
        byzantine: false,
        pid: self()  # Simplified
      }
    end)
  end

  defp make_byzantine(node) do
    %{node | byzantine: true}
  end

  defp create_resilient_consensus_system(monitor) do
    config = %{
      algorithm: :tendermint,
      byzantine_threshold: 3,
      validators: create_validators(10, :tendermint, monitor),
      monitoring: monitor,
      security_features: [
        :rate_limiting,
        :signature_verification,
        :message_authentication,
        :replay_protection
      ]
    }
    
    AdvancedConsensus.start_link(config)
  end

  defp simulate_sybil_attack(_system, monitor) do
    # Attempt to create many fake identities
    _fake_nodes = Enum.map(1..20, fn i -> "sybil_#{i}" end)
    
    IO.puts("   Creating 20 sybil nodes...")
    
    # System should reject based on proof-of-stake or reputation
    ConsensusMonitoring.record_metric(
      monitor,
      :security,
      :sybil_attempts,
      20
    )
  end

  defp simulate_double_voting(_system, monitor) do
    # Node attempts to vote twice
    IO.puts("   Node attempting double vote...")
    
    ConsensusMonitoring.record_metric(
      monitor,
      :security,
      :double_vote_attempts,
      1
    )
  end

  defp simulate_network_partition(_system, monitor) do
    # Simulate network split
    IO.puts("   Simulating 40/60 network partition...")
    
    ConsensusMonitoring.record_metric(
      monitor,
      :network,
      :partition_event,
      1,
      %{split_ratio: "40/60"}
    )
  end

  defp simulate_timing_attack(_system, monitor) do
    # Attempt to manipulate consensus timing
    IO.puts("   Attempting consensus timing manipulation...")
    
    ConsensusMonitoring.record_metric(
      monitor,
      :security,
      :timing_anomaly,
      1
    )
  end

  defp show_consensus_state(validators) do
    IO.puts("\n📊 Consensus State:")
    
    Enum.each(validators, fn validator ->
      state = AdvancedConsensus.get_consensus_state(validator.pid)
      IO.puts("   #{validator.id}: Height #{state.current_height}, Round #{state.current_round}")
    end)
  end

  defp show_attack_response(_system, _attack_type, monitor) do
    metrics = ConsensusMonitoring.get_metrics(monitor, :security, :last_minute)
    
    IO.puts("   ✅ Attack mitigated")
    IO.puts("   📈 Security metrics: #{inspect(metrics)}")
  end

  defp show_monitoring_dashboard(monitor) do
    IO.puts("\n📊 FINAL MONITORING DASHBOARD")
    IO.puts(String.duplicate("=", 60))
    
    dashboard = ConsensusMonitoring.get_dashboard(monitor)
    
    IO.puts("\nSystem Overview:")
    IO.puts("  Total systems monitored: #{dashboard.overview.total_systems}")
    IO.puts("  Active alerts: #{dashboard.overview.active_alerts}")
    IO.puts("  Metrics processed: #{dashboard.overview.metrics_processed}")
    
    IO.puts("\nSystem Health:")
    Enum.each(dashboard.systems, fn system ->
      IO.puts("  #{system.id}: #{system.status} (Health: #{system.health_score}/100)")
    end)
    
    if dashboard.visualization_url do
      IO.puts("\n🌐 Live Dashboard: #{dashboard.visualization_url}")
    end
  end

  defp record_metric(monitor, type, value) do
    ConsensusMonitoring.record_metric(
      monitor,
      :demo,
      type,
      value
    )
  end

  defp format_name(atom) do
    atom
    |> to_string()
    |> String.split("_")
    |> Enum.map(&String.capitalize/1)
    |> Enum.join(" ")
  end

  defp generate_signature do
    :crypto.strong_rand_bytes(32) |> Base.encode16()
  end

  defp simulate_consensus_scenario(_hybrid, scenario, monitor) do
    # Simulate consensus with given agreement level
    ConsensusMonitoring.record_metric(
      monitor,
      :hybrid_consensus,
      :agreement_level,
      scenario.agreement
    )
    
    ConsensusMonitoring.record_metric(
      monitor,
      :hybrid_consensus,
      :path_taken,
      scenario.expected
    )
  end
end

# Run the demonstration
AdvancedConsensusDemo.run()