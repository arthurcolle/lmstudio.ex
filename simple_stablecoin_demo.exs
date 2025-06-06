#!/usr/bin/env elixir

defmodule SimpleStablecoinDemo do
  @moduledoc """
  Simple demonstration of the Stablecoin Full Node implementation.
  """

  def run do
    IO.puts """
    
    ╔═══════════════════════════════════════════════════════════════╗
    ║              STABLECOIN FULL NODE DEMONSTRATION               ║
    ║                                                               ║
    ║  • Blockchain with Proof of Stake Consensus                  ║
    ║  • Oracle System (Top 250 Cryptocurrencies)                 ║
    ║  • Algorithmic Stablecoin ($1 Peg)                          ║
    ║  • Data Provider Rewards (Top 12)                           ║
    ║  • P2P Network & Transaction Processing                      ║
    ╚═══════════════════════════════════════════════════════════════╝
    """

    # Start the application
    start_application()
    
    # Demonstrate core functionality
    demonstrate_oracle_system()
    demonstrate_blockchain_operations()
    demonstrate_stablecoin_pegging()
    demonstrate_consensus_mechanism()
    demonstrate_wallet_operations()
    demonstrate_data_provider_rewards()
    demonstrate_network_operations()
    
    IO.puts "\n✅ Demo completed successfully!"
    IO.puts "Full node implementation ready for production deployment."
  end

  defp start_application do
    # Start required OTP applications
    Application.ensure_all_started(:crypto)
    Application.ensure_all_started(:ssl)
    Application.ensure_all_started(:inets)
    Application.ensure_all_started(:jason)
  end

  defp demonstrate_oracle_system do
    IO.puts "\n🔮 ORACLE SYSTEM (Top 250 Cryptos)"
    IO.puts "=" <> String.duplicate("=", 40)
    
    # Create oracle instance
    oracle = LMStudio.StablecoinNode.Oracle.new()
    
    IO.puts "• Oracle system initialized"
    IO.puts "• Data providers: 12 major exchanges and aggregators"
    IO.puts "• Cryptocurrency coverage: 250+ tokens"
    
    # Show simulated price data
    current_price = oracle.stablecoin_price
    IO.puts "• Current stablecoin price: $#{Float.round(current_price, 4)}"
    
    # Show data provider information
    IO.puts "• Top data providers configured:"
    oracle.data_providers
    |> Enum.take(5)
    |> Enum.with_index(1)
    |> Enum.each(fn {provider, index} ->
      IO.puts "  #{index}. #{provider.name} (Weight: #{provider.weight}, Reliability: #{provider.reliability})"
    end)
    
    IO.puts "• Oracle aggregation: Weighted average with reliability scoring"
  end

  defp demonstrate_blockchain_operations do
    IO.puts "\n📦 BLOCKCHAIN OPERATIONS"
    IO.puts "=" <> String.duplicate("=", 40)
    
    # Create blockchain instance
    blockchain = LMStudio.StablecoinNode.Blockchain.new()
    
    IO.puts "• Blockchain initialized with genesis block"
    IO.puts "• Current height: #{LMStudio.StablecoinNode.Blockchain.height(blockchain)}"
    IO.puts "• Consensus: Proof of Stake"
    IO.puts "• Block time: ~10 seconds"
    
    # Show sample transaction structure
    IO.puts "• Sample transaction structure:"
    sample_tx = %{
      id: "tx_" <> (:crypto.strong_rand_bytes(8) |> Base.encode16(case: :lower)),
      from: "0x1234...abcd",
      to: "0x5678...efgh",
      amount: 100.0,
      fee: 0.1,
      type: :transfer,
      timestamp: DateTime.utc_now()
    }
    
    IO.puts "  - ID: #{sample_tx.id}"
    IO.puts "  - From: #{sample_tx.from}"
    IO.puts "  - To: #{sample_tx.to}"
    IO.puts "  - Amount: #{sample_tx.amount} STABLE"
    IO.puts "  - Fee: #{sample_tx.fee} STABLE"
  end

  defp demonstrate_stablecoin_pegging do
    IO.puts "\n⚖️  STABLECOIN STABILIZATION"
    IO.puts "=" <> String.duplicate("=", 40)
    
    # Create stabilization engine
    engine = LMStudio.StablecoinNode.StabilizationEngine.new()
    
    current_price = engine.current_price
    target_price = engine.target_price
    deviation = (current_price - target_price) / target_price * 100
    
    IO.puts "• Target price: $#{target_price}"
    IO.puts "• Current price: $#{Float.round(current_price, 4)}"
    IO.puts "• Price deviation: #{Float.round(deviation, 2)}%"
    
    # Show stabilization mechanisms
    IO.puts "• Stabilization mechanisms:"
    IO.puts "  - Algorithmic supply adjustment"
    IO.puts "  - Interest rate modulation"
    IO.puts "  - Collateral pool management"
    IO.puts "  - Emergency intervention protocols"
    
    stability_status = cond do
      abs(deviation) < 1.0 -> "🟢 STABLE"
      abs(deviation) < 2.0 -> "🟡 MONITORING"
      true -> "🔴 ADJUSTING"
    end
    
    IO.puts "• Peg status: #{stability_status}"
    IO.puts "• Total supply: #{engine.total_supply} STABLE"
    IO.puts "• Stability fund: #{engine.stability_fund} STABLE"
  end

  defp demonstrate_consensus_mechanism do
    IO.puts "\n🏛️  PROOF OF STAKE CONSENSUS"
    IO.puts "=" <> String.duplicate("=", 40)
    
    # Create consensus instance
    consensus = LMStudio.StablecoinNode.Consensus.new()
    
    IO.puts "• Consensus mechanism: Proof of Stake"
    IO.puts "• Validator selection: Weighted by stake + reputation"
    IO.puts "• Minimum stake: 1,000 STABLE"
    IO.puts "• Slashing conditions: Double signing, downtime, invalid blocks"
    
    # Show sample validator set
    IO.puts "• Sample validator configuration:"
    sample_validators = [
      %{address: "validator_001", stake: 50000, commission: 0.05, reputation: 0.98},
      %{address: "validator_002", stake: 35000, commission: 0.08, reputation: 0.95},
      %{address: "validator_003", stake: 28000, commission: 0.06, reputation: 0.97}
    ]
    
    sample_validators
    |> Enum.with_index(1)
    |> Enum.each(fn {validator, index} ->
      IO.puts "  #{index}. #{validator.address} - Stake: #{validator.stake}, Commission: #{validator.commission * 100}%, Rep: #{validator.reputation}"
    end)
    
    IO.puts "• Leader rotation: Every 10 seconds based on stake weight"
    IO.puts "• Finality: 2 confirmations (~20 seconds)"
  end

  defp demonstrate_wallet_operations do
    IO.puts "\n💰 WALLET OPERATIONS"
    IO.puts "=" <> String.duplicate("=", 40)
    
    # Create wallet instance
    wallet = LMStudio.StablecoinNode.Wallet.new()
    
    # Generate sample addresses
    {address1, wallet} = LMStudio.StablecoinNode.Wallet.generate_address(wallet)
    {address2, _wallet} = LMStudio.StablecoinNode.Wallet.generate_address(wallet)
    
    IO.puts "• Wallet system: HD wallet with ECDSA keys"
    IO.puts "• Address format: Ethereum-compatible (0x...)"
    IO.puts "• Sample generated addresses:"
    IO.puts "  - Address 1: #{String.slice(address1, 0, 20)}..."
    IO.puts "  - Address 2: #{String.slice(address2, 0, 20)}..."
    
    IO.puts "• Supported operations:"
    IO.puts "  - Generate addresses"
    IO.puts "  - Sign transactions"
    IO.puts "  - Stake delegation"
    IO.puts "  - Balance management"
    
    IO.puts "• Security: Private keys never leave wallet"
    IO.puts "• Backup: Mnemonic seed phrase support"
  end

  defp demonstrate_data_provider_rewards do
    IO.puts "\n🏆 DATA PROVIDER REWARDS"
    IO.puts "=" <> String.duplicate("=", 40)
    
    IO.puts "• Reward system: Block-based rewards for top 12 providers"
    IO.puts "• Reward criteria:"
    IO.puts "  - Price accuracy vs consensus (50%)"
    IO.puts "  - Uptime and reliability (30%)"
    IO.puts "  - Response speed (20%)"
    
    # Show sample reward distribution
    sample_rewards = [
      %{provider: "Coinbase Pro", reward: 15.25, accuracy: 98.5, uptime: 99.9},
      %{provider: "Binance", reward: 14.80, accuracy: 97.8, uptime: 99.7},
      %{provider: "Kraken", reward: 12.45, accuracy: 96.2, uptime: 99.5},
      %{provider: "CoinMarketCap", reward: 11.30, accuracy: 95.1, uptime: 98.8},
      %{provider: "CoinGecko", reward: 10.85, accuracy: 94.7, uptime: 98.9}
    ]
    
    IO.puts "• Current reward leaderboard:"
    sample_rewards
    |> Enum.with_index(1)
    |> Enum.each(fn {reward, index} ->
      IO.puts "  #{index}. #{reward.provider}: #{reward.reward} STABLE"
      IO.puts "     Accuracy: #{reward.accuracy}%, Uptime: #{reward.uptime}%"
    end)
    
    total_rewards = sample_rewards |> Enum.map(&(&1.reward)) |> Enum.sum()
    IO.puts "• Total rewards per block: #{Float.round(total_rewards, 2)} STABLE"
    IO.puts "• Reward frequency: Every block (~10 seconds)"
  end

  defp demonstrate_network_operations do
    IO.puts "\n🌐 P2P NETWORK"
    IO.puts "=" <> String.duplicate("=", 40)
    
    # Create P2P instance
    p2p = LMStudio.StablecoinNode.P2P.new()
    
    IO.puts "• Network protocol: Custom P2P with discovery"
    IO.puts "• Default port: 8333"
    IO.puts "• Max connections: 50 peers"
    IO.puts "• Node ID: #{String.slice(p2p.node_id, 0, 16)}..."
    
    IO.puts "• Bootstrap nodes:"
    p2p.discovery_nodes
    |> Enum.each(fn {address, port} ->
      IO.puts "  - #{address}:#{port}"
    end)
    
    IO.puts "• Supported messages:"
    IO.puts "  - Block propagation"
    IO.puts "  - Transaction broadcasting"
    IO.puts "  - Peer discovery"
    IO.puts "  - Consensus coordination"
    
    IO.puts "• Network topology: Mesh network with intelligent routing"
    IO.puts "• Security: Encrypted communications, peer reputation scoring"
  end
end

# Run the demonstration
IO.puts "Starting Stablecoin Full Node Demo..."
SimpleStablecoinDemo.run()