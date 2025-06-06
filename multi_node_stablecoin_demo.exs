#!/usr/bin/env elixir

Mix.install([
  {:jason, "~> 1.4"}
])

defmodule MultiNodeStablecoinDemo do
  @moduledoc """
  Demonstrates running multiple trusted stablecoin nodes locally that connect to each other.
  Creates a local blockchain network with 4 nodes acting as validators and data providers.
  """

  require Logger

  defmodule NodeRunner do
    use GenServer
    require Logger

    def start_link(config, name) do
      GenServer.start_link(__MODULE__, config, name: name)
    end

    def init(config) do
      Logger.info("Starting StablecoinNode #{config.id}...")

      state = %{
        node_id: config.id,
        port: config.port,
        mining: config.mining,
        data_provider: config.data_provider,
        bootstrap: config.bootstrap,
        peers: [],
        blockchain_height: 0,
        mempool_size: 0,
        stablecoin_price: 1.000 + :rand.uniform() * 0.01 - 0.005,
        connected_peers: [],
        last_block_time: DateTime.utc_now(),
        total_supply: 1_000_000.0,
        stability_reserves: 500_000.0,
        oracle_feeds: generate_initial_oracle_feeds(),
        started_at: DateTime.utc_now()
      }

      # Start P2P listener
      {:ok, listen_socket} = :gen_tcp.listen(config.port, [
        :binary, 
        packet: 4, 
        active: false, 
        reuseaddr: true
      ])
      
      Logger.info("Node #{config.id} listening on port #{config.port}")
      
      # Start accepting connections
      spawn(fn -> accept_loop(listen_socket, state) end)
      
      # Schedule periodic tasks
      :timer.send_interval(5000, self(), :mine_block)
      :timer.send_interval(3000, self(), :update_oracle_data)
      :timer.send_interval(10000, self(), :stability_check)
      :timer.send_interval(15000, self(), :peer_discovery)
      
      new_state = Map.put(state, :listen_socket, listen_socket)
      {:ok, new_state}
    end

    def handle_info(:mine_block, state) do
      if state.mining do
        new_height = state.blockchain_height + 1
        timestamp = DateTime.utc_now()
        
        # Simulate mining delay
        :timer.sleep(100 + :rand.uniform(200))
        
        Logger.info("🔨 Node #{state.node_id} mined block ##{new_height}")
        
        # Broadcast block to peers
        block = %{
          height: new_height,
          timestamp: timestamp,
          miner: state.node_id,
          transactions: :rand.uniform(10),
          oracle_data: state.oracle_feeds,
          stablecoin_price: state.stablecoin_price
        }
        
        broadcast_to_peers(state, {:new_block, block})
        
        new_state = %{state | 
          blockchain_height: new_height,
          last_block_time: timestamp,
          mempool_size: max(0, state.mempool_size - :rand.uniform(5))
        }
        {:noreply, new_state}
      else
        {:noreply, state}
      end
    end

    def handle_info(:update_oracle_data, state) do
      if state.data_provider do
        # Simulate price feed updates
        price_change = (:rand.uniform() - 0.5) * 0.002  # ±0.1% change
        new_price = max(0.95, min(1.05, state.stablecoin_price + price_change))
        
        new_feeds = update_oracle_feeds(state.oracle_feeds)
        
        # Broadcast price update
        price_update = %{
          provider: state.node_id,
          price: new_price,
          timestamp: DateTime.utc_now(),
          feeds: new_feeds
        }
        
        broadcast_to_peers(state, {:price_update, price_update})
        
        new_state = %{state | 
          stablecoin_price: new_price,
          oracle_feeds: new_feeds
        }
        {:noreply, new_state}
      else
        {:noreply, state}
      end
    end

    def handle_info(:stability_check, state) do
      deviation = abs(state.stablecoin_price - 1.0)
      
      if deviation > 0.01 do  # 1% deviation threshold
        action = if state.stablecoin_price > 1.0, do: "mint", else: "burn"
        amount = deviation * 10000
        
        Logger.info("💰 Node #{state.node_id} executing #{action} of #{Float.round(amount, 2)} tokens for stabilization")
        
        # Simulate stabilization mechanism
        new_supply = if action == "mint" do
          state.total_supply + amount
        else
          state.total_supply - amount
        end
        
        new_state = %{state | total_supply: new_supply}
        {:noreply, new_state}
      else
        {:noreply, state}
      end
    end

    def handle_info(:peer_discovery, state) do
      # Try to discover new peers
      if length(state.connected_peers) < 3 do
        # Discovery logic would go here
        Logger.debug("Node #{state.node_id} discovering peers...")
      end
      {:noreply, state}
    end

    def handle_info({:tcp, _socket, data}, state) do
      case :erlang.binary_to_term(data) do
        {:new_block, block} ->
          if block.height > state.blockchain_height do
            Logger.info("📦 Node #{state.node_id} received new block ##{block.height} from node #{block.miner}")
            new_state = %{state | 
              blockchain_height: block.height,
              last_block_time: block.timestamp
            }
            {:noreply, new_state}
          else
            {:noreply, state}
          end
          
        {:price_update, update} ->
          Logger.info("📊 Node #{state.node_id} received price update: $#{Float.round(update.price, 4)} from provider #{update.provider}")
          new_state = %{state | stablecoin_price: update.price}
          {:noreply, new_state}
          
        {:peer_handshake, peer_info} ->
          if not Enum.member?(state.connected_peers, peer_info.node_id) do
            Logger.info("🤝 Node #{state.node_id} connected to peer #{peer_info.node_id}")
            new_peers = [peer_info.node_id | state.connected_peers]
            new_state = %{state | connected_peers: new_peers}
            {:noreply, new_state}
          else
            {:noreply, state}
          end
          
        _ ->
          {:noreply, state}
      end
    end

    def handle_info({:tcp_closed, _socket}, state) do
      {:noreply, state}
    end

    def handle_call(:status, _from, state) do
      status = %{
        node_id: state.node_id,
        port: state.port,
        blockchain_height: state.blockchain_height,
        stablecoin_price: state.stablecoin_price,
        connected_peers: length(state.connected_peers),
        mining: state.mining,
        data_provider: state.data_provider,
        mempool_size: state.mempool_size,
        total_supply: state.total_supply,
        uptime: DateTime.diff(DateTime.utc_now(), state.started_at)
      }
      {:reply, status, state}
    end

    defp accept_loop(listen_socket, state) do
      case :gen_tcp.accept(listen_socket) do
        {:ok, client_socket} ->
          spawn(fn -> handle_peer_connection(client_socket, state) end)
          accept_loop(listen_socket, state)
        {:error, _reason} ->
          accept_loop(listen_socket, state)
      end
    end

    defp handle_peer_connection(socket, state) do
      # Send handshake
      handshake = {:peer_handshake, %{node_id: state.node_id, port: state.port}}
      :gen_tcp.send(socket, :erlang.term_to_binary(handshake))
      
      # Keep connection alive
      :gen_tcp.controlling_process(socket, self())
      :inet.setopts(socket, [active: true])
    end

    defp broadcast_to_peers(_state, message) do
      # In a real implementation, this would send to actual peer sockets
      # For demo purposes, we simulate broadcasting
      _encoded = :erlang.term_to_binary(message)
      # This is simplified - actual peer sockets would be used
    end

    defp generate_initial_oracle_feeds do
      [
        %{symbol: "BTC", price: 45000 + :rand.uniform(5000), volume: :rand.uniform(1000000)},
        %{symbol: "ETH", price: 3200 + :rand.uniform(800), volume: :rand.uniform(500000)},
        %{symbol: "ADA", price: 1.2 + :rand.uniform(), volume: :rand.uniform(100000)},
        %{symbol: "SOL", price: 180 + :rand.uniform(40), volume: :rand.uniform(200000)},
        %{symbol: "AVAX", price: 75 + :rand.uniform(25), volume: :rand.uniform(150000)}
      ]
    end

    defp update_oracle_feeds(feeds) do
      Enum.map(feeds, fn feed ->
        price_change = (:rand.uniform() - 0.5) * 0.05 * feed.price
        volume_change = (:rand.uniform() - 0.5) * 0.2 * feed.volume
        
        %{feed |
          price: max(feed.price * 0.9, feed.price + price_change),
          volume: max(0, feed.volume + volume_change)
        }
      end)
    end
  end

  def run do
    Logger.configure(level: :info)
    
    IO.puts """
    
    ╔═══════════════════════════════════════════════════════════════════════════════╗
    ║                    🏦 Multi-Node Stablecoin Network Demo 🏦                   ║
    ║                                                                               ║
    ║ Starting 4 trusted stablecoin nodes locally:                                 ║
    ║ • Node 1: Port 8333 (Bootstrap/Mining)                                       ║
    ║ • Node 2: Port 8334 (Validator/Data Provider)                                ║
    ║ • Node 3: Port 8335 (Validator/Data Provider)                                ║
    ║ • Node 4: Port 8336 (Mining/Data Provider)                                   ║
    ╚═══════════════════════════════════════════════════════════════════════════════╝
    
    """

    # Start 4 nodes with different configurations
    nodes = [
      %{id: 1, port: 8333, mining: true, data_provider: true, bootstrap: true},
      %{id: 2, port: 8334, mining: false, data_provider: true, bootstrap: false},
      %{id: 3, port: 8335, mining: false, data_provider: true, bootstrap: false},
      %{id: 4, port: 8336, mining: true, data_provider: true, bootstrap: false}
    ]

    # Start each node in its own process
    node_pids = start_nodes(nodes)
    
    # Wait for nodes to initialize
    :timer.sleep(2000)
    
    # Connect nodes to each other
    connect_nodes(nodes)
    
    # Wait for connections to establish
    :timer.sleep(3000)
    
    # Show network status
    show_network_status(node_pids)
    
    # Demonstrate transactions and consensus
    demonstrate_network_functionality(node_pids)
    
    # Keep the demo running
    IO.puts "\n🔄 Network is running... Press Ctrl+C to stop\n"
    
    # Monitor the network
    monitor_network(node_pids)
  end

  defp start_nodes(nodes) do
    Enum.map(nodes, fn node_config ->
      IO.puts "🚀 Starting Node #{node_config.id} on port #{node_config.port}..."
      
      # Start each node with unique name
      node_name = String.to_atom("stablecoin_node_#{node_config.id}")
      
      {:ok, pid} = NodeRunner.start_link(node_config, node_name)
      
      {node_config.id, pid, node_name}
    end)
  end

  defp connect_nodes(nodes) do
    IO.puts "🔗 Connecting nodes to each other..."
    
    # Connect each node to others
    for node1 <- nodes, node2 <- nodes, node1.id != node2.id do
      spawn(fn ->
        connect_peer(node1.port, node2.port)
      end)
    end
  end

  defp connect_peer(from_port, to_port) do
    case :gen_tcp.connect(~c"localhost", to_port, [:binary, packet: 4, active: true], 5000) do
      {:ok, socket} ->
        handshake = {:peer_handshake, %{node_id: from_port, port: from_port}}
        :gen_tcp.send(socket, :erlang.term_to_binary(handshake))
        Logger.debug("Connected #{from_port} -> #{to_port}")
      {:error, reason} ->
        Logger.debug("Failed to connect #{from_port} -> #{to_port}: #{inspect(reason)}")
    end
  end

  defp show_network_status(node_pids) do
    IO.puts """
    
    ╔═══════════════════════════════════════════════════════════════════════════════╗
    ║                            🌐 Network Status 🌐                              ║
    ╚═══════════════════════════════════════════════════════════════════════════════╝
    """
    
    Enum.each(node_pids, fn {node_id, _pid, node_name} ->
      try do
        status = GenServer.call(node_name, :status)
        
        IO.puts """
        
        📊 Node #{node_id} Status:
        ├── Port: #{status.port}
        ├── Blockchain Height: #{status.blockchain_height}
        ├── Stablecoin Price: $#{Float.round(status.stablecoin_price, 4)}
        ├── Connected Peers: #{status.connected_peers}
        ├── Mining: #{if status.mining, do: "✅", else: "❌"}
        ├── Data Provider: #{if status.data_provider, do: "✅", else: "❌"}
        ├── Mempool Size: #{status.mempool_size}
        ├── Total Supply: #{Float.round(status.total_supply, 0)}
        └── Uptime: #{status.uptime}s
        """
      rescue
        _ -> IO.puts "❌ Node #{node_id} not responding"
      end
    end)
  end

  defp demonstrate_network_functionality(_node_pids) do
    IO.puts """
    
    ╔═══════════════════════════════════════════════════════════════════════════════╗
    ║                      🎯 Network Functionality Demo 🎯                        ║
    ╚═══════════════════════════════════════════════════════════════════════════════╝
    """
    
    # Simulate some transactions
    IO.puts "💸 Simulating cross-node transactions..."
    
    transactions = [
      %{from: "wallet_1", to: "wallet_2", amount: 100.50, fee: 0.01},
      %{from: "wallet_3", to: "wallet_1", amount: 75.25, fee: 0.01},
      %{from: "wallet_2", to: "wallet_4", amount: 200.00, fee: 0.02}
    ]
    
    Enum.each(transactions, fn tx ->
      IO.puts "  📤 Transaction: #{tx.from} → #{tx.to} (#{tx.amount} coins, fee: #{tx.fee})"
      :timer.sleep(1000)
    end)
    
    IO.puts "\n🏗️  Mining transactions into blocks..."
    :timer.sleep(3000)
    
    IO.puts "\n📈 Oracle price feeds updating across network..."
    :timer.sleep(2000)
    
    IO.puts "\n⚖️  Stability mechanism maintaining peg..."
  end

  defp monitor_network(node_pids) do
    # Continuously monitor and display network statistics
    monitor_loop(node_pids, 0)
  end

  defp monitor_loop(node_pids, iteration) do
    :timer.sleep(30000)  # Wait 30 seconds
    
    if rem(iteration, 2) == 0 do
      IO.puts "\n" <> String.duplicate("═", 80)
      IO.puts "🔄 Network Status Update ##{div(iteration, 2) + 1} (#{DateTime.utc_now() |> DateTime.to_string()})"
      IO.puts String.duplicate("═", 80)
      
      show_network_status(node_pids)
      
      # Show aggregate network stats
      total_height = node_pids
      |> Enum.map(fn {_id, _pid, node_name} ->
        try do
          GenServer.call(node_name, :status).blockchain_height
        rescue
          _ -> 0
        end
      end)
      |> Enum.max()
      
      IO.puts """
      
      🌐 Network Summary:
      ├── Max Blockchain Height: #{total_height}
      ├── Active Nodes: #{length(node_pids)}
      ├── Network Consensus: #{if total_height > 0, do: "✅ Active", else: "⏳ Syncing"}
      └── Stability Status: ✅ Maintaining Peg
      """
    end
    
    monitor_loop(node_pids, iteration + 1)
  end
end

# Run the demo
MultiNodeStablecoinDemo.run()