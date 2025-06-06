defmodule LMStudio.StablecoinNode.P2P do
  @moduledoc """
  Peer-to-peer networking layer for the stablecoin blockchain network.
  Handles node discovery, block propagation, and transaction broadcasting.
  """

  use GenServer
  require Logger

  @default_port 8333
  @max_peers 50
  @peer_discovery_interval 30_000
  @heartbeat_interval 60_000

  defstruct [
    :node_id,
    :port,
    :peers,
    :listening_socket,
    :connected_peers,
    :peer_scores,
    :message_handlers,
    :discovery_nodes
  ]

  defmodule Peer do
    defstruct [
      :id,
      :address,
      :port,
      :socket,
      :connected_at,
      :last_seen,
      :version,
      :services,
      :height,
      :score
    ]
  end

  def new do
    %__MODULE__{
      node_id: generate_node_id(),
      port: @default_port,
      peers: %{},
      listening_socket: nil,
      connected_peers: %{},
      peer_scores: %{},
      message_handlers: %{},
      discovery_nodes: default_discovery_nodes()
    }
  end

  def start_link do
    GenServer.start_link(__MODULE__, [], name: __MODULE__)
  end

  def broadcast_block(p2p, block) do
    GenServer.cast(__MODULE__, {:broadcast_block, block})
  end

  def broadcast_transaction(p2p, transaction) do
    GenServer.cast(__MODULE__, {:broadcast_transaction, transaction})
  end

  def discover_peers(p2p) do
    GenServer.call(__MODULE__, :discover_peers)
  end

  def get_connected_peers(p2p) do
    GenServer.call(__MODULE__, :get_connected_peers)
  end

  def connect_to_peer(address, port) do
    GenServer.cast(__MODULE__, {:connect_to_peer, address, port})
  end

  def init(_) do
    state = new()
    
    # Try to find an available port starting from default
    {port, socket} = find_available_port(state.port)
    
    case socket do
      {:ok, listening_socket} ->
        Logger.info("P2P node listening on port #{port}")
        new_state = %{state | listening_socket: listening_socket, port: port}
        
        # Start accepting connections
        spawn(fn -> accept_connections(listening_socket) end)
        
        # Schedule periodic tasks
        :timer.send_interval(@peer_discovery_interval, self(), :discover_peers)
        :timer.send_interval(@heartbeat_interval, self(), :heartbeat)
        
        # Connect to bootstrap nodes
        Enum.each(state.discovery_nodes, fn {address, port} ->
          connect_to_peer(address, port)
        end)
        
        {:ok, new_state}
        
      {:error, reason} ->
        Logger.error("Failed to start P2P listener: #{inspect(reason)}")
        {:stop, reason}
    end
  end

  def handle_call(:discover_peers, _from, state) do
    new_peers = perform_peer_discovery(state)
    {:reply, Map.keys(new_peers), %{state | peers: new_peers}}
  end

  def handle_call(:get_connected_peers, _from, state) do
    connected = state.connected_peers
    |> Enum.map(fn {_id, peer} -> 
      %{
        id: peer.id,
        address: peer.address,
        port: peer.port,
        height: peer.height,
        connected_at: peer.connected_at,
        score: peer.score
      }
    end)
    {:reply, connected, state}
  end

  def handle_cast({:broadcast_block, block}, state) do
    message = %{
      type: :new_block,
      data: block,
      timestamp: DateTime.utc_now(),
      sender: state.node_id
    }
    
    broadcast_message(state, message)
    {:noreply, state}
  end

  def handle_cast({:broadcast_transaction, transaction}, state) do
    message = %{
      type: :new_transaction,
      data: transaction,
      timestamp: DateTime.utc_now(),
      sender: state.node_id
    }
    
    broadcast_message(state, message)
    {:noreply, state}
  end

  def handle_cast({:connect_to_peer, address, port}, state) do
    case connect_peer(address, port, state.node_id) do
      {:ok, peer} ->
        new_connected_peers = Map.put(state.connected_peers, peer.id, peer)
        Logger.info("Connected to peer #{peer.id} at #{address}:#{port}")
        {:noreply, %{state | connected_peers: new_connected_peers}}
        
      {:error, reason} ->
        Logger.warning("Failed to connect to peer #{address}:#{port} - #{inspect(reason)}")
        {:noreply, state}
    end
  end

  def handle_info(:discover_peers, state) do
    new_peers = perform_peer_discovery(state)
    {:noreply, %{state | peers: new_peers}}
  end

  def handle_info(:heartbeat, state) do
    new_state = send_heartbeats(state)
    {:noreply, new_state}
  end

  def handle_info({:tcp, socket, data}, state) do
    case decode_message(data) do
      {:ok, message} ->
        new_state = handle_peer_message(state, socket, message)
        {:noreply, new_state}
        
      {:error, reason} ->
        Logger.warning("Failed to decode message: #{inspect(reason)}")
        {:noreply, state}
    end
  end

  def handle_info({:tcp_closed, socket}, state) do
    new_state = handle_peer_disconnect(state, socket)
    {:noreply, new_state}
  end

  defp accept_connections(socket) do
    case :gen_tcp.accept(socket) do
      {:ok, client_socket} ->
        Logger.info("New peer connection accepted")
        spawn(fn -> handle_peer_connection(client_socket) end)
        accept_connections(socket)
        
      {:error, reason} ->
        Logger.error("Failed to accept connection: #{inspect(reason)}")
        accept_connections(socket)
    end
  end

  defp handle_peer_connection(socket) do
    case :gen_tcp.recv(socket, 0, 30_000) do
      {:ok, data} ->
        case decode_message(data) do
          {:ok, %{type: :handshake} = message} ->
            Logger.info("Received handshake from peer #{message.sender}")
            send_handshake_response(socket)
            peer_communication_loop(socket, message.sender)
            
          _ ->
            :gen_tcp.close(socket)
        end
        
      {:error, _reason} ->
        :gen_tcp.close(socket)
    end
  end

  defp peer_communication_loop(socket, peer_id) do
    case :gen_tcp.recv(socket, 0, 60_000) do
      {:ok, data} ->
        case decode_message(data) do
          {:ok, message} ->
            send(self(), {:tcp, socket, data})
            peer_communication_loop(socket, peer_id)
            
          {:error, _reason} ->
            :gen_tcp.close(socket)
        end
        
      {:error, _reason} ->
        :gen_tcp.close(socket)
    end
  end

  defp connect_peer(address, port, node_id) do
    case :gen_tcp.connect(String.to_charlist(address), port, [:binary, packet: 4, active: true], 10_000) do
      {:ok, socket} ->
        # Send handshake
        handshake = %{
          type: :handshake,
          sender: node_id,
          version: "1.0.0",
          services: [:full_node],
          height: 0,
          timestamp: DateTime.utc_now()
        }
        
        case send_message(socket, handshake) do
          :ok ->
            peer = %Peer{
              id: generate_peer_id(address, port),
              address: address,
              port: port,
              socket: socket,
              connected_at: DateTime.utc_now(),
              last_seen: DateTime.utc_now(),
              version: "unknown",
              services: [],
              height: 0,
              score: 1.0
            }
            {:ok, peer}
            
          {:error, reason} ->
            :gen_tcp.close(socket)
            {:error, reason}
        end
        
      {:error, reason} ->
        {:error, reason}
    end
  end

  defp perform_peer_discovery(state) do
    # Request peer lists from connected peers
    discovery_message = %{
      type: :get_peers,
      sender: state.node_id,
      timestamp: DateTime.utc_now()
    }
    
    broadcast_message(state, discovery_message)
    
    # Also try to connect to new random peers
    state.connected_peers
    |> Enum.take(5)  # Ask 5 random peers for their peer lists
    |> Enum.each(fn {_id, peer} ->
      send_message(peer.socket, discovery_message)
    end)
    
    state.peers
  end

  defp send_heartbeats(state) do
    heartbeat = %{
      type: :heartbeat,
      sender: state.node_id,
      timestamp: DateTime.utc_now(),
      height: 0  # Would be actual blockchain height
    }
    
    # Send heartbeat to all connected peers
    Enum.each(state.connected_peers, fn {_id, peer} ->
      send_message(peer.socket, heartbeat)
    end)
    
    # Remove stale peers (no heartbeat in 5 minutes)
    cutoff_time = DateTime.add(DateTime.utc_now(), -300, :second)
    active_peers = state.connected_peers
    |> Enum.filter(fn {_id, peer} ->
      DateTime.compare(peer.last_seen, cutoff_time) == :gt
    end)
    |> Map.new()
    
    %{state | connected_peers: active_peers}
  end

  defp handle_peer_message(state, socket, message) do
    case message.type do
      :handshake ->
        handle_handshake(state, socket, message)
        
      :new_block ->
        handle_new_block(state, message)
        
      :new_transaction ->
        handle_new_transaction(state, message)
        
      :get_peers ->
        handle_get_peers(state, socket, message)
        
      :peers_response ->
        handle_peers_response(state, message)
        
      :heartbeat ->
        handle_heartbeat(state, socket, message)
        
      _ ->
        Logger.warning("Unknown message type: #{message.type}")
        state
    end
  end

  defp handle_handshake(state, socket, message) do
    peer = %Peer{
      id: message.sender,
      address: "unknown",  # Would extract from socket
      port: 0,
      socket: socket,
      connected_at: DateTime.utc_now(),
      last_seen: DateTime.utc_now(),
      version: message.version,
      services: message.services,
      height: message.height,
      score: 1.0
    }
    
    new_connected_peers = Map.put(state.connected_peers, peer.id, peer)
    Logger.info("Handshake completed with peer #{peer.id}")
    
    %{state | connected_peers: new_connected_peers}
  end

  defp handle_new_block(state, message) do
    # Forward to blockchain module for processing
    Logger.info("Received new block from peer #{message.sender}")
    # Would call LMStudio.StablecoinNode.process_block(message.data)
    state
  end

  defp handle_new_transaction(state, message) do
    # Forward to mempool for processing
    Logger.info("Received new transaction from peer #{message.sender}")
    # Would call LMStudio.StablecoinNode.Mempool.add_transaction(message.data)
    state
  end

  defp handle_get_peers(state, socket, message) do
    peer_list = state.connected_peers
    |> Enum.take(20)  # Send up to 20 peers
    |> Enum.map(fn {_id, peer} ->
      %{id: peer.id, address: peer.address, port: peer.port}
    end)
    
    response = %{
      type: :peers_response,
      sender: state.node_id,
      data: peer_list,
      timestamp: DateTime.utc_now()
    }
    
    send_message(socket, response)
    state
  end

  defp handle_peers_response(state, message) do
    # Add new peers to our peer list
    new_peers = message.data
    |> Enum.reduce(state.peers, fn peer_info, acc ->
      Map.put(acc, peer_info.id, peer_info)
    end)
    
    # Try to connect to some new peers if we have capacity
    if map_size(state.connected_peers) < @max_peers do
      new_candidates = new_peers
      |> Enum.filter(fn {id, _peer} -> not Map.has_key?(state.connected_peers, id) end)
      |> Enum.take(3)
      
      Enum.each(new_candidates, fn {_id, peer_info} ->
        connect_to_peer(peer_info.address, peer_info.port)
      end)
    end
    
    %{state | peers: new_peers}
  end

  defp handle_heartbeat(state, socket, message) do
    # Update peer's last seen time
    peer_id = message.sender
    new_connected_peers = Map.update(state.connected_peers, peer_id, nil, fn peer ->
      if peer do
        %{peer | last_seen: DateTime.utc_now(), height: message.height}
      else
        peer
      end
    end)
    
    %{state | connected_peers: new_connected_peers}
  end

  defp handle_peer_disconnect(state, socket) do
    # Find and remove the disconnected peer
    disconnected_peer = state.connected_peers
    |> Enum.find(fn {_id, peer} -> peer.socket == socket end)
    
    case disconnected_peer do
      {peer_id, peer} ->
        Logger.info("Peer #{peer_id} disconnected")
        new_connected_peers = Map.delete(state.connected_peers, peer_id)
        %{state | connected_peers: new_connected_peers}
        
      nil ->
        state
    end
  end

  defp broadcast_message(state, message) do
    Enum.each(state.connected_peers, fn {_id, peer} ->
      send_message(peer.socket, message)
    end)
  end

  defp send_message(socket, message) do
    encoded_message = encode_message(message)
    :gen_tcp.send(socket, encoded_message)
  end

  defp send_handshake_response(socket) do
    response = %{
      type: :handshake_response,
      sender: generate_node_id(),
      version: "1.0.0",
      services: [:full_node],
      height: 0,
      timestamp: DateTime.utc_now()
    }
    send_message(socket, response)
  end

  defp encode_message(message) do
    :erlang.term_to_binary(message)
  end

  defp decode_message(data) do
    try do
      message = :erlang.binary_to_term(data)
      {:ok, message}
    rescue
      _ -> {:error, :invalid_message}
    end
  end

  defp generate_node_id do
    :crypto.strong_rand_bytes(16) |> Base.encode16(case: :lower)
  end

  defp generate_peer_id(address, port) do
    :crypto.hash(:sha256, "#{address}:#{port}") |> Base.encode16(case: :lower)
  end

  defp find_available_port(start_port, max_attempts \\ 100) do
    find_available_port_recursive(start_port, max_attempts, 0)
  end

  defp find_available_port_recursive(port, max_attempts, attempt) when attempt < max_attempts do
    case :gen_tcp.listen(port, [:binary, packet: 4, active: false, reuseaddr: true]) do
      {:ok, socket} ->
        {port, {:ok, socket}}
      {:error, :eaddrinuse} ->
        find_available_port_recursive(port + 1, max_attempts, attempt + 1)
      {:error, reason} ->
        {port, {:error, reason}}
    end
  end

  defp find_available_port_recursive(_port, _max_attempts, _attempt) do
    {nil, {:error, :no_available_port}}
  end

  defp default_discovery_nodes do
    [
      {"seed1.stablecoin.network", 8333},
      {"seed2.stablecoin.network", 8333},
      {"seed3.stablecoin.network", 8333}
    ]
  end
end