#!/usr/bin/env elixir

defmodule SimpleEvolutionTest do
  @moduledoc """
  Simple self-evolving system that demonstrates:
  - Continuous mutation
  - Error correction
  - Performance tracking
  """

  defmodule Agent do
    use GenServer
    
    def start_link(name) do
      GenServer.start_link(__MODULE__, name, name: String.to_atom(name))
    end

    def init(name) do
      state = %{
        name: name,
        data: "Initial knowledge: I am learning",
        mutations: 0,
        errors: 0,
        generation: 1
      }
      
      IO.puts("🧠 #{name} started")
      
      # Start mutation cycle
      Process.send_after(self(), :mutate, 2000)
      
      {:ok, state}
    end

    def handle_info(:mutate, state) do
      new_state = try do
        # Perform mutation
        mutated_data = state.data <> " [Mutation #{state.mutations + 1}: Learning more]"
        
        updated_state = %{state | 
          data: mutated_data,
          mutations: state.mutations + 1
        }
        
        # Sometimes cause an error for testing
        if :rand.uniform() < 0.2 do
          raise "Simulated error"
        end
        
        IO.puts("✅ #{state.name}: Mutation #{updated_state.mutations}")
        updated_state
        
      rescue
        error ->
          IO.puts("❌ #{state.name}: Error occurred - #{inspect(error)}")
          corrected_state = apply_error_correction(state)
          IO.puts("🔧 #{state.name}: Error corrected")
          corrected_state
      end
      
      # Schedule next mutation
      Process.send_after(self(), :mutate, 3000)
      
      {:noreply, new_state}
    end

    def handle_call(:get_state, _from, state) do
      {:reply, state, state}
    end

    defp apply_error_correction(state) do
      %{state | 
        errors: state.errors + 1,
        data: state.data <> " [Error corrected at #{DateTime.utc_now()}]",
        generation: state.generation + 1
      }
    end

    def get_state(agent_name) do
      GenServer.call(String.to_atom(agent_name), :get_state)
    end
  end

  def run do
    IO.puts """
    🚀 Simple Evolution Test
    ========================
    
    Starting self-evolving agents...
    """

    # Start agents
    agents = ["Alpha", "Beta"]
    
    started_agents = Enum.map(agents, fn name ->
      case Agent.start_link(name) do
        {:ok, _pid} -> name
        {:error, reason} -> 
          IO.puts("Failed to start #{name}: #{inspect(reason)}")
          nil
      end
    end)
    |> Enum.filter(&(&1 != nil))

    if length(started_agents) > 0 do
      IO.puts("✅ Started #{length(started_agents)} agents")
      
      # Monitor loop
      monitor_loop(started_agents)
    else
      IO.puts("❌ No agents started")
    end
  end

  defp monitor_loop(agents) do
    IO.puts("\n📊 Status Report:")
    
    Enum.each(agents, fn name ->
      try do
        state = Agent.get_state(name)
        IO.puts("🤖 #{name}: Gen #{state.generation}, Mutations #{state.mutations}, Errors #{state.errors}")
      rescue
        error ->
          IO.puts("❌ #{name}: Status error - #{inspect(error)}")
      end
    end)
    
    receive after
      5000 ->
        monitor_loop(agents)
    end
  end
end

SimpleEvolutionTest.run()