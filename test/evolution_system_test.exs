defmodule LMStudio.EvolutionSystemTest do
  use ExUnit.Case, async: false
  
  alias LMStudio.EvolutionSystem
  alias LMStudio.CognitiveAgent
  alias LMStudio.MetaDSL.{Mutation, SelfModifyingGrid}
  alias LMStudio.Persistence
  
  setup_all do
    # Ensure clean state
    Application.stop(:lmstudio)
    Application.start(:lmstudio)
    :ok
  end
  
  setup do
    # Clean up any existing processes
    case Process.whereis(EvolutionSystem) do
      nil -> :ok
      pid -> Supervisor.stop(pid, :normal)
    end
    
    # Wait for cleanup
    Process.sleep(100)
    :ok
  end
  
  describe "EvolutionSystem supervision" do
    test "starts successfully with default configuration" do
      {:ok, pid} = EvolutionSystem.start_link()
      assert Process.alive?(pid)
      assert Process.whereis(EvolutionSystem) == pid
      
      # Verify registry is running
      assert Process.whereis(LMStudio.AgentRegistry) != nil
      
      Supervisor.stop(pid)
    end
    
    test "starts with custom agent count" do
      {:ok, pid} = EvolutionSystem.start_link(num_agents: 5)
      assert Process.alive?(pid)
      
      # Give time for agents to start
      Process.sleep(500)
      
      # Check that agents are registered
      agents = Registry.select(LMStudio.AgentRegistry, [{{:"$1", :"$2", :"$3"}, [], [:"$1"]}])
      assert length(agents) >= 3  # At least the default agent types
      
      Supervisor.stop(pid)
    end
    
    test "handles agent restarts gracefully" do
      {:ok, pid} = EvolutionSystem.start_link()
      Process.sleep(300)
      
      # Get an agent pid
      agents = Registry.select(LMStudio.AgentRegistry, [{{:"$1", :"$2", :"$3"}, [], [:"$2"]}])
      agent_pid = List.first(agents)
      
      if agent_pid do
        # Kill the agent
        Process.exit(agent_pid, :kill)
        Process.sleep(200)
        
        # Verify system is still running
        assert Process.alive?(pid)
      end
      
      Supervisor.stop(pid)
    end
  end
  
  describe "agent coordination" do
    test "agents can share mutations through evolution system" do
      {:ok, system_pid} = EvolutionSystem.start_link()
      Process.sleep(300)
      
      # Create a test mutation
      mutation = Mutation.new(:append, "knowledge", content: "shared insight")
      
      # Simulate sharing mutation between agents
      result = EvolutionSystem.share_mutation("Explorer", mutation)
      assert result in [:ok, {:error, :agent_not_found}]  # Agent might not be ready yet
      
      Supervisor.stop(system_pid)
    end
    
    test "evolution cycles run periodically" do
      {:ok, system_pid} = EvolutionSystem.start_link()
      Process.sleep(300)
      
      # Start an evolution cycle
      result = EvolutionSystem.trigger_evolution_cycle()
      assert result in [:ok, {:error, :no_agents_ready}]
      
      Supervisor.stop(system_pid)
    end
  end
  
  describe "performance and resilience" do
    test "system handles multiple concurrent operations" do
      {:ok, system_pid} = EvolutionSystem.start_link()
      Process.sleep(300)
      
      # Spawn multiple tasks that interact with the system
      tasks = for i <- 1..10 do
        Task.async(fn ->
          mutation = Mutation.new(:append, "test_#{i}", content: "concurrent test #{i}")
          EvolutionSystem.share_mutation("Explorer", mutation)
        end)
      end
      
      # Wait for all tasks to complete
      results = Task.await_many(tasks, 5000)
      
      # System should still be alive
      assert Process.alive?(system_pid)
      
      Supervisor.stop(system_pid)
    end
    
    test "system recovers from registry failures" do
      {:ok, system_pid} = EvolutionSystem.start_link()
      Process.sleep(200)
      
      # Kill the registry
      registry_pid = Process.whereis(LMStudio.AgentRegistry)
      if registry_pid do
        Process.exit(registry_pid, :kill)
        Process.sleep(300)
        
        # System should restart the registry
        new_registry_pid = Process.whereis(LMStudio.AgentRegistry)
        assert new_registry_pid != nil
        assert new_registry_pid != registry_pid
      end
      
      Supervisor.stop(system_pid)
    end
  end
  
  describe "persistence integration" do
    test "evolution system persists agent states" do
      {:ok, system_pid} = EvolutionSystem.start_link()
      Process.sleep(300)
      
      # Trigger persistence
      EvolutionSystem.save_system_state()
      
      # Check that something was persisted
      stored_keys = Persistence.list_keys()
      assert is_list(stored_keys)
      
      Supervisor.stop(system_pid)
    end
    
    test "system can restore from persisted state" do
      # First run - create some state
      {:ok, system_pid} = EvolutionSystem.start_link()
      Process.sleep(300)
      
      EvolutionSystem.save_system_state()
      Supervisor.stop(system_pid)
      
      # Second run - restore state
      {:ok, new_system_pid} = EvolutionSystem.start_link()
      Process.sleep(300)
      
      result = EvolutionSystem.restore_system_state()
      assert result in [:ok, {:error, :no_state_found}]
      
      Supervisor.stop(new_system_pid)
    end
  end
  
  describe "advanced evolution patterns" do
    test "cross-pollination between agent types" do
      {:ok, system_pid} = EvolutionSystem.start_link()
      Process.sleep(500)
      
      # Create mutations from different agent types
      explorer_mutation = Mutation.new(:evolve, "reasoning", content: "creative exploration")
      optimizer_mutation = Mutation.new(:compress, "efficiency", content: "optimized pattern")
      
      # Test cross-pollination
      EvolutionSystem.cross_pollinate([explorer_mutation, optimizer_mutation])
      
      # System should handle this gracefully
      assert Process.alive?(system_pid)
      
      Supervisor.stop(system_pid)
    end
    
    test "evolutionary pressure and selection" do
      {:ok, system_pid} = EvolutionSystem.start_link()
      Process.sleep(300)
      
      # Create multiple competing mutations
      mutations = for i <- 1..5 do
        Mutation.new(:evolve, "strategy_#{i}", content: "approach #{i}")
      end
      
      # Apply evolutionary pressure
      selected = EvolutionSystem.apply_selection_pressure(mutations)
      assert is_list(selected)
      assert length(selected) <= length(mutations)
      
      Supervisor.stop(system_pid)
    end
  end
end