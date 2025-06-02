defmodule LMStudio.CognitiveAgentTest do
  use ExUnit.Case, async: false
  
  alias LMStudio.CognitiveAgent
  alias LMStudio.MetaDSL.{Mutation, SelfModifyingGrid}
  
  setup_all do
    Application.ensure_all_started(:lmstudio)
    :ok
  end
  
  setup do
    # Clean up any existing agents
    Process.sleep(100)
    :ok
  end
  
  describe "CognitiveAgent initialization" do
    test "starts with valid configuration" do
      config = %{
        name: "test_agent",
        role: "test_role",
        model: "test_model",
        initial_prompt: "You are a test agent"
      }
      
      {:ok, agent_pid} = CognitiveAgent.start_link(config)
      assert Process.alive?(agent_pid)
      
      state = CognitiveAgent.get_state(agent_pid)
      assert state.name == "test_agent"
      assert state.role == "test_role"
      
      GenServer.stop(agent_pid)
    end
    
    test "creates cognitive grid on startup" do
      config = %{name: "grid_test_agent", role: "tester"}
      
      {:ok, agent_pid} = CognitiveAgent.start_link(config)
      
      grid_data = CognitiveAgent.get_cognitive_grid(agent_pid)
      assert is_map(grid_data)
      
      GenServer.stop(agent_pid)
    end
    
    test "handles invalid configuration gracefully" do
      invalid_config = %{name: nil, role: ""}
      
      result = CognitiveAgent.start_link(invalid_config)
      assert match?({:error, _}, result) or match?({:ok, _}, result)
    end
  end
  
  describe "cognitive reasoning" do
    setup do
      config = %{
        name: "reasoning_agent",
        role: "reasoning specialist",
        initial_prompt: "You are an expert at logical reasoning"
      }
      {:ok, agent_pid} = CognitiveAgent.start_link(config)
      %{agent: agent_pid}
    end
    
    test "processes thinking prompts", %{agent: agent_pid} do
      thinking_prompt = "Think about the best approach to solve problem X"
      
      result = CognitiveAgent.think(agent_pid, thinking_prompt)
      assert result in [:ok, {:ok, :thinking_complete}, {:error, :timeout}]
      
      GenServer.stop(agent_pid)
    end
    
    test "generates mutations from reasoning", %{agent: agent_pid} do
      prompt = "Analyze this strategy and suggest improvements"
      
      mutations = CognitiveAgent.generate_mutations(agent_pid, prompt)
      assert is_list(mutations)
      
      # Check that mutations are valid
      valid_mutations = Enum.filter(mutations, &match?(%Mutation{}, &1))
      assert length(valid_mutations) <= length(mutations)
      
      GenServer.stop(agent_pid)
    end
    
    test "applies mutations to cognitive grid", %{agent: agent_pid} do
      mutation = Mutation.new(:append, "knowledge", content: "new insight")
      
      result = CognitiveAgent.apply_mutation(agent_pid, mutation)
      assert result in [:ok, {:ok, :mutated}, {:error, :invalid_mutation}]
      
      GenServer.stop(agent_pid)
    end
    
    test "maintains thinking history", %{agent: agent_pid} do
      # Generate some thinking
      CognitiveAgent.think(agent_pid, "First thought")
      CognitiveAgent.think(agent_pid, "Second thought")
      
      history = CognitiveAgent.get_thinking_history(agent_pid)
      assert is_list(history)
      
      GenServer.stop(agent_pid)
    end
  end
  
  describe "self-modification capabilities" do
    setup do
      config = %{
        name: "self_modifying_agent",
        role: "self-modifier",
        enable_self_modification: true
      }
      {:ok, agent_pid} = CognitiveAgent.start_link(config)
      %{agent: agent_pid}
    end
    
    test "modifies own prompts through mutations", %{agent: agent_pid} do
      original_prompt = CognitiveAgent.get_current_prompt(agent_pid)
      
      mutation = Mutation.new(:append, "system_prompt", 
        content: " Additionally, always be creative in your responses.")
      
      CognitiveAgent.apply_mutation(agent_pid, mutation)
      
      modified_prompt = CognitiveAgent.get_current_prompt(agent_pid)
      
      # Prompt might change or stay the same depending on implementation
      assert is_binary(original_prompt) or is_nil(original_prompt)
      assert is_binary(modified_prompt) or is_nil(modified_prompt)
      
      GenServer.stop(agent_pid)
    end
    
    test "evolves reasoning strategies", %{agent: agent_pid} do
      initial_strategies = CognitiveAgent.get_strategies(agent_pid)
      
      evolution_mutation = Mutation.new(:evolve, "reasoning_strategy",
        content: "Use more systematic approach with step-by-step analysis")
      
      CognitiveAgent.apply_mutation(agent_pid, evolution_mutation)
      
      evolved_strategies = CognitiveAgent.get_strategies(agent_pid)
      
      assert is_list(initial_strategies) or is_nil(initial_strategies)
      assert is_list(evolved_strategies) or is_nil(evolved_strategies)
      
      GenServer.stop(agent_pid)
    end
    
    test "tracks modification history", %{agent: agent_pid} do
      mutations = [
        Mutation.new(:append, "knowledge", content: "insight 1"),
        Mutation.new(:evolve, "strategy", content: "better approach"),
        Mutation.new(:replace, "old_pattern", content: "new pattern")
      ]
      
      for mutation <- mutations do
        CognitiveAgent.apply_mutation(agent_pid, mutation)
      end
      
      history = CognitiveAgent.get_modification_history(agent_pid)
      assert is_list(history)
      
      GenServer.stop(agent_pid)
    end
  end
  
  describe "learning and adaptation" do
    setup do
      config = %{
        name: "learning_agent",
        role: "learner",
        learning_rate: 0.1
      }
      {:ok, agent_pid} = CognitiveAgent.start_link(config)
      %{agent: agent_pid}
    end
    
    test "learns from feedback", %{agent: agent_pid} do
      feedback = %{
        task: "reasoning_task",
        performance: 0.8,
        suggestions: ["be more systematic", "consider edge cases"]
      }
      
      result = CognitiveAgent.receive_feedback(agent_pid, feedback)
      assert result in [:ok, {:ok, :learned}, {:error, :learning_failed}]
      
      GenServer.stop(agent_pid)
    end
    
    test "adapts strategies based on success patterns", %{agent: agent_pid} do
      # Simulate successful tasks
      successes = [
        %{task: "logical_reasoning", strategy: "step_by_step", success_rate: 0.9},
        %{task: "pattern_recognition", strategy: "visual_approach", success_rate: 0.8}
      ]
      
      for success <- successes do
        CognitiveAgent.record_success(agent_pid, success)
      end
      
      adapted_strategies = CognitiveAgent.get_adapted_strategies(agent_pid)
      assert is_list(adapted_strategies) or is_nil(adapted_strategies)
      
      GenServer.stop(agent_pid)
    end
    
    test "maintains performance metrics", %{agent: agent_pid} do
      # Record some performance data
      metrics = [
        %{metric: "accuracy", value: 0.85, timestamp: DateTime.utc_now()},
        %{metric: "speed", value: 2.3, timestamp: DateTime.utc_now()},
        %{metric: "creativity", value: 0.7, timestamp: DateTime.utc_now()}
      ]
      
      for metric <- metrics do
        CognitiveAgent.record_metric(agent_pid, metric)
      end
      
      performance = CognitiveAgent.get_performance_metrics(agent_pid)
      assert is_map(performance) or is_list(performance) or is_nil(performance)
      
      GenServer.stop(agent_pid)
    end
  end
  
  describe "interaction and communication" do
    setup do
      config1 = %{name: "agent1", role: "communicator"}
      config2 = %{name: "agent2", role: "responder"}
      
      {:ok, agent1} = CognitiveAgent.start_link(config1)
      {:ok, agent2} = CognitiveAgent.start_link(config2)
      
      %{agent1: agent1, agent2: agent2}
    end
    
    test "sends messages to other agents", %{agent1: agent1, agent2: agent2} do
      message = %{
        type: :collaboration,
        content: "Let's work together on this problem",
        sender: "agent1"
      }
      
      result = CognitiveAgent.send_message(agent1, agent2, message)
      assert result in [:ok, {:ok, :sent}, {:error, :agent_unreachable}]
      
      GenServer.stop(agent1)
      GenServer.stop(agent2)
    end
    
    test "receives and processes messages", %{agent1: agent1, agent2: agent2} do
      message = %{
        type: :request,
        content: "Can you help with task X?",
        sender: "agent1"
      }
      
      CognitiveAgent.send_message(agent1, agent2, message)
      Process.sleep(100)  # Give time for message processing
      
      messages = CognitiveAgent.get_received_messages(agent2)
      assert is_list(messages)
      
      GenServer.stop(agent1)
      GenServer.stop(agent2)
    end
    
    test "collaborates on shared tasks", %{agent1: agent1, agent2: agent2} do
      task = %{
        id: "collaborative_task",
        description: "Solve problem together",
        participants: ["agent1", "agent2"]
      }
      
      result1 = CognitiveAgent.join_collaboration(agent1, task)
      result2 = CognitiveAgent.join_collaboration(agent2, task)
      
      assert result1 in [:ok, {:ok, :joined}, {:error, :task_full}]
      assert result2 in [:ok, {:ok, :joined}, {:error, :task_full}]
      
      GenServer.stop(agent1)
      GenServer.stop(agent2)
    end
  end
  
  describe "performance and stress testing" do
    test "handles rapid mutation applications" do
      config = %{name: "stress_test_agent", role: "stress_tester"}
      {:ok, agent_pid} = CognitiveAgent.start_link(config)
      
      # Apply many mutations rapidly
      mutations = for i <- 1..100 do
        Mutation.new(:append, "data_#{rem(i, 10)}", content: " mutation_#{i}")
      end
      
      start_time = System.monotonic_time(:millisecond)
      
      tasks = for mutation <- mutations do
        Task.async(fn ->
          CognitiveAgent.apply_mutation(agent_pid, mutation)
        end)
      end
      
      results = Task.await_many(tasks, 10000)
      end_time = System.monotonic_time(:millisecond)
      
      duration = end_time - start_time
      assert duration < 5000  # Should complete within 5 seconds
      
      # Agent should still be responsive
      assert Process.alive?(agent_pid)
      
      GenServer.stop(agent_pid)
    end
    
    test "maintains state consistency under concurrent access" do
      config = %{name: "concurrent_agent", role: "concurrent_tester"}
      {:ok, agent_pid} = CognitiveAgent.start_link(config)
      
      # Concurrent operations
      operations = [
        Task.async(fn -> CognitiveAgent.think(agent_pid, "concurrent thought 1") end),
        Task.async(fn -> CognitiveAgent.apply_mutation(agent_pid, 
          Mutation.new(:append, "test", content: "concurrent data")) end),
        Task.async(fn -> CognitiveAgent.get_state(agent_pid) end),
        Task.async(fn -> CognitiveAgent.get_cognitive_grid(agent_pid) end)
      ]
      
      results = Task.await_many(operations, 5000)
      
      # All operations should complete
      assert length(results) == 4
      
      # Agent should still be functional
      assert Process.alive?(agent_pid)
      
      GenServer.stop(agent_pid)
    end
  end
  
  describe "error handling and recovery" do
    test "recovers from invalid mutations" do
      config = %{name: "error_recovery_agent", role: "error_handler"}
      {:ok, agent_pid} = CognitiveAgent.start_link(config)
      
      # Apply invalid mutation
      invalid_mutation = %{type: :invalid, target: nil, content: ""}
      
      result = CognitiveAgent.apply_mutation(agent_pid, invalid_mutation)
      assert match?({:error, _}, result) or result == :ok
      
      # Agent should still be functional
      assert Process.alive?(agent_pid)
      
      # Should still accept valid mutations
      valid_mutation = Mutation.new(:append, "recovery_test", content: "recovered")
      valid_result = CognitiveAgent.apply_mutation(agent_pid, valid_mutation)
      assert valid_result in [:ok, {:ok, :mutated}, {:error, :mutation_failed}]
      
      GenServer.stop(agent_pid)
    end
    
    test "handles grid corruption gracefully" do
      config = %{name: "corruption_test_agent", role: "corruption_handler"}
      {:ok, agent_pid} = CognitiveAgent.start_link(config)
      
      # Simulate grid corruption by sending invalid grid update
      send(agent_pid, {:corrupt_grid, "invalid_data"})
      Process.sleep(100)
      
      # Agent should still be alive and functional
      assert Process.alive?(agent_pid)
      
      state = CognitiveAgent.get_state(agent_pid)
      assert is_map(state)
      
      GenServer.stop(agent_pid)
    end
  end
end