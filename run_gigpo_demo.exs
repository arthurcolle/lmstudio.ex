#!/usr/bin/env elixir

# Quick Start GiGPO Demo
# Run this script to see GiGPO in action!

Mix.install([
  {:jason, "~> 1.4"}
])

Code.require_file("lib/lmstudio.ex")
Code.require_file("lib/lmstudio/quantum_classical_hybrid.ex")
Code.require_file("lib/lmstudio/gigpo_trainer.ex")

defmodule QuickGiGPODemo do
  @moduledoc """
  Quick demonstration of GiGPO for LLM agent training.
  
  Just run: elixir run_gigpo_demo.exs
  """
  
  alias LMStudio.{GiGPOTrainer, QuantumClassicalHybrid}
  
  def run do
    IO.puts """
    
    🚀 GiGPO Quick Demo
    ==================
    
    Group-in-Group Policy Optimization (GiGPO) for LLM Agent Training
    
    Key Features:
    ✓ Hierarchical advantage estimation (episode + step level)
    ✓ Anchor state grouping for fine-grained credit assignment  
    ✓ Critic-free, memory-efficient training
    ✓ Works with sparse rewards and long horizons
    
    """
    
    # Start required systems
    {:ok, _qc_pid} = QuantumClassicalHybrid.start_link()
    {:ok, _pid} = GiGPOTrainer.start_link([
      group_size: 4,
      discount_factor: 0.95,
      step_weight: 1.0
    ])
    
    # Quick demo scenarios
    demo_simple_task()
    demo_complex_task()
    show_results_summary()
    
    IO.puts "\n🎉 GiGPO Demo Complete! Check out the full demos for more details."
  end
  
  def demo_simple_task do
    IO.puts """
    
    🏠 Demo 1: Simple Kitchen Task
    =============================
    
    Task: "Heat an egg and put it on countertop"
    
    This demonstrates GiGPO's ability to:
    • Group actions from the same kitchen state
    • Learn which appliances work better for heating
    • Assign fine-grained credit to individual actions
    """
    
    env_config = %{
      type: :kitchen,
      initial_state: %{location: "kitchen", step_count: 0},
      action_space: ["go to fridge", "take egg", "go to microwave", "heat egg", "put egg"],
      reward_function: :task_completion,
      max_steps: 20
    }
    
    {:ok, result} = GiGPOTrainer.train_episode_group(
      "heat an egg and put it on countertop",
      env_config
    )
    
    display_quick_results("Kitchen Task", result)
  end
  
  def demo_complex_task do
    IO.puts """
    
    🛒 Demo 2: Online Shopping Task  
    ==============================
    
    Task: "Find wireless headphones under $100"
    
    This demonstrates GiGPO's ability to:
    • Group actions from the same webpage state
    • Learn optimal click sequences
    • Handle sparse rewards (success only at purchase)
    """
    
    env_config = %{
      type: :shopping,
      initial_state: %{page: "search", step_count: 0},
      action_space: ["search[query]", "click[item]", "add to cart", "buy now"],
      reward_function: :purchase_success,
      max_steps: 15
    }
    
    {:ok, result} = GiGPOTrainer.train_episode_group(
      "find wireless headphones under $100",
      env_config
    )
    
    display_quick_results("Shopping Task", result)
  end
  
  def show_results_summary do
    IO.puts """
    
    📊 What Just Happened?
    =====================
    
    GiGPO Training Process:
    
    1. 📋 Rollout Phase
       • Generated 4 agent trajectories per task
       • Each trajectory tried different action sequences
       • Collected states, actions, and rewards
    
    2. 🎯 Anchor State Grouping  
       • Identified repeated states across trajectories
       • Grouped actions taken from the same state
       • Created step-level groups for comparison
    
    3. 📊 Hierarchical Advantages
       • Episode advantages: "How good was this trajectory overall?"
       • Step advantages: "How good was this action in this context?"
       • Combined both for fine-grained credit assignment
    
    4. 🔄 Policy Update
       • Updated agent policy using combined advantages
       • Improved both overall strategy and specific actions
       • Maintained computational efficiency
    
    Key Benefits Demonstrated:
    ✓ Better credit assignment than episode-level only methods
    ✓ Learns from repeated situations across trajectories  
    ✓ No extra memory or computational overhead
    ✓ Suitable for long-horizon, sparse-reward tasks
    
    Paper Results:
    • 13.3% improvement on ALFWorld tasks
    • 10.6% improvement on WebShop tasks
    • Same efficiency as GRPO baseline
    """
  end
  
  defp display_quick_results(task_name, result) do
    if result.step_metrics.total_anchor_states > 0 do
      IO.puts """
      
      ✓ #{task_name} Results:
        • Success Rate: #{Float.round(result.episode_metrics.success_rate * 100, 1)}%
        • Anchor States Found: #{result.step_metrics.total_anchor_states}
        • Actions Grouped: #{result.step_metrics.total_grouped_steps}
        • Training Time: #{result.timing.total_time}ms
      
      🎯 Anchor state grouping enabled fine-grained learning!
      """
    else
      IO.puts """
      
      ✓ #{task_name} Results:
        • Success Rate: #{Float.round(result.episode_metrics.success_rate * 100, 1)}%
        • No repeated states (fell back to episode-level only)
        • Training Time: #{result.timing.total_time}ms
      """
    end
  end
end

# Run the quick demo
QuickGiGPODemo.run()