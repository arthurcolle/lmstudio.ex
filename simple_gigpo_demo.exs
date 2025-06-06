#!/usr/bin/env elixir

# Simple GiGPO Demonstration
# Shows the core concepts of Group-in-Group Policy Optimization
# in an easy-to-understand format

Mix.install([
  {:jason, "~> 1.4"}
])

Code.require_file("lib/lmstudio.ex")
Code.require_file("lib/lmstudio/gigpo_trainer.ex")

defmodule SimpleGiGPODemo do
  @moduledoc """
  Simple demonstration of GiGPO's key concepts:
  1. How anchor states are identified across trajectories
  2. How step-level groups are formed
  3. How advantages are computed hierarchically
  4. How this improves long-horizon learning
  """
  
  alias LMStudio.GiGPOTrainer
  
  def run do
    IO.puts """
    
    🚀 Simple GiGPO Demo: Understanding Hierarchical Advantages
    ==========================================================
    
    GiGPO = Group-in-Group Policy Optimization
    
    Key Innovation: Two-level advantage estimation
    📊 Episode Level: How good is the entire trajectory?
    🎯 Step Level: How good is each action in context?
    
    """
    
    # Start the trainer
    {:ok, _pid} = GiGPOTrainer.start_link([
      group_size: 4,  # Smaller for demo clarity
      discount_factor: 0.9,
      step_weight: 1.0
    ])
    
    demonstrate_core_concept()
    demonstrate_anchor_state_grouping()
    demonstrate_advantage_computation()
    show_learning_improvement()
    
    IO.puts "\n✅ Simple GiGPO Demo Complete!"
  end
  
  def demonstrate_core_concept do
    IO.puts """
    
    💡 Core GiGPO Concept
    ====================
    
    Problem: Traditional RL assigns credit to entire trajectories
    - Hard to know which specific actions were good/bad
    - Especially difficult in long episodes (50+ steps)
    
    GiGPO Solution: Hierarchical credit assignment
    1. Episode advantages: "Was this trajectory successful overall?"
    2. Step advantages: "Was this action good in this specific situation?"
    
    Magic: No extra rollouts needed! Uses repeated states across trajectories.
    """
  end
  
  def demonstrate_anchor_state_grouping do
    IO.puts """
    
    🎯 Anchor State Grouping Example
    ===============================
    
    Imagine 4 agent trajectories trying to "find and heat an egg":
    
    Trajectory 1: Kitchen → Fridge → [No egg] → Countertop → [Found egg!] → Microwave → Success ✓
    Trajectory 2: Kitchen → Countertop → [Found egg!] → Microwave → Success ✓  
    Trajectory 3: Kitchen → Fridge → [No egg] → Countertop → [Found egg!] → Stove → Failure ✗
    Trajectory 4: Kitchen → Countertop → [Found egg!] → Stove → Failure ✗
    
    GiGPO identifies repeated states (anchor states):
    • "Countertop with egg visible" appears in all 4 trajectories
    • "Microwave available" appears in trajectories 1 & 2  
    • "Stove available" appears in trajectories 3 & 4
    
    Now we can group actions by context and compare their outcomes!
    """
    
    # Simulate this scenario
    simulate_egg_heating_scenario()
  end
  
  def demonstrate_advantage_computation do
    IO.puts """
    
    📊 Hierarchical Advantage Computation
    ====================================
    
    Step 1: Episode-Level Advantages (like standard GRPO)
    • Compare total returns across trajectories
    • Trajectory 1: +10 reward → High episode advantage
    • Trajectory 2: +10 reward → High episode advantage  
    • Trajectory 3: +0 reward → Low episode advantage
    • Trajectory 4: +0 reward → Low episode advantage
    
    Step 2: Step-Level Advantages (GiGPO innovation)
    • For "Countertop with egg" state:
      - Action "go to microwave": Returns [+10, +10] → High step advantage
      - Action "go to stove": Returns [+0, +0] → Low step advantage
    
    Step 3: Combined Advantages
    • Final advantage = Episode advantage + Step advantage
    • This gives fine-grained feedback on both trajectory success AND action quality
    """
  end
  
  def simulate_egg_heating_scenario do
    IO.puts "\n🧪 Simulating Egg Heating Scenario..."
    
    # Create a simple kitchen environment
    kitchen_config = %{
      type: :alfworld,
      initial_state: %{
        location: "kitchen",
        step_count: 0,
        observation: "You are in a kitchen. You can see a fridge, countertop, microwave, and stove."
      },
      action_space: ["go to fridge", "go to countertop", "go to microwave", "go to stove", "take egg", "heat egg"],
      reward_function: :task_completion,
      max_steps: 10
    }
    
    # Train with GiGPO
    {:ok, result} = GiGPOTrainer.train_episode_group(
      "find and heat an egg",
      kitchen_config
    )
    
    # Show the results
    IO.puts """
    
    📈 Results from Egg Heating Simulation:
    
    Episode Metrics:
    • Success Rate: #{Float.round(result.episode_metrics.success_rate * 100, 1)}%
    • Mean Return: #{Float.round(result.episode_metrics.mean_return, 2)}
    
    Step-Level Grouping:
    • Found #{result.step_metrics.total_anchor_states} repeated states
    • #{result.step_metrics.total_grouped_steps} actions were grouped for comparison
    • Average #{Float.round(result.step_metrics.average_group_size, 1)} actions per anchor state
    
    The agent learned which appliances work better for heating eggs!
    """
  end
  
  def show_learning_improvement do
    IO.puts """
    
    📈 Why This Improves Learning
    ============================
    
    Traditional GRPO: "Trajectory 1 was good, Trajectory 3 was bad"
    → Agent learns: "Kitchen → Fridge → Countertop → Microwave is better than Kitchen → Fridge → Countertop → Stove"
    → But doesn't know specifically that "Microwave > Stove for heating"
    
    GiGPO: "Trajectory 1 was good AND at countertop state, microwave action was better than stove action"  
    → Agent learns: "Overall strategy matters AND microwave is better than stove for heating"
    → Much faster learning, especially for long episodes!
    
    Real Benefits:
    ✓ 13.3% improvement on ALFWorld tasks
    ✓ 10.6% improvement on WebShop tasks  
    ✓ Same computational cost as GRPO
    ✓ No extra memory overhead
    ✓ Works with sparse rewards
    """
  end
end

# Run the simple demo
SimpleGiGPODemo.run()