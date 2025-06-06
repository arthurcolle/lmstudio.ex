defmodule LMStudio.GiGPOTrainer do
  @moduledoc """
  Group-in-Group Policy Optimization (GiGPO) for LLM Agent Training
  
  Implements the hierarchical advantage estimation algorithm from:
  "GiGPO: Group-in-Group Policy Optimization for LLM Agent Training"
  
  Key Features:
  - Episode-level advantages for trajectory-wide feedback
  - Step-level advantages via anchor state grouping
  - Fine-grained credit assignment for long-horizon tasks
  - Critic-free, memory-efficient training
  - Compatible with sparse reward environments
  """
  
  use GenServer
  require Logger
  
  alias LMStudio.QuantumClassicalHybrid
  
  defmodule TrajectoryStep do
    @moduledoc "Single step in an agent trajectory"
    defstruct [
      :state,
      :action,
      :reward,
      :next_state,
      :log_prob,
      :discounted_return,
      :step_index,
      :trajectory_id
    ]
  end
  
  defmodule Trajectory do
    @moduledoc "Complete agent trajectory"
    defstruct [
      :id,
      :task_description,
      :steps,
      :total_return,
      :success,
      :length,
      :episode_advantage
    ]
  end
  
  defmodule EpisodeGroup do
    @moduledoc "Group of trajectories for episode-level advantage computation"
    defstruct [
      :trajectories,
      :task_description,
      :group_size,
      :mean_return,
      :std_return,
      :success_rate
    ]
  end
  
  defmodule StepGroup do
    @moduledoc "Group of steps from same anchor state for step-level advantages"
    defstruct [
      :anchor_state,
      :steps,
      :mean_return,
      :std_return,
      :group_size
    ]
  end
  
  defmodule GiGPOConfig do
    @moduledoc "Configuration for GiGPO training"
    defstruct [
      group_size: 8,
      discount_factor: 0.95,
      episode_weight: 1.0,
      step_weight: 1.0,
      normalization_factor: :std,  # :std or :fixed
      clip_epsilon: 0.2,
      kl_penalty: 0.01,
      learning_rate: 1.0e-6,
      max_trajectory_length: 50,
      state_hash_precision: 6
    ]
  end
  
  defmodule TrainingState do
    @moduledoc "Internal training state"
    defstruct [
      :config,
      :policy_model,
      :reference_model,
      :episode_groups,
      :step_groups,
      :anchor_state_map,
      :training_metrics,
      :iteration_count
    ]
  end
  
  # Public API
  
  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end
  
  def train_episode_group(task_description, environment_configs, opts \\ []) do
    GenServer.call(__MODULE__, {:train_episode_group, task_description, environment_configs, opts}, 60_000)
  end
  
  def evaluate_policy(task_description, environment_configs, opts \\ []) do
    GenServer.call(__MODULE__, {:evaluate_policy, task_description, environment_configs, opts})
  end
  
  def get_training_metrics do
    GenServer.call(__MODULE__, :get_training_metrics)
  end
  
  def update_config(new_config) do
    GenServer.call(__MODULE__, {:update_config, new_config})
  end
  
  # GenServer Implementation
  
  @impl true
  def init(opts) do
    config = struct(GiGPOConfig, opts)
    
    state = %TrainingState{
      config: config,
      policy_model: initialize_policy_model(opts),
      reference_model: initialize_reference_model(opts),
      episode_groups: [],
      step_groups: %{},
      anchor_state_map: %{},
      training_metrics: initialize_metrics(),
      iteration_count: 0
    }
    
    Logger.info("GiGPO Trainer initialized with config: #{inspect(config)}")
    {:ok, state}
  end
  
  @impl true
  def handle_call({:train_episode_group, task_description, environment_configs, opts}, _from, state) do
    Logger.info("Starting GiGPO training iteration #{state.iteration_count + 1}")
    
    # Step 1: Rollout trajectories
    {trajectories, rollout_time} = time_operation(fn ->
      rollout_trajectory_group(state, task_description, environment_configs)
    end)
    
    # Step 2: Compute episode-level advantages
    {episode_group, episode_time} = time_operation(fn ->
      compute_episode_advantages(trajectories, state.config)
    end)
    
    # Step 3: Build anchor state groups
    {step_groups, grouping_time} = time_operation(fn ->
      build_anchor_state_groups(episode_group.trajectories, state.config)
    end)
    
    # Step 4: Compute step-level advantages
    {enhanced_trajectories, step_time} = time_operation(fn ->
      compute_step_advantages(episode_group.trajectories, step_groups, state.config)
    end)
    
    # Step 5: Combine advantages and update policy
    {policy_update_result, update_time} = time_operation(fn ->
      update_policy_with_gigpo(enhanced_trajectories, state)
    end)
    
    # Step 6: Update training state
    updated_state = %{state |
      episode_groups: [episode_group | state.episode_groups],
      step_groups: step_groups,
      training_metrics: update_training_metrics(state.training_metrics, %{
        iteration: state.iteration_count + 1,
        rollout_time: rollout_time,
        episode_computation_time: episode_time,
        grouping_time: grouping_time,
        step_computation_time: step_time,
        policy_update_time: update_time,
        episode_group: episode_group,
        step_groups_count: map_size(step_groups),
        policy_update: policy_update_result
      }),
      iteration_count: state.iteration_count + 1
    }
    
    result = %{
      iteration: updated_state.iteration_count,
      episode_metrics: extract_episode_metrics(episode_group),
      step_metrics: extract_step_metrics(step_groups),
      policy_metrics: policy_update_result,
      timing: %{
        total_time: rollout_time + episode_time + grouping_time + step_time + update_time,
        rollout_time: rollout_time,
        computation_time: episode_time + step_time,
        grouping_time: grouping_time,
        update_time: update_time
      }
    }
    
    {:reply, {:ok, result}, updated_state}
  end
  
  @impl true
  def handle_call({:evaluate_policy, task_description, environment_configs, opts}, _from, state) do
    evaluation_trajectories = rollout_trajectory_group(state, task_description, environment_configs, evaluation: true)
    
    metrics = %{
      success_rate: calculate_success_rate(evaluation_trajectories),
      average_return: calculate_average_return(evaluation_trajectories),
      average_length: calculate_average_length(evaluation_trajectories),
      step_efficiency: calculate_step_efficiency(evaluation_trajectories)
    }
    
    {:reply, {:ok, metrics}, state}
  end
  
  @impl true
  def handle_call(:get_training_metrics, _from, state) do
    {:reply, state.training_metrics, state}
  end
  
  @impl true
  def handle_call({:update_config, new_config}, _from, state) do
    updated_config = struct(state.config, new_config)
    updated_state = %{state | config: updated_config}
    Logger.info("Updated GiGPO config: #{inspect(updated_config)}")
    {:reply, :ok, updated_state}
  end
  
  # Core GiGPO Algorithm Implementation
  
  defp rollout_trajectory_group(state, task_description, environment_configs, opts \\ []) do
    is_evaluation = Keyword.get(opts, :evaluation, false)
    group_size = if is_evaluation, do: state.config.group_size * 2, else: state.config.group_size
    
    Logger.info("Rolling out #{group_size} trajectories for task: #{String.slice(task_description, 0, 50)}...")
    
    # Create identical initial environments
    environments = create_identical_environments(environment_configs, group_size)
    
    # Rollout trajectories in parallel
    trajectories = Enum.map(0..(group_size - 1), fn trajectory_id ->
      rollout_single_trajectory(
        state.policy_model,
        task_description,
        Enum.at(environments, trajectory_id),
        trajectory_id,
        state.config
      )
    end)
    
    Logger.info("Completed rollout: #{length(trajectories)} trajectories, average length: #{calculate_average_length(trajectories)}")
    trajectories
  end
  
  defp rollout_single_trajectory(policy_model, task_description, environment, trajectory_id, config) do
    steps = []
    current_state = environment.initial_state
    total_return = 0.0
    
    final_result = Enum.reduce_while(0..(config.max_trajectory_length - 1), {steps, current_state, total_return}, 
      fn step_index, {acc_steps, state, acc_return} ->
        # Generate action using policy
        action_result = generate_action(policy_model, state, task_description)
        
        # Execute action in environment
        step_result = execute_action(environment, state, action_result.action)
        
        # Create trajectory step
        trajectory_step = %TrajectoryStep{
          state: normalize_state(state, config),
          action: action_result.action,
          reward: step_result.reward,
          next_state: step_result.next_state,
          log_prob: action_result.log_prob,
          step_index: step_index,
          trajectory_id: trajectory_id
        }
        
        new_steps = [trajectory_step | acc_steps]
        new_return = acc_return + step_result.reward
        
        if step_result.done do
          {:halt, {Enum.reverse(new_steps), step_result.next_state, new_return}}
        else
          {:cont, {new_steps, step_result.next_state, new_return}}
        end
      end)
    
    {final_steps, _final_state, final_return} = final_result
    
    # Compute discounted returns for each step
    enhanced_steps = compute_discounted_returns(final_steps, config.discount_factor)
    
    %Trajectory{
      id: trajectory_id,
      task_description: task_description,
      steps: enhanced_steps,
      total_return: final_return,
      success: final_return > 0,
      length: length(enhanced_steps)
    }
  end
  
  defp compute_episode_advantages(trajectories, config) do
    Logger.info("Computing episode-level advantages for #{length(trajectories)} trajectories")
    
    returns = Enum.map(trajectories, & &1.total_return)
    mean_return = Enum.sum(returns) / length(returns)
    
    {normalization_factor, std_return} = case config.normalization_factor do
      :std ->
        std = calculate_standard_deviation(returns, mean_return)
        {max(std, 1.0e-8), std}
      :fixed ->
        {1.0, 0.0}
    end
    
    enhanced_trajectories = Enum.map(trajectories, fn trajectory ->
      episode_advantage = (trajectory.total_return - mean_return) / normalization_factor
      %{trajectory | episode_advantage: episode_advantage}
    end)
    
    %EpisodeGroup{
      trajectories: enhanced_trajectories,
      task_description: hd(trajectories).task_description,
      group_size: length(trajectories),
      mean_return: mean_return,
      std_return: std_return,
      success_rate: calculate_success_rate(trajectories)
    }
  end
  
  defp build_anchor_state_groups(trajectories, config) do
    Logger.info("Building anchor state groups...")
    
    # Collect all steps with their states
    all_steps = trajectories
    |> Enum.flat_map(& &1.steps)
    
    # Group steps by normalized state (anchor states)
    anchor_groups = all_steps
    |> Enum.group_by(fn step -> 
      hash_state(step.state, config.state_hash_precision)
    end)
    |> Enum.filter(fn {_state_hash, steps} -> 
      # Only consider states that appear multiple times
      length(steps) > 1
    end)
    |> Map.new(fn {state_hash, steps} ->
      returns = Enum.map(steps, & &1.discounted_return)
      mean_return = Enum.sum(returns) / length(returns)
      std_return = calculate_standard_deviation(returns, mean_return)
      
      step_group = %StepGroup{
        anchor_state: state_hash,
        steps: steps,
        mean_return: mean_return,
        std_return: max(std_return, 1.0e-8),
        group_size: length(steps)
      }
      
      {state_hash, step_group}
    end)
    
    Logger.info("Created #{map_size(anchor_groups)} anchor state groups, covering #{
      anchor_groups |> Map.values() |> Enum.map(& &1.group_size) |> Enum.sum()
    } steps")
    
    anchor_groups
  end
  
  defp compute_step_advantages(trajectories, step_groups, config) do
    Logger.info("Computing step-level advantages...")
    
    enhanced_trajectories = Enum.map(trajectories, fn trajectory ->
      enhanced_steps = Enum.map(trajectory.steps, fn step ->
        state_hash = hash_state(step.state, config.state_hash_precision)
        
        step_advantage = case Map.get(step_groups, state_hash) do
          nil ->
            # No anchor group for this state (appears only once)
            0.0
          
          step_group ->
            # Compute step advantage using group statistics
            normalization_factor = case config.normalization_factor do
              :std -> step_group.std_return
              :fixed -> 1.0
            end
            
            (step.discounted_return - step_group.mean_return) / normalization_factor
        end
        
        Map.put(step, :step_advantage, step_advantage)
      end)
      
      %{trajectory | steps: enhanced_steps}
    end)
    
    enhanced_trajectories
  end
  
  defp update_policy_with_gigpo(trajectories, state) do
    Logger.info("Updating policy using GiGPO objective...")
    
    # Collect all steps with combined advantages
    training_data = trajectories
    |> Enum.flat_map(fn trajectory ->
      Enum.map(trajectory.steps, fn step ->
        # Combine episode and step advantages
        combined_advantage = trajectory.episode_advantage + 
                            state.config.step_weight * Map.get(step, :step_advantage, 0.0)
        
        %{
          state: step.state,
          action: step.action,
          old_log_prob: step.log_prob,
          advantage: combined_advantage,
          trajectory_id: trajectory.id
        }
      end)
    end)
    
    # Compute policy update using clipped objective
    update_result = compute_policy_gradient_update(training_data, state)
    
    Logger.info("Policy update completed: #{inspect(Map.take(update_result, [:loss, :kl_divergence, :clipped_fraction]))}")
    update_result
  end
  
  # Helper Functions
  
  defp initialize_policy_model(opts) do
    model_name = Keyword.get(opts, :model_name, "qwen2.5-1.5b-instruct")
    
    %{
      name: model_name,
      parameters: initialize_model_parameters(),
      tokenizer: load_tokenizer(model_name),
      generation_config: %{
        max_tokens: 512,
        temperature: 1.0,
        top_p: 0.9
      }
    }
  end
  
  defp initialize_reference_model(opts) do
    # Reference model is typically a copy of the initial policy
    initialize_policy_model(opts)
  end
  
  defp initialize_model_parameters do
    # Simplified parameter initialization
    %{
      embedding_dim: 2048,
      hidden_dim: 2048,
      num_layers: 24,
      vocab_size: 50000
    }
  end
  
  defp load_tokenizer(model_name) do
    # Mock tokenizer implementation
    %{
      vocab_size: 50000,
      pad_token: "<pad>",
      eos_token: "</s>",
      model_name: model_name
    }
  end
  
  defp create_identical_environments(environment_configs, count) do
    Enum.map(1..count, fn _i ->
      %{
        type: environment_configs.type,
        initial_state: environment_configs.initial_state,
        action_space: environment_configs.action_space,
        reward_function: environment_configs.reward_function,
        max_steps: environment_configs.max_steps || 50
      }
    end)
  end
  
  defp generate_action(policy_model, state, task_description) do
    # Mock action generation using quantum-enhanced reasoning
    quantum_state = QuantumClassicalHybrid.quantum_feature_map(
      state_to_features(state), 
      :angle_encoding
    )
    
    # Simplified action sampling
    action_logits = simulate_policy_forward(policy_model, state, task_description)
    action_probs = softmax(action_logits)
    
    selected_action = sample_from_distribution(action_probs)
    log_prob = :math.log(Enum.at(action_probs, selected_action))
    
    %{
      action: format_action(selected_action, state),
      log_prob: log_prob,
      action_probs: action_probs
    }
  end
  
  defp execute_action(environment, state, action) do
    # Mock environment step
    case environment.type do
      :alfworld ->
        execute_alfworld_action(state, action, environment)
      :webshop ->
        execute_webshop_action(state, action, environment)
      :sokoban ->
        execute_sokoban_action(state, action, environment)
      _ ->
        execute_generic_action(state, action, environment)
    end
  end
  
  defp execute_alfworld_action(state, action, environment) do
    # Simplified ALFWorld simulation
    next_state = update_alfworld_state(state, action)
    reward = calculate_alfworld_reward(state, action, next_state)
    done = check_alfworld_termination(next_state, environment)
    
    %{
      next_state: next_state,
      reward: reward,
      done: done,
      info: %{action_valid: true}
    }
  end
  
  defp execute_webshop_action(state, action, environment) do
    # Simplified WebShop simulation
    next_state = update_webshop_state(state, action)
    reward = calculate_webshop_reward(state, action, next_state)
    done = check_webshop_termination(next_state, environment)
    
    %{
      next_state: next_state,
      reward: reward,
      done: done,
      info: %{action_valid: true}
    }
  end
  
  defp execute_sokoban_action(state, action, environment) do
    # Simplified Sokoban simulation
    next_state = update_sokoban_state(state, action)
    reward = calculate_sokoban_reward(state, action, next_state)
    done = check_sokoban_termination(next_state, environment)
    
    %{
      next_state: next_state,
      reward: reward,
      done: done,
      info: %{action_valid: true}
    }
  end
  
  defp execute_generic_action(state, action, environment) do
    # Generic environment simulation
    next_state = Map.update!(state, :step_count, &(&1 + 1))
    reward = :rand.uniform() - 0.5  # Random reward
    done = next_state.step_count >= environment.max_steps
    
    %{
      next_state: next_state,
      reward: reward,
      done: done,
      info: %{action_valid: true}
    }
  end
  
  defp compute_discounted_returns(steps, discount_factor) do
    # Compute discounted returns for each step
    reversed_steps = Enum.reverse(steps)
    
    {enhanced_steps, _} = Enum.reduce(reversed_steps, {[], 0.0}, fn step, {acc_steps, future_return} ->
      discounted_return = step.reward + discount_factor * future_return
      enhanced_step = Map.put(step, :discounted_return, discounted_return)
      {[enhanced_step | acc_steps], discounted_return}
    end)
    
    enhanced_steps
  end
  
  defp normalize_state(state, config) do
    # Create a normalized, hashable representation of the state
    case is_map(state) do
      true ->
        state
        |> Map.take([:observation, :inventory, :location, :step_count])
        |> normalize_map_values(config.state_hash_precision)
      
      false ->
        %{raw_state: state}
    end
  end
  
  defp normalize_map_values(map, precision) when is_map(map) do
    Map.new(map, fn {key, value} ->
      normalized_value = case value do
        v when is_float(v) -> Float.round(v, precision)
        v when is_binary(v) -> String.slice(v, 0, 100)  # Truncate long strings
        v when is_list(v) -> Enum.take(v, 10)  # Limit list length
        v -> v
      end
      {key, normalized_value}
    end)
  end
  
  defp hash_state(state, precision) do
    # Create a hash of the normalized state for grouping
    state
    |> normalize_map_values(precision)
    |> :erlang.term_to_binary()
    |> then(&:crypto.hash(:sha256, &1))
    |> Base.encode16()
  end
  
  defp compute_policy_gradient_update(training_data, state) do
    # Simplified policy gradient computation
    total_loss = 0.0
    total_kl = 0.0
    clipped_count = 0
    total_count = length(training_data)
    
    update_stats = Enum.reduce(training_data, %{loss: 0.0, kl: 0.0, clipped: 0}, fn data_point, acc ->
      # Compute importance sampling ratio
      new_log_prob = compute_new_log_prob(state.policy_model, data_point.state, data_point.action)
      ratio = :math.exp(new_log_prob - data_point.old_log_prob)
      
      # Clipped objective
      clipped_ratio = clip_ratio(ratio, state.config.clip_epsilon)
      policy_loss = -min(ratio * data_point.advantage, clipped_ratio * data_point.advantage)
      
      # KL divergence (simplified)
      kl_div = abs(new_log_prob - data_point.old_log_prob)
      
      # Update statistics
      is_clipped = if abs(ratio - clipped_ratio) > 1.0e-8, do: 1, else: 0
      
      %{
        loss: acc.loss + policy_loss,
        kl: acc.kl + kl_div,
        clipped: acc.clipped + is_clipped
      }
    end)
    
    # Final metrics
    %{
      loss: update_stats.loss / total_count,
      kl_divergence: update_stats.kl / total_count,
      clipped_fraction: update_stats.clipped / total_count,
      training_samples: total_count,
      learning_rate: state.config.learning_rate
    }
  end
  
  # Utility Functions
  
  defp calculate_success_rate(trajectories) do
    successful = Enum.count(trajectories, & &1.success)
    successful / length(trajectories)
  end
  
  defp calculate_average_return(trajectories) do
    total_return = Enum.sum(Enum.map(trajectories, & &1.total_return))
    total_return / length(trajectories)
  end
  
  defp calculate_average_length(trajectories) do
    total_length = Enum.sum(Enum.map(trajectories, & &1.length))
    total_length / length(trajectories)
  end
  
  defp calculate_step_efficiency(trajectories) do
    # Efficiency = success rate / average steps
    success_rate = calculate_success_rate(trajectories)
    avg_length = calculate_average_length(trajectories)
    if avg_length > 0, do: success_rate / avg_length, else: 0.0
  end
  
  defp calculate_standard_deviation(values, mean) do
    if length(values) <= 1 do
      1.0
    else
      variance = Enum.reduce(values, 0.0, fn value, acc ->
        acc + :math.pow(value - mean, 2)
      end) / (length(values) - 1)
      :math.sqrt(variance)
    end
  end
  
  defp time_operation(fun) do
    start_time = System.monotonic_time(:millisecond)
    result = fun.()
    end_time = System.monotonic_time(:millisecond)
    {result, end_time - start_time}
  end
  
  defp initialize_metrics do
    %{
      iterations: 0,
      total_training_time: 0,
      success_rates: [],
      average_returns: [],
      step_efficiency: [],
      policy_losses: [],
      kl_divergences: []
    }
  end
  
  defp update_training_metrics(metrics, iteration_data) do
    %{metrics |
      iterations: iteration_data.iteration,
      total_training_time: metrics.total_training_time + Map.get(iteration_data, :total_time, 0),
      success_rates: [iteration_data.episode_group.success_rate | metrics.success_rates],
      average_returns: [iteration_data.episode_group.mean_return | metrics.average_returns],
      policy_losses: [iteration_data.policy_update.loss | metrics.policy_losses],
      kl_divergences: [iteration_data.policy_update.kl_divergence | metrics.kl_divergences]
    }
  end
  
  defp extract_episode_metrics(episode_group) do
    %{
      group_size: episode_group.group_size,
      mean_return: episode_group.mean_return,
      std_return: episode_group.std_return,
      success_rate: episode_group.success_rate,
      task_description: String.slice(episode_group.task_description, 0, 50)
    }
  end
  
  defp extract_step_metrics(step_groups) do
    if map_size(step_groups) > 0 do
      group_sizes = step_groups |> Map.values() |> Enum.map(& &1.group_size)
      
      %{
        total_anchor_states: map_size(step_groups),
        average_group_size: Enum.sum(group_sizes) / length(group_sizes),
        max_group_size: Enum.max(group_sizes),
        min_group_size: Enum.min(group_sizes),
        total_grouped_steps: Enum.sum(group_sizes)
      }
    else
      %{
        total_anchor_states: 0,
        average_group_size: 0,
        max_group_size: 0,
        min_group_size: 0,
        total_grouped_steps: 0
      }
    end
  end
  
  # Mock implementations for environment-specific functions
  
  defp update_alfworld_state(state, action) do
    # Mock ALFWorld state update
    Map.update!(state, :step_count, &(&1 + 1))
  end
  
  defp calculate_alfworld_reward(state, action, next_state) do
    # Mock reward calculation
    if String.contains?(action, "take") and :rand.uniform() > 0.7 do
      1.0
    else
      -0.01
    end
  end
  
  defp check_alfworld_termination(state, environment) do
    state.step_count >= environment.max_steps or 
    Map.get(state, :task_completed, false)
  end
  
  defp update_webshop_state(state, action) do
    Map.update!(state, :step_count, &(&1 + 1))
  end
  
  defp calculate_webshop_reward(state, action, next_state) do
    cond do
      String.contains?(action, "buy") -> 10.0
      String.contains?(action, "search") -> 0.1
      true -> -0.01
    end
  end
  
  defp check_webshop_termination(state, environment) do
    state.step_count >= environment.max_steps or
    Map.get(state, :purchase_completed, false)
  end
  
  defp update_sokoban_state(state, action) do
    Map.update!(state, :step_count, &(&1 + 1))
  end
  
  defp calculate_sokoban_reward(state, action, next_state) do
    # Mock Sokoban reward
    if Map.get(next_state, :boxes_on_goals, 0) > Map.get(state, :boxes_on_goals, 0) do
      1.0
    else
      -0.01
    end
  end
  
  defp check_sokoban_termination(state, environment) do
    state.step_count >= environment.max_steps or
    Map.get(state, :all_boxes_placed, false)
  end
  
  defp state_to_features(state) do
    # Convert state to numerical features for quantum processing
    features = case state do
      %{observation: obs} when is_binary(obs) ->
        # Convert text observation to numerical features
        obs |> String.codepoints() |> Enum.take(10) |> Enum.map(&(String.to_charlist(&1) |> hd() |> Kernel./(128)))
      
      %{step_count: count} ->
        [count / 50.0]  # Normalize step count
      
      _ ->
        [0.0, 0.0, 0.0]  # Default features
    end
    
    # Ensure we have a consistent feature vector length
    features
    |> Enum.take(8)
    |> Kernel.++(List.duplicate(0.0, 8))
    |> Enum.take(8)
  end
  
  defp simulate_policy_forward(policy_model, state, task_description) do
    # Mock policy forward pass
    features = state_to_features(state)
    task_embedding = task_description |> String.codepoints() |> Enum.take(5) |> length() |> Kernel./(10)
    
    # Simple linear combination with random weights
    Enum.map(0..9, fn i ->
      base_logit = Enum.at(features, rem(i, length(features))) + task_embedding
      base_logit + (:rand.uniform() - 0.5) * 0.1
    end)
  end
  
  defp softmax(logits) do
    max_logit = Enum.max(logits)
    shifted_logits = Enum.map(logits, &(&1 - max_logit))
    exp_logits = Enum.map(shifted_logits, &:math.exp/1)
    sum_exp = Enum.sum(exp_logits)
    Enum.map(exp_logits, &(&1 / sum_exp))
  end
  
  defp sample_from_distribution(probs) do
    random_val = :rand.uniform()
    
    {_final_prob, action_index} = Enum.reduce_while(Enum.with_index(probs), {0.0, 0}, 
      fn {prob, index}, {cumulative_prob, _current_index} ->
        new_cumulative = cumulative_prob + prob
        if random_val <= new_cumulative do
          {:halt, {new_cumulative, index}}
        else
          {:cont, {new_cumulative, index}}
        end
      end)
    
    action_index
  end
  
  defp format_action(action_index, state) do
    # Convert action index to environment-specific action format
    actions = [
      "go north", "go south", "go east", "go west",
      "take object", "use object", "examine object",
      "search", "click", "buy"
    ]
    
    Enum.at(actions, rem(action_index, length(actions)))
  end
  
  defp compute_new_log_prob(policy_model, state, action) do
    # Mock new log probability computation
    logits = simulate_policy_forward(policy_model, state, "task")
    probs = softmax(logits)
    
    # Find action index (simplified)
    action_index = rem(:erlang.phash2(action), length(probs))
    action_prob = Enum.at(probs, action_index)
    
    :math.log(max(action_prob, 1.0e-8))
  end
  
  defp clip_ratio(ratio, epsilon) do
    max(min(ratio, 1.0 + epsilon), 1.0 - epsilon)
  end
end