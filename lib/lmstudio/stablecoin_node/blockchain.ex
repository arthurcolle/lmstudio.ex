defmodule LMStudio.StablecoinNode.Blockchain do
  @moduledoc """
  Blockchain implementation for the stablecoin network with
  Proof of Stake consensus and data provider rewards.
  """

  require Logger

  defstruct [
    :chain,
    :height,
    :difficulty,
    :last_block_hash,
    :pending_transactions,
    :balances,
    :validators,
    :stake_pool,
    :genesis_block
  ]

  defmodule Block do
    defstruct [
      :height,
      :timestamp,
      :previous_hash,
      :merkle_root,
      :transactions,
      :oracle_data,
      :validator,
      :signature,
      :hash,
      :nonce,
      :difficulty,
      :block_reward,
      :data_provider_rewards
    ]
  end

  defmodule Transaction do
    defstruct [
      :id,
      :from,
      :to,
      :amount,
      :fee,
      :timestamp,
      :signature,
      :type,
      :data,
      :nonce
    ]
  end

  def new do
    genesis_block = create_genesis_block()
    
    %__MODULE__{
      chain: [genesis_block],
      height: 0,
      difficulty: 4,
      last_block_hash: genesis_block.hash,
      pending_transactions: [],
      balances: %{
        "genesis" => 1_000_000,  # Initial supply
        "stability_fund" => 100_000
      },
      validators: %{},
      stake_pool: 0,
      genesis_block: genesis_block
    }
  end

  def height(blockchain) do
    blockchain.height
  end

  def get_balance(blockchain, address) do
    Map.get(blockchain.balances, address, 0)
  end

  def validate_transaction(blockchain, transaction) do
    cond do
      transaction.type == :data_provider_reward ->
        # Data provider rewards are always valid
        true
        
      transaction.type == :block_reward ->
        # Block rewards are always valid
        true
        
      transaction.type == :stake ->
        # Validate staking transaction
        validate_stake_transaction(blockchain, transaction)
        
      true ->
        # Regular transaction validation
        validate_regular_transaction(blockchain, transaction)
    end
  end

  def mine_block(blockchain, transactions, oracle_data) do
    case create_block(blockchain, transactions, oracle_data) do
      {:ok, block} ->
        case validate_block(blockchain, block) do
          true ->
            new_blockchain = add_block(blockchain, block)
            {:ok, block, new_blockchain}
          false ->
            {:error, :invalid_block}
        end
      {:error, reason} ->
        {:error, reason}
    end
  end

  def add_validator(blockchain, validator_address, stake_amount) do
    if stake_amount >= minimum_stake() do
      new_validators = Map.put(blockchain.validators, validator_address, %{
        stake: stake_amount,
        joined_at: DateTime.utc_now(),
        blocks_produced: 0,
        reputation: 1.0
      })
      
      new_balances = Map.update(blockchain.balances, validator_address, 0, &(&1 - stake_amount))
      new_stake_pool = blockchain.stake_pool + stake_amount
      
      %{blockchain |
        validators: new_validators,
        balances: new_balances,
        stake_pool: new_stake_pool
      }
    else
      {:error, :insufficient_stake}
    end
  end

  def select_validator(blockchain) do
    if map_size(blockchain.validators) == 0 do
      nil
    else
      # Weighted random selection based on stake
      total_stake = blockchain.validators
      |> Enum.map(fn {_addr, validator} -> validator.stake end)
      |> Enum.sum()
      
      target = :rand.uniform() * total_stake
      
      blockchain.validators
      |> Enum.reduce_while({0, nil}, fn {address, validator}, {acc_stake, _} ->
        new_acc = acc_stake + validator.stake
        if new_acc >= target do
          {:halt, {new_acc, address}}
        else
          {:cont, {new_acc, nil}}
        end
      end)
      |> elem(1)
    end
  end

  defp create_genesis_block do
    %Block{
      height: 0,
      timestamp: DateTime.utc_now(),
      previous_hash: "0",
      merkle_root: "",
      transactions: [],
      oracle_data: %{},
      validator: "genesis",
      signature: "",
      hash: calculate_hash("genesis_block"),
      nonce: 0,
      difficulty: 4,
      block_reward: 0,
      data_provider_rewards: []
    }
  end

  defp create_block(blockchain, transactions, oracle_data) do
    validator = select_validator(blockchain)
    
    if validator do
      block_reward = calculate_block_reward(blockchain.height + 1)
      data_provider_rewards = extract_data_provider_rewards(transactions)
      
      # Filter out reward transactions from regular transactions
      regular_transactions = Enum.filter(transactions, fn tx ->
        tx.type not in [:data_provider_reward, :block_reward]
      end)
      
      # Add block reward transaction
      reward_tx = %Transaction{
        id: generate_transaction_id(),
        from: "network",
        to: validator,
        amount: block_reward,
        fee: 0,
        timestamp: DateTime.utc_now(),
        signature: "",
        type: :block_reward,
        data: %{height: blockchain.height + 1},
        nonce: 0
      }
      
      all_transactions = [reward_tx | regular_transactions] ++ data_provider_rewards
      
      block = %Block{
        height: blockchain.height + 1,
        timestamp: DateTime.utc_now(),
        previous_hash: blockchain.last_block_hash,
        merkle_root: calculate_merkle_root(all_transactions),
        transactions: all_transactions,
        oracle_data: oracle_data,
        validator: validator,
        signature: "",
        hash: "",
        nonce: 0,
        difficulty: blockchain.difficulty,
        block_reward: block_reward,
        data_provider_rewards: data_provider_rewards
      }
      
      # Calculate block hash
      block_hash = calculate_block_hash(block)
      signed_block = %{block | hash: block_hash, signature: sign_block(block, validator)}
      
      {:ok, signed_block}
    else
      {:error, :no_validators}
    end
  end

  defp validate_block(blockchain, block) do
    # Validate block structure and content
    cond do
      block.height != blockchain.height + 1 ->
        Logger.error("Invalid block height")
        false
        
      block.previous_hash != blockchain.last_block_hash ->
        Logger.error("Invalid previous hash")
        false
        
      not validate_merkle_root(block) ->
        Logger.error("Invalid merkle root")
        false
        
      not validate_validator(blockchain, block.validator) ->
        Logger.error("Invalid validator")
        false
        
      not validate_block_transactions(blockchain, block.transactions) ->
        Logger.error("Invalid transactions")
        false
        
      true ->
        true
    end
  end

  defp add_block(blockchain, block) do
    # Update balances based on transactions
    new_balances = apply_block_transactions(blockchain.balances, block.transactions)
    
    # Update validator stats
    new_validators = update_validator_stats(blockchain.validators, block.validator)
    
    %{blockchain |
      chain: [block | blockchain.chain],
      height: block.height,
      last_block_hash: block.hash,
      pending_transactions: [],
      balances: new_balances,
      validators: new_validators
    }
  end

  defp validate_regular_transaction(blockchain, transaction) do
    sender_balance = get_balance(blockchain, transaction.from)
    total_amount = transaction.amount + transaction.fee
    
    cond do
      sender_balance < total_amount ->
        false
        
      transaction.amount <= 0 ->
        false
        
      transaction.fee < 0 ->
        false
        
      not valid_signature?(transaction) ->
        false
        
      true ->
        true
    end
  end

  defp validate_stake_transaction(blockchain, transaction) do
    stake_amount = transaction.amount
    sender_balance = get_balance(blockchain, transaction.from)
    
    cond do
      stake_amount < minimum_stake() ->
        false
        
      sender_balance < stake_amount ->
        false
        
      Map.has_key?(blockchain.validators, transaction.from) ->
        false  # Already a validator
        
      true ->
        true
    end
  end

  defp validate_block_transactions(blockchain, transactions) do
    Enum.all?(transactions, &validate_transaction(blockchain, &1))
  end

  defp validate_merkle_root(block) do
    calculated_root = calculate_merkle_root(block.transactions)
    block.merkle_root == calculated_root
  end

  defp validate_validator(blockchain, validator_address) do
    Map.has_key?(blockchain.validators, validator_address)
  end

  defp apply_block_transactions(balances, transactions) do
    Enum.reduce(transactions, balances, fn transaction, acc_balances ->
      case transaction.type do
        :data_provider_reward ->
          # Credit reward to data provider
          Map.update(acc_balances, transaction.to, transaction.amount, &(&1 + transaction.amount))
          
        :block_reward ->
          # Credit block reward to validator
          Map.update(acc_balances, transaction.to, transaction.amount, &(&1 + transaction.amount))
          
        _ ->
          # Regular transaction - debit from sender, credit to receiver
          acc_balances
          |> Map.update(transaction.from, 0, &(&1 - transaction.amount - transaction.fee))
          |> Map.update(transaction.to, 0, &(&1 + transaction.amount))
          |> Map.update("fee_pool", 0, &(&1 + transaction.fee))
      end
    end)
  end

  defp update_validator_stats(validators, validator_address) do
    Map.update(validators, validator_address, %{}, fn validator ->
      %{validator | blocks_produced: validator.blocks_produced + 1}
    end)
  end

  defp calculate_block_reward(height) do
    # Halving every 100,000 blocks
    base_reward = 50
    halvings = div(height, 100_000)
    base_reward / :math.pow(2, halvings)
  end

  defp extract_data_provider_rewards(transactions) do
    Enum.filter(transactions, fn tx -> tx.type == :data_provider_reward end)
  end

  defp calculate_merkle_root(transactions) do
    transaction_hashes = Enum.map(transactions, &calculate_transaction_hash/1)
    build_merkle_tree(transaction_hashes)
  end

  defp build_merkle_tree([]), do: ""
  defp build_merkle_tree([single_hash]), do: single_hash
  defp build_merkle_tree(hashes) do
    paired_hashes = hashes
    |> Enum.chunk_every(2, 2, [List.last(hashes)])  # Duplicate last if odd
    |> Enum.map(fn
      [left, right] -> calculate_hash("#{left}#{right}")
      [single] -> single
    end)
    
    build_merkle_tree(paired_hashes)
  end

  defp calculate_transaction_hash(transaction) do
    data = "#{transaction.from}#{transaction.to}#{transaction.amount}#{transaction.timestamp}#{transaction.nonce}"
    calculate_hash(data)
  end

  defp calculate_block_hash(block) do
    data = "#{block.height}#{block.timestamp}#{block.previous_hash}#{block.merkle_root}#{block.validator}#{block.nonce}"
    calculate_hash(data)
  end

  defp calculate_hash(data) do
    :crypto.hash(:sha256, data) |> Base.encode16(case: :lower)
  end

  defp sign_block(block, validator) do
    # In a real implementation, this would use the validator's private key
    data = "#{block.hash}#{validator}"
    calculate_hash(data)
  end

  defp valid_signature?(_transaction) do
    # Placeholder - would validate cryptographic signature
    true
  end

  defp generate_transaction_id do
    :crypto.strong_rand_bytes(16) |> Base.encode16(case: :lower)
  end

  defp minimum_stake do
    1000  # Minimum 1000 tokens to become a validator
  end
end