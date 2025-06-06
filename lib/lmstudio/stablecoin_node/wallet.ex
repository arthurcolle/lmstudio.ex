defmodule LMStudio.StablecoinNode.Wallet do
  @moduledoc """
  Wallet functionality for managing addresses, keys, and transactions.
  """

  require Logger

  defstruct [
    :addresses,
    :private_keys,
    :public_keys,
    :balances,
    :transaction_history,
    :nonces
  ]

  defmodule Address do
    defstruct [
      :address,
      :public_key,
      :private_key,
      :balance,
      :nonce,
      :created_at
    ]
  end

  def new do
    %__MODULE__{
      addresses: %{},
      private_keys: %{},
      public_keys: %{},
      balances: %{},
      transaction_history: %{},
      nonces: %{}
    }
  end

  def generate_address(wallet) do
    # Generate new key pair
    {public_key, private_key} = generate_key_pair()
    address = derive_address(public_key)
    
    address_info = %Address{
      address: address,
      public_key: public_key,
      private_key: private_key,
      balance: 0,
      nonce: 0,
      created_at: DateTime.utc_now()
    }
    
    new_wallet = %{wallet |
      addresses: Map.put(wallet.addresses, address, address_info),
      private_keys: Map.put(wallet.private_keys, address, private_key),
      public_keys: Map.put(wallet.public_keys, address, public_key),
      balances: Map.put(wallet.balances, address, 0),
      transaction_history: Map.put(wallet.transaction_history, address, []),
      nonces: Map.put(wallet.nonces, address, 0)
    }
    
    Logger.info("Generated new address: #{address}")
    {address, new_wallet}
  end

  def import_address(wallet, private_key) do
    case validate_private_key(private_key) do
      {:ok, public_key} ->
        address = derive_address(public_key)
        
        address_info = %Address{
          address: address,
          public_key: public_key,
          private_key: private_key,
          balance: 0,
          nonce: 0,
          created_at: DateTime.utc_now()
        }
        
        new_wallet = %{wallet |
          addresses: Map.put(wallet.addresses, address, address_info),
          private_keys: Map.put(wallet.private_keys, address, private_key),
          public_keys: Map.put(wallet.public_keys, address, public_key),
          balances: Map.put(wallet.balances, address, 0),
          transaction_history: Map.put(wallet.transaction_history, address, []),
          nonces: Map.put(wallet.nonces, address, 0)
        }
        
        Logger.info("Imported address: #{address}")
        {:ok, address, new_wallet}
        
      {:error, reason} ->
        {:error, reason}
    end
  end

  def get_balance(wallet, address) do
    Map.get(wallet.balances, address, 0)
  end

  def update_balance(wallet, address, new_balance) do
    new_balances = Map.put(wallet.balances, address, new_balance)
    %{wallet | balances: new_balances}
  end

  def create_transaction(wallet, from_address, to_address, amount, fee) do
    case validate_transaction_params(wallet, from_address, to_address, amount, fee) do
      :ok ->
        nonce = get_next_nonce(wallet, from_address)
        
        transaction = %{
          id: generate_transaction_id(),
          from: from_address,
          to: to_address,
          amount: amount,
          fee: fee,
          nonce: nonce,
          timestamp: DateTime.utc_now(),
          type: :transfer,
          data: nil,
          signature: nil
        }
        
        # Sign the transaction
        case sign_transaction(wallet, transaction) do
          {:ok, signed_transaction} ->
            # Update wallet state
            new_wallet = %{wallet |
              nonces: Map.put(wallet.nonces, from_address, nonce + 1),
              transaction_history: Map.update(wallet.transaction_history, from_address, [signed_transaction], fn history ->
                [signed_transaction | history]
              end)
            }
            
            {:ok, signed_transaction, new_wallet}
            
          {:error, reason} ->
            {:error, reason}
        end
        
      {:error, reason} ->
        {:error, reason}
    end
  end

  def create_stake_transaction(wallet, validator_address, stake_amount) do
    case Map.keys(wallet.addresses) do
      [] ->
        {:error, :no_addresses}
        
      [from_address | _] ->
        case validate_stake_params(wallet, from_address, stake_amount) do
          :ok ->
            nonce = get_next_nonce(wallet, from_address)
            
            transaction = %{
              id: generate_transaction_id(),
              from: from_address,
              to: validator_address,
              amount: stake_amount,
              fee: 1.0,  # Standard staking fee
              nonce: nonce,
              timestamp: DateTime.utc_now(),
              type: :stake,
              data: %{validator: validator_address},
              signature: nil
            }
            
            case sign_transaction(wallet, transaction) do
              {:ok, signed_transaction} ->
                new_wallet = %{wallet |
                  nonces: Map.put(wallet.nonces, from_address, nonce + 1),
                  transaction_history: Map.update(wallet.transaction_history, from_address, [signed_transaction], fn history ->
                    [signed_transaction | history]
                  end)
                }
                
                {:ok, signed_transaction, new_wallet}
                
              {:error, reason} ->
                {:error, reason}
            end
            
          {:error, reason} ->
            {:error, reason}
        end
    end
  end

  def get_transaction_history(wallet, address) do
    Map.get(wallet.transaction_history, address, [])
  end

  def get_all_addresses(wallet) do
    Map.keys(wallet.addresses)
  end

  def get_total_balance(wallet) do
    wallet.balances
    |> Map.values()
    |> Enum.sum()
  end

  defp validate_transaction_params(wallet, from_address, to_address, amount, fee) do
    cond do
      not Map.has_key?(wallet.addresses, from_address) ->
        {:error, :address_not_found}
        
      amount <= 0 ->
        {:error, :invalid_amount}
        
      fee < 0 ->
        {:error, :invalid_fee}
        
      get_balance(wallet, from_address) < (amount + fee) ->
        {:error, :insufficient_balance}
        
      from_address == to_address ->
        {:error, :same_address}
        
      true ->
        :ok
    end
  end

  defp validate_stake_params(wallet, from_address, stake_amount) do
    cond do
      not Map.has_key?(wallet.addresses, from_address) ->
        {:error, :address_not_found}
        
      stake_amount < 1000 ->  # Minimum stake
        {:error, :insufficient_stake}
        
      get_balance(wallet, from_address) < (stake_amount + 1.0) ->  # Amount + fee
        {:error, :insufficient_balance}
        
      true ->
        :ok
    end
  end

  defp sign_transaction(wallet, transaction) do
    case Map.get(wallet.private_keys, transaction.from) do
      nil ->
        {:error, :private_key_not_found}
        
      private_key ->
        # Create transaction hash for signing
        transaction_hash = create_transaction_hash(transaction)
        
        # Sign the hash
        signature = sign_hash(transaction_hash, private_key)
        
        signed_transaction = %{transaction | signature: signature}
        {:ok, signed_transaction}
    end
  end

  defp create_transaction_hash(transaction) do
    # Create deterministic hash of transaction data (excluding signature)
    data_to_hash = %{
      from: transaction.from,
      to: transaction.to,
      amount: transaction.amount,
      fee: transaction.fee,
      nonce: transaction.nonce,
      timestamp: transaction.timestamp,
      type: transaction.type,
      data: transaction.data
    }
    
    serialized_data = :erlang.term_to_binary(data_to_hash, [:deterministic])
    :crypto.hash(:sha256, serialized_data)
  end

  defp sign_hash(hash, private_key) do
    # Use ECDSA signing (simplified implementation)
    :crypto.sign(:ecdsa, :sha256, hash, [private_key, :secp256k1])
  end

  defp get_next_nonce(wallet, address) do
    Map.get(wallet.nonces, address, 0)
  end

  defp generate_key_pair do
    # Generate ECDSA key pair on secp256k1 curve
    {public_key, private_key} = :crypto.generate_key(:ecdh, :secp256k1)
    {public_key, private_key}
  end

  defp derive_address(public_key) do
    # Create address from public key using Ethereum-style derivation
    public_key_hash = :crypto.hash(:keccak, public_key)
    address_bytes = binary_part(public_key_hash, 12, 20)  # Take last 20 bytes
    "0x" <> Base.encode16(address_bytes, case: :lower)
  end

  defp validate_private_key(private_key) do
    try do
      # Validate by deriving public key
      {public_key, _} = :crypto.generate_key(:ecdh, :secp256k1, private_key)
      {:ok, public_key}
    rescue
      _ -> {:error, :invalid_private_key}
    end
  end

  defp generate_transaction_id do
    :crypto.strong_rand_bytes(32) |> Base.encode16(case: :lower)
  end
end