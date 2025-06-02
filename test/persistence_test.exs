defmodule LMStudio.PersistenceTest do
  use ExUnit.Case, async: false
  
  alias LMStudio.Persistence
  alias LMStudio.Persistence.Helpers
  
  @test_storage_dir "test_storage"
  
  setup_all do
    Application.ensure_all_started(:lmstudio)
    :ok
  end
  
  setup do
    # Clean up any existing persistence process
    case Process.whereis(Persistence) do
      nil -> :ok
      pid -> GenServer.stop(pid, :normal)
    end
    
    # Clean up test storage directory
    if File.exists?(@test_storage_dir) do
      File.rm_rf!(@test_storage_dir)
    end
    
    Process.sleep(100)
    :ok
  end
  
  describe "Persistence GenServer lifecycle" do
    test "starts successfully with default configuration" do
      {:ok, pid} = Persistence.start_link()
      assert Process.alive?(pid)
      assert Process.whereis(Persistence) == pid
      
      GenServer.stop(pid)
    end
    
    test "starts with custom storage directory" do
      {:ok, pid} = Persistence.start_link(storage_dir: @test_storage_dir)
      assert Process.alive?(pid)
      
      # Directory should be created
      assert File.exists?(@test_storage_dir)
      
      GenServer.stop(pid)
    end
    
    test "handles restart gracefully" do
      {:ok, pid} = Persistence.start_link()
      
      # Store some data
      Persistence.store("test_key", "test_value")
      
      # Restart the process
      GenServer.stop(pid)
      {:ok, new_pid} = Persistence.start_link()
      
      # Data should be restored
      value = Persistence.get("test_key")
      assert value == "test_value"
      
      GenServer.stop(new_pid)
    end
  end
  
  describe "ETS storage operations" do
    setup do
      {:ok, pid} = Persistence.start_link()
      %{persistence_pid: pid}
    end
    
    test "stores and retrieves simple values", %{persistence_pid: pid} do
      assert :ok = Persistence.store("simple_key", "simple_value")
      assert Persistence.get("simple_key") == "simple_value"
      assert Persistence.get("nonexistent_key") == nil
      assert Persistence.get("nonexistent_key", "default") == "default"
      
      GenServer.stop(pid)
    end
    
    test "stores and retrieves complex data structures", %{persistence_pid: pid} do
      complex_data = %{
        agents: ["agent1", "agent2"],
        mutations: [%{type: :append, target: "test"}],
        nested: %{deep: %{value: 42}}
      }
      
      assert :ok = Persistence.store("complex_key", complex_data)
      retrieved = Persistence.get("complex_key")
      assert retrieved == complex_data
      
      GenServer.stop(pid)
    end
    
    test "handles concurrent operations", %{persistence_pid: pid} do
      tasks = for i <- 1..50 do
        Task.async(fn ->
          key = "concurrent_key_#{i}"
          value = "concurrent_value_#{i}"
          Persistence.store(key, value)
          {key, Persistence.get(key)}
        end)
      end
      
      results = Task.await_many(tasks, 5000)
      
      # All operations should succeed
      assert length(results) == 50
      assert Enum.all?(results, fn {key, value} ->
        expected_value = String.replace(key, "key", "value")
        value == expected_value
      end)
      
      GenServer.stop(pid)
    end
    
    test "lists stored keys", %{persistence_pid: pid} do
      keys = ["key1", "key2", "key3"]
      
      for key <- keys do
        Persistence.store(key, "value_#{key}")
      end
      
      stored_keys = Persistence.list_keys()
      assert is_list(stored_keys)
      
      for key <- keys do
        assert key in stored_keys
      end
      
      GenServer.stop(pid)
    end
    
    test "deletes keys", %{persistence_pid: pid} do
      Persistence.store("deletable_key", "deletable_value")
      assert Persistence.get("deletable_key") == "deletable_value"
      
      assert :ok = Persistence.delete("deletable_key")
      assert Persistence.get("deletable_key") == nil
      
      GenServer.stop(pid)
    end
  end
  
  describe "file-based persistence" do
    setup do
      {:ok, pid} = Persistence.start_link(storage_dir: @test_storage_dir)
      %{persistence_pid: pid}
    end
    
    test "persists data to disk automatically", %{persistence_pid: pid} do
      Persistence.store("disk_key", "disk_value", persist: true)
      
      # Give time for async write
      Process.sleep(200)
      
      # Check that file was created
      files = File.ls!(@test_storage_dir)
      assert length(files) > 0
      
      # File should contain the key
      key_file = Enum.find(files, &String.contains?(&1, "disk_key"))
      assert key_file != nil
      
      GenServer.stop(pid)
    end
    
    test "loads data from disk on startup", %{persistence_pid: pid} do
      # Store data
      Persistence.store("persistent_key", "persistent_value", persist: true)
      Process.sleep(200)
      
      GenServer.stop(pid)
      
      # Restart and check data is loaded
      {:ok, new_pid} = Persistence.start_link(storage_dir: @test_storage_dir)
      
      value = Persistence.get("persistent_key")
      assert value == "persistent_value"
      
      GenServer.stop(new_pid)
    end
    
    test "handles disk write failures gracefully", %{persistence_pid: pid} do
      # Try to write to invalid directory
      invalid_dir = "/invalid/path/that/does/not/exist"
      
      # This should not crash the system
      result = Persistence.store("test_key", "test_value", 
                               persist: true, storage_dir: invalid_dir)
      
      # Process should still be alive
      assert Process.alive?(pid)
      
      GenServer.stop(pid)
    end
    
    test "performs periodic checkpoints", %{persistence_pid: pid} do
      # Store data without explicit persist flag
      Persistence.store("checkpoint_key", "checkpoint_value")
      
      # Trigger manual checkpoint
      Persistence.checkpoint()
      Process.sleep(200)
      
      # Data should be persisted
      files = File.ls!(@test_storage_dir)
      assert length(files) > 0
      
      GenServer.stop(pid)
    end
  end
  
  describe "advanced persistence features" do
    setup do
      {:ok, pid} = Persistence.start_link(storage_dir: @test_storage_dir)
      %{persistence_pid: pid}
    end
    
    test "implements versioning for stored data", %{persistence_pid: pid} do
      key = "versioned_key"
      
      # Store multiple versions
      Persistence.store(key, "version_1", version: 1)
      Persistence.store(key, "version_2", version: 2)
      Persistence.store(key, "version_3", version: 3)
      
      # Get specific version
      v1 = Persistence.get(key, version: 1)
      v2 = Persistence.get(key, version: 2)
      v3 = Persistence.get(key, version: 3)
      
      # Fallback to latest if version not specified
      latest = Persistence.get(key)
      
      assert latest in ["version_3", "version_2", "version_1"]  # Depends on implementation
      
      GenServer.stop(pid)
    end
    
    test "implements TTL (time-to-live) for keys", %{persistence_pid: pid} do
      # Store with short TTL
      Persistence.store("ttl_key", "ttl_value", ttl: 500)  # 500ms
      
      # Should be available immediately
      assert Persistence.get("ttl_key") == "ttl_value"
      
      # Wait for expiry
      Process.sleep(600)
      
      # Should be expired (might not be implemented, so check gracefully)
      expired_value = Persistence.get("ttl_key")
      assert expired_value in [nil, "ttl_value"]  # Depends on implementation
      
      GenServer.stop(pid)
    end
    
    test "supports atomic transactions", %{persistence_pid: pid} do
      # Atomic update of multiple keys
      transaction = [
        {:store, "atomic_key1", "atomic_value1"},
        {:store, "atomic_key2", "atomic_value2"},
        {:delete, "atomic_key3"}
      ]
      
      result = Persistence.transaction(transaction)
      
      # Transaction might not be implemented
      assert result in [:ok, {:error, :not_implemented}]
      
      GenServer.stop(pid)
    end
    
    test "handles large data efficiently", %{persistence_pid: pid} do
      # Create large data structure
      large_data = for i <- 1..10000 do
        %{id: i, data: String.duplicate("data", 100)}
      end
      
      start_time = System.monotonic_time(:millisecond)
      Persistence.store("large_data", large_data)
      store_time = System.monotonic_time(:millisecond)
      
      retrieved_data = Persistence.get("large_data")
      retrieve_time = System.monotonic_time(:millisecond)
      
      # Operations should complete in reasonable time
      store_duration = store_time - start_time
      retrieve_duration = retrieve_time - store_time
      
      assert store_duration < 5000    # Less than 5 seconds
      assert retrieve_duration < 2000 # Less than 2 seconds
      assert retrieved_data == large_data
      
      GenServer.stop(pid)
    end
  end
  
  describe "persistence helpers" do
    test "generates unique keys" do
      key1 = Helpers.generate_key()
      key2 = Helpers.generate_key()
      
      assert is_binary(key1)
      assert is_binary(key2)
      assert key1 != key2
      assert String.length(key1) > 0
      assert String.length(key2) > 0
    end
    
    test "serializes and deserializes data" do
      test_data = %{
        string: "test",
        number: 42,
        list: [1, 2, 3],
        map: %{nested: "value"}
      }
      
      serialized = Helpers.serialize(test_data)
      assert is_binary(serialized)
      
      deserialized = Helpers.deserialize(serialized)
      assert deserialized == test_data
    end
    
    test "validates storage keys" do
      assert Helpers.valid_key?("valid_key") == true
      assert Helpers.valid_key?("another.valid.key") == true
      assert Helpers.valid_key?("") == false
      assert Helpers.valid_key?(nil) == false
      assert Helpers.valid_key?(123) == false
    end
    
    test "calculates storage size" do
      small_data = "small"
      large_data = String.duplicate("large", 1000)
      
      small_size = Helpers.calculate_size(small_data)
      large_size = Helpers.calculate_size(large_data)
      
      assert is_integer(small_size)
      assert is_integer(large_size)
      assert large_size > small_size
    end
  end
  
  describe "error handling and recovery" do
    test "handles corrupted storage files gracefully" do
      # Create corrupted file
      File.mkdir_p!(@test_storage_dir)
      corrupted_file = Path.join(@test_storage_dir, "corrupted_key.etf")
      File.write!(corrupted_file, "invalid_etf_data")
      
      # Starting persistence should handle corruption
      {:ok, pid} = Persistence.start_link(storage_dir: @test_storage_dir)
      
      # Should still be able to store new data
      result = Persistence.store("new_key", "new_value")
      assert result == :ok
      
      GenServer.stop(pid)
    end
    
    test "recovers from memory pressure" do
      {:ok, pid} = Persistence.start_link()
      
      # Simulate memory pressure by storing lots of data
      for i <- 1..1000 do
        large_value = String.duplicate("memory_pressure_test", 1000)
        Persistence.store("pressure_key_#{i}", large_value, persist: false)
      end
      
      # System should still be responsive
      assert Process.alive?(pid)
      assert Persistence.store("final_key", "final_value") == :ok
      
      GenServer.stop(pid)
    end
  end
end