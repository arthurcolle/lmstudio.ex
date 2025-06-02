defmodule MetaDSLTest do
  use ExUnit.Case, async: false
  doctest LMStudio.MetaDSL

  alias LMStudio.MetaDSL.{SelfModifyingGrid, Mutation, MutationType}
  alias LMStudio.MutationParser

  describe "MutationType" do
    test "has correct types" do
      types = MutationType.types()
      assert length(types) > 0
      assert :append in types
      assert :evolve in types
      assert :replace in types
      assert :delete in types
      assert :insert in types
      assert :compress in types
      assert :expand in types
      assert :merge in types
      assert :fork in types
      assert :mutate_strategy in types
    end

    test "string conversion works for all types" do
      valid_types = ["append", "replace", "delete", "insert", "compress", 
                     "expand", "evolve", "merge", "fork", "mutate_strategy"]
      
      for type_str <- valid_types do
        {:ok, type_atom} = MutationType.from_string(type_str)
        assert type_atom == String.to_atom(type_str)
      end

      {:error, _} = MutationType.from_string("invalid")
    end
  end

  describe "Mutation" do
    test "creation with all parameters" do
      mutation = Mutation.new(:append, "test_target", content: "test content")
      assert mutation.type == :append
      assert mutation.target == "test_target"
      assert mutation.content == "test content"
      assert %DateTime{} = mutation.timestamp
    end

    test "creation with metadata" do
      metadata = %{priority: :high, source: "test"}
      mutation = Mutation.new(:evolve, "strategy", content: "new approach", metadata: metadata)
      assert mutation.metadata == metadata
    end

    test "validates mutation type" do
      assert_raise FunctionClauseError, fn ->
        Mutation.new(:invalid_type, "target", content: "content")
      end
    end
  end

  describe "SelfModifyingGrid advanced operations" do
    setup do
      initial_data = %{
        "knowledge" => "base knowledge",
        "strategies" => ["strategy1", "strategy2"],
        "patterns" => %{"pattern1" => "value1"}
      }
      {:ok, grid_pid} = SelfModifyingGrid.start_link(initial_data: initial_data)
      %{grid: grid_pid, initial_data: initial_data}
    end

    test "append mutation", %{grid: grid_pid} do
      append_mutation = Mutation.new(:append, "knowledge", content: " + new insight")
      {:ok, :mutated} = SelfModifyingGrid.mutate(grid_pid, append_mutation)

      updated_data = SelfModifyingGrid.get_data(grid_pid)
      assert updated_data["knowledge"] == "base knowledge + new insight"
    end

    test "replace mutation", %{grid: grid_pid} do
      replace_mutation = Mutation.new(:replace, "knowledge", content: "completely new knowledge")
      {:ok, :mutated} = SelfModifyingGrid.mutate(grid_pid, replace_mutation)

      updated_data = SelfModifyingGrid.get_data(grid_pid)
      assert updated_data["knowledge"] == "completely new knowledge"
    end

    test "delete mutation", %{grid: grid_pid} do
      delete_mutation = Mutation.new(:delete, "patterns", content: "")
      {:ok, :mutated} = SelfModifyingGrid.mutate(grid_pid, delete_mutation)

      updated_data = SelfModifyingGrid.get_data(grid_pid)
      refute Map.has_key?(updated_data, "patterns")
    end

    test "evolve mutation with complex transformation", %{grid: grid_pid} do
      evolve_mutation = Mutation.new(:evolve, "strategies", 
        content: "transform into advanced strategy system")
      {:ok, :mutated} = SelfModifyingGrid.mutate(grid_pid, evolve_mutation)

      updated_data = SelfModifyingGrid.get_data(grid_pid)
      assert is_binary(updated_data["strategies"])
      assert String.contains?(updated_data["strategies"], "advanced")
    end

    test "concurrent mutations", %{grid: grid_pid} do
      mutations = [
        Mutation.new(:append, "knowledge", content: " concurrent1"),
        Mutation.new(:append, "knowledge", content: " concurrent2"),
        Mutation.new(:append, "knowledge", content: " concurrent3")
      ]

      tasks = for mutation <- mutations do
        Task.async(fn ->
          SelfModifyingGrid.mutate(grid_pid, mutation)
        end)
      end

      results = Task.await_many(tasks, 5000)
      assert Enum.all?(results, &match?({:ok, :mutated}, &1))

      final_data = SelfModifyingGrid.get_data(grid_pid)
      knowledge = final_data["knowledge"]
      assert String.contains?(knowledge, "concurrent1")
      assert String.contains?(knowledge, "concurrent2")
      assert String.contains?(knowledge, "concurrent3")
    end

    test "mutation history tracking", %{grid: grid_pid} do
      mutations = [
        Mutation.new(:append, "knowledge", content: " step1"),
        Mutation.new(:append, "knowledge", content: " step2"),
        Mutation.new(:evolve, "knowledge", content: "evolved knowledge")
      ]

      for mutation <- mutations do
        SelfModifyingGrid.mutate(grid_pid, mutation)
      end

      history = SelfModifyingGrid.get_mutation_history(grid_pid)
      assert length(history) == 3
      assert Enum.all?(history, &match?(%Mutation{}, &1))
    end

    test "rollback capability", %{grid: grid_pid} do
      original_data = SelfModifyingGrid.get_data(grid_pid)
      
      # Apply several mutations
      mutations = [
        Mutation.new(:append, "knowledge", content: " temp1"),
        Mutation.new(:replace, "strategies", content: "temp strategies")
      ]

      for mutation <- mutations do
        SelfModifyingGrid.mutate(grid_pid, mutation)
      end

      # Rollback
      {:ok, :rolled_back} = SelfModifyingGrid.rollback(grid_pid, 2)
      
      rolled_back_data = SelfModifyingGrid.get_data(grid_pid)
      assert rolled_back_data == original_data
    end
  end

  describe "MutationParser advanced parsing" do
    test "parses nested mutations" do
      text = """
      <evolve target="system">
        <append target="knowledge">New insight</append>
        <compress target="old_data">Simplified version</compress>
      </evolve>
      """

      mutations = MutationParser.parse(text)
      assert length(mutations) >= 1
      
      evolve_mutations = Enum.filter(mutations, &(&1.type == :evolve))
      assert length(evolve_mutations) >= 1
    end

    test "handles malformed mutation tags gracefully" do
      text = """
      <append target="knowledge">Valid content</append>
      <invalid_mutation>This should be ignored</invalid_mutation>
      <replace target="data" content="valid replacement"/>
      <append>Missing target</append>
      """

      mutations = MutationParser.parse(text)
      valid_mutations = Enum.filter(mutations, fn mutation ->
        mutation.type in MutationType.types() and 
        is_binary(mutation.target) and 
        byte_size(mutation.target) > 0
      end)
      
      assert length(valid_mutations) == 2  # Only the valid ones
    end

    test "extracts metadata from mutation tags" do
      text = """
      <append target="knowledge" priority="high" source="user">Important insight</append>
      <evolve target="strategy" complexity="medium">New approach</evolve>
      """

      mutations = MutationParser.parse(text)
      assert length(mutations) == 2

      append_mutation = Enum.find(mutations, &(&1.type == :append))
      assert append_mutation.metadata[:priority] == "high"
      assert append_mutation.metadata[:source] == "user"
    end

    test "handles large mutation content" do
      large_content = String.duplicate("Lorem ipsum dolor sit amet. ", 1000)
      text = "<append target=\"large_data\">#{large_content}</append>"

      mutations = MutationParser.parse(text)
      assert length(mutations) == 1
      assert String.length(List.first(mutations).content) > 10000
    end
  end

  describe "performance and stress testing" do
    test "grid handles many mutations efficiently" do
      {:ok, grid_pid} = SelfModifyingGrid.start_link(initial_data: %{})
      
      start_time = System.monotonic_time(:millisecond)
      
      # Apply 1000 mutations
      for i <- 1..1000 do
        mutation = Mutation.new(:append, "data_#{rem(i, 10)}", content: " #{i}")
        SelfModifyingGrid.mutate(grid_pid, mutation)
      end
      
      end_time = System.monotonic_time(:millisecond)
      duration = end_time - start_time
      
      # Should complete in reasonable time (less than 10 seconds)
      assert duration < 10000
      
      # Verify final state
      final_data = SelfModifyingGrid.get_data(grid_pid)
      assert map_size(final_data) <= 10  # Should have data_0 through data_9
    end

    test "mutation parser handles complex nested structures" do
      complex_text = """
      <evolve target="system">
        This is a complex evolution containing:
        <append target="sub1">Nested content 1</append>
        Some text in between
        <replace target="sub2" content="Replacement content"/>
        <compress target="sub3">
          Multi-line
          content with
          special characters: !@#$%^&*()
        </compress>
      </evolve>
      """

      mutations = MutationParser.parse(complex_text)
      assert length(mutations) >= 1
      
      # Should handle the parsing without errors
      assert Enum.all?(mutations, &match?(%Mutation{}, &1))
    end
  end
end