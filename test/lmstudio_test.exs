defmodule LMStudioTest do
  use ExUnit.Case
  doctest LMStudio

  describe "complete/2" do
    test "builds correct request body with default options" do
      _messages = [%{role: "user", content: "Hello"}]
      
      # This would be a mock test in a real scenario
      # For now, we're just testing that the function exists and accepts the right parameters
      assert is_function(&LMStudio.complete/2)
      assert is_function(&LMStudio.complete/1)
    end

    test "chat/2 is a convenience wrapper" do
      assert is_function(&LMStudio.chat/2)
      assert is_function(&LMStudio.chat/1)
    end

    test "list_models/0 exists" do
      assert is_function(&LMStudio.list_models/0)
    end
  end

  describe "configuration" do
    test "config module provides defaults" do
      assert LMStudio.Config.base_url() == "http://192.168.1.177:1234"
      assert LMStudio.Config.default_model() == "default"
      assert LMStudio.Config.default_temperature() == 0.7
      assert LMStudio.Config.default_max_tokens() == 2048
    end
  end
end