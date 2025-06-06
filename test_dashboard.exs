#!/usr/bin/env elixir

# Test script to verify the dashboard structure
Code.require_file("lib/lmstudio/consensus_monitoring.ex")

alias LMStudio.ConsensusMonitoring

{:ok, monitor} = ConsensusMonitoring.start_link(enable_visualization: true)

# Register a test system
:ok = ConsensusMonitoring.register_system(monitor, "test_system", %{type: :validator})

# Get the dashboard
dashboard = ConsensusMonitoring.get_dashboard(monitor)

IO.puts("Dashboard structure:")
IO.inspect(dashboard, pretty: true)

# Test the specific fields
IO.puts("\nOverview field present? #{Map.has_key?(dashboard, :overview)}")
if Map.has_key?(dashboard, :overview) do
  IO.puts("Overview: #{inspect(dashboard.overview)}")
end

GenServer.stop(monitor)