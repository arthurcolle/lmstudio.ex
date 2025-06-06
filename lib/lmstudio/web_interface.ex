defmodule LMStudio.WebInterface do
  @moduledoc """
  Simple web interface for LMStudio system.
  """

  use Plug.Router

  plug(:match)
  plug(:dispatch)

  get "/" do
    send_resp(conn, 200, """
    <html>
      <head><title>LMStudio System</title></head>
      <body>
        <h1>LMStudio System</h1>
        <p>Welcome to the LMStudio Self-Modifying MetaDSL System</p>
        <h2>Available Features:</h2>
        <ul>
          <li>Multi-Agent Systems</li>
          <li>Cognitive Agents</li>
          <li>Evolution System</li>
          <li>Neural Architecture</li>
          <li>Quantum Reasoning</li>
          <li>Stablecoin Node</li>
        </ul>
        <h2>API Endpoints:</h2>
        <ul>
          <li><a href="/api/status">GET /api/status</a> - System status</li>
          <li><a href="/api/health">GET /api/health</a> - Health check</li>
        </ul>
      </body>
    </html>
    """)
  end

  get "/api/status" do
    status = %{
      system: "LMStudio",
      version: "0.1.0",
      status: "running",
      timestamp: DateTime.utc_now() |> DateTime.to_iso8601(),
      features: [
        "multi_agent_systems",
        "cognitive_agents", 
        "evolution_system",
        "neural_architecture",
        "quantum_reasoning",
        "stablecoin_node"
      ]
    }

    conn
    |> put_resp_content_type("application/json")
    |> send_resp(200, Jason.encode!(status))
  end

  get "/api/health" do
    health = %{
      status: "healthy",
      timestamp: DateTime.utc_now() |> DateTime.to_iso8601()
    }

    conn
    |> put_resp_content_type("application/json")
    |> send_resp(200, Jason.encode!(health))
  end

  match _ do
    send_resp(conn, 404, "Not found")
  end
end