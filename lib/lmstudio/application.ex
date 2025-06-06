defmodule LMStudio.Application do
  @moduledoc """
  Application module for the LMStudio system.
  """

  use Application

  def start(_type, _args) do
    children = [
      # Start a simple HTTP server for basic functionality
      {Plug.Cowboy, scheme: :http, plug: LMStudio.WebInterface, options: [port: 4000]}
    ]

    opts = [strategy: :one_for_one, name: LMStudio.Supervisor]
    Supervisor.start_link(children, opts)
  end
end