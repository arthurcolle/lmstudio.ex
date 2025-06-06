defmodule Lmstudio.MixProject do
  use Mix.Project

  def project do
    [
      app: :lmstudio,
      version: "0.1.0",
      elixir: "~> 1.18",
      start_permanent: Mix.env() == :prod,
      deps: deps(),
      description: "A self-modifying MetaDSL system for interacting with LM Studio",
      package: package()
    ]
  end

  # Run "mix help compile.app" to learn about applications.
  def application do
    [
      extra_applications: [:logger, :inets, :ssl, :crypto, :plug, :cowboy],
      mod: {LMStudio.Application, []}
    ]
  end

  # Run "mix help deps" to learn about dependencies.
  defp deps do
    [
      {:jason, "~> 1.4"},
      {:plug, "~> 1.15"},
      {:plug_cowboy, "~> 2.6"},
      {:httpoison, "~> 2.0"},
      {:websocket_client, "~> 1.5"},
      {:ex_doc, ">= 0.0.0", only: :dev, runtime: false}
    ]
  end

  defp package do
    [
      licenses: ["MIT"],
      links: %{"GitHub" => "https://github.com/arthurcolle/lmstudio"}
    ]
  end
end
