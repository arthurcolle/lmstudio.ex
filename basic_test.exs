IO.puts("🚀 Basic Evolution Demo Started")

# Simple loop with mutations and error correction
defmodule Demo do
  def run(count \\ 0) do
    new_count = count + 1
    
    result = if :rand.uniform() < 0.3 do
      "🔧 Error #{new_count} corrected"
    else
      "✅ Mutation #{new_count} applied"
    end
    
    IO.puts(result)
    
    Process.sleep(1500)
    run(new_count)
  end
end

Demo.run()