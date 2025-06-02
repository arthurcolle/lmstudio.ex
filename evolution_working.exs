IO.puts("🧬 Continuous Evolution Demo Started")

spawn(fn ->
  count = 0
  loop = fn loop_fn, n ->
    new_n = n + 1
    
    result = if :rand.uniform() < 0.3 do
      "🔧 Error #{new_n} corrected"
    else
      "✅ Mutation #{new_n} successful"
    end
    
    IO.puts(result)
    Process.sleep(1500)
    loop_fn.(loop_fn, new_n)
  end
  
  loop.(loop, count)
end)

Process.sleep(30000)
IO.puts("Demo completed successfully!")