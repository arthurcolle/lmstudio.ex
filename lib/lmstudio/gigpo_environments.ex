defmodule LMStudio.GiGPOEnvironments do
  @moduledoc """
  Environment implementations and utilities for GiGPO training.
  
  Provides realistic environment simulations for:
  - ALFWorld (embodied household tasks)
  - WebShop (e-commerce interactions)  
  - Sokoban (puzzle solving)
  - Custom environments
  
  Each environment supports the GiGPO training paradigm with:
  - Deterministic state representations for anchor state grouping
  - Appropriate reward structures (sparse/dense)
  - Realistic action spaces and dynamics
  """
  
  require Logger
  
  defmodule ALFWorldEnvironment do
    @moduledoc "ALFWorld embodied task environment simulation"
    
    defstruct [
      :task_type,
      :rooms,
      :objects,
      :inventory,
      :current_room,
      :target_object,
      :target_location,
      :step_count,
      :max_steps,
      :completed
    ]
    
    def new(task_description, opts \\ []) do
      {task_type, target_object, target_location} = parse_task_description(task_description)
      
      %__MODULE__{
        task_type: task_type,
        rooms: initialize_rooms(),
        objects: initialize_objects(task_type),
        inventory: [],
        current_room: "kitchen",
        target_object: target_object,
        target_location: target_location,
        step_count: 0,
        max_steps: Keyword.get(opts, :max_steps, 50),
        completed: false
      }
    end
    
    def step(env, action) do
      case parse_action(action) do
        {:go, room} ->
          execute_movement(env, room)
        
        {:take, object} ->
          execute_take(env, object)
        
        {:put, object, location} ->
          execute_put(env, object, location)
        
        {:heat, object} ->
          execute_heat(env, object)
        
        {:cool, object} ->
          execute_cool(env, object)
        
        {:clean, object} ->
          execute_clean(env, object)
        
        {:examine, object} ->
          execute_examine(env, object)
        
        :invalid ->
          execute_invalid_action(env)
      end
    end
    
    def get_observation(env) do
      room_objects = get_room_objects(env, env.current_room)
      inventory_text = if length(env.inventory) > 0 do
        " You are carrying: #{Enum.join(env.inventory, ", ")}."
      else
        " Your hands are empty."
      end
      
      "You are in the #{env.current_room}. You can see: #{Enum.join(room_objects, ", ")}.#{inventory_text}"
    end
    
    def get_valid_actions(env) do
      movement_actions = get_movement_actions(env)
      object_actions = get_object_actions(env)
      
      movement_actions ++ object_actions ++ ["examine room", "inventory"]
    end
    
    def is_completed?(env) do
      env.completed or check_task_completion(env)
    end
    
    # Private helper functions
    
    defp parse_task_description(description) do
      cond do
        String.contains?(description, "heat") and String.contains?(description, "egg") ->
          {:heat_and_place, "egg", "countertop"}
        
        String.contains?(description, "clean") and String.contains?(description, "mug") ->
          {:clean_and_place, "mug", "coffeemachine"}
        
        String.contains?(description, "cool") and String.contains?(description, "apple") ->
          {:cool_and_place, "apple", "countertop"}
        
        String.contains?(description, "examine") and String.contains?(description, "book") ->
          {:examine_with_light, "book", "desklamp"}
        
        true ->
          {:generic_task, "object", "location"}
      end
    end
    
    defp initialize_rooms do
      %{
        "kitchen" => %{
          objects: ["fridge", "countertop", "microwave", "stove", "sink"],
          connections: ["living_room", "dining_room"]
        },
        "living_room" => %{
          objects: ["sofa", "tv", "coffee_table", "lamp"],
          connections: ["kitchen", "bedroom"]
        },
        "bedroom" => %{
          objects: ["bed", "dresser", "desk", "desklamp"],
          connections: ["living_room", "bathroom"]
        },
        "bathroom" => %{
          objects: ["toilet", "sink", "mirror", "bathtub"],
          connections: ["bedroom"]
        },
        "dining_room" => %{
          objects: ["dining_table", "chairs", "cabinet"],
          connections: ["kitchen"]
        }
      }
    end
    
    defp initialize_objects(task_type) do
      base_objects = %{
        "fridge" => %{room: "kitchen", contains: ["apple", "milk", "leftovers"]},
        "countertop" => %{room: "kitchen", contains: ["egg", "mug", "plate"]},
        "microwave" => %{room: "kitchen", contains: []},
        "stove" => %{room: "kitchen", contains: []},
        "desk" => %{room: "bedroom", contains: ["book", "pen"]},
        "desklamp" => %{room: "bedroom", contains: []},
        "coffee_table" => %{room: "living_room", contains: ["remote", "magazine"]},
        "sink" => %{room: "kitchen", contains: []},
        "coffeemachine" => %{room: "kitchen", contains: []}
      }
      
      # Customize objects based on task type
      case task_type do
        :heat_and_place ->
          put_in(base_objects["countertop"].contains, ["egg", "mug", "plate", "spatula"])
        
        :clean_and_place ->
          put_in(base_objects["sink"].contains, ["dirty_mug", "sponge"])
        
        _ ->
          base_objects
      end
    end
    
    defp parse_action(action) do
      action = String.downcase(action)
      
      cond do
        String.contains?(action, "go to") ->
          room = String.replace(action, "go to ", "")
          {:go, room}
        
        String.contains?(action, "take") ->
          object = String.replace(action, ~r/take\s+/, "") |> String.replace(~r/\s+from.*/, "")
          {:take, object}
        
        String.contains?(action, "put") ->
          parts = String.split(action, " ")
          object = Enum.at(parts, 1, "")
          location = List.last(parts)
          {:put, object, location}
        
        String.contains?(action, "heat") ->
          object = String.replace(action, ~r/heat\s+/, "")
          {:heat, object}
        
        String.contains?(action, "cool") ->
          object = String.replace(action, ~r/cool\s+/, "")
          {:cool, object}
        
        String.contains?(action, "clean") ->
          object = String.replace(action, ~r/clean\s+/, "")
          {:clean, object}
        
        String.contains?(action, "examine") ->
          object = String.replace(action, ~r/examine\s+/, "")
          {:examine, object}
        
        true ->
          :invalid
      end
    end
    
    defp execute_movement(env, target_room) do
      current_room_data = Map.get(env.rooms, env.current_room)
      
      if target_room in current_room_data.connections do
        new_env = %{env | current_room: target_room, step_count: env.step_count + 1}
        {new_env, 0.0, false, "You move to the #{target_room}."}
      else
        {%{env | step_count: env.step_count + 1}, -0.1, false, "You cannot go to #{target_room} from here."}
      end
    end
    
    defp execute_take(env, object) do
      room_objects = get_available_objects(env, env.current_room)
      
      if object in room_objects do
        new_inventory = [object | env.inventory]
        new_objects = remove_object_from_room(env.objects, object, env.current_room)
        new_env = %{env | inventory: new_inventory, objects: new_objects, step_count: env.step_count + 1}
        {new_env, 0.1, false, "You take the #{object}."}
      else
        {%{env | step_count: env.step_count + 1}, -0.1, false, "You cannot take #{object}."}
      end
    end
    
    defp execute_put(env, object, location) do
      if object in env.inventory do
        location_available = location in get_room_objects(env, env.current_room)
        
        if location_available do
          new_inventory = List.delete(env.inventory, object)
          new_objects = add_object_to_location(env.objects, object, location)
          new_env = %{env | inventory: new_inventory, objects: new_objects, step_count: env.step_count + 1}
          
          # Check if this completes the task
          task_completed = check_task_completion(new_env)
          reward = if task_completed, do: 10.0, else: 1.0
          
          {%{new_env | completed: task_completed}, reward, task_completed, "You put the #{object} in/on the #{location}."}
        else
          {%{env | step_count: env.step_count + 1}, -0.1, false, "You cannot put #{object} there."}
        end
      else
        {%{env | step_count: env.step_count + 1}, -0.1, false, "You are not carrying #{object}."}
      end
    end
    
    defp execute_heat(env, object) do
      if object in env.inventory and env.current_room == "kitchen" do
        # Check if microwave or stove is available
        can_heat = "microwave" in get_room_objects(env, "kitchen") or "stove" in get_room_objects(env, "kitchen")
        
        if can_heat do
          # Heat the object (mark it as heated)
          heated_object = "heated_#{object}"
          new_inventory = List.delete(env.inventory, object) |> List.insert_at(0, heated_object)
          new_env = %{env | inventory: new_inventory, step_count: env.step_count + 1}
          {new_env, 0.5, false, "You heat the #{object}."}
        else
          {%{env | step_count: env.step_count + 1}, -0.1, false, "You need a microwave or stove to heat #{object}."}
        end
      else
        {%{env | step_count: env.step_count + 1}, -0.1, false, "You cannot heat #{object}."}
      end
    end
    
    defp execute_cool(env, object) do
      if object in env.inventory and "fridge" in get_room_objects(env, env.current_room) do
        cooled_object = "cooled_#{object}"
        new_inventory = List.delete(env.inventory, object) |> List.insert_at(0, cooled_object)
        new_env = %{env | inventory: new_inventory, step_count: env.step_count + 1}
        {new_env, 0.5, false, "You cool the #{object} in the fridge."}
      else
        {%{env | step_count: env.step_count + 1}, -0.1, false, "You cannot cool #{object} here."}
      end
    end
    
    defp execute_clean(env, object) do
      if object in env.inventory and "sink" in get_room_objects(env, env.current_room) do
        cleaned_object = "clean_#{object}"
        new_inventory = List.delete(env.inventory, object) |> List.insert_at(0, cleaned_object)
        new_env = %{env | inventory: new_inventory, step_count: env.step_count + 1}
        {new_env, 0.5, false, "You clean the #{object} in the sink."}
      else
        {%{env | step_count: env.step_count + 1}, -0.1, false, "You cannot clean #{object} here."}
      end
    end
    
    defp execute_examine(env, object) do
      if object in get_room_objects(env, env.current_room) do
        # Special case for examine with desklamp
        if object == "book" and "desklamp" in get_room_objects(env, env.current_room) do
          new_env = %{env | completed: true, step_count: env.step_count + 1}
          {new_env, 10.0, true, "You examine the book under the desklamp and complete your task."}
        else
          {%{env | step_count: env.step_count + 1}, 0.1, false, "You examine the #{object}."}
        end
      else
        {%{env | step_count: env.step_count + 1}, -0.1, false, "You cannot examine #{object} here."}
      end
    end
    
    defp execute_invalid_action(env) do
      {%{env | step_count: env.step_count + 1}, -0.1, false, "Invalid action."}
    end
    
    defp get_room_objects(env, room) do
      env.rooms[room].objects ++ get_available_objects(env, room)
    end
    
    defp get_available_objects(env, room) do
      env.objects
      |> Enum.filter(fn {_name, data} -> data.room == room end)
      |> Enum.flat_map(fn {_name, data} -> data.contains end)
    end
    
    defp get_movement_actions(env) do
      current_room_data = Map.get(env.rooms, env.current_room)
      Enum.map(current_room_data.connections, &("go to #{&1}"))
    end
    
    defp get_object_actions(env) do
      room_objects = get_available_objects(env, env.current_room)
      inventory_actions = Enum.map(env.inventory, &("put #{&1}"))
      take_actions = Enum.map(room_objects, &("take #{&1}"))
      
      take_actions ++ inventory_actions ++ ["heat object", "cool object", "clean object"]
    end
    
    defp check_task_completion(env) do
      case env.task_type do
        :heat_and_place ->
          heated_egg_placed = "heated_egg" in get_available_objects(env, "kitchen") and
                             "countertop" in get_room_objects(env, "kitchen")
          heated_egg_placed
        
        :clean_and_place ->
          clean_mug_placed = "clean_mug" in get_available_objects(env, "kitchen") and
                            "coffeemachine" in get_room_objects(env, "kitchen")
          clean_mug_placed
        
        :cool_and_place ->
          cooled_apple_placed = "cooled_apple" in get_available_objects(env, "kitchen") and
                               "countertop" in get_room_objects(env, "kitchen")
          cooled_apple_placed
        
        _ ->
          false
      end
    end
    
    defp remove_object_from_room(objects, object, room) do
      Map.new(objects, fn {name, data} ->
        if data.room == room do
          {name, %{data | contains: List.delete(data.contains, object)}}
        else
          {name, data}
        end
      end)
    end
    
    defp add_object_to_location(objects, object, location) do
      Map.update(objects, location, %{room: "unknown", contains: [object]}, fn data ->
        %{data | contains: [object | data.contains]}
      end)
    end
  end
  
  defmodule WebShopEnvironment do
    @moduledoc "WebShop e-commerce environment simulation"
    
    defstruct [
      :search_query,
      :current_page,
      :search_results,
      :selected_item,
      :cart,
      :step_count,
      :max_steps,
      :purchase_completed,
      :target_criteria
    ]
    
    def new(task_description, opts \\ []) do
      criteria = parse_shopping_criteria(task_description)
      
      %__MODULE__{
        search_query: "",
        current_page: "search",
        search_results: [],
        selected_item: nil,
        cart: [],
        step_count: 0,
        max_steps: Keyword.get(opts, :max_steps, 15),
        purchase_completed: false,
        target_criteria: criteria
      }
    end
    
    def step(env, action) do
      case parse_shopping_action(action) do
        {:search, query} ->
          execute_search(env, query)
        
        {:click, item} ->
          execute_click(env, item)
        
        {:select_option, option} ->
          execute_select_option(env, option)
        
        {:add_to_cart} ->
          execute_add_to_cart(env)
        
        {:buy_now} ->
          execute_buy_now(env)
        
        {:back} ->
          execute_back(env)
        
        :invalid ->
          execute_invalid_shopping_action(env)
      end
    end
    
    def get_observation(env) do
      case env.current_page do
        "search" ->
          "Search page. Enter your search query."
        
        "results" ->
          format_search_results(env.search_results)
        
        "product" ->
          format_product_page(env.selected_item)
        
        "cart" ->
          format_cart_page(env.cart)
        
        _ ->
          "Unknown page"
      end
    end
    
    def get_valid_actions(env) do
      case env.current_page do
        "search" ->
          ["search[#{generate_search_suggestion(env.target_criteria)}]"]
        
        "results" ->
          item_actions = Enum.map(env.search_results, fn item ->
            "click[#{item.id}]"
          end)
          item_actions ++ ["search[new query]", "next page", "previous page"]
        
        "product" ->
          ["add to cart", "buy now", "back", "select color", "select size"]
        
        "cart" ->
          ["buy now", "continue shopping", "remove item"]
        
        _ ->
          ["back"]
      end
    end
    
    def is_completed?(env) do
      env.purchase_completed
    end
    
    # Private helper functions
    
    defp parse_shopping_criteria(description) do
      # Extract shopping criteria from task description
      %{
        product_type: extract_product_type(description),
        color: extract_color(description),
        size: extract_size(description),
        price_limit: extract_price_limit(description),
        features: extract_features(description)
      }
    end
    
    defp parse_shopping_action(action) do
      action = String.downcase(action)
      
      cond do
        String.contains?(action, "search[") ->
          query = Regex.run(~r/search\[(.*?)\]/, action, capture: :all_but_first) |> List.first()
          {:search, query}
        
        String.contains?(action, "click[") ->
          item = Regex.run(~r/click\[(.*?)\]/, action, capture: :all_but_first) |> List.first()
          {:click, item}
        
        String.contains?(action, "add to cart") ->
          {:add_to_cart}
        
        String.contains?(action, "buy now") ->
          {:buy_now}
        
        String.contains?(action, "back") ->
          {:back}
        
        String.contains?(action, "select") ->
          option = String.replace(action, "select ", "")
          {:select_option, option}
        
        true ->
          :invalid
      end
    end
    
    defp execute_search(env, query) do
      # Generate mock search results
      results = generate_search_results(query, env.target_criteria)
      
      new_env = %{env |
        search_query: query,
        current_page: "results",
        search_results: results,
        step_count: env.step_count + 1
      }
      
      reward = if length(results) > 0, do: 0.1, else: -0.1
      {new_env, reward, false, "Search completed. Found #{length(results)} results."}
    end
    
    defp execute_click(env, item_id) do
      selected_item = Enum.find(env.search_results, &(&1.id == item_id))
      
      if selected_item do
        new_env = %{env |
          current_page: "product",
          selected_item: selected_item,
          step_count: env.step_count + 1
        }
        {new_env, 0.1, false, "Viewing product: #{selected_item.name}"}
      else
        {%{env | step_count: env.step_count + 1}, -0.1, false, "Item not found."}
      end
    end
    
    defp execute_add_to_cart(env) do
      if env.selected_item do
        new_cart = [env.selected_item | env.cart]
        new_env = %{env | cart: new_cart, step_count: env.step_count + 1}
        {new_env, 0.5, false, "Added #{env.selected_item.name} to cart."}
      else
        {%{env | step_count: env.step_count + 1}, -0.1, false, "No item selected."}
      end
    end
    
    defp execute_buy_now(env) do
      if length(env.cart) > 0 or env.selected_item do
        # Check if purchase meets criteria
        item_to_buy = env.selected_item || hd(env.cart)
        meets_criteria = item_matches_criteria(item_to_buy, env.target_criteria)
        
        reward = if meets_criteria, do: 10.0, else: 0.0
        new_env = %{env | purchase_completed: true, step_count: env.step_count + 1}
        
        message = if meets_criteria do
          "Purchase successful! Item matches your requirements."
        else
          "Purchase completed, but item doesn't fully match requirements."
        end
        
        {new_env, reward, true, message}
      else
        {%{env | step_count: env.step_count + 1}, -0.1, false, "No items to purchase."}
      end
    end
    
    defp execute_back(env) do
      new_page = case env.current_page do
        "product" -> "results"
        "cart" -> "product"
        _ -> "search"
      end
      
      new_env = %{env | current_page: new_page, step_count: env.step_count + 1}
      {new_env, 0.0, false, "Navigated back."}
    end
    
    defp execute_select_option(env, option) do
      # Mock option selection (color, size, etc.)
      {%{env | step_count: env.step_count + 1}, 0.1, false, "Selected #{option}."}
    end
    
    defp execute_invalid_shopping_action(env) do
      {%{env | step_count: env.step_count + 1}, -0.1, false, "Invalid action."}
    end
    
    defp generate_search_results(query, criteria) do
      # Generate realistic product results
      base_products = [
        %{id: "B001", name: "Men's Tuxedo Shirt", price: 29.99, color: "white", size: "L"},
        %{id: "B002", name: "Bluetooth Headphones", price: 79.99, color: "black", size: "one size"},
        %{id: "B003", name: "Running Shoes", price: 129.99, color: "blue", size: "9"},
        %{id: "B004", name: "Coffee Maker", price: 159.99, color: "stainless", size: "12 cup"},
        %{id: "B005", name: "Wireless Mouse", price: 39.99, color: "black", size: "standard"}
      ]
      
      # Filter and modify based on query and criteria
      base_products
      |> Enum.filter(fn product ->
        String.contains?(String.downcase(product.name), String.downcase(query))
      end)
      |> Enum.take(5)
    end
    
    defp item_matches_criteria(item, criteria) do
      price_ok = item.price <= (criteria.price_limit || 1000.0)
      color_ok = criteria.color == nil or String.contains?(String.downcase(item.color), String.downcase(criteria.color || ""))
      size_ok = criteria.size == nil or String.contains?(String.downcase(item.size), String.downcase(criteria.size || ""))
      
      price_ok and color_ok and size_ok
    end
    
    defp extract_product_type(description) do
      cond do
        String.contains?(description, "shirt") -> "shirt"
        String.contains?(description, "headphone") -> "headphones"
        String.contains?(description, "shoes") -> "shoes"
        String.contains?(description, "coffee") -> "coffee maker"
        true -> "generic"
      end
    end
    
    defp extract_color(description) do
      colors = ["blue", "red", "black", "white", "green", "yellow", "grey", "gray"]
      Enum.find(colors, fn color ->
        String.contains?(String.downcase(description), color)
      end)
    end
    
    defp extract_size(description) do
      sizes = ["xs", "s", "m", "l", "xl", "xxl", "small", "medium", "large", "x-large", "xx-large"]
      Enum.find(sizes, fn size ->
        String.contains?(String.downcase(description), size)
      end)
    end
    
    defp extract_price_limit(description) do
      case Regex.run(~r/under\s+\$?(\d+)/, String.downcase(description)) do
        [_, price_str] -> String.to_float(price_str)
        _ -> nil
      end
    end
    
    defp extract_features(description) do
      String.split(description, ",") |> Enum.map(&String.trim/1)
    end
    
    defp generate_search_suggestion(criteria) do
      "#{criteria.product_type} #{criteria.color} #{criteria.size}"
    end
    
    defp format_search_results(results) do
      result_text = Enum.map_join(results, "\n", fn item ->
        "#{item.id}: #{item.name} - $#{item.price}"
      end)
      
      "Search Results:\n#{result_text}\nActions: click[item_id], next page, previous page"
    end
    
    defp format_product_page(item) do
      if item do
        "Product: #{item.name}\nPrice: $#{item.price}\nColor: #{item.color}\nSize: #{item.size}\nActions: add to cart, buy now, back"
      else
        "No product selected"
      end
    end
    
    defp format_cart_page(cart) do
      if length(cart) > 0 do
        cart_text = Enum.map_join(cart, "\n", fn item ->
          "#{item.name} - $#{item.price}"
        end)
        "Shopping Cart:\n#{cart_text}\nActions: buy now, continue shopping"
      else
        "Cart is empty"
      end
    end
  end
  
  # Factory function for creating environments
  def create_environment(type, task_description, opts \\ []) do
    case type do
      :alfworld ->
        ALFWorldEnvironment.new(task_description, opts)
      
      :webshop ->
        WebShopEnvironment.new(task_description, opts)
      
      _ ->
        raise "Unknown environment type: #{type}"
    end
  end
  
  # Environment step wrapper
  def step_environment(env, action) do
    case env do
      %ALFWorldEnvironment{} ->
        ALFWorldEnvironment.step(env, action)
      
      %WebShopEnvironment{} ->
        WebShopEnvironment.step(env, action)
      
      _ ->
        raise "Unknown environment type"
    end
  end
  
  # Get environment observation
  def get_observation(env) do
    case env do
      %ALFWorldEnvironment{} ->
        ALFWorldEnvironment.get_observation(env)
      
      %WebShopEnvironment{} ->
        WebShopEnvironment.get_observation(env)
      
      _ ->
        "Unknown environment"
    end
  end
  
  # Get valid actions
  def get_valid_actions(env) do
    case env do
      %ALFWorldEnvironment{} ->
        ALFWorldEnvironment.get_valid_actions(env)
      
      %WebShopEnvironment{} ->
        WebShopEnvironment.get_valid_actions(env)
      
      _ ->
        []
    end
  end
  
  # Check if environment episode is complete
  def is_completed?(env) do
    case env do
      %ALFWorldEnvironment{} ->
        ALFWorldEnvironment.is_completed?(env)
      
      %WebShopEnvironment{} ->
        WebShopEnvironment.is_completed?(env)
      
      _ ->
        false
    end
  end
end