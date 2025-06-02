# MetaDSL Script Fix

## Problem
When running `elixir self_modifying_metadsl_example.exs`, you get:
```
** (UndefinedFunctionError) function LMStudio.MetaDSL.SelfModifyingGrid.start_link/1 is undefined
```

## Solution
Use `mix run` instead of `elixir` directly:
```bash
mix run self_modifying_metadsl_example.exs
```

## Why This Works
- `elixir` runs scripts without loading compiled project modules
- `mix run` loads the compiled project modules before running the script
- The MetaDSL modules (SelfModifyingGrid, Mutation) are defined in `lib/lmstudio/meta_dsl.ex`

## Additional Fixes Applied
- Updated `demonstrate_single_agent` to handle already-started Registry
- Updated `run_custom_query` to handle already-started Registry

## Quick Test
Run the test script to verify modules are working:
```bash
mix run test_metadsl.exs
```

## Running Options
1. Full demo: `mix run self_modifying_metadsl_example.exs`
2. Interactive mode: `mix run self_modifying_metadsl_example.exs --interactive`
3. Use the helper script: `./run_metadsl_demo.sh`