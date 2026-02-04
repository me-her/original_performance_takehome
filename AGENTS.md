# AGENTS.md - Repository Guide for Coding Agents

This file contains essential information for AI agents working on this performance optimization take-home project.

## Project Overview

This is Anthropic's Original Performance Take-Home - a performance engineering challenge involving optimizing a VLIW SIMD kernel that traverses a binary tree structure. The goal is to minimize clock cycles in the custom simulator.

## Commands

### Testing Commands
- **Run all tests**: `uv run python perf_takehome.py`
- **Run single test**: `uv run python perf_takehome.py Tests.test_kernel_cycles`
- **Run submission validation**: `uv run python tests/submission_tests.py`
- **Run trace generation**: `uv run python perf_takehome.py Tests.test_kernel_trace`

### Development Commands
- **Start trace viewer**: `uv run python watch_trace.py` (opens browser at localhost:8000)
- **Validate submission**: 
  ```bash
  git diff origin/main tests/  # Should be empty
  uv run python tests/submission_tests.py
  ```

### Python Environment
- **Package Manager**: `uv` (required for all commands)
- **Python version**: Python 3.12+ (managed by `uv`)
- **Command**: `uv run python`

## Code Structure

### Key Files
- `perf_takehome.py` - Main implementation with `KernelBuilder.build_kernel()` to optimize
- `problem.py` - Simulator and problem definition (DO NOT MODIFY)
- `tests/frozen_problem.py` - Frozen copy for submission testing
- `tests/submission_tests.py` - Official performance and correctness tests
- `watch_trace.py` - Trace visualization server

### Core Classes
- **KernelBuilder**: Build optimized instructions in `build_kernel()`
- **Machine**: Custom VLIW SIMD simulator
- **Tree/Input**: Data structures for the problem

## Code Style Guidelines

### Imports
- Use standard library imports first, then local imports
- Prefer specific imports over `import *`
- typing imports use `from typing import Any, Literal`

### Type Annotations
- Use Python 3.9+ type hints: `list[int]`, `dict[str, int]`
- Function parameters require type hints
- Return types should be annotated
- Use `Literal` for string unions: `Engine = Literal["alu", "load", "store", "flow"]`

### Naming Conventions
- **Variables**: `snake_case`
- **Functions**: `snake_case` 
- **Classes**: `PascalCase`
- **Constants**: `UPPER_SNAKE_CASE`
- **Private methods**: `_snake_case`

### Code Formatting
- **Indentation**: 4 spaces
- **Line length**: Prefer under 100 characters
- **Spacing**: Around operators, after commas

### Error Handling
- Use assertions for invariants: `assert condition, "message"`
- Raise specific exceptions: `NotImplementedError(f"Unknown op {op}")`
- Include descriptive error messages

## Architecture Guidelines

### VLIW Instruction Format
Instructions are dictionaries mapping engines to slot lists:
```python
{
    "alu": [("+", dest, src1, src2)],
    "load": [("load", dest, addr)],
    "flow": [("pause",)]
}
```

### Available Engines
- **alu**: Scalar arithmetic (12 slots/cycle)
- **valu**: Vector arithmetic (6 slots/cycle) 
- **load**: Memory loads (2 slots/cycle)
- **store**: Memory stores (2 slots/cycle)
- **flow**: Control flow (1 slot/cycle)
- **debug**: Debug operations (64 slots/cycle, ignored in submission)

### Optimization Strategy
- Focus on `KernelBuilder.build_kernel()` method
- Minimize instruction count and maximize parallelism
- Use vector operations when beneficial (VLEN = 8)
- Consider scratch space usage (1536 word limit)
- DO NOT modify anything in `tests/` directory

### Memory Layout
- Memory contains tree values, input indices, and input values
- Header contains metadata at fixed positions
- Use scratch space for temporary variables and constants

## Performance Benchmarks
Target cycle counts (baseline: 147734):
- Updated starter code: <18532 cycles
- Claude Opus 4.5 casual: <1790 cycles  
- Current best: <1363 cycles

## Testing Requirements
- **Correctness**: Must match reference kernel exactly
- **Performance**: Lower cycles = better
- **No test modifications**: `tests/` directory must remain unchanged
- **Submission validation**: Run `uv run python tests/submission_tests.py`

## Debugging Tools
- **Trace visualization**: Use `watch_trace.py` for instruction-level traces
- **Debug comparisons**: Use `debug` engine slots with `compare` operations
- **Value tracing**: Leverage `Machine.value_trace` for reference comparisons

## Important Constraints
- **Single core**: N_CORES = 1 (multicore disabled)
- **Vector length**: VLEN = 8
- **Scratch limit**: 1536 words
- **32-bit arithmetic**: All operations modulo 2**32
- **Hash function**: Cannot modify the `HASH_STAGES` implementation

## File Structure Rules
- `perf_takehome.py`: Main implementation (can modify)
- `problem.py`: Core simulator (read-only reference)
- `tests/`: Frozen for submission (DO NOT MODIFY)
- `watch_trace.html`: Trace viewer UI
- Generated `trace.json`: Trace output (gitignored)

## Best Practices
- Use `KernelBuilder` helper methods (`alloc_scratch`, `scratch_const`)
- Maintain debug information with `debug_info()`
- Consider instruction scheduling for VLIW efficiency
- Profile with trace viewer frequently
- Test with both small and large inputs
- Preserve algorithm correctness while optimizing