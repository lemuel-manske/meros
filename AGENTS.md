# Conventions

These rules apply across the Meros codebase.

- **Follow existing patterns.** Prefer the repository's current architecture, naming,
  and dependencies over introducing new frameworks or abstractions for a small change.

- **Separate code blocks with blank lines.** Keep a blank line between logical
  steps, branches, loops, and groups of statements whenever it improves readability.
  **When in doubt, always add a blank line.**

- **Keep experimental work lightweight.** Exploratory scripts should expose
  adjustable settings as module-level constants rather than add CLI flags
  unless a command-line interface is explicitly needed.

- **Keep filesystem access behind adapters.** Processing and application code
  should use `Project` and its media/metadata stores instead of constructing
  persistence paths directly. Adapter-specific conventions belong in that
  module's own rules.

- **Keep documentation durable.** Do not update READMEs for every experiment,
  prototype, parameter change, or temporary workflow. Update documentation
  when an architectural convention, public interface, or established workflow
  is intended to remain. A module README must contain rules relevant to that
  module; global rules belong here.

- **Avoid unnecessary validation and scaffolding.** Do not add speculative
  validation layers, extra abstractions, or unrelated cleanup when implementing
  a focused change.

- **Keep experimental changes isolated.** Do not alter established pipelines,
  unrelated scripts, or existing data unless the task calls for it. Before
  editing a shared branch, read its current files and preserve concurrent changes.

- **Tests are not required as permanent repository artifacts for now.** Temporary
  tests may be used during development, but do not add or retain test files
  unless that policy changes.

- **Run `make check` after code changes.** Report accurately when checks
  could not be executed; do not claim validation from code inspection alone.

- **Clean up regularly.** This project is about experimentation and multiple tries
  that may not succeed. Remove unused code, data, and artifacts to keep the repository
  manageable and reduce confusion.|

- **Use visual debugging strategies.** It should be possible for humans to understand
  the behavior of algorithms and data processing, not just with `print` statements
  but also with plots. You can also use `src/meros/sandbox` to add rapid & quick scripts
  to demonstrate things, but use visual debugging on ml and processing modules.

# Code style

Python docs should be like

```python
def my_function(param1: int, param2: str) -> bool:
    """
    This function does something important.
    """
    # Implementation goes here
```

Or at module level:

```python
"""
This module provides utilities for processing images.

It includes functions for resizing, filtering, and transforming images.
"""
```

- Never use `#` for module-level documentation. Use triple quotes instead.

- Never write right after `"""` on the same line. Always start a new line for the docstring content.

# Validation

> `make test`, `make typecheck`, `make lint`, `make format-check`, and `make cli-check` can also run separately. `make format` applies the formatting rules.

```sh
make check
```
