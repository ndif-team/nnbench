"""Tasks: free-form workload code + a dataset (the 2026-07 task decision, design.md §12.8).

A task is one function `task(be, model, m, items, **knobs)`: `be` is the backend interface,
`m` the family's ModelProfile, `items` the task's dataset verbatim (whatever JSON it wants,
no schema), knobs are task-level configuration. Portability comes only from `be` and `m`;
the framework never models what a task is.
"""
