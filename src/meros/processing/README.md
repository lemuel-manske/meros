# Processing

Modules expose a preparation step and reusable image functions. Preparation runs in a fixed order and uses each step's completion check to reuse valid outputs. Rebuilding a step rebuilds the steps after it. Alignment stores transforms; composite construction applies them in memory. The matcher returns comparison statistics.
