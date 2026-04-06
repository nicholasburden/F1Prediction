---
name: Prefer explicit lookups over positional assumptions
description: When accessing items from a collection by key/name, use the explicit key rather than assuming position
type: feedback
---

Use explicit key lookups (e.g. `results["mae"]`) rather than positional assumptions (e.g. `results[0]` or hoping the first element is the right one).

**Why:** The user corrected an approach where the first metric was assumed to be MAE. Implicit ordering is fragile and unclear.

**How to apply:** Whenever retrieving a specific item from a dict or list, use the named key or filter explicitly.
