# Code Style

- Do not write comments that explain trivial or self-evident code. Only comment where the logic is non-obvious.
- Use precise type annotations. `Callable` alone is not acceptable — always specify signature, e.g. `Callable[[pl.LazyFrame], pl.LazyFrame]` or `Callable[[], pl.Expr]`.
