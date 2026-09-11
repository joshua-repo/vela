"""Turning a run into artifacts on disk.

Layering rule, as in antlia and the prototype: `vela.run` imports the rest of
vela; nothing in vela imports `vela.run`. Provenance is a consumer of the
engine, never a dependency of it.

`vela.run.cli` is deliberately **not** re-exported here. It is an entry point,
not part of the package API, and importing it from this file makes
`python -m vela.run.cli` warn -- runpy finds the module already loaded and
executes it a second time.
"""

from vela.run.artifacts import summary, write_run

__all__ = ["summary", "write_run"]
