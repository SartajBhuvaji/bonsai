"""Structured pruning: turning importance scores into a smaller model.

`bonsai.importance` scores which components matter; this package acts on
those scores -- depth, width, and (later) expert pruning each get their own
module here, mirroring `bonsai.importance`'s own per-axis split, since the
papers treat each axis as a structurally different operation (2407.14679
Section 2.3). Each pruning function mutates a model in place and returns
nothing, matching plan.md's Phase 2 framing: the deliverable of each step is
"a pruned-but-not-yet-retrained model plus its zero-shot loss" --
`bonsai.importance.depth.mean_loss` (or the equivalent for other axes)
supplies that second half.
"""

from __future__ import annotations
