"""
M1: Centroid-targeted retrieval attack.

Instead of prepending the query verbatim, optimize the adversarial document
to be close to the centroid of a query class embedding. This makes a single
blocker document retrieved across all paraphrases of the same query.

Two variants:
  - simple: use the representative (centroid) query during BBO
  - hotflip: gradient-based token flips toward the centroid embedding

Not yet implemented. See CONTRIBUTING.md for how to add a new attack.
"""

raise NotImplementedError("m1_retrieval is not yet implemented.")
