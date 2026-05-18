"""
Perplexity-based defense filter.

Computes perplexity of each retrieved document under a small LM (e.g., GPT-2).
Documents with perplexity above a threshold are flagged as adversarial and
removed from the retrieved context before generation.

Not yet implemented. See CONTRIBUTING.md for how to add a new defense.
"""

raise NotImplementedError("perplexity_filter is not yet implemented.")
