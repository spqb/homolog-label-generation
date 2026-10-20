"""Shared length policy for residue-tokenized protein language models."""

DEFAULT_MAX_RESIDUES = 512
LENGTH_HELP = (
    "Maximum number of residues (default: 512). Shorter inputs use the smallest "
    "power of two covering the longest sequence, capped at this limit. "
    "Model-specific special tokens are added separately."
)


def resolve_token_length(sequences, tokenizer, max_length=DEFAULT_MAX_RESIDUES, *, report=False):
    """Round the residue budget up to a power of two, then apply the user cap."""
    if max_length is None:
        max_length = DEFAULT_MAX_RESIDUES
    if not isinstance(max_length, int) or isinstance(max_length, bool) or max_length <= 0:
        raise ValueError("--max_length must be a positive number of residues.")
    lengths = [len(str(sequence)) for sequence in sequences]
    if not lengths or max(lengths) == 0:
        raise ValueError("Cannot encode a dataset with no residues.")
    special = tokenizer.num_special_tokens_to_add(pair=False)
    rounded_length = 1 << (max(lengths) - 1).bit_length()
    limit = min(rounded_length, max_length) + special
    model_limit = getattr(tokenizer, "model_max_length", None)
    if isinstance(model_limit, int) and model_limit > 0:
        limit = min(limit, model_limit)
    if limit <= special:
        raise ValueError("Tokenizer length limit leaves no room for residues.")
    if report:
        truncated = sum(length > limit - special for length in lengths)
        print(f"Sequence length: longest={max(lengths)} residues; "
              f"using up to {limit - special} residues + {special} special tokens "
              f"({limit} tokens); truncated sequences={truncated}.")
    return limit
