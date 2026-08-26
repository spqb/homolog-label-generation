from collections.abc import Iterable

import torch


def boundary_special_token_ids(tokenizer) -> list[int]:
    """Return tokenizer IDs that must not contribute to residue pooling."""
    token_ids = {
        token_id
        for token_id in (
            tokenizer.bos_token_id,
            tokenizer.cls_token_id,
            tokenizer.eos_token_id,
            tokenizer.sep_token_id,
            tokenizer.pad_token_id,
        )
        if token_id is not None
    }
    return sorted(token_ids)


def residue_mean_pool(
    last_hidden_state: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    special_token_ids: Iterable[int],
) -> torch.Tensor:
    """Mean-pool amino-acid tokens, excluding padding and boundary tokens."""
    residue_mask = attention_mask.bool()
    for token_id in special_token_ids:
        residue_mask &= input_ids.ne(token_id)

    counts = residue_mask.sum(dim=1)
    if torch.any(counts == 0):
        raise ValueError("Cannot pool a sequence with no residue tokens.")

    weights = residue_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    return (last_hidden_state * weights).sum(dim=1) / weights.sum(dim=1)
