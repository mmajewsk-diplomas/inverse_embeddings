"""Lean SIPIT – only recovery logic, no trajectory / rank evolution / loss history."""
import torch
import torch.nn.functional as F
import time
from typing import Dict, Any, Optional, List
from transformers import PreTrainedModel, PreTrainedTokenizer
from transformers.cache_utils import DynamicCache


def _fast_dists(
    embedding_matrix: torch.Tensor,
    vector: torch.Tensor,
    metric: str,
    emb_norms_sq: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if metric == 'l2':
        if emb_norms_sq is not None:
            dots = embedding_matrix.mv(vector)
            return emb_norms_sq - 2.0 * dots + vector.dot(vector)
        diff = embedding_matrix - vector
        return (diff * diff).sum(dim=-1)
    elif metric == 'l1':
        return (embedding_matrix - vector).abs().sum(dim=-1)
    elif metric == 'cosine':
        return 1.0 - F.cosine_similarity(embedding_matrix, vector.unsqueeze(0), dim=-1)
    elif metric == 'dot':
        return -embedding_matrix.mv(vector)
    else:
        raise ValueError(f"Nieznana metryka: {metric}")


def _freeze_cache(past_key_values):
    """Extract detached (key, value) tuples from a DynamicCache or tuple-based cache."""
    if past_key_values is None:
        return None
    return [
        (layer[0].detach(), layer[1].detach())
        for layer in past_key_values
    ]


def _make_cache(frozen):
    """Build a fresh DynamicCache from frozen (key, value) tuples."""
    if frozen is None:
        return None
    return DynamicCache([(k, v, None) for k, v in frozen])


def _expand_cache(frozen, batch_size):
    """Build a batched DynamicCache from frozen (key, value) tuples."""
    if frozen is None:
        return None
    return DynamicCache([
        (k.expand(batch_size, -1, -1, -1), v.expand(batch_size, -1, -1, -1), None)
        for k, v in frozen
    ])


def sipit(
    model: PreTrainedModel,
    target_hidden_states: torch.Tensor,
    target_ids: torch.Tensor,
    tokenizer: PreTrainedTokenizer,
    lr: float = 0.05,
    steps: int = 500,
    loss_th: float = 1e-4,
    loss_th_hard: float = 1e-8,
    max_candidates: int = 2000,
    reg_weight: float = 1e-5,
    discrete_check_interval: int = 10,
    verbose: bool = False,
    metric: str = 'l1',
    candidate_batch_size: int = 64,
) -> Dict[str, Any]:

    device = next(model.parameters()).device
    embedding_matrix = model.transformer.wte.weight.detach()
    vocab_size, emb_dim = embedding_matrix.shape
    seq_len = target_hidden_states.shape[0]

    emb_norms_sq = None
    if metric == 'l2':
        emb_norms_sq = (embedding_matrix * embedding_matrix).sum(dim=-1)

    single_input = torch.zeros((1, 1), dtype=torch.long, device=device)

    recovered_ids: List[int] = []
    frozen_cache = None
    per_token_times: List[float] = []
    prompt_start_time = time.perf_counter()

    for t in range(seq_len):
        token_start_time = time.perf_counter()
        target_h = target_hidden_states[t].to(device)

        proxy_emb = torch.zeros((1, 1, emb_dim), device=device, requires_grad=True)
        optimizer = torch.optim.Adam([proxy_emb], lr=lr, weight_decay=reg_weight)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)

        banned_mask = torch.zeros(vocab_size, dtype=torch.bool, device=device)
        found_token: Optional[int] = None
        prev_loss = float('inf')

        for step in range(steps):
            optimizer.zero_grad(set_to_none=True)

            out = model(
                inputs_embeds=proxy_emb,
                past_key_values=_make_cache(frozen_cache),
                use_cache=False,
                output_hidden_states=True,
            )
            h_pred = out.hidden_states[-1][0, -1, :]

            loss = F.mse_loss(h_pred, target_h)
            loss_val = loss.item()
            prev_loss = loss_val

            if loss_val < loss_th:
                break

            if step > 0 and step % discrete_check_interval == 0:
                with torch.no_grad():
                    current_vec = proxy_emb.data.squeeze()
                    dists = _fast_dists(embedding_matrix, current_vec, metric, emb_norms_sq)
                    dists[banned_mask] = float('inf')
                    closest_cand = dists.argmin().item()

                    single_input[0, 0] = closest_cand
                    out_test = model(
                        input_ids=single_input,
                        past_key_values=_make_cache(frozen_cache),
                        use_cache=False,
                        output_hidden_states=True,
                    )
                    h_test = out_test.hidden_states[-1][0, -1, :]
                    hard_error = F.mse_loss(h_test, target_h).item()

                    if hard_error < loss_th_hard:
                        found_token = closest_cand
                        if verbose:
                            print(f"Token {t+1}: Early stop at step {step}. "
                                  f"Found token '{tokenizer.decode([found_token])}'")
                        break
                    elif hard_error < prev_loss:
                        proxy_emb.data.copy_(
                            embedding_matrix[closest_cand].view(1, 1, -1)
                        )
                        prev_loss = hard_error
                        continue
                    else:
                        banned_mask[closest_cand] = True

            loss.backward()
            torch.nn.utils.clip_grad_norm_([proxy_emb], max_norm=1.0)
            optimizer.step()
            scheduler.step()

        # --- batched candidate verification ---
        if found_token is None:
            with torch.no_grad():
                optimized_vec = proxy_emb.data.squeeze()
                dists = _fast_dists(embedding_matrix, optimized_vec, metric, emb_norms_sq)
                dists[banned_mask] = float('inf')
                _, top_idx = dists.topk(max_candidates, largest=False)
                candidates = top_idx.tolist()

            found_token = candidates[0]

            for batch_start in range(0, len(candidates), candidate_batch_size):
                batch_ids = candidates[batch_start:batch_start + candidate_batch_size]
                batch_input = torch.tensor(
                    batch_ids, dtype=torch.long, device=device
                ).unsqueeze(1)

                with torch.no_grad():
                    out_batch = model(
                        input_ids=batch_input,
                        past_key_values=_expand_cache(frozen_cache, len(batch_ids)),
                        use_cache=False,
                        output_hidden_states=True,
                    )
                    h_batch = out_batch.hidden_states[-1][:, -1, :]
                    errors = ((h_batch - target_h) ** 2).mean(dim=-1)

                match_pos = (errors < loss_th).nonzero(as_tuple=False)
                if match_pos.numel() > 0:
                    first = match_pos[0, 0].item()
                    found_token = batch_ids[first]
                    if verbose:
                        print(f"Token {t+1}: Found candidate "
                              f"{tokenizer.decode([found_token])} at rank "
                              f"{batch_start + first} (post-opt)")
                    break

        recovered_ids.append(found_token)

        # --- KV cache update ---
        with torch.no_grad():
            single_input[0, 0] = found_token
            out_final = model(
                input_ids=single_input,
                past_key_values=_make_cache(frozen_cache),
                use_cache=True,
            )
            frozen_cache = _freeze_cache(out_final.past_key_values)

        per_token_times.append(time.perf_counter() - token_start_time)

    total_time_s = time.perf_counter() - prompt_start_time
    return {
        "recovered_ids": recovered_ids,
        "recovered_text": tokenizer.decode(recovered_ids),
        "per_token_times": per_token_times,
        "total_time_s": total_time_s,
    }
