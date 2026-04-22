import torch
import torch.nn.functional as F
import time
from typing import List, Dict, Any
from transformers import PreTrainedModel, PreTrainedTokenizer

def compute_distances(matrix: torch.Tensor, vector: torch.Tensor, metric: str) -> torch.Tensor:
    if metric == 'l2':
        return torch.norm(matrix - vector, p=2, dim=-1)
    elif metric == 'l1':
        return torch.norm(matrix - vector, p=1, dim=-1)
    elif metric == 'cosine':
        return 1.0 - F.cosine_similarity(matrix, vector.unsqueeze(0), dim=-1)
    elif metric == 'dot':
        return -torch.matmul(matrix, vector)
    else:
        raise ValueError(f"Nieznana metryka: {metric}")

def detach_past(past_key_values):
    if past_key_values is None: return None
    return tuple(tuple(t.detach() for t in layer) for layer in past_key_values)

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
    rank_check_interval: int = 20,
    discrete_check_interval: int = 10,
    verbose: bool = False,
    metric: str = 'l1'
) -> Dict[str, Any]:
    
    device = next(model.parameters()).device
    embedding_matrix = model.transformer.wte.weight.detach()
    seq_len = target_hidden_states.shape[0]
    
    recovered_ids = []
    past_key_values = None 
    trajectory_data = []
    per_token_times: List[float] = []
    prompt_start_time = time.perf_counter()

    for t in range(seq_len):
        token_start_time = time.perf_counter()
        target_h = target_hidden_states[t].to(device)
        true_token_id = target_ids[t].item()
        
        proxy_emb = torch.zeros((1, 1, model.config.n_embd), device=device, requires_grad=True)
        optimizer = torch.optim.Adam([proxy_emb], lr=lr, weight_decay=reg_weight)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)

        step_losses = []
        vector_history = [] 
        early_stop_step = steps
        
        banned_tokens = set()
        found_token = None
        found_rank = -1
        
        prev_loss = float('inf')

        for step in range(steps):
            optimizer.zero_grad()
            
            if step % rank_check_interval == 0:
                vector_history.append((step, proxy_emb.detach().clone()))

            out = model(inputs_embeds=proxy_emb, past_key_values=past_key_values, use_cache=True, output_hidden_states=True)
            h_pred = out.hidden_states[-1][0, -1, :]
            
            loss = F.mse_loss(h_pred, target_h)
            loss_val = loss.item()
            step_losses.append(loss_val)
            
            prev_loss = loss_val

            if loss_val < loss_th:
                early_stop_step = step
                vector_history.append((step, proxy_emb.detach().clone()))
                break
                
            if step > 0 and step % discrete_check_interval == 0:
                with torch.no_grad():
                    current_vec = proxy_emb.detach().squeeze()
                    dists = compute_distances(embedding_matrix, current_vec, metric=metric)
                    
                    if banned_tokens:
                        dists[list(banned_tokens)] = float('inf')
                        
                    closest_cand = torch.argmin(dists).item()
                    
                    inp_test = torch.tensor([[closest_cand]], device=device)
                    out_test = model(input_ids=inp_test, past_key_values=past_key_values, use_cache=True, output_hidden_states=True)
                    h_test = out_test.hidden_states[-1][0, -1, :]
                    hard_error = F.mse_loss(h_test, target_h).item()

                    if hard_error < loss_th_hard:
                        found_token = closest_cand
                        early_stop_step = step
                        vector_history.append((step, proxy_emb.detach().clone()))
                        if verbose:
                            print(f"Token {t+1}: Early stop at step {step}. Found token '{tokenizer.decode([found_token])}'")
                        break
                    elif hard_error < prev_loss:
                        proxy_emb.data = embedding_matrix[closest_cand].clone().view(1, 1, -1).data
                        prev_loss = hard_error
                        continue
                    else:
                        banned_tokens.add(closest_cand)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_([proxy_emb], max_norm=1.0)
            optimizer.step()
            scheduler.step()

        if found_token is None:
            with torch.no_grad():
                optimized_vec = proxy_emb.detach().squeeze()
                dists = compute_distances(embedding_matrix, optimized_vec, metric=metric)
                
                if banned_tokens:
                    dists[list(banned_tokens)] = float('inf')
                    
                candidates = torch.argsort(dists)[:max_candidates].tolist()

            found_token = candidates[0]
            found_rank = 0

            for rank, cand_id in enumerate(candidates):
                inp = torch.tensor([[cand_id]], device=device)
                with torch.no_grad():
                    out_test = model(input_ids=inp, past_key_values=past_key_values, use_cache=True, output_hidden_states=True)
                    h_test = out_test.hidden_states[-1][0, -1, :]
                    error = F.mse_loss(h_test, target_h).item()

                if error < loss_th:
                    found_token = cand_id
                    found_rank = rank
                    if verbose:
                        print(f"Token {t+1}: Found candidate {tokenizer.decode([cand_id])} at rank {rank} (post-opt)")
                    break
                else:
                    banned_tokens.add(cand_id)

        recovered_ids.append(found_token)

        rank_evolution_winner = []
        rank_evolution_true = [] 

        # with torch.no_grad():
        #     winner_vec = embedding_matrix[found_token]
        #     true_vec = embedding_matrix[true_token_id]

        #     for step_num, hist_vec in vector_history:
        #         hist_vec = hist_vec.squeeze()
                
        #         dists_t = compute_distances(embedding_matrix, hist_vec, metric=metric)
                
        #         dist_to_winner = compute_distances(winner_vec.unsqueeze(0), hist_vec, metric=metric).item()
        #         rank_at_step_winner = (dists_t < dist_to_winner).sum().item()
        #         rank_evolution_winner.append(rank_at_step_winner)

        #         dist_to_true = compute_distances(true_vec.unsqueeze(0), hist_vec, metric=metric).item()
        #         rank_at_step_true = (dists_t < dist_to_true).sum().item()
        #         rank_evolution_true.append(rank_at_step_true)

        if verbose:
            print(f"Token {t+1} | True: '{tokenizer.decode([true_token_id])}' | Rec: '{tokenizer.decode([found_token])}' | Banned: {len(banned_tokens)}")

        with torch.no_grad():
            inp_discrete = torch.tensor([[found_token]], device=device)
            out_discrete = model(input_ids=inp_discrete, past_key_values=past_key_values, use_cache=True, output_hidden_states=True)
            h_discrete = out_discrete.hidden_states[-1][0, -1, :]
            discrete_loss = F.mse_loss(h_discrete, target_h).item()

        token_stats = {
            "token_str": tokenizer.decode([found_token]),
            "true_token_str": tokenizer.decode([true_token_id]),
            "rank_final": found_rank,
            "loss_history": step_losses,
            "discrete_loss": discrete_loss,
            "rank_evolution_winner": rank_evolution_winner, 
            "rank_evolution_true": rank_evolution_true,
            "rank_evolution_steps": [x[0] for x in vector_history],
            "banned_tokens_count": len(banned_tokens)
        }
        trajectory_data.append(token_stats)
        
        with torch.no_grad():
            inp_final = torch.tensor([[found_token]], device=device)
            out_final = model(input_ids=inp_final, past_key_values=past_key_values, use_cache=True)
            past_key_values = detach_past(out_final.past_key_values)

        per_token_times.append(time.perf_counter() - token_start_time)

    total_time_s = time.perf_counter() - prompt_start_time
    return {
        "recovered_ids": recovered_ids,
        "recovered_text": tokenizer.decode(recovered_ids),
        "trajectory": trajectory_data,
        "per_token_times": per_token_times,
        "total_time_s": total_time_s,
    }