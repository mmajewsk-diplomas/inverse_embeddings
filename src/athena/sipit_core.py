import torch
import torch.nn.functional as F
from typing import List, Tuple, Dict, Any
from transformers import PreTrainedModel, PreTrainedTokenizer

def detach_past(past_key_values):
    if past_key_values is None: return None
    return tuple(tuple(t.detach() for t in layer) for layer in past_key_values)

def sipit_hpc(
    model: PreTrainedModel,
    target_hidden_states: torch.Tensor,
    tokenizer: PreTrainedTokenizer,
    lr: float = 0.05,
    steps: int = 500,
    loss_th: float = 1e-4,
    max_candidates: int = 2000,
    reg_weight: float = 1e-5,
    verbose: bool = False
) -> Dict[str, Any]:
    
    device = next(model.parameters()).device
    embedding_matrix = model.transformer.wte.weight.detach()
    seq_len = target_hidden_states.shape[0]
    
    recovered_ids = []
    past_key_values = None 
    
    trajectory_data = []

    for t in range(seq_len):
        target_h = target_hidden_states[t].to(device)
        
        proxy_emb = torch.zeros((1, 1, model.config.n_embd), device=device, requires_grad=True)
        
        optimizer = torch.optim.Adam([proxy_emb], lr=lr, weight_decay=reg_weight)
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)

        step_losses = []
        early_stop_step = steps

        for step in range(steps):
            optimizer.zero_grad()
            
            out = model(inputs_embeds=proxy_emb, past_key_values=past_key_values, use_cache=True, output_hidden_states=True)
            h_pred = out.hidden_states[-1][0, -1, :]
            
            loss = F.mse_loss(h_pred, target_h)
            loss_val = loss.item()
            step_losses.append(loss_val)

            if loss_val < loss_th:
                early_stop_step = step
                break
            
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_([proxy_emb], max_norm=1.0)
            
            optimizer.step()
            scheduler.step()

        with torch.no_grad():
            optimized_vec = proxy_emb.detach().squeeze()
            
            cos_sim = F.cosine_similarity(optimized_vec.unsqueeze(0), embedding_matrix)
            
            dists = torch.norm(embedding_matrix - optimized_vec, dim=1)
            candidates = torch.argsort(dists)[:max_candidates].tolist()
            
            final_l2 = torch.norm(optimized_vec).item()
            target_l2 = torch.norm(embedding_matrix[candidates[0]]).item()

        found_token = candidates[0]
        found_rank = 0
        final_verification_loss = 100.0

        for rank, cand_id in enumerate(candidates):
            inp = torch.tensor([[cand_id]], device=device)
            with torch.no_grad():
                out_test = model(input_ids=inp, past_key_values=past_key_values, use_cache=True, output_hidden_states=True)
                h_test = out_test.hidden_states[-1][0, -1, :]
                error = F.mse_loss(h_test, target_h).item()

            if error < loss_th:
                found_token = cand_id
                found_rank = rank
                final_verification_loss = error
                break
            
            if rank == 0:
                final_verification_loss = error

        recovered_ids.append(found_token)

        token_stats = {
            "position": t,
            "token_id": found_token,
            "token_str": tokenizer.decode([found_token]),
            "rank": found_rank,
            "steps_taken": early_stop_step,
            "final_opt_loss": step_losses[-1],
            "verification_loss": final_verification_loss,
            "vector_l2_norm": final_l2,
            "loss_history": step_losses
        }
        trajectory_data.append(token_stats)
        
        if verbose:
            print(f"T{t+1}: '{token_stats['token_str']}' | Rank: {found_rank} | Steps: {early_stop_step}")

        with torch.no_grad():
            inp_final = torch.tensor([[found_token]], device=device)
            out_final = model(input_ids=inp_final, past_key_values=past_key_values, use_cache=True)
            past_key_values = detach_past(out_final.past_key_values)

    return {
        "recovered_ids": recovered_ids,
        "recovered_text": tokenizer.decode(recovered_ids),
        "trajectory": trajectory_data
    }