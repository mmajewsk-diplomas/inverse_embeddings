import torch
import torch.nn.functional as F
from typing import List, Dict, Any
from transformers import PreTrainedModel, PreTrainedTokenizer

def detach_past(past_key_values):
    if past_key_values is None: return None
    return tuple(tuple(t.detach() for t in layer) for layer in past_key_values)

def sipit(
    model: PreTrainedModel,
    target_hidden_states: torch.Tensor,
    target_ids: torch.Tensor,  # <--- NOWY ARGUMENT: Prawdziwe ID tokenów
    tokenizer: PreTrainedTokenizer,
    lr: float = 0.05,
    steps: int = 500,
    loss_th: float = 1e-4,
    max_candidates: int = 2000,
    reg_weight: float = 1e-5,
    rank_check_interval: int = 20,
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
        true_token_id = target_ids[t].item() # Pobieramy ID prawdziwego tokena
        
        proxy_emb = torch.zeros((1, 1, model.config.n_embd), device=device, requires_grad=True)
        optimizer = torch.optim.Adam([proxy_emb], lr=lr, weight_decay=reg_weight)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)

        step_losses = []
        vector_history = [] 
        early_stop_step = steps

        # --- OPTYMALIZACJA ---
        for step in range(steps):
            optimizer.zero_grad()
            
            # Zapisujemy historię wektora co N kroków
            if step % rank_check_interval == 0:
                vector_history.append((step, proxy_emb.detach().clone()))

            out = model(inputs_embeds=proxy_emb, past_key_values=past_key_values, use_cache=True, output_hidden_states=True)
            h_pred = out.hidden_states[-1][0, -1, :]
            
            loss = F.mse_loss(h_pred, target_h)
            loss_val = loss.item()
            step_losses.append(loss_val)

            if loss_val < loss_th:
                early_stop_step = step
                vector_history.append((step, proxy_emb.detach().clone()))
                break
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_([proxy_emb], max_norm=1.0)
            optimizer.step()
            scheduler.step()

        # --- WYSZUKIWANIE KANDYDATA (Na koniec) ---
        with torch.no_grad():
            optimized_vec = proxy_emb.detach().squeeze()
            dists = torch.norm(embedding_matrix - optimized_vec, dim=1)
            candidates = torch.argsort(dists)[:max_candidates].tolist()

        found_token = candidates[0]
        found_rank = 0

        # Sprawdzenie, który kandydat daje najmniejszy błąd na wyjściu
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
                    print(f"Token {t+1}: Found candidate {tokenizer.decode([cand_id])} at rank {rank}")
                break

        recovered_ids.append(found_token)

        # --- ANALIZA POST-FACTUM (Tracking Rank) ---
        rank_evolution_winner = []
        rank_evolution_true = [] # <--- Tu będziemy zapisywać ranking poprawnego tokena

        with torch.no_grad():
            winner_vec = embedding_matrix[found_token]
            true_vec = embedding_matrix[true_token_id]

            for step_num, hist_vec in vector_history:
                hist_vec = hist_vec.squeeze()
                
                # Obliczamy odległości do wszystkich słów w słowniku
                dists_t = torch.norm(embedding_matrix - hist_vec, dim=1)
                
                # 1. Ranking zwycięzcy (tego, co algorytm wybrał)
                dist_to_winner = torch.norm(winner_vec - hist_vec).item()
                rank_at_step_winner = (dists_t < dist_to_winner).sum().item()
                rank_evolution_winner.append(rank_at_step_winner)

                # 2. Ranking prawdziwego tokena (Ground Truth)
                dist_to_true = torch.norm(true_vec - hist_vec).item()
                rank_at_step_true = (dists_t < dist_to_true).sum().item()
                rank_evolution_true.append(rank_at_step_true)

        if verbose:
            print(f"Token {t+1} | True: '{tokenizer.decode([true_token_id])}' | Rec: '{tokenizer.decode([found_token])}'")

        token_stats = {
            "token_str": tokenizer.decode([found_token]),
            "true_token_str": tokenizer.decode([true_token_id]),
            "rank_final": found_rank,
            "loss_history": step_losses,
            
            # Zapisujemy obie ewolucje
            "rank_evolution_winner": rank_evolution_winner, 
            "rank_evolution_true": rank_evolution_true,
            
            "rank_evolution_steps": [x[0] for x in vector_history] 
        }
        trajectory_data.append(token_stats)
        
        # Aktualizacja KV Cache dla następnego tokena
        with torch.no_grad():
            inp_final = torch.tensor([[found_token]], device=device)
            out_final = model(input_ids=inp_final, past_key_values=past_key_values, use_cache=True)
            past_key_values = detach_past(out_final.past_key_values)

    return {
        "recovered_ids": recovered_ids,
        "recovered_text": tokenizer.decode(recovered_ids),
        "trajectory": trajectory_data
    }