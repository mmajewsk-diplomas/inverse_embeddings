import torch
import torch.nn.functional as F
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
    lr: float = 0.1,           # Zwiększony domyślny LR, optymalny dla SGD
    steps: int = 2000,         # Zwiększona liczba kroków (SIPIT zazwyczaj potrzebuje więcej iteracji)
    loss_th: float = 1e-4,
    reg_weight: float = 1e-5,
    rank_check_interval: int = 20,
    verbose: bool = False,
    metric: str = 'l2',
    projection_iters_base: int = 100,  # Nowy parametr z SIPIT
    vocab_scale_factor: int = 25000   # Nowy parametr z SIPIT
) -> Dict[str, Any]:
    
    device = next(model.parameters()).device
    
    # Używamy uogólnionej metody pobierania embeddingów, aby działało na LLaMA/GPT itd.
    embedding_matrix = model.get_input_embeddings().weight.detach() 
    seq_len = target_hidden_states.shape[0]
    vocab_size = embedding_matrix.size(0)
    
    recovered_ids = []
    past_key_values = None 
    trajectory_data = []

    for t in range(seq_len):
        target_h = target_hidden_states[t].to(device)
        true_token_id = target_ids[t].item()
        
        # Obliczenie częstotliwości twardego rzutowania wg logiki SIPIT
        reset_extra = vocab_size // vocab_scale_factor
        reset_every = projection_iters_base * (1 + reset_extra * int(t == 0))
        
        # Kopia macierzy do mechanizmu Tabu Search
        copy_embedding_matrix = embedding_matrix.clone()
        
        # Inicjalizacja z losowego dyskretnego tokenu (zamiast wektora zerowego)
        current_token_id = int(torch.randint(0, vocab_size, (1,)).item())
        proxy_emb = copy_embedding_matrix[current_token_id].clone().view(1, 1, -1).requires_grad_(True)
        
        # Zmiana optymalizatora na SGD zgodnie z SIPIT
        optimizer = torch.optim.SGD([proxy_emb], lr=lr, weight_decay=reg_weight)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, 
            T_0=reset_every, 
            T_mult=1
        )

        step_losses = []
        vector_history = [] 
        
        best_token_id = current_token_id
        best_discrete_loss = float('inf')

        for step in range(steps):
            optimizer.zero_grad()
            
            if step % rank_check_interval == 0:
                vector_history.append((step, proxy_emb.detach().clone()))

            # Krok 1: Forward po przestrzeni ciągłej
            out = model(inputs_embeds=proxy_emb, past_key_values=past_key_values, use_cache=True, output_hidden_states=True)
            h_pred = out.hidden_states[-1][0, -1, :]
            
            loss = F.mse_loss(h_pred, target_h)
            loss_val = loss.item()
            step_losses.append(loss_val)
            
            # Krok 2: Backward i agresywne skalowanie gradientu
            loss.backward()
            grad_norm = proxy_emb.grad.norm().item()
            
            # ZMIANA: Zawsze normalizujemy gradient do długości 1.0!
            # Dzięki temu wektor ciągły robi realne kroki równe dokładnie "lr" (np. 0.5) 
            # i ucieka z początkowego "krateru" przed rzutowaniem.
            if grad_norm > 0:
                proxy_emb.grad = proxy_emb.grad / grad_norm
                
            optimizer.step()
            scheduler.step()

            # Krok 3: Weryfikacja Dyskretna aktualnego kandydata (bezpośrednio z modelu)
            with torch.no_grad():
                inp_discrete = torch.tensor([[current_token_id]], device=device)
                out_discrete = model(input_ids=inp_discrete, past_key_values=past_key_values, use_cache=True, output_hidden_states=True)
                h_discrete = out_discrete.hidden_states[-1][0, -1, :]
                discrete_loss = F.mse_loss(h_discrete, target_h).item()
                
                # Zapisujemy najlepszy token, gdybyśmy nie osiągnęli progu `loss_th`
                if discrete_loss < best_discrete_loss:
                    best_discrete_loss = discrete_loss
                    best_token_id = current_token_id

                # Warunek szybkiego wyjścia
                if discrete_loss < loss_th:
                    vector_history.append((step, proxy_emb.detach().clone()))
                    break
            
            # Krok 4: Tabu Search i Snapping (Rzutowanie)
            with torch.no_grad():
                # Eliminujemy z przestrzeni poszukiwań sprawdzony już token (1e9 zapobiega ew. błędom NaN przy liczeniu dystansu dla niektórych metryk)
                copy_embedding_matrix[current_token_id] = 1e9 
                
                # Szukamy najbliższego z wciąż "dostępnych" tokenów
                distances = compute_distances(copy_embedding_matrix, proxy_emb.squeeze(), metric=metric)
                current_token_id = torch.argmin(distances).item()
                
                # Rzutowanie (Snap) wektora ciągłego na idealny wektor dyskretny co N iteracji
                if (step + 1) % reset_every == 0:
                    proxy_emb.data = copy_embedding_matrix[current_token_id].clone().view(1, 1, -1).data

        # Przypisujemy token z najniższym błędem dyskretnym
        found_token = best_token_id
        recovered_ids.append(found_token)

        # -- Analiza Ewolucji Rangi --
        with torch.no_grad():
            final_dists = compute_distances(embedding_matrix, proxy_emb.squeeze(), metric=metric)
            # Obliczamy końcową rangę na podstawie oryginalnej macierzy
            found_rank = (final_dists < final_dists[found_token]).sum().item()

        rank_evolution_winner = []
        rank_evolution_true = [] 

        with torch.no_grad():
            winner_vec = embedding_matrix[found_token]
            true_vec = embedding_matrix[true_token_id]

            for step_num, hist_vec in vector_history:
                hist_vec = hist_vec.squeeze()
                dists_t = compute_distances(embedding_matrix, hist_vec, metric=metric)
                
                dist_to_winner = compute_distances(winner_vec.unsqueeze(0), hist_vec, metric=metric).item()
                rank_at_step_winner = (dists_t < dist_to_winner).sum().item()
                rank_evolution_winner.append(rank_at_step_winner)

                dist_to_true = compute_distances(true_vec.unsqueeze(0), hist_vec, metric=metric).item()
                rank_at_step_true = (dists_t < dist_to_true).sum().item()
                rank_evolution_true.append(rank_at_step_true)

        if verbose:
            print(f"Token {t+1} | True: '{tokenizer.decode([true_token_id])}' | Rec: '{tokenizer.decode([found_token])}' | Steps: {step+1} | Disc Loss: {best_discrete_loss:.2e} | Rank: {found_rank}")

        token_stats = {
            "token_str": tokenizer.decode([found_token]),
            "true_token_str": tokenizer.decode([true_token_id]),
            "rank_final": found_rank,
            "loss_history": step_losses,
            "discrete_loss": best_discrete_loss,
            "rank_evolution_winner": rank_evolution_winner, 
            "rank_evolution_true": rank_evolution_true,
            "rank_evolution_steps": [x[0] for x in vector_history] 
        }
        trajectory_data.append(token_stats)
        
        with torch.no_grad():
            # Przepychamy docelowy "twardy" token dla zachowania kontekstu (past_key_values)
            inp_final = torch.tensor([[found_token]], device=device)
            out_final = model(input_ids=inp_final, past_key_values=past_key_values, use_cache=True)
            past_key_values = detach_past(out_final.past_key_values)

    return {
        "recovered_ids": recovered_ids,
        "recovered_text": tokenizer.decode(recovered_ids),
        "trajectory": trajectory_data
    }