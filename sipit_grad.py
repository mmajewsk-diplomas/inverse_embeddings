import torch
import torch.nn.functional as F

# Odpinanie histori, żeby cuda nie dawala bledu
def detach_past(past_key_values):
    if past_key_values is None:
        return None
    return tuple(tuple(t.detach() for t in layer) for layer in past_key_values)

def sipit(model, target_hidden_states, tokenizer, max_candidates=100, verbose=True):
    device = next(model.parameters()).device
    embedding_matrix = model.transformer.wte.weight.detach()
    
    T = target_hidden_states.shape[0]
    recovered_ids = []
    past = None 

    print(f"Inversing {T}...")

    # Petla po tokenach
    for t in range(T):
        target_h = target_hidden_states[t].to(device)

        # miekki wektor do optymalizacji
        proxy_emb = torch.zeros((1, 1, model.config.n_embd), device=device, requires_grad=True)
        # TODO: przeniesc lr do arg fun
        optimizer = torch.optim.Adam([proxy_emb], lr=0.03)

        # opt gradient
        for _ in range(1000):
            optimizer.zero_grad()
            
            out = model(
                inputs_embeds=proxy_emb,
                past_key_values=past,
                use_cache=True,
                output_hidden_states=True
            )
            
            h_pred = out.hidden_states[-1][0, -1, :]
            loss = F.mse_loss(h_pred, target_h)
            
            loss.backward()
            optimizer.step()

        # wybor najblizszego tokenu
        with torch.no_grad():
            optimized_vec = proxy_emb.detach().squeeze()
            # dyst euk do slow w slowniku
            dists = torch.norm(embedding_matrix - optimized_vec, dim=1)
            # wybor max_cand kandydatow
            candidates = torch.argsort(dists)[:max_candidates].tolist()

        found_token = None

        # wybor najlepszego tokeny z kandydatow
        for cand_id in candidates:
            inp = torch.tensor([[cand_id]], device=device)
            
            with torch.no_grad():
                out_test = model(
                    input_ids=inp,
                    past_key_values=past,
                    use_cache=True,
                    output_hidden_states=True
                )
                h_test = out_test.hidden_states[-1][0, -1, :]
                error = F.mse_loss(h_test, target_h).item()

            # TODO: przeniesc do arg fun
            if error < 1e-4:
                found_token = cand_id
                if verbose:
                    token_str = tokenizer.decode([cand_id])
                    print(f"Token {t+1}: '{token_str}' (ID: {cand_id}) | loss: {error:.2e}")
                break
                
        # nie ma dobrego tokeny <E brany pierwszy
        if found_token is None:
            found_token = candidates[0]

        recovered_ids.append(found_token)

        # zatwiedzamy token i aktualizujemy pam do nast kroku
        with torch.no_grad():
            inp_final = torch.tensor([[found_token]], device=device)
            out_final = model(
                input_ids=inp_final, 
                past_key_values=past,
                use_cache=True
            )
            
            past = detach_past(out_final.past_key_values)

    return recovered_ids