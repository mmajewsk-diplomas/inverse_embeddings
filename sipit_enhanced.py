import torch
import torch.nn.functional as F
import torch.nn as nn
from typing import List, Tuple, Optional, Union
from transformers import PreTrainedModel, PreTrainedTokenizer

def detach_past(past_key_values: Optional[Tuple]) -> Optional[Tuple]:
    """
    Detaches past key values from the computation graph.
    """
    if past_key_values is None:
        return None
    return tuple(tuple(t.detach() for t in layer) for layer in past_key_values)

import torch

class VectorSpaceModel(torch.nn.Module):

    def __init__(self):
        super(TinyModel, self).__init__()
        vocab_size = ...
        self.linear1 = torch.nn.Linear(vocab_size, vocab_size)

    def forward(self, x):
        x = self.linear1(x)
        return x

vsmodel = VectorSpaceModel()
triplet_loss = nn.TripletMarginLoss(margin=0.2, p=2)
optimizer = torch.optim.Adam(vsmodel.parameters(), lr=...)

def sipit(
    model: PreTrainedModel,
    target_hidden_states: torch.Tensor,
    tokenizer: PreTrainedTokenizer,
    max_candidates: int = 10000,
    learning_rate: float = 0.03,
    num_optimization_steps: int = 1000,
    loss_threshold: float = 1e-4,
    verbose: bool = True,
    return_loss_history: bool = False
) -> Union[List[int], Tuple[List[int], List[List[float]]]]:
    """
    Reconstructs input tokens using gradient-based optimization.
    
    Args:
        return_loss_history (bool): If True, returns a tuple (recovered_ids, loss_history).
                                    loss_history is a list of lists [token_idx][step_loss].
    """
    device = next(model.parameters()).device
    embedding_matrix = model.transformer.wte.weight.detach()
    
    seq_len = target_hidden_states.shape[0]
    recovered_ids: List[int] = []
    past_key_values = None 
    
    full_loss_history: List[List[float]] = []
    rank_history = []

    if verbose:
        print(f"Starting inversion for sequence length: {seq_len}...")

    for t in range(seq_len):
        target_h = target_hidden_states[t].to(device)
        
        # Initialize proxy embedding
        proxy_emb = torch.zeros((1, 1, model.config.n_embd), device=device, requires_grad=True)
        optimizer = torch.optim.Adam([proxy_emb], lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_optimization_steps)

        token_loss_history: List[float] = []

        # Gradient optimization loop
        for _ in range(num_optimization_steps):
            optimizer.zero_grad()
            out = model(inputs_embeds=proxy_emb, past_key_values=past_key_values, use_cache=True, output_hidden_states=True)
            h_pred = out.hidden_states[-1][0, -1, :]
            loss = F.mse_loss(h_pred, target_h)

            proxy_emb_copy = proxy_emb.detach().clone()
            v_pull = vsmodel(proxy_emb_copy)
            new_h_pred = proxy_emb_copy + v_pull
            with torch.no_grad():
                optimized_vec = proxy_emb.copy().detach().squeeze()
                dists = torch.norm(embedding_matrix - optimized_vec, dim=1)
                candidates = torch.argsort(dists)[:max_candidates].tolist()
            second_best_fit_token = ... # if second best is not same as target_h
            # else first
            # or simply remove target_h from embedding_matrix and get best
            triplet_loss = (target_h, new_h_pred, second_best_fit_token)

            
            if loss < loss_threshold:
                break
            
            token_loss_history.append(loss.item())
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_([proxy_emb], max_norm=1.0)
            
            optimizer.step()
            scheduler.step()

        full_loss_history.append(token_loss_history)

        # Selection of the closest discrete token
        with torch.no_grad():
            optimized_vec = proxy_emb.copy().detach().squeeze()
            dists = torch.norm(embedding_matrix - optimized_vec, dim=1)
            candidates = torch.argsort(dists)[:max_candidates].tolist()

        found_token = None

        # Verify candidates
        for i, cand_id in enumerate(candidates):
            inp = torch.tensor([[cand_id]], device=device)
            
            with torch.no_grad():
                out_test = model(
                    input_ids=inp,
                    past_key_values=past_key_values,
                    use_cache=True,
                    output_hidden_states=True
                )
                h_test = out_test.hidden_states[-1][0, -1, :]
                error = F.mse_loss(h_test, target_h).item()

            if error < loss_threshold:
                found_token = cand_id
                found_rank = i
                if verbose:
                    token_str = tokenizer.decode([cand_id]).replace('\n', '\\n')
                    print(f"Token {t+1}: '{token_str}' | Rank: {i} | Loss: {error:.2e}")
                break
                
        if found_token is None:
            found_token = candidates[0]
            if verbose:
                print(f"Token {t+1}/{seq_len}: Fallback to ID {found_token}")

        recovered_ids.append(found_token)
        rank_history.append(found_rank)

        # Update context
        with torch.no_grad():
            inp_final = torch.tensor([[found_token]], device=device)
            out_final = model(
                input_ids=inp_final, 
                past_key_values=past_key_values,
                use_cache=True
            )
            past_key_values = detach_past(out_final.past_key_values)

    if return_loss_history:
        return recovered_ids, full_loss_history, rank_history
    
    return recovered_ids
