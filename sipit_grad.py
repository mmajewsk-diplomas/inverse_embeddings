import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional, Any
from transformers import PreTrainedModel, PreTrainedTokenizer

def detach_past(past_key_values: Optional[Tuple]) -> Optional[Tuple]:
    """
    Detaches past key values from the computation graph to prevent memory leaks
    and gradient issues during sequential generation.
    """
    if past_key_values is None:
        return None
    return tuple(tuple(t.detach() for t in layer) for layer in past_key_values)

def sipit(
    model: PreTrainedModel,
    target_hidden_states: torch.Tensor,
    tokenizer: PreTrainedTokenizer,
    max_candidates: int = 100,
    learning_rate: float = 0.03,
    num_optimization_steps: int = 1000,
    loss_threshold: float = 1e-4,
    verbose: bool = True
) -> List[int]:
    """
    Reconstructs input tokens from their hidden states using gradient-based optimization (SIPIT).

    Args:
        model: The language model (e.g., GPT-2).
        target_hidden_states: Tensor of shape (seq_len, hidden_size) containing target embeddings.
        tokenizer: The tokenizer corresponding to the model.
        max_candidates: Number of nearest neighbor tokens to evaluate.
        learning_rate: Learning rate for the Adam optimizer.
        num_optimization_steps: Number of gradient steps per token.
        loss_threshold: MSE loss threshold to accept a candidate token immediately.
        verbose: If True, prints progress for each token.

    Returns:
        List[int]: A list of recovered token IDs.
    """
    device = next(model.parameters()).device
    embedding_matrix = model.transformer.wte.weight.detach()
    
    seq_len = target_hidden_states.shape[0]
    recovered_ids: List[int] = []
    past_key_values = None 

    if verbose:
        print(f"Starting inversion for sequence length: {seq_len}...")

    # Loop over each time step (token) in the sequence
    for t in range(seq_len):
        target_h = target_hidden_states[t].to(device)

        # Initialize a soft embedding vector for optimization
        # We optimize this vector to produce the target hidden state
        proxy_emb = torch.zeros((1, 1, model.config.n_embd), device=device, requires_grad=True)
        optimizer = torch.optim.Adam([proxy_emb], lr=learning_rate)

        # Gradient optimization loop
        for _ in range(num_optimization_steps):
            optimizer.zero_grad()
            
            out = model(
                inputs_embeds=proxy_emb,
                past_key_values=past_key_values,
                use_cache=True,
                output_hidden_states=True
            )
            
            # Extract the last hidden state of the last token
            h_pred = out.hidden_states[-1][0, -1, :]
            loss = F.mse_loss(h_pred, target_h)
            
            loss.backward()
            optimizer.step()

        # Selection of the closest discrete token
        with torch.no_grad():
            optimized_vec = proxy_emb.detach().squeeze()
            # Calculate Euclidean distance to all token embeddings
            dists = torch.norm(embedding_matrix - optimized_vec, dim=1)
            # Select top-k candidates
            candidates = torch.argsort(dists)[:max_candidates].tolist()

        found_token = None

        # Verify candidates by feeding them into the model
        for cand_id in candidates:
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
                if verbose:
                    token_str = tokenizer.decode([cand_id])
                    # Escape newlines for cleaner printing
                    safe_str = token_str.replace('\n', '\\n')
                    print(f"Token {t+1}/{seq_len}: '{safe_str}' (ID: {cand_id}) | Loss: {error:.2e}")
                break
                
        # Fallback: if no token meets the threshold, take the closest one
        if found_token is None:
            found_token = candidates[0]
            if verbose:
                print(f"Token {t+1}/{seq_len}: Fallback to ID {found_token}")

        recovered_ids.append(found_token)

        # Update past_key_values with the chosen token for the next step
        with torch.no_grad():
            inp_final = torch.tensor([[found_token]], device=device)
            out_final = model(
                input_ids=inp_final, 
                past_key_values=past_key_values,
                use_cache=True
            )
            
            past_key_values = detach_past(out_final.past_key_values)

    return recovered_ids
    