# Inverse embeddings

This repo is an implementation of "Language Models are Injective and Hence Invertible" for GPT-2. It demonstrates a privacy attack/interpretability method where input tokens are reconstructed by optimizing a proxy embedding vector via gradient descent to match target hidden states. (https://arxiv.org/abs/2510.15511)

## Usage

Run the main script to demonstrate the inversion process on a sample sentence:

```bash
python gpt2_grad.py
```

## Demo

Given only the **final layer hidden states** of GPT-2, the algorithm recovers the text token-by-token.

| Step | Type | Content |
| :--- | :--- | :--- |
| **Input** | *Hidden States* | `Tensor(seq_len, hidden_size)` (No text access) |
| **Output** | *Recovered Text* | `"Hi, my name is Mikolaj Slowikowski"` |

```bash
# Example Output Log
--------------------------------------------------
Original Text: Hi, my name is Mikolaj
Original IDs:  [17250, 11, 616, 1438, 318, 17722, 349, 1228]
--------------------------------------------------
Target Hidden Shape: torch.Size([8, 768])
--------------------------------------------------
Starting inversion for sequence length: 8...
Token 1/8: 'Hi' (ID: 17250) | Loss: 2.33e-12
Token 2/8: ',' (ID: 11) | Loss: 3.16e-12
Token 3/8: ' my' (ID: 616) | Loss: 7.35e-13
Token 4/8: ' name' (ID: 1438) | Loss: 5.58e-12
Token 5/8: ' is' (ID: 318) | Loss: 3.87e-13
Token 6/8: ' Mik' (ID: 17722) | Loss: 1.46e-12
Token 7/8: 'ol' (ID: 349) | Loss: 1.80e-12
Token 8/8: 'aj' (ID: 1228) | Loss: 2.31e-12
--------------------------------------------------
Reconstructed IDs:  [17250, 11, 616, 1438, 318, 17722, 349, 1228]
Reconstructed Text: Hi, my name is Mikolaj
--------------------------------------------------
```