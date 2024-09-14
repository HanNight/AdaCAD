# AdaCAD (Adaptive Context Aware Decoding)
Code for the paper [AdaCAD: Adaptively Decoding to Balance Conflicts between Contextual and Parametric Knowledge](https://arxiv.org/abs/2409.07394
).

by [Han Wang](https://hannight.github.io/), [Archiki Prasad](https://archiki.github.io/), [Elias Stengel-Eskin](https://esteng.github.io/), [Mohit Bansal](https://www.cs.unc.edu/~mbansal/).

![image](https://github.com/user-attachments/assets/0df89574-1dd7-40f7-8187-7652e0ea05ed)

## Requirements
You can install all required packages by running the following command:
```bash
pip install -r requirements.txt
```

## Data
We provide two sample input files `nq_swap_2_-1.jsonl` and `nq_synth_2_-1.jsonl` in `data` folder. The details are described in `data/README.md`.

## Run AdaCAD
### For Question Answering
```bash
HF_TOKEN=your_huggingface_token # User Access Token to authenticate to the Hub.
HF_HUB_CACHE=your_cache_path # where repositories from the Hub will be cached locally (models, datasets and spaces).
bash run_nq.sh /path/to/your/input/file
```
As an exampe, run the following command:
```bash
bash run_nq.sh data/nq_swap_2_-1.json
```
We explain the arguments in `run_nq.sh` as follows:
- `GLOBALLEN`: the maximum sequence length of the model.
- `MAXCTXLEN`: the maximum input context length.
- `GENLEN`: the maximun generation length, should be `GENLEN = GLOBALLEN - MAXCTXLEN`.
- `SEED`: random seed.
- `DEVICE`: the GPU device ids, for example, `0,1`.
- `TOPP`: top-p sampling, set to 0.0 for greedy decoding.
- `GPUS`: number of gpus.
- `FLAG`: whether to use int4 quantization to load the model.

**Note:** Remember to use your own huggingface token and set your local cache path.

### For Summarization
Coming soon, please stay tuned.

### How to incorporate AdaCAD into your own decoding method:
You can use the following code snippet to compute the JSD value and then adjust the output probability distribution during decoding. 
```python
def get_jsd(p, q):
    p = F.softmax(p, dim=-1)
    q = F.softmax(q, dim=-1)
    p, q = p.view(-1, p.size(-1)), q.view(-1, q.size(-1))
    if ((p + q) == 0).any():
        m = (0.5 * (p + q)).clamp_min(1e-9).log()
    else:
        m = (0.5 * (p + q)).log()
    if torch.any(p <= 0):
        p = p.clamp_min(1e-9)
    if torch.any(q <= 0):
        q = q.clamp_min(1e-9)
    return 0.5 * (F.kl_div(m, p, reduction='batchmean', log_target=False) + F.kl_div(m, q, reduction='batchmean', log_target=False))

# logits1 is the output logits of the input with context
# logits2 is the output logits of the input without context
alpha = get_jsd(logits1, logits2)
new_logits1 = (1 + alpha) * logits1 + (0 - alpha) * logits2
```

## Acknowledgement
We sincerely thank the authors of [CAD](https://github.com/xhan77/context-aware-decoding/tree/main) for their public code release.

## Citation
```bibtex
@article{wang2024adacad,
  title={AdaCAD: Adaptively Decoding to Balance Conflicts between Contextual and Parametric Knowledge},
  author={Han Wang and Archiki Prasad and Elias Stengel-Eskin and Mohit Bansal},
  year={2024},
  journal={arXiv preprint arXiv:2409.07394}
}
```
