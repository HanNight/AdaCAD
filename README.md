# AdaCAD (Adaptive Context Aware Decoding)
Code for the paper [AdaCAD: Adaptively Decoding to Balance Conflicts between Contextual and Parametric Knowledge]().

by [Han Wang](https://hannight.github.io/), [Archiki Prasad](https://archiki.github.io/), [Elias Stengel-Eskin](https://esteng.github.io/), [Mohit Bansal](https://www.cs.unc.edu/~mbansal/).

![image](https://github.com/user-attachments/assets/0df89574-1dd7-40f7-8187-7652e0ea05ed)

## Requirements
You can install all required packages by running the following command:
```bash
pip install -r requirements.txt
```

## Data
We provide a sample input file `data/nq_swap_2_-1.json`.

The format of the input file is as follow:
```json
{
    "input_index": 0, // instances that decode together should have the same input_index
    "assigned_model": "meta-llama/Meta-Llama-3-8B", // same model for all instances in context-aware decoding, but can use different models here, e.g., DExperts, contrastive decoding, proxy tuning, etc.
    "assigned_process": 0, // which GPU should take this instance
    "context_string": "The fourth season of Chicago Fire , an American drama television series with executive producer Dick Wolf , and producers Derek Haas , Michael Brandt , and Matt Olmstead , was ordered on February 5 , 2015 , by NBC , and premiered on October 13 , 2015 and concluded on May 17 , 2016 . The season contained 1078 episodes . \nUsing only the references listed above, answer the following question: \nQuestion: How many episodes are in chicago fire season 4 ?\nAnswer:", // the input with context
    "assigned_weight": 2, // weight for current instance/process (1+alpha, weights should add up to 1 by default, but can also incorporate sampling temperature if needed)
    "filter_p": 1.0, // optional filtering for low-probablity tokens, disabled by default
}
{
    "input_index": 0, // instances that decode together should have the same input_index
    "assigned_model": "meta-llama/Meta-Llama-3-8B", // same model for all instances in context-aware decoding, but can use different models here, e.g., DExperts, contrastive decoding, proxy tuning, etc.
    "assigned_process": 1, // which GPU should take this instance
    "context_string": "Answer the following question: \nQuestion: How many episodes are in chicago fire season 4 ?\nAnswer:", // the input without context
    "assigned_weight": -1, // weight for current instance/process (-alpha, weights should add up to 1 by default, but can also incorporate sampling temperature if needed)
}
...
```

## Run AdaCAD
### For Question Answering
```bash
bash run_nq.sh /path/to/your/input/file
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

**Note:** Remember to use your own huggingface token (User Access Token to authenticate to the Hub) as `hf_token` in `group_decode_fileio_adacad.py`.

## Acknowledgement
We sincerely thank the authors of [CAD](https://github.com/xhan77/context-aware-decoding/tree/main) for their public code release.

## Citation
```bibtex
@article{wang2024adacad,
  title={AdaCAD: Adaptively Decoding to Balance Conflicts between Contextual and Parametric Knowledge},
  author={Han Wang and Archiki Prasad and Elias Stengel-Eskin and Mohit Bansal},
  year={2024},
```