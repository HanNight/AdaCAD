## Data
We provide two sample input files `nq_swap_2_-1.jsonl`, `nq_synth_2_-1.jsonl`, and `tofu_1.5_-0.5.jsonl`.

The format of the input file is as follows:
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

**Note:** The `assigned_weight` here is used for CAD method. We just use it to distinguish the input with context and without context, and dynamically compute the alpha/weight during each decoding step in AdaCAD.