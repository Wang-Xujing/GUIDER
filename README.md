# GUIDER: Uncertainty Guided Dynamic Re-ranking for Large Language Models Based Recommender Systems

This is the official repository for the paper **GUIDER: Uncertainty Guided Dynamic Re-ranking for Large Language Models Based Recommender Systems**, accepted for **Oral Presentation** at **AAAI 2026**.

## Abstract

Large Language Models (LLMs) are increasingly integral to recommendation systems, but their practical application is often hindered by data sparsity, unreliable (hallucinated) recommendations, and a lack of transparency.

To address these limitations, we propose **GUIDER**. This framework innovatively leverages the `logits` produced by LLMs as "evidence" for recommended items. By employing a Dirichlet distribution, GUIDER decomposes the total predictive uncertainty into two distinct components:

1. **Data Uncertainty (DU):** Reflects the inherent ambiguity and noise within the data.
2. **Model Uncertainty (MU):** Indicates the model's own conviction or lack of knowledge.

This principled decomposition is achieved within a single inference pass, enhancing transparency and trustworthiness.

## Project Structure

```
.
├── main.py                     # Main script to run experiments
├── models/
│   ├── llm_recommender.py        # Core GUIDER model: uncertainty calculation & 4-quadrant strategy
│   └── llm_generator.py          # (Optional) Model for generating uncertainty-aware explanations
├── utils/
│   ├── data_loader.py          # Data loader (supports MovieLens)
│   └── evaluation.py           # Evaluation metrics (NDCG, Recall@K, etc.)
└── data/
    └── ml-10m/                 # (Requires download) MovieLens-10M dataset
        ├── ratings.dat
        └── movies.dat
```

## Installation

You will need the following Python libraries:

```
pip install torch transformers pandas numpy scipy matplotlib tqdm
```

## How to Run

### 1. Prepare Data

Download the [MovieLens 10M dataset](https://grouplens.org/datasets/movielens/10m/) and place the `ratings.dat` and `movies.dat` files into the `data/ml-10m/` directory.

### 2. Prepare LLM

This code requires you to have pre-trained LLMs available locally. Please update the `MODEL_PATHS` dictionary in `models/llm_recommender.py` to point to the local paths of your models (e.g., Qwen, Llama3, Mistral).

```
# In models/llm_recommender.py
class LLMRecommender:
    MODEL_PATHS = {
        "llama3": "/path/to/your/Llama-3",
        "gemma": "/path/to/your/gemma",
        "mistral": "/path/to/your/mistral",
        "qwen": "/path/to/your/qwen2.5"
    }
    ...
```

### 3. Run Experiment

Use `main.py` to run the evaluation.

```
python main.py \
    --data-path data/ml-10m \
    --model qwen \
    --top-k 20 \
    --max-test-users 1000 \
    --num-neg-samples 29
```

#### Command-Line Arguments

- `--data-path`: Path to the MovieLens dataset.
- `--model`: Name of the LLM to use (must be a key in `MODEL_PATHS`).
- `--top-k`: The 'K' value for evaluation metrics (e.g., NDCG@K).
- `--max-test-users`: Maximum number of users to evaluate from the test set.
- `--num-neg-samples`: Number of negative samples to use per positive instance.
- `--du-threshold`: (Optional) Float threshold for High/Low Data Uncertainty.
- `--mu-threshold`: (Optional) Float threshold for High/Low Model Uncertainty.

## Citation

If you find this work helpful in your research, please consider citing our paper:

```
@inproceedings{xu2026guider,
  title={GUIDER: Uncertainty Guided Dynamic Re-ranking for Large Language Models Based Recommender Systems},
  author={Xu, Cai and Wang, Xujing and Guan, Ziyu and Zhao, Wei and Yan, Meng},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2026}
}
```