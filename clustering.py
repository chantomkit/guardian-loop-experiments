#!/usr/bin/env python3
"""
Configurable clustering experiment script based on mutation_experiment.ipynb
Performs embeddings-based clustering of unsafe prompts from safety datasets
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch
import yaml
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.cluster import HDBSCAN
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer


def setup_logging(log_level: str = "INFO") -> None:
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def load_and_filter_dataset(dataset_name: str, subset: Optional[str] = None, split: Optional[str] = None, num_samples: Optional[int] = None) -> List[str]:
    """Load dataset and extract prompts"""
    logger = logging.getLogger(__name__)
    logger.info(f"Loading dataset: {dataset_name}, subset: {subset}, split: {split}")

    if dataset_name == "PKU-Alignment/BeaverTails":
        dataset = load_dataset(dataset_name)
        df = dataset[split].data.to_pandas()
        logger.info(f"Dataset shape: {df.shape}")
        queries = df.loc[~df["is_safe"], "prompt"]
        logger.info(f"Filtered to {len(queries)} unsafe prompts")

    elif (dataset_name == "AI-Secure/PolyGuard"):
        dataset = load_dataset(dataset_name, subset)
        if subset == "finance_input" or subset == "law_input":
            unsafe_splits = [col for col in dataset.column_names.keys() if "unsafe" in col]
            df_list = [dataset[split].data.to_pandas() for split in unsafe_splits]
            df = pd.concat(df_list, ignore_index=True)
            queries = df["original instance"]
        elif subset == "cyber":
            df_list = [dataset[split].data.to_pandas() for split in dataset.column_names.keys()]
            df = pd.concat(df_list, ignore_index=True)
            df = df.loc[df.label == "unsafe", :]
            queries = df["prompt"]
        elif subset == "education":
            unsafe_splits = [col for col in dataset.column_names.keys() if "unsafe" in col]
            df_list = [dataset[split].data.to_pandas() for split in unsafe_splits]
            df = pd.concat(df_list, ignore_index=True)
            queries = df["instance"]

    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    print("Before dropping duplicates:", len(queries))
    queries = queries.drop_duplicates(keep="first")
    print("After dropping duplicates:", len(queries))

    if num_samples:
        queries = queries.iloc[:num_samples]
        logger.info(f"Limited to {len(queries)} samples")
    
    return queries.tolist()


def generate_embeddings(queries: List[str], model_name: str, domain: str, device: str = "auto", 
                       save_embeddings_path: Optional[Path] = None) -> torch.Tensor:
    """Generate embeddings for queries using sentence transformer"""
    logger = logging.getLogger(__name__)
    logger.info(f"Loading model: {model_name}")
    
    model = SentenceTransformer(model_name)
    
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    
    logger.info(f"Using device: {device}")
    logger.info(f"Encoding {len(queries)} queries...")
    
    TASK_PROMPT = """
    Represent the following harmful user query from the {DOMAIN} safety domain as an embedding.
    The embedding should emphasize:
    - Harmful intent, unsafe action, or sensitive target.
    - Risk characteristics and potential rule violations.
    - Semantic similarity of harmful context, so that related harmful queries cluster together.
    Ignore irrelevant details such as politeness, filler text, or non-harmful context.
    Input query: 
    """
    prompt = TASK_PROMPT.format(DOMAIN=domain)
    print("TASK PROMPT:" + prompt)

    embeddings = model.encode(queries, convert_to_tensor=True, prompt=prompt, device=device, normalize_embeddings=True)
    logger.info(f"Generated embeddings shape: {embeddings.shape}")
    
    if save_embeddings_path:
        torch.save(embeddings, save_embeddings_path)
        logger.info(f"Embeddings saved to {save_embeddings_path}")
    
    return embeddings


def perform_clustering(embeddings: torch.Tensor, min_cluster_size: int = 3, 
                      metric: str = "cosine", leaf_size: int = 10, 
                      min_samples: Optional[int] = None) -> HDBSCAN:
    """Perform HDBSCAN clustering on embeddings"""
    logger = logging.getLogger(__name__)
    logger.info(f"Performing HDBSCAN clustering with min_cluster_size={min_cluster_size}")
    
    cluster_params = {
        'min_cluster_size': min_cluster_size,
        'metric': metric,
        'leaf_size': leaf_size
    }
    
    if min_samples is not None:
        cluster_params['min_samples'] = min_samples
    
    hdb = HDBSCAN(**cluster_params)
    hdb.fit(embeddings.cpu().numpy())

    n_clusters = len(set(hdb.labels_)) - (1 if -1 in hdb.labels_ else 0)
    n_noise = list(hdb.labels_).count(-1)

    logger.info(f"Found {n_clusters} clusters, {n_noise} noise points")

    return hdb


def build_cluster_dict(hdb: HDBSCAN, queries: List[str]) -> Dict[str, List[str]]:
    """Build dictionary mapping cluster IDs to queries.

    Dict format:
    {
        "cluster_id": [
            (query, probability),
            ...
        ],
        ...
    }
    """
    cluster_dict = {}
    
    for cluster_id in set(hdb.labels_):
        if cluster_id == -1:  # Skip noise
            continue
            
        cluster_str = str(cluster_id)
        cluster_dict[cluster_str] = []

        for i, label in enumerate(hdb.labels_):
            if label == cluster_id:
                cluster_dict[cluster_str].append((queries[i], hdb.probabilities_[i]))

    return cluster_dict


def save_results(cluster_dict: Dict[str, List[str]], output_path: Path,  indent: int = 2) -> None:
    """Save clustering results to JSON file"""
    logger = logging.getLogger(__name__)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(cluster_dict, f, indent=indent, ensure_ascii=False)

    logger.info(f"Results saved to {output_path}")

    # Log cluster statistics
    total_queries = sum(len(queries) for queries in cluster_dict.values())
    logger.info(f"Clustered {total_queries} queries into {len(cluster_dict)} clusters")

    cluster_sizes = [len(queries) for queries in cluster_dict.values()]
    if cluster_sizes:
        logger.info(f"Cluster sizes - Min: {min(cluster_sizes)}, Max: {max(cluster_sizes)}, "
                   f"Mean: {sum(cluster_sizes)/len(cluster_sizes):.1f}")


def mutate_queries(queries: List[str], batch_size: int = 16) -> List[str]:
    """
    Mutate a list of queries by paraphrasing them while preserving meaning.
    Only activated when config['dataset']['mutate'] is True.
    
    Args:
        queries (list): List of input queries to mutate
        batch_size (int): Number of queries to process in each batch
        
    Returns:
        list: List of mutated queries
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Mutating {len(queries)} queries")
    
    model_name = "Qwen/Qwen3-4B-Instruct-2507"
    
    # Load the tokenizer and model
    logger.info(f"Loading mutation model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    
    # Set padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    TASK_PROMPT = """
Paraphrase the entire text below.
Preserve all meaning and details. Do not shorten, omit, or summarize.
You may reword, restructure sentences, or adjust phrasing.

Text: {input}
"""

    print("mutation task prompt:")
    print(TASK_PROMPT)

    mutated_queries = []
    
    # Create prompts for each query
    all_prompts = []
    for query in queries:
        prompt = TASK_PROMPT.format(input=query)
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        all_prompts.append(text)
    
    # Process in batches
    for i in range(0, len(all_prompts), batch_size):
        batch_prompts = all_prompts[i:i + batch_size]
        logger.debug(f"Processing batch {i//batch_size + 1}/{(len(all_prompts) + batch_size - 1)//batch_size}")
        
        # Tokenize batch
        model_inputs = tokenizer(
            batch_prompts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True,
            max_length=4096
        ).to(model.device)
        
        # Generate in batch
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=1024,
            do_sample=True,
            temperature=0.7,
            top_p=0.8,
            top_k=20,
            min_p=0,
            pad_token_id=tokenizer.pad_token_id
        )
        
        # Decode each output, extracting only the generated portion
        for j, gen_ids in enumerate(generated_ids):
            input_length = len(model_inputs.input_ids[j])
            output_ids = gen_ids[input_length:].tolist()
            mutated_query = tokenizer.decode(output_ids, skip_special_tokens=True)
            mutated_queries.append(mutated_query.strip())
    
    logger.info(f"Generated {len(mutated_queries)} mutated queries")
    return mutated_queries


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        print(f"Error: Config file '{config_path}' not found")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error parsing config file: {e}")
        sys.exit(1)


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    config = load_config(config_path)
    
    setup_logging(config['output']['log_level'])
    logger = logging.getLogger(__name__)
    
    try:
        # Load and filter dataset
        queries = load_and_filter_dataset(
            config['dataset']['name'],
            config['dataset']['subset'],
            config['dataset']['split'],
            config['dataset']['num_samples'],
        )

        output_dir = Path(config['output']['dir_name'])
        output_dir.mkdir(parents=True, exist_ok=True)

        # mutate queries
        if config['dataset']['mutate']:
            queries = mutate_queries(queries)

        queries_path = output_dir / Path("queries.json")
        json.dump(queries, open(queries_path, "w"), indent=2)
        print(f"Filtered queries saved to {queries_path}")

        if not queries:
            logger.error("No queries found after filtering")
            return 1
        
        # Generate embeddings
        embeddings_path = output_dir / Path("embeddings.pt")
        embeddings = generate_embeddings(
            queries,
            config['model']['name'],
            config['model']['domain'],
            config['model']['device'],
            embeddings_path,
        )
        
        # Perform clustering
        hdb = perform_clustering(
            embeddings,
            config['clustering']['min_cluster_size'],
            config['clustering']['metric'],
            config['clustering']['leaf_size'],
            config['clustering']['min_samples']
        )
        
        # Build results
        cluster_dict = build_cluster_dict(hdb, queries)

        # Save results
        cluster_path = output_dir / Path("clusters.json")
        save_results(cluster_dict, cluster_path)

        logger.info("Experiment completed successfully")
        return 0
        
    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())