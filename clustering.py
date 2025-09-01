#!/usr/bin/env python3
"""
Configurable clustering experiment script based on mutation_experiment.ipynb
Performs embeddings-based clustering of unsafe prompts from safety datasets
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch
import yaml
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.cluster import HDBSCAN
import pandas as pd


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

    elif dataset_name == "AI-Secure/PolyGuard":
        dataset = load_dataset(dataset_name, subset)
        unsafe_splits = [col for col in dataset.column_names.keys() if "unsafe" in col]
        df_list = [dataset[split].data.to_pandas() for split in unsafe_splits]
        df = pd.concat(df_list, ignore_index=True)
        queries = df["original instance"]

    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    print("Before dropping duplicates:", len(queries))
    queries = queries.drop_duplicates(keep="first")
    print("After dropping duplicates:", len(queries))

    if num_samples:
        queries = queries.iloc[:num_samples]
        logger.info(f"Limited to {len(queries)} samples")
    
    return queries.tolist()


def generate_embeddings(queries: List[str], model_name: str, device: str = "auto", 
                       save_embeddings_path: Optional[str] = None) -> torch.Tensor:
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
    
    embeddings = model.encode(queries, convert_to_tensor=True, device=device)
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
        
        if not queries:
            logger.error("No queries found after filtering")
            return 1
        
        # Generate embeddings
        embeddings = generate_embeddings(
            queries,
            config['model']['name'],
            config['model']['device'],
            config['model']['save_embeddings']
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
        save_results(cluster_dict, config['output']['path'], config['output']['indent'])

        logger.info("Experiment completed successfully")
        return 0
        
    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())