#!/usr/bin/env python3
"""
Configurable clustering experiment script based on mutation_experiment.ipynb
Performs embeddings-based clustering of unsafe prompts from safety datasets
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import torch
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


def load_and_filter_dataset(dataset_name: str, split: str, num_samples: Optional[int] = None, 
                           filter_unsafe: bool = True) -> List[str]:
    """Load dataset and extract prompts"""
    logger = logging.getLogger(__name__)
    logger.info(f"Loading dataset: {dataset_name}, split: {split}")
    
    dataset = load_dataset(dataset_name)
    df = dataset[split].data.to_pandas()
    
    logger.info(f"Dataset shape: {df.shape}")
    
    if filter_unsafe and 'is_safe' in df.columns:
        queries = df.loc[~df["is_safe"], "prompt"]
        logger.info(f"Filtered to {len(queries)} unsafe prompts")
    else:
        queries = df["prompt"]
        logger.info(f"Using all {len(queries)} prompts")
    
    if num_samples:
        queries = queries.iloc[:num_samples]
        logger.info(f"Limited to {len(queries)} samples")
    
    return queries.tolist()


def generate_embeddings(queries: List[str], model_name: str, device: str = "auto") -> torch.Tensor:
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


def build_cluster_queries_dict(hdb: HDBSCAN, queries: List[str]) -> Dict[str, List[str]]:
    """Build dictionary mapping cluster IDs to queries"""
    cluster_dict = {}
    
    for cluster_id in set(hdb.labels_):
        if cluster_id == -1:  # Skip noise
            continue
            
        cluster_str = str(cluster_id)
        cluster_dict[cluster_str] = []

        for i, label in enumerate(hdb.labels_):
            if label == cluster_id:
                cluster_dict[cluster_str].append(queries[i])
    
    return cluster_dict


def save_results(cluster_queries: Dict[str, List[str]], output_path: Path, 
                dataset_name: str, indent: int = 2) -> None:
    """Save clustering results to JSON file"""
    logger = logging.getLogger(__name__)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(cluster_queries, f, indent=indent, ensure_ascii=False)
    
    logger.info(f"Results saved to {output_path}")
    
    # Log cluster statistics
    total_queries = sum(len(queries) for queries in cluster_queries.values())
    logger.info(f"Clustered {total_queries} queries into {len(cluster_queries)} clusters")
    
    cluster_sizes = [len(queries) for queries in cluster_queries.values()]
    if cluster_sizes:
        logger.info(f"Cluster sizes - Min: {min(cluster_sizes)}, Max: {max(cluster_sizes)}, "
                   f"Mean: {sum(cluster_sizes)/len(cluster_sizes):.1f}")


def main():
    parser = argparse.ArgumentParser(
        description="Configurable clustering experiment for safety prompts",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Dataset config
    parser.add_argument("--dataset-name", default="PKU-Alignment/BeaverTails",
                       help="HuggingFace dataset name")
    parser.add_argument("--dataset-split", default="330k_train",
                       help="Dataset split to use")
    parser.add_argument("--num-samples", type=int,
                       help="Number of samples to use (default: all)")
    parser.add_argument("--filter-unsafe", action="store_true", default=True,
                       help="Only use unsafe prompts (requires 'is_safe' column)")
    parser.add_argument("--no-filter-unsafe", dest="filter_unsafe", action="store_false",
                       help="Use all prompts regardless of safety")
    
    # Model config
    parser.add_argument("--model-name", default="Qwen/Qwen3-Embedding-0.6B",
                       help="SentenceTransformer model name")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"],
                       help="Device to use for embeddings")
    
    # HDBSCAN config
    parser.add_argument("--min-cluster-size", type=int, default=50,
                       help="Minimum cluster size for HDBSCAN")
    parser.add_argument("--metric", default="cosine", choices=["cosine", "euclidean", "manhattan"],
                       help="Distance metric for clustering")
    parser.add_argument("--leaf-size", type=int, default=500,
                       help="Leaf size for HDBSCAN")
    parser.add_argument("--min-samples", type=int,
                       help="Minimum samples for HDBSCAN (default: min_cluster_size)")
    
    # Output config
    parser.add_argument("--output-path", type=Path, default=Path("."),
                       help="Output file path (directory or full path)")
    parser.add_argument("--indent", type=int, default=2,
                       help="JSON indentation")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    
    args = parser.parse_args()
    
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    try:
        # Load and filter dataset
        queries = load_and_filter_dataset(
            args.dataset_name, 
            args.dataset_split, 
            args.num_samples,
            args.filter_unsafe
        )
        
        if not queries:
            logger.error("No queries found after filtering")
            return 1
        
        # Generate embeddings
        embeddings = generate_embeddings(queries, args.model_name, args.device)
        
        # Perform clustering
        hdb = perform_clustering(
            embeddings,
            args.min_cluster_size,
            args.metric,
            args.leaf_size,
            args.min_samples
        )
        
        # Build results
        cluster_queries = build_cluster_queries_dict(hdb, queries)
        
        # Save results
        output_path = f"{args.dataset_name.replace('/', '_')}_clusters.json"
        save_results(cluster_queries, output_path, args.dataset_name, args.indent)
        
        logger.info("Experiment completed successfully")
        return 0
        
    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())