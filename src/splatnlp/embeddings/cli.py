import argparse
import io
import logging
import os
import sqlite3

import boto3
import numpy as np
import orjson
import pandas as pd
import torch
from gensim.models import Doc2Vec

from splatnlp.embeddings.clustering import cluster_vectors
from splatnlp.embeddings.dimensionality_reduction import (
    reduce_doc2vec_dimensions_by_tag,
)
from splatnlp.embeddings.inference import infer_doc2vec_vectors
from splatnlp.embeddings.load import (
    load_doc2vec_model,
    load_tokenized_data,
    load_vocab_json,
)
from splatnlp.embeddings.train import train_doc2vec_embeddings
from splatnlp.preprocessing.datasets.generate_datasets import (
    generate_tokenized_datasets,
)


def load_data(data_path, table_name=None):
    """Load data from various sources (local file, S3, SQLite)."""
    if data_path.startswith(("http://", "https://", "s3://")):
        if data_path.startswith("s3://"):
            s3 = boto3.client("s3")
            bucket, key = data_path[5:].split("/", 1)
            response = s3.get_object(Bucket=bucket, Key=key)
            content = response["Body"].read()
        else:
            import requests

            response = requests.get(data_path)
            content = response.content
        return pd.read_csv(io.BytesIO(content), sep="\t", header=0)
    elif data_path.endswith(".db") or data_path.endswith(".sqlite"):
        conn = sqlite3.connect(data_path)
        if table_name:
            return pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
        else:
            return pd.read_sql_query("SELECT * FROM data", conn)
    else:
        return pd.read_csv(data_path, sep="\t")


def main():
    parser = argparse.ArgumentParser(
        description="Train and use Doc2Vec embeddings for SplatNLP"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to the tokenized dataset (local file, S3 path, or SQLite DB)",
    )
    parser.add_argument(
        "--vocab_path",
        type=str,
        required=True,
        help="Path to the vocabulary JSON file (local file or S3 path)",
    )
    parser.add_argument(
        "--weapon_vocab_path",
        type=str,
        required=True,
        help="Path to the weapon vocabulary JSON file (local file or S3 path)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save the trained model and outputs",
    )
    parser.add_argument(
        "--vector_size",
        type=int,
        default=100,
        help="Dimension of the Doc2Vec vectors",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=5,
        help="Context window size for Doc2Vec",
    )
    parser.add_argument(
        "--min_count",
        type=int,
        default=1,
        help="Minimum frequency for tokens in Doc2Vec",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--dm",
        type=int,
        default=1,
        help="Doc2Vec training algorithm (1=PV-DM, 0=PV-DBOW)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of worker threads for training",
    )
    parser.add_argument(
        "--table_name",
        type=str,
        help="Table name for SQLite database (optional)",
    )
    parser.add_argument(
        "--verbose",
        type=bool,
        default=False,
        help="Enable verbose output",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "infer", "reduce", "cluster"],
        default="train",
        help="Operation mode: train, infer, reduce, or cluster",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        help="Path to a trained Doc2Vec model (required for infer/reduce/cluster modes)",
    )
    parser.add_argument(
        "--n_components",
        type=int,
        default=2,
        help="Number of components for dimensionality reduction",
    )
    parser.add_argument(
        "--perplexity",
        type=float,
        default=30.0,
        help="Perplexity for t-SNE",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=1000,
        help="Number of iterations for t-SNE",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=0.5,
        help="DBSCAN epsilon parameter",
    )
    parser.add_argument(
        "--min_samples",
        type=int,
        default=5,
        help="DBSCAN min_samples parameter",
    )
    parser.add_argument(
        "--umap_neighbors",
        type=int,
        default=15,
        help="Number of neighbors for UMAP",
    )
    parser.add_argument(
        "--umap_min_dist",
        type=float,
        default=0.1,
        help="Minimum distance for UMAP",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        help="Random seed for reproducibility",
    )

    args = parser.parse_args()

    # Configure logging based on verbose flag
    if args.verbose:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )
    else:
        logging.basicConfig(
            level=logging.WARNING,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )

    logger = logging.getLogger()

    logger.info("Loading data and vocabulary...")

    # Load data and vocabulary
    df = load_data(args.data_path, args.table_name)
    logger.info(f"Loaded {len(df)} rows from {args.data_path}")
    logger.info(f"Columns: {df.columns}")
    logger.info(f"Heads: {df.head()}")

    # Parse ability_tags if they're strings
    if isinstance(df["ability_tags"].iloc[0], str):
        df["ability_tags"] = df["ability_tags"].apply(orjson.loads)

    # Load vocabularies
    vocab = load_vocab_json(args.vocab_path)
    weapon_vocab = load_vocab_json(args.weapon_vocab_path)

    logger.info("Generating datasets...")

    # Generate datasets using the existing function
    train_df, val_df, test_df = generate_tokenized_datasets(
        df,
        frac=1.0,  # Use full dataset
        validation_size=0.1,
        test_size=0.2,
        random_state=args.random_state,
    )

    if args.mode == "train":
        logger.info("Training Doc2Vec model...")

        # Train the model
        model = train_doc2vec_embeddings(
            df=train_df,
            vector_size=args.vector_size,
            window=args.window,
            min_count=args.min_count,
            epochs=args.epochs,
            dm=args.dm,
            workers=args.workers,
        )

        # Save the model
        os.makedirs(args.output_dir, exist_ok=True)
        model.save(os.path.join(args.output_dir, "doc2vec.model"))
        logger.info(f"Model saved to {args.output_dir}")

    elif args.mode in ["infer", "reduce", "cluster"]:
        if not args.model_path:
            raise ValueError(
                "--model_path is required for infer/reduce/cluster modes"
            )

        logger.info(f"Loading model from {args.model_path}...")
        model = load_doc2vec_model(args.model_path)

        if args.mode == "infer":
            logger.info("Inferring vectors...")
            vectors = infer_doc2vec_vectors(model, test_df["ability_tags"])

            # Save inferred vectors
            os.makedirs(args.output_dir, exist_ok=True)
            np.save(
                os.path.join(args.output_dir, "inferred_vectors.npy"), vectors
            )
            logger.info(f"Inferred vectors saved to {args.output_dir}")

        elif args.mode == "reduce":
            logger.info("Reducing dimensions...")
            reduced_vectors, tags = reduce_doc2vec_dimensions_by_tag(
                model=model,
                tags_to_reduce=None,  # Use all tags
                n_components=args.n_components,
                perplexity=args.perplexity,
                n_iter=args.n_iter,
                random_state=args.random_state,
            )

            # Save reduced vectors and tags
            os.makedirs(args.output_dir, exist_ok=True)
            np.save(
                os.path.join(args.output_dir, "reduced_vectors.npy"),
                reduced_vectors,
            )
            with open(
                os.path.join(args.output_dir, "reduced_tags.json"), "wb"
            ) as f:
                f.write(orjson.dumps(tags))
            logger.info(f"Reduced vectors and tags saved to {args.output_dir}")

        elif args.mode == "cluster":
            logger.info("Clustering vectors...")
            # First reduce dimensions
            reduced_vectors, tags = reduce_doc2vec_dimensions_by_tag(
                model=model,
                tags_to_reduce=None,  # Use all tags
                n_components=args.n_components,
                perplexity=args.perplexity,
                n_iter=args.n_iter,
                random_state=args.random_state,
            )

            # Then cluster
            clustered_vectors, cluster_labels = cluster_vectors(
                vector_array=reduced_vectors,
                n_dimensions=args.n_components,
                eps=args.eps,
                min_samples=args.min_samples,
                umap_neighbors=args.umap_neighbors,
                umap_min_dist=args.umap_min_dist,
                random_state=args.random_state,
            )

            # Save results
            os.makedirs(args.output_dir, exist_ok=True)
            np.save(
                os.path.join(args.output_dir, "clustered_vectors.npy"),
                clustered_vectors,
            )
            np.save(
                os.path.join(args.output_dir, "cluster_labels.npy"),
                cluster_labels,
            )
            with open(
                os.path.join(args.output_dir, "cluster_tags.json"), "wb"
            ) as f:
                f.write(orjson.dumps(tags))
            logger.info(f"Clustering results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
