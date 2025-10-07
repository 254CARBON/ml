"""Financial domain-specific embedding model training."""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, AutoConfig
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
import mlflow
import mlflow.pytorch
import structlog

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from libs.common.config import BaseConfig
from libs.common.logging import configure_logging

logger = structlog.get_logger("financial_embedding_model")


class FinancialTextDataset(Dataset):
    """Dataset for financial text embeddings."""
    
    def __init__(self, texts: List[str], labels: Optional[List[int]] = None):
        """Create a simple text dataset.

        Parameters
        - texts: List of raw text strings
        - labels: Optional integer labels (defaults to zeros)
        """
        self.texts = texts
        self.labels = labels or [0] * len(texts)
    
    def __len__(self):
        """Return number of samples."""
        return len(self.texts)
    
    def __getitem__(self, idx):
        """Return item at index as mapping with text and label."""
        return {
            "text": self.texts[idx],
            "label": self.labels[idx]
        }


class FinancialEmbeddingModel(nn.Module):
    """Custom financial embedding model."""
    
    def __init__(
        self,
        base_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        embedding_dim: int = 384,
        financial_vocab_size: int = 1000,
        dropout: float = 0.1
    ):
        """Initialize embedding model with domain adapter layers.

        Parameters
        - base_model_name: HuggingFace model identifier to start from
        - embedding_dim: Output embedding dimension
        - financial_vocab_size: Size of auxiliary domain vocabulary
        - dropout: Dropout rate in adapter stack
        """
        super().__init__()
        
        self.base_model_name = base_model_name
        self.embedding_dim = embedding_dim
        
        # Load base transformer
        self.transformer = AutoModel.from_pretrained(base_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        
        # Financial domain adaptation layers
        transformer_dim = self.transformer.config.hidden_size
        
        self.domain_adapter = nn.Sequential(
            nn.Linear(transformer_dim, embedding_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.Tanh()
        )
        
        # Financial term embedding layer
        self.financial_embeddings = nn.Embedding(financial_vocab_size, embedding_dim)
        
        # Attention mechanism for combining embeddings
        self.attention = nn.MultiheadAttention(embedding_dim, num_heads=8, batch_first=True)
        
        # Output projection
        self.output_projection = nn.Linear(embedding_dim, embedding_dim)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(embedding_dim)
    
    def forward(self, input_ids, attention_mask, financial_terms=None):
        """Forward pass."""
        
        # Get transformer embeddings
        transformer_outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = transformer_outputs.last_hidden_state
        
        # Pool transformer embeddings (mean pooling)
        masked_output = sequence_output * attention_mask.unsqueeze(-1)
        pooled_output = masked_output.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
        
        # Apply domain adapter
        domain_embedding = self.domain_adapter(pooled_output)
        
        # If financial terms are provided, incorporate them
        if financial_terms is not None:
            financial_emb = self.financial_embeddings(financial_terms)
            financial_pooled = financial_emb.mean(dim=1)  # Pool financial term embeddings
            
            # Combine using attention
            combined_input = torch.stack([domain_embedding, financial_pooled], dim=1)
            attended_output, _ = self.attention(combined_input, combined_input, combined_input)
            final_embedding = attended_output.mean(dim=1)
        else:
            final_embedding = domain_embedding
        
        # Apply output projection and normalization
        output = self.output_projection(final_embedding)
        output = self.layer_norm(output)
        
        return output
    
    def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Encode texts to embeddings."""
        
        self.eval()
        embeddings = []
        
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                
                # Tokenize
                encoded = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt"
                )
                
                # Forward pass
                batch_embeddings = self.forward(
                    input_ids=encoded["input_ids"],
                    attention_mask=encoded["attention_mask"]
                )
                
                embeddings.append(batch_embeddings.cpu().numpy())
        
        return np.vstack(embeddings)


class FinancialEmbeddingTrainer:
    """Trainer for financial domain embedding models."""
    
    def __init__(self, config: Dict[str, Any]):
        """Create a trainer with config dict.

        Expected keys are used for dataset paths, training params, etc.
        """
        self.config = config
        self.financial_vocabulary = self._build_financial_vocabulary()
        self.model: Optional[FinancialEmbeddingModel] = None
        self.sentence_transformer: Optional[SentenceTransformer] = None
    
    def _build_financial_vocabulary(self) -> Dict[str, int]:
        """Build financial domain vocabulary."""
        
        # Common financial terms and their variations
        financial_terms = [
            # Instruments
            "futures", "forward", "option", "swap", "bond", "equity", "derivative",
            "contract", "security", "instrument", "asset",
            
            # Commodities
            "crude", "oil", "gas", "natural", "heating", "gasoline", "gold", "silver",
            "copper", "aluminum", "wheat", "corn", "soybean", "coffee", "sugar",
            
            # Energy
            "wti", "brent", "henry", "hub", "nymex", "ice", "rbob", "ulsd",
            "pipeline", "refinery", "storage", "inventory",
            
            # Rates and FX
            "yield", "rate", "treasury", "government", "corporate", "municipal",
            "libor", "sofr", "fed", "funds", "discount", "prime",
            "dollar", "euro", "pound", "yen", "franc", "currency", "forex",
            
            # Market terms
            "volatility", "volume", "open", "interest", "settlement", "expiry",
            "maturity", "tenor", "duration", "convexity", "spread", "basis",
            
            # Trading terms
            "bid", "ask", "mid", "last", "high", "low", "close", "settlement",
            "margin", "collateral", "clearing", "exchange", "otc",
            
            # Risk terms
            "var", "stress", "scenario", "monte", "carlo", "backtest",
            "correlation", "covariance", "beta", "alpha", "sharpe",
            
            # Time periods
            "daily", "weekly", "monthly", "quarterly", "annual",
            "overnight", "intraday", "front", "back", "calendar",
            
            # Market conditions
            "bull", "bear", "volatile", "stable", "trending", "ranging",
            "liquid", "illiquid", "tight", "wide", "active", "inactive"
        ]
        
        # Create vocabulary mapping
        vocab = {"<UNK>": 0, "<PAD>": 1}
        for i, term in enumerate(financial_terms, 2):
            vocab[term] = i
            # Add variations
            vocab[term.upper()] = len(vocab)
            vocab[term.capitalize()] = len(vocab)
        
        logger.info("Financial vocabulary built", size=len(vocab))
        return vocab
    
    def create_training_data(self, real_data_path: Optional[str] = None) -> List[InputExample]:
        """Create training data for financial embeddings."""
        
        training_examples = []
        
        if real_data_path and os.path.exists(real_data_path):
            # Load real financial data
            logger.info("Loading real financial data", path=real_data_path)
            df = pd.read_parquet(real_data_path)
            
            # Extract text descriptions and create pairs
            if "text" in df.columns and "entity_type" in df.columns:
                # Create positive pairs (same entity type)
                entity_groups = df.groupby("entity_type")
                
                for entity_type, group in entity_groups:
                    texts = group["text"].tolist()
                    
                    # Create positive pairs within same entity type
                    for i in range(len(texts)):
                        for j in range(i + 1, min(i + 5, len(texts))):  # Limit pairs per text
                            training_examples.append(
                                InputExample(texts=[texts[i], texts[j]], label=0.8)
                            )
                
                # Create negative pairs (different entity types)
                entity_types = df["entity_type"].unique()
                for i, type1 in enumerate(entity_types):
                    for type2 in entity_types[i + 1:]:
                        texts1 = df[df["entity_type"] == type1]["text"].sample(min(10, len(df))).tolist()
                        texts2 = df[df["entity_type"] == type2]["text"].sample(min(10, len(df))).tolist()
                        
                        for t1 in texts1:
                            for t2 in texts2[:2]:  # Limit negative pairs
                                training_examples.append(
                                    InputExample(texts=[t1, t2], label=0.2)
                                )
        
        else:
            # Generate synthetic financial text pairs
            logger.info("Generating synthetic financial training data")
            
            # Instrument descriptions
            instruments = [
                "Natural gas Henry Hub futures contract front month",
                "Crude oil WTI futures contract front month",
                "US Treasury 10-year bond yield",
                "EUR/USD foreign exchange spot rate",
                "S&P 500 equity index futures",
                "Gold futures commodity contract",
                "Heating oil futures contract",
                "RBOB gasoline futures contract",
                "Silver futures commodity contract",
                "Copper futures contract"
            ]
            
            # Curve descriptions
            curves = [
                "US Dollar interest rate swap curve",
                "Euro government bond yield curve",
                "UK Gilt yield curve",
                "Japanese government bond yield curve",
                "Corporate credit spread curve",
                "Municipal bond yield curve"
            ]
            
            # Scenario descriptions
            scenarios = [
                "Federal Reserve interest rate hike scenario",
                "Oil supply disruption stress test",
                "Currency devaluation scenario",
                "Credit spread widening scenario",
                "Inflation spike scenario",
                "Recession scenario analysis"
            ]
            
            all_texts = instruments + curves + scenarios
            
            # Create positive pairs (similar financial concepts)
            for i, text1 in enumerate(all_texts):
                for j, text2 in enumerate(all_texts):
                    if i != j:
                        # Calculate semantic similarity based on shared terms
                        similarity = self._calculate_text_similarity(text1, text2)
                        if similarity > 0.3:  # Threshold for positive pairs
                            training_examples.append(
                                InputExample(texts=[text1, text2], label=similarity)
                            )
        
        logger.info("Training data created", examples=len(training_examples))
        return training_examples
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple text similarity for synthetic data."""
        
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        # Jaccard similarity
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def fine_tune_sentence_transformer(
        self,
        base_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        training_data: Optional[List[InputExample]] = None,
        output_path: str = "models/financial_embedding_model",
        epochs: int = 4,
        batch_size: int = 16,
        learning_rate: float = 2e-5
    ) -> SentenceTransformer:
        """Fine-tune sentence transformer on financial data."""
        
        logger.info("Fine-tuning sentence transformer", 
                   base_model=base_model,
                   epochs=epochs,
                   batch_size=batch_size)
        
        # Load base model
        model = SentenceTransformer(base_model)
        
        # Create training data if not provided
        if training_data is None:
            training_data = self.create_training_data()
        
        # Create data loader
        train_dataloader = DataLoader(training_data, shuffle=True, batch_size=batch_size)
        
        # Define loss function
        train_loss = losses.CosineSimilarityLoss(model)
        
        # Create evaluation data
        eval_examples = training_data[:100]  # Use first 100 for evaluation
        evaluator = EmbeddingSimilarityEvaluator.from_input_examples(
            eval_examples, 
            name="financial_eval"
        )
        
        # Start MLflow run
        with mlflow.start_run(run_name="financial_embedding_fine_tune"):
            # Log parameters
            mlflow.log_param("base_model", base_model)
            mlflow.log_param("epochs", epochs)
            mlflow.log_param("batch_size", batch_size)
            mlflow.log_param("learning_rate", learning_rate)
            mlflow.log_param("training_examples", len(training_data))
            
            # Fine-tune model
            model.fit(
                train_objectives=[(train_dataloader, train_loss)],
                epochs=epochs,
                evaluator=evaluator,
                evaluation_steps=500,
                warmup_steps=100,
                output_path=output_path,
                optimizer_params={"lr": learning_rate}
            )
            
            # Evaluate final model
            final_score = evaluator(model)
            mlflow.log_metric("final_evaluation_score", final_score)
            
            # Log model
            mlflow.pytorch.log_model(
                pytorch_model=model,
                artifact_path="financial_embedding_model",
                registered_model_name="financial_embeddings"
            )
            
            # Test embedding generation
            test_texts = [
                "Natural gas Henry Hub futures contract",
                "US Treasury 10-year bond yield",
                "EUR/USD foreign exchange rate"
            ]
            
            embeddings = model.encode(test_texts)
            mlflow.log_metric("embedding_dimension", embeddings.shape[1])
            mlflow.log_metric("test_embedding_norm", np.linalg.norm(embeddings[0]))
            
            logger.info("Fine-tuning completed", 
                       final_score=final_score,
                       output_path=output_path)
        
        self.sentence_transformer = model
        return model
    
    def train_custom_model(
        self,
        training_data: Optional[List[InputExample]] = None,
        epochs: int = 10,
        batch_size: int = 32,
        learning_rate: float = 1e-4
    ) -> FinancialEmbeddingModel:
        """Train custom financial embedding model."""
        
        logger.info("Training custom financial embedding model")
        
        # Create model
        model = FinancialEmbeddingModel(
            financial_vocab_size=len(self.financial_vocabulary)
        )
        
        # Create training data if not provided
        if training_data is None:
            training_data = self.create_training_data()
        
        # Convert to dataset format
        texts = []
        labels = []
        for example in training_data:
            texts.extend(example.texts)
            labels.extend([example.label] * len(example.texts))
        
        dataset = FinancialTextDataset(texts, labels)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Training setup
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        
        # Start MLflow run
        with mlflow.start_run(run_name="custom_financial_embedding"):
            mlflow.log_param("model_type", "custom_financial")
            mlflow.log_param("epochs", epochs)
            mlflow.log_param("batch_size", batch_size)
            mlflow.log_param("learning_rate", learning_rate)
            
            # Training loop
            model.train()
            for epoch in range(epochs):
                epoch_loss = 0
                batch_count = 0
                
                for batch in dataloader:
                    optimizer.zero_grad()
                    
                    # Tokenize batch
                    encoded = model.tokenizer(
                        batch["text"],
                        padding=True,
                        truncation=True,
                        max_length=512,
                        return_tensors="pt"
                    )
                    
                    # Forward pass
                    embeddings = model(
                        input_ids=encoded["input_ids"],
                        attention_mask=encoded["attention_mask"]
                    )
                    
                    # Create target embeddings (simplified)
                    targets = torch.randn_like(embeddings)  # In reality, use contrastive learning
                    
                    # Calculate loss
                    loss = criterion(embeddings, targets)
                    
                    # Backward pass
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                    batch_count += 1
                
                avg_loss = epoch_loss / batch_count
                mlflow.log_metric("epoch_loss", avg_loss, step=epoch)
                
                logger.info(f"Epoch {epoch + 1}/{epochs}", loss=avg_loss)
            
            # Save model
            model_path = "models/custom_financial_embedding.pth"
            torch.save(model.state_dict(), model_path)
            
            # Log model artifact
            mlflow.log_artifact(model_path)
            
            logger.info("Custom model training completed")
        
        self.model = model
        return model
    
    def evaluate_embedding_quality(
        self,
        model: SentenceTransformer,
        test_data: Optional[List[Tuple[str, str, float]]] = None
    ) -> Dict[str, float]:
        """Evaluate embedding quality on financial domain tasks."""
        
        logger.info("Evaluating embedding quality")
        
        if test_data is None:
            # Create test pairs
            test_data = [
                ("Natural gas futures contract", "Gas futures commodity", 0.9),
                ("US Treasury bond yield", "Government bond rate", 0.8),
                ("EUR/USD exchange rate", "Euro dollar currency pair", 0.9),
                ("Crude oil price", "Natural gas price", 0.6),
                ("Stock index futures", "Bond yield curve", 0.2),
                ("Gold commodity price", "Silver precious metal", 0.7)
            ]
        
        # Generate embeddings for test pairs
        texts1 = [pair[0] for pair in test_data]
        texts2 = [pair[1] for pair in test_data]
        expected_similarities = [pair[2] for pair in test_data]
        
        embeddings1 = model.encode(texts1)
        embeddings2 = model.encode(texts2)
        
        # Calculate cosine similarities
        similarities = []
        for emb1, emb2 in zip(embeddings1, embeddings2):
            similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
            similarities.append(similarity)
        
        # Calculate evaluation metrics
        correlation = np.corrcoef(similarities, expected_similarities)[0, 1]
        mae = np.mean(np.abs(np.array(similarities) - np.array(expected_similarities)))
        rmse = np.sqrt(np.mean((np.array(similarities) - np.array(expected_similarities)) ** 2))
        
        metrics = {
            "correlation": correlation,
            "mae": mae,
            "rmse": rmse,
            "avg_similarity": np.mean(similarities),
            "similarity_std": np.std(similarities)
        }
        
        logger.info("Embedding quality evaluation completed", **metrics)
        return metrics
    
    def create_multi_modal_embeddings(
        self,
        text_data: List[str],
        numerical_data: np.ndarray,
        output_dim: int = 512
    ) -> np.ndarray:
        """Create multi-modal embeddings combining text and numerical data."""
        
        logger.info("Creating multi-modal embeddings", 
                   text_samples=len(text_data),
                   numerical_features=numerical_data.shape[1])
        
        # Generate text embeddings
        if self.sentence_transformer:
            text_embeddings = self.sentence_transformer.encode(text_data)
        else:
            # Use default model
            model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            text_embeddings = model.encode(text_data)
        
        # Normalize numerical data
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        numerical_normalized = scaler.fit_transform(numerical_data)
        
        # Combine embeddings
        text_dim = text_embeddings.shape[1]
        numerical_dim = numerical_normalized.shape[1]
        
        # Create projection layers to common dimension
        text_projection = np.random.randn(text_dim, output_dim // 2)
        numerical_projection = np.random.randn(numerical_dim, output_dim // 2)
        
        # Project to common space
        text_projected = text_embeddings @ text_projection
        numerical_projected = numerical_normalized @ numerical_projection
        
        # Concatenate
        multi_modal_embeddings = np.concatenate([text_projected, numerical_projected], axis=1)
        
        # Normalize final embeddings
        norms = np.linalg.norm(multi_modal_embeddings, axis=1, keepdims=True)
        multi_modal_embeddings = multi_modal_embeddings / norms
        
        logger.info("Multi-modal embeddings created", output_dim=multi_modal_embeddings.shape[1])
        return multi_modal_embeddings


async def main():
    """Main training function."""
    
    # Configure logging
    configure_logging("financial_embedding_trainer", "INFO", "json")
    
    # Create trainer
    config = {
        "base_model": "sentence-transformers/all-MiniLM-L6-v2",
        "embedding_dim": 384,
        "epochs": 4,
        "batch_size": 16
    }
    
    trainer = FinancialEmbeddingTrainer(config)
    
    # Fine-tune sentence transformer
    model = trainer.fine_tune_sentence_transformer(
        base_model=config["base_model"],
        epochs=config["epochs"],
        batch_size=config["batch_size"]
    )
    
    # Evaluate model
    quality_metrics = trainer.evaluate_embedding_quality(model)
    print("Embedding Quality Metrics:")
    for metric, value in quality_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Test multi-modal embeddings
    test_texts = [
        "Natural gas futures contract",
        "US Treasury bond yield",
        "EUR/USD exchange rate"
    ]
    
    test_numerical = np.random.randn(3, 10)  # 3 samples, 10 features
    
    multi_modal_emb = trainer.create_multi_modal_embeddings(test_texts, test_numerical)
    print(f"Multi-modal embeddings shape: {multi_modal_emb.shape}")


if __name__ == "__main__":
    asyncio.run(main())
