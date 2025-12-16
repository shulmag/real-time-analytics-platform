"""
Siamese Network Module for Municipal Bond Embeddings
Handles network architecture and training with proper temporal alignment
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import pickle
from typing import Tuple, Dict, List, Optional
from .feature_engineering import engineer_features_complete

# -- Add this block near the top (after imports) --
from tensorflow import keras
from tensorflow.keras import layers, Model, Input
import tensorflow as tf

from tensorflow.keras.utils import register_keras_serializable 

@register_keras_serializable(package="ficc")
class L2Normalize(layers.Layer):
    def __init__(self, axis=1, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis

    def call(self, x):
        return tf.nn.l2_normalize(x, axis=self.axis)

    def get_config(self):
        return {"axis": self.axis, **super().get_config()}



# ===========================
# NETWORK ARCHITECTURE
# ===========================

def create_base_network(input_dim: int, embedding_dim: int = 128) -> Model:
    inputs = Input(shape=(input_dim,), name="input_layer")

    x = layers.Dense(512, activation='relu', name="dense")(inputs)
    x = layers.BatchNormalization(name="batch_normalization")(x)
    x = layers.Dropout(0.1, name="dropout")(x)

    x = layers.Dense(256, activation='relu', name="dense_1")(x)
    x = layers.BatchNormalization(name="batch_normalization_1")(x)
    x = layers.Dropout(0.1, name="dropout_1")(x)

    x = layers.Dense(256, activation='relu', name="dense_2")(x)
    x = layers.BatchNormalization(name="batch_normalization_2")(x)
    x = layers.Dropout(0.1, name="dropout_2")(x)

    embeddings = layers.Dense(embedding_dim, activation='linear', name='embeddings')(x)

    outputs = L2Normalize(axis=1, name="l2norm")(embeddings)

    model = Model(inputs=inputs, outputs=outputs, name='base_network')
    return model



def create_siamese_network(input_dim: int, embedding_dim: int = 128) -> Tuple[Model, Model]:
    """
    Create the full Siamese network
    
    Args:
        input_dim: Dimension of input features
        embedding_dim: Dimension of embeddings
    
    Returns:
        Tuple of (siamese_model, base_network)
    """
    # Create shared base network
    base_network = create_base_network(input_dim, embedding_dim)
    
    # Create inputs for two CUSIPs
    input_a = Input(shape=(input_dim,), name='input_a')
    input_b = Input(shape=(input_dim,), name='input_b')
    
    # Generate embeddings using shared weights
    embedding_a = base_network(input_a)
    embedding_b = base_network(input_b)
    
    # Compute cosine similarity (dot product of L2-normalized vectors)
    cosine_similarity = layers.Dot(axes=1, normalize=False)([embedding_a, embedding_b])
    
    # Full Siamese model
    siamese_model = Model(inputs=[input_a, input_b], outputs=cosine_similarity)
    
    return siamese_model, base_network

def contrastive_loss(y_true, y_pred, margin: float = 0.5):
    """
    Contrastive loss for Siamese network
    
    Args:
        y_true: True labels (1 for similar, 0 for dissimilar)
        y_pred: Predicted similarity (cosine similarity)
        margin: Margin for negative pairs
    
    Returns:
        Loss value
    """
    # Distance is 1 - cosine_similarity
    y_pred_dist = 1 - y_pred
    
    # Loss for positive pairs: want distance to be 0
    pos_loss = y_true * tf.square(y_pred_dist)
    
    # Loss for negative pairs: want distance to be at least margin
    neg_loss = (1 - y_true) * tf.square(tf.maximum(0.0, margin - y_pred_dist))
    
    return tf.reduce_mean(pos_loss + neg_loss)

# ===========================
# DATA PREPARATION WITH TEMPORAL ALIGNMENT
# ===========================

def prepare_pairs_for_training_temporal(
    features_df: pd.DataFrame, 
    pairs_df: pd.DataFrame, 
    artifacts: Optional[Dict] = None
) -> Tuple[Tuple[np.ndarray, np.ndarray], np.ndarray, Dict, List[str]]:
    """
    Prepare pairs for training with TEMPORAL ALIGNMENT
    This is the KEY function that ensures point-in-time consistency
    
    Args:
        features_df: DataFrame with all trades and their features
        pairs_df: DataFrame with pairs including temporal metadata
        artifacts: Pre-fitted feature engineering artifacts
    
    Returns:
        Tuple of ((features_a, features_b), labels, artifacts, feature_names)
    """
    print("Preparing temporally-aligned pairs for training...")
    
    # Ensure date columns are datetime
    date_columns = [
        'refund_date', 'accrual_date', 'dated_date', 'next_sink_date',
        'delivery_date', 'trade_date', 'trade_datetime', 'par_call_date',
        'maturity_date', 'settlement_date', 'next_call_date',
        'previous_coupon_payment_date', 'next_coupon_payment_date',
        'first_coupon_date', 'last_period_accrues_from_date'
    ]
    
    for col in date_columns:
        if col in features_df.columns:
            features_df[col] = pd.to_datetime(features_df[col], errors='coerce')
    
    # Create a mapping from (CUSIP, datetime) to feature vector
    print("Creating temporal feature index...")
    
    # First, engineer features for ALL trades
    X_all, artifacts, feature_names = engineer_features_complete(
        features_df, 
        fit=(artifacts is None), 
        artifacts=artifacts
    )
    
    print(f"Feature dimension: {X_all.shape[1]}")
    print(f"Total trades processed: {X_all.shape[0]}")
    
    # Create temporal index: (cusip, trade_datetime) -> feature_vector
    temporal_feature_index = {}
    
    # Reset index to ensure alignment between features_df and X_all
    features_df_reset = features_df.reset_index(drop=True)
    
    for idx in range(len(features_df_reset)):
        row = features_df_reset.iloc[idx]
        # Use trade_datetime as the temporal key
        key = (row['cusip'], pd.Timestamp(row['trade_datetime']))
        temporal_feature_index[key] = X_all[idx]
    
    print(f"Created temporal index with {len(temporal_feature_index)} entries")
    
    # Now prepare pairs using temporal alignment
    features_a = []
    features_b = []
    labels = []
    missing_temporal_matches = 0
    successful_matches = 0
    
    for _, pair_row in pairs_df.iterrows():
        cusip1 = pair_row['cusip1']
        cusip2 = pair_row['cusip2']
        
        # Use the specific datetime for each CUSIP in the pair
        cusip1_dt = pd.Timestamp(pair_row.get('cusip1_datetime', pair_row.get('window_datetime')))
        cusip2_dt = pd.Timestamp(pair_row.get('cusip2_datetime', pair_row.get('window_datetime')))
        
        key1 = (cusip1, cusip1_dt)
        key2 = (cusip2, cusip2_dt)
        
        # Try exact match first
        if key1 in temporal_feature_index and key2 in temporal_feature_index:
            features_a.append(temporal_feature_index[key1])
            features_b.append(temporal_feature_index[key2])
            labels.append(pair_row['label'])
            successful_matches += 1
        else:
            # If exact match fails, try to find closest timestamp within window
            # This handles minor timestamp misalignments
            found_match = False
            
            # Find closest match for cusip1
            cusip1_matches = [(k, v) for k, v in temporal_feature_index.items() 
                            if k[0] == cusip1 and abs((k[1] - cusip1_dt).total_seconds()) < 60]
            cusip2_matches = [(k, v) for k, v in temporal_feature_index.items() 
                            if k[0] == cusip2 and abs((k[1] - cusip2_dt).total_seconds()) < 60]
            
            if cusip1_matches and cusip2_matches:
                # Use closest match
                features_a.append(cusip1_matches[0][1])
                features_b.append(cusip2_matches[0][1])
                labels.append(pair_row['label'])
                successful_matches += 1
                found_match = True
            
            if not found_match:
                missing_temporal_matches += 1
    
    print(f"\nTemporal matching results:")
    print(f"  Successful matches: {successful_matches:,}")
    print(f"  Missing temporal matches: {missing_temporal_matches:,}")
    print(f"  Match rate: {successful_matches / len(pairs_df):.1%}")
    
    if successful_matches == 0:
        raise ValueError("No successful temporal matches found! Check data alignment.")
    
    # Convert to numpy arrays
    X_pairs = (np.array(features_a, dtype=np.float32), 
               np.array(features_b, dtype=np.float32))
    y = np.array(labels, dtype=np.float32)
    
    print(f"\nFinal training data shape:")
    print(f"  Features A: {X_pairs[0].shape}")
    print(f"  Features B: {X_pairs[1].shape}")
    print(f"  Labels: {y.shape}")
    print(f"  Positive ratio: {y.mean():.2%}")
    
    return X_pairs, y, artifacts, feature_names

# ===========================
# TRAINING PIPELINE
# ===========================

def train_siamese_network(
    X_train: Tuple[np.ndarray, np.ndarray], 
    y_train: np.ndarray, 
    X_val: Tuple[np.ndarray, np.ndarray], 
    y_val: np.ndarray,
    input_dim: int, 
    embedding_dim: int = 128,
    epochs: int = 100, 
    batch_size: int = 256
) -> Tuple[Model, Model, keras.callbacks.History]:
    """
    Train the Siamese network
    
    Args:
        X_train: Training features (tuple of two arrays)
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        input_dim: Input feature dimension
        embedding_dim: Embedding dimension
        epochs: Number of epochs
        batch_size: Batch size
    
    Returns:
        Tuple of (siamese_model, base_network, history)
    """
    # Create network
    siamese_model, base_network = create_siamese_network(input_dim, embedding_dim)
    
    # Compile with custom loss
    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    siamese_model.compile(
        optimizer=optimizer,
        loss=contrastive_loss,
        metrics=['mae']
    )
    
    print("\nBase network summary:")
    base_network.summary()
    
    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss', 
            patience=15, 
            restore_best_weights=True, 
            verbose=1
        ),
        ModelCheckpoint(
            'best_siamese_model.keras', 
            save_best_only=True, 
            monitor='val_loss', 
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss', 
            factor=0.5, 
            patience=5, 
            min_lr=1e-6, 
            verbose=1
        )
    ]
    
    print("\nStarting training...")
    history = siamese_model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    return siamese_model, base_network, history

def get_embeddings(
    cusip_features_df: pd.DataFrame, 
    base_network: Model, 
    artifacts: Dict
) -> pd.DataFrame:
    """
    Generate embeddings for CUSIPs
    
    Args:
        cusip_features_df: DataFrame with CUSIP features
        base_network: Trained base network
        artifacts: Feature engineering artifacts
    
    Returns:
        DataFrame with CUSIP and embedding columns
    """
    # Ensure date columns are datetime
    date_columns = [
        'refund_date', 'accrual_date', 'dated_date', 'next_sink_date',
        'delivery_date', 'trade_date', 'trade_datetime', 'par_call_date',
        'maturity_date', 'settlement_date', 'next_call_date',
        'previous_coupon_payment_date', 'next_coupon_payment_date',
        'first_coupon_date', 'last_period_accrues_from_date'
    ]
    
    for col in date_columns:
        if col in cusip_features_df.columns:
            cusip_features_df[col] = pd.to_datetime(cusip_features_df[col], errors='coerce')
    
    # Engineer features
    X_features, _, _ = engineer_features_complete(
        cusip_features_df, 
        fit=False, 
        artifacts=artifacts
    )
    
    # Generate embeddings
    embeddings = base_network.predict(X_features, batch_size=256)
    
    # Create dataframe
    embedding_cols = [f'emb_{i}' for i in range(embeddings.shape[1])]
    embeddings_df = pd.DataFrame(embeddings, columns=embedding_cols)
    embeddings_df['cusip'] = cusip_features_df['cusip'].values
    
    # Add temporal metadata if available
    if 'trade_datetime' in cusip_features_df.columns:
        embeddings_df['trade_datetime'] = cusip_features_df['trade_datetime'].values
    
    # Reorder columns
    cols = ['cusip']
    if 'trade_datetime' in embeddings_df.columns:
        cols.append('trade_datetime')
    cols.extend(embedding_cols)
    embeddings_df = embeddings_df[cols]
    
    return embeddings_df

def run_training_pipeline(
    features_df: pd.DataFrame, 
    pairs_df: pd.DataFrame, 
    test_size: float = 0.2, 
    embedding_dim: int = 128, 
    epochs: int = 100
) -> Tuple[Model, Dict, keras.callbacks.History]:
    """
    Run the complete training pipeline with temporal alignment
    
    Args:
        features_df: DataFrame with all trade features
        pairs_df: DataFrame with pairs including temporal metadata
        test_size: Validation split ratio
        embedding_dim: Embedding dimension
        epochs: Number of training epochs
    
    Returns:
        Tuple of (base_network, artifacts, history)
    """
    print("=" * 60)
    print("SIAMESE NETWORK TRAINING PIPELINE")
    print("=" * 60)
    
    # Prepare features and pairs with temporal alignment
    X_pairs, y, artifacts, feature_names = prepare_pairs_for_training_temporal(
        features_df, 
        pairs_df
    )
    
    # Split data
    print("\nSplitting data...")
    X_train_a, X_val_a, X_train_b, X_val_b, y_train, y_val = train_test_split(
        X_pairs[0], X_pairs[1], y, 
        test_size=test_size, 
        random_state=42, 
        stratify=y
    )
    X_train = (X_train_a, X_train_b)
    X_val = (X_val_a, X_val_b)
    
    input_dim = X_train[0].shape[1]
    
    print(f"\nData summary:")
    print(f"  Input dimension: {input_dim}")
    print(f"  Number of features: {len(feature_names)}")
    print(f"  Training samples: {len(y_train):,}")
    print(f"  Validation samples: {len(y_val):,}")
    print(f"  Positive ratio in train: {y_train.mean():.2%}")
    print(f"  Positive ratio in val: {y_val.mean():.2%}")
    
    # Train network
    siamese_model, base_network, history = train_siamese_network(
        X_train, y_train, X_val, y_val,
        input_dim=input_dim,
        embedding_dim=embedding_dim,
        epochs=epochs
    )
    
    print("\nSaving model and artifacts...")


    # --- Always pull the BEST weights from the checkpointed Siamese ---
    from tensorflow import keras
    from modules.siamese_network import L2Normalize, contrastive_loss  # ensure registered
    
    best_siamese = keras.models.load_model(
        "best_siamese_model.keras",
        compile=False,
        custom_objects={"L2Normalize": L2Normalize, "contrastive_loss": contrastive_loss},
    )
    best_base = best_siamese.get_layer("base_network")
    
    # 1) Primary artifact: Keras format (single file)
    best_base.save("cusip_embedding_model_temporal.keras")  # <-- no save_format arg
    
    # 2) (Optional) Export a TF-Serving SavedModel folder for serving/infra
    #    Note: exported SavedModel isn't meant to be reloaded via keras.models.load_model.
    try:
        best_base.export("cusip_embedding_model_temporal_tf")  # creates a folder
    except AttributeError:
        # Older TF/Keras may not have .export(); you can skip this or fall back:
        import tensorflow as tf
        tf.saved_model.save(best_base, "cusip_embedding_model_temporal_tf")
    
    # Feature artifacts
    with open("feature_artifacts_temporal.pkl", "wb") as f:
        pickle.dump(artifacts, f)
    with open("feature_names_temporal.pkl", "wb") as f:
        pickle.dump(feature_names, f)

    print("Training complete!")
    print(f"Final validation loss: {history.history['val_loss'][-1]:.4f}")
    
    return base_network, artifacts, history