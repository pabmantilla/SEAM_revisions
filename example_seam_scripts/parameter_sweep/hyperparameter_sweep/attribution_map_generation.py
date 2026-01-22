"""
Attribution Map Generation Script
=================================
This script generates attribution maps using different methods (deepshap, saliency, ism, intgrad)
from the same 100K mutagenesis library. It follows the exact same subset ordering and content
as library_size_optimization.py to ensure consistency.

Processing Order:
    1. deepshap (if all files don't exist)
    2. saliency (if all files don't exist)
    3. ism (if all files don't exist)
    4. intgrad (if all files don't exist)

For each method:
    - Process both Dev and Hk tasks
    - For each sequence, compute 100K attributions first (with checkpointing)
    - Then extract subsets using the EXACT same subset_idx from mutagenesis library
    - This ensures subset content matches library_size_optimization.py exactly

Directory Structure:
    attribution_map_libraries/
    ├── deepSHAP/Dev/seq_X/100K.h5, 75K.h5, ...
    ├── saliency/Dev/seq_X/100K.h5, 75K.h5, ...
    ├── ism/Dev/seq_X/100K.h5, 75K.h5, ...
    └── intgrad/Dev/seq_X/100K.h5, 75K.h5, ...
"""

# =============================================================================
# IMPORTS
# =============================================================================

# Core imports for file I/O and numerical operations
import os
import numpy as np
import h5py
import time
import pickle

# TensorFlow/Keras imports for model loading and attribution computation
import tensorflow as tf
from keras.models import model_from_json

# SEAM imports for attribution computation
import seam
from seam import Attributer


# =============================================================================
# CONFIGURATION
# =============================================================================

# Attribution methods to process (in order)
# Each method will be processed completely before moving to the next
METHODS = ['deepshap', 'saliency', 'ism', 'intgrad']

# Subset sizes - MUST match library_size_optimization.py exactly
# The order matters: 100K first, then decreasing sizes
# This dictionary preserves insertion order (Python 3.7+)
SUBSET_SIZES = {
    '100K': 100000,
    '75K': 75000,
    '50K': 50000,
    '25K': 25000,
    '10K': 10000,
    '5K': 5000,
    '1K': 1000
}

# Tasks to process (Dev = developmental enhancers, Hk = housekeeping enhancers)
TASKS = ['Dev', 'Hk']

# GPU device to use for attribution computation
GPU_DEVICE = 0

# Checkpoint frequency - save progress every N samples
# This allows resumption if computation is interrupted
CHECKPOINT_EVERY = 5000


# =============================================================================
# LOAD MODEL AND LIBRARIES
# =============================================================================

# Load the hyperparam libraries (contains dev_loci and hk_loci)
# These define which sequences to process
with open('../libraries/hyperparam_libraries.pkl', 'rb') as f:
    libraries = pickle.load(f)
    dev_loci = libraries['dev']
    hk_loci = libraries['hk']

# Validate library sizes
if len(dev_loci) == 8 and len(hk_loci) == 8:
    print("Size of test library validated.")
else:
    print("Wrong Library!!!")

# Load DeepSTARR model
# This model predicts enhancer activity for Dev and Hk tasks
model_dir = '../models/deepstarr'
model_json_file = os.path.join(model_dir, 'deepstarr.model.json')
model_weights_file = os.path.join(model_dir, 'deepstarr.model.h5')

# Read model architecture from JSON
with open(model_json_file, 'r') as f:
    model_json = f.read()

# Create model and load weights
# custom_objects needed for Keras Functional API compatibility
model = model_from_json(model_json, custom_objects={'Functional': tf.keras.Model})
model.load_weights(model_weights_file)
print("Model loaded successfully!")


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def load_library_100k(task, seq_idx):
    """
    Load the full 100K mutagenesis library for a given task and sequence.

    Args:
        task: 'Dev' or 'Hk' - which enhancer task
        seq_idx: Index of the sequence (0-7)

    Returns:
        sequences: One-hot encoded sequences (100000, L, 4)
        predictions: Model predictions for each sequence (100000,)
        original_idx: Index in the original dataset
    """
    filepath = f'mutagenisis_library/{task}/seq_{seq_idx}/100K.h5'
    with h5py.File(filepath, 'r') as f:
        sequences = f['sequences'][:]
        predictions = f['predictions'][:]
        original_idx = f.attrs['original_idx']
    return sequences, predictions, original_idx


def load_library(task, seq_idx, size_label):
    """
    Load a specific size subset library including the subset_idx.

    The subset_idx is CRITICAL - it tells us which rows from the 100K library
    are included in this subset. We use this to extract the corresponding
    attributions from the 100K attribution maps.

    Args:
        task: 'Dev' or 'Hk'
        seq_idx: Index of the sequence (0-7)
        size_label: '100K', '75K', '50K', '25K', '10K', '5K', or '1K'

    Returns:
        sequences: One-hot encoded sequences for this subset
        predictions: Model predictions for this subset
        original_idx: Index in the original dataset
        subset_idx: Indices mapping this subset to the 100K library (None for 100K)
    """
    filepath = f'mutagenisis_library/{task}/seq_{seq_idx}/{size_label}.h5'
    with h5py.File(filepath, 'r') as f:
        sequences = f['sequences'][:]
        predictions = f['predictions'][:]
        original_idx = f.attrs['original_idx']
        # subset_idx only exists for sizes < 100K
        subset_idx = f['subset_idx'][:] if 'subset_idx' in f else None
    return sequences, predictions, original_idx, subset_idx


def save_attributions(filepath, attributions, original_idx, subset_idx=None):
    """
    Save attribution maps to an HDF5 file with metadata.

    Args:
        filepath: Path to save the HDF5 file
        attributions: Attribution maps array (N, L, 4)
        original_idx: Index in the original dataset
        subset_idx: Indices mapping to 100K library (None for 100K)
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    with h5py.File(filepath, 'w') as f:
        # Save attributions with gzip compression to reduce file size
        f.create_dataset('attributions', data=attributions,
                        compression='gzip', compression_opts=4)
        # Save subset_idx if provided (for subsets < 100K)
        if subset_idx is not None:
            f.create_dataset('subset_idx', data=subset_idx,
                           compression='gzip', compression_opts=4)
        # Store metadata as attributes
        f.attrs['original_idx'] = original_idx
        f.attrs['n_samples'] = len(attributions)


def load_attributions(filepath):
    """
    Load attribution maps from an HDF5 file.

    Args:
        filepath: Path to the HDF5 file

    Returns:
        attributions: Attribution maps array (N, L, 4)
    """
    with h5py.File(filepath, 'r') as f:
        return f['attributions'][:]


def attributions_exist(method, task, seq_idx):
    """
    Check if 100K attributions already exist for a given method/task/seq.

    Args:
        method: Attribution method name ('deepshap', 'saliency', etc.)
        task: 'Dev' or 'Hk'
        seq_idx: Sequence index

    Returns:
        bool: True if the 100K attribution file exists
    """
    filepath = f'attribution_map_libraries/{method}/{task}/seq_{seq_idx}/100K.h5'
    return os.path.exists(filepath)


def all_attributions_exist(method, task, seq_idx, subset_sizes):
    """
    Check if ALL attribution files exist for a given method/task/seq.

    This is used for the quick skip check - if all files exist, we can
    skip this entire task/seq combination.

    Args:
        method: Attribution method name
        task: 'Dev' or 'Hk'
        seq_idx: Sequence index
        subset_sizes: Dictionary of size labels to sizes

    Returns:
        bool: True if ALL attribution files exist for all sizes
    """
    for size_label in subset_sizes.keys():
        attr_path = f'attribution_map_libraries/{method}/{task}/seq_{seq_idx}/{size_label}.h5'
        if not os.path.exists(attr_path):
            return False
    return True


# =============================================================================
# DEEPSHAP ATTRIBUTION FUNCTION
# =============================================================================

def compute_deepshap(x_mut, task_index, checkpoint_path=None, checkpoint_every=5000):
    """
    Compute DeepSHAP attributions with checkpointing.

    DeepSHAP requires special TensorFlow setup because it was designed for TF 1.x.
    This function handles:
    1. Disabling eager execution for SHAP compatibility
    2. Setting up SHAP operation handlers
    3. Reloading the model after TF setup
    4. Computing attributions with checkpointing for resumability

    Args:
        x_mut: Input sequences (N, L, 4)
        task_index: 0 for Dev, 1 for Hk
        checkpoint_path: Path to save/load checkpoint
        checkpoint_every: Save checkpoint every N samples

    Returns:
        attributions: Attribution maps (N, L, 4)
    """
    # Use reference sequences as themselves (self-reference)
    x_ref = x_mut
    print(f"Computing DeepSHAP attributions for task_index: {task_index}")

    import random

    # -------------------------------------------------------------------------
    # CHECKPOINT RECOVERY
    # Check if we have a partial computation we can resume from
    # -------------------------------------------------------------------------
    if checkpoint_path and os.path.exists(checkpoint_path):
        with h5py.File(checkpoint_path, 'r') as f:
            start_idx = f.attrs['last_completed_idx'] + 1
            attributions_partial = f['attributions'][:start_idx]
        print(f"Resuming from checkpoint at index {start_idx}")
    else:
        start_idx = 0
        attributions_partial = None

    # If computation is already complete, just return the checkpoint
    if start_idx >= len(x_mut):
        print("Attributions already complete, loading from checkpoint")
        with h5py.File(checkpoint_path, 'r') as f:
            return f['attributions'][:]

    # -------------------------------------------------------------------------
    # TENSORFLOW SETUP FOR DEEPSHAP
    # DeepSHAP requires TF 1.x style execution
    # -------------------------------------------------------------------------
    try:
        # Disable eager execution (required for DeepSHAP)
        tf.compat.v1.disable_eager_execution()
        tf.compat.v1.disable_v2_behavior()
        print("TensorFlow eager execution disabled for DeepSHAP compatibility")

        # Import SHAP and configure operation handlers
        try:
            import shap
        except ImportError:
            raise ImportError("SHAP package required for DeepSHAP attribution")

        # Add handler for AddV2 operation (needed for modern TF)
        shap.explainers.deep.deep_tf.op_handlers["AddV2"] = shap.explainers.deep.deep_tf.passthrough

        # Reload model after disabling eager execution
        # This is necessary because the model needs to be rebuilt in graph mode
        keras_model_json = '../models/deepstarr/deepstarr.model.json'
        keras_model_weights = '../models/deepstarr/deepstarr.model.h5'

        keras_model = model_from_json(open(keras_model_json).read(),
                                      custom_objects={'Functional': tf.keras.Model})
        # Set random seeds for reproducibility
        np.random.seed(113)
        random.seed(0)
        keras_model.load_weights(keras_model_weights)
        model_for_deepshap = keras_model

        # Rebuild model graph by passing a dummy input
        _ = model_for_deepshap(tf.keras.Input(shape=model_for_deepshap.input_shape[1:]))

    except ImportError:
        raise
    except Exception as e:
        print(f"Warning: Could not setup TensorFlow for DeepSHAP. Error: {e}")
        print("DeepSHAP may not work properly.")

    # -------------------------------------------------------------------------
    # CREATE ATTRIBUTER WITH COMPRESSION FUNCTION
    # -------------------------------------------------------------------------

    def deepstarr_compress(x):
        """
        Compress model output for DeepSHAP.

        For DeepSHAP, when x is the model, we need to extract the appropriate
        output head and reduce it to a scalar.
        """
        if hasattr(x, 'outputs'):
            # x is the model - extract the appropriate task output
            return tf.reduce_sum(x.outputs[task_index], axis=-1)
        else:
            return x

    attributer = Attributer(
        model_for_deepshap,
        method='deepshap',
        task_index=task_index,
        compress_fun=deepstarr_compress
    )

    # Show attribution parameters for logging
    attributer.show_params('deepshap')

    # -------------------------------------------------------------------------
    # COMPUTE ATTRIBUTIONS WITH CHECKPOINTING
    # -------------------------------------------------------------------------
    t1 = time.time()

    n_samples = len(x_mut)
    all_attributions = []

    # Add previously computed attributions if resuming from checkpoint
    if attributions_partial is not None:
        all_attributions.append(attributions_partial)

    # Process in chunks for checkpointing
    for chunk_start in range(start_idx, n_samples, checkpoint_every):
        chunk_end = min(chunk_start + checkpoint_every, n_samples)
        print(f"\nProcessing samples {chunk_start} to {chunk_end} of {n_samples}")

        x_chunk = x_mut[chunk_start:chunk_end]
        x_ref_chunk = x_chunk  # Self-reference

        # Compute attributions for this chunk
        chunk_attributions = attributer.compute(
            x_ref=x_ref_chunk,
            x=x_chunk,
            save_window=None,
            batch_size=64,
            gpu=GPU_DEVICE,
        )

        all_attributions.append(chunk_attributions)

        # Save checkpoint after each chunk
        if checkpoint_path:
            attributions_so_far = np.concatenate(all_attributions, axis=0)
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            with h5py.File(checkpoint_path, 'w') as f:
                f.create_dataset('attributions', data=attributions_so_far,
                               compression='gzip', compression_opts=4)
                f.attrs['last_completed_idx'] = chunk_end - 1
                f.attrs['n_samples'] = n_samples
            print(f"Checkpoint saved at index {chunk_end - 1}")

    # Concatenate all chunks into final result
    attributions = np.concatenate(all_attributions, axis=0)

    t2 = time.time() - t1
    print(f'DeepSHAP attribution time: {t2/60:.2f} minutes')

    return attributions


# =============================================================================
# SALIENCY ATTRIBUTION FUNCTION
# =============================================================================

def compute_saliency(x_mut, task_index, checkpoint_path=None, checkpoint_every=5000):
    """
    Compute Saliency map attributions with checkpointing.

    Saliency maps compute the gradient of the output with respect to the input.
    This is the simplest gradient-based attribution method.

    Args:
        x_mut: Input sequences (N, L, 4)
        task_index: 0 for Dev, 1 for Hk
        checkpoint_path: Path to save/load checkpoint
        checkpoint_every: Save checkpoint every N samples

    Returns:
        attributions: Attribution maps (N, L, 4)
    """
    print(f"Computing Saliency attributions for task_index: {task_index}")

    # -------------------------------------------------------------------------
    # CHECKPOINT RECOVERY
    # -------------------------------------------------------------------------
    if checkpoint_path and os.path.exists(checkpoint_path):
        with h5py.File(checkpoint_path, 'r') as f:
            start_idx = f.attrs['last_completed_idx'] + 1
            attributions_partial = f['attributions'][:start_idx]
        print(f"Resuming from checkpoint at index {start_idx}")
    else:
        start_idx = 0
        attributions_partial = None

    if start_idx >= len(x_mut):
        print("Attributions already complete, loading from checkpoint")
        with h5py.File(checkpoint_path, 'r') as f:
            return f['attributions'][:]

    # -------------------------------------------------------------------------
    # CREATE ATTRIBUTER
    # -------------------------------------------------------------------------
    attributer = Attributer(
        model,
        method='saliency',
        task_index=task_index,
        compress_fun=tf.math.reduce_mean,
        gpu=True
    )

    # -------------------------------------------------------------------------
    # COMPUTE ATTRIBUTIONS WITH CHECKPOINTING
    # -------------------------------------------------------------------------
    t1 = time.time()

    n_samples = len(x_mut)
    all_attributions = []

    if attributions_partial is not None:
        all_attributions.append(attributions_partial)

    for chunk_start in range(start_idx, n_samples, checkpoint_every):
        chunk_end = min(chunk_start + checkpoint_every, n_samples)
        print(f"\nProcessing samples {chunk_start} to {chunk_end} of {n_samples}")

        x_chunk = x_mut[chunk_start:chunk_end]

        # Compute saliency maps for this chunk
        chunk_attributions = attributer.compute(
            x=x_chunk,
            batch_size=256,
            gpu=GPU_DEVICE,
        )

        all_attributions.append(chunk_attributions)

        # Save checkpoint
        if checkpoint_path:
            attributions_so_far = np.concatenate(all_attributions, axis=0)
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            with h5py.File(checkpoint_path, 'w') as f:
                f.create_dataset('attributions', data=attributions_so_far,
                               compression='gzip', compression_opts=4)
                f.attrs['last_completed_idx'] = chunk_end - 1
                f.attrs['n_samples'] = n_samples
            print(f"Checkpoint saved at index {chunk_end - 1}")

    attributions = np.concatenate(all_attributions, axis=0)

    t2 = time.time() - t1
    print(f'Saliency attribution time: {t2/60:.2f} minutes')

    return attributions


# =============================================================================
# ISM (IN-SILICO MUTAGENESIS) ATTRIBUTION FUNCTION
# =============================================================================

def compute_ism(x_mut, task_index, checkpoint_path=None, checkpoint_every=5000):
    """
    Compute ISM (In-Silico Mutagenesis) attributions with checkpointing.

    ISM computes the effect of mutating each position to each possible base.
    This is a forward-pass method (no gradients), so it requires pred_fun.

    Args:
        x_mut: Input sequences (N, L, 4)
        task_index: 0 for Dev, 1 for Hk
        checkpoint_path: Path to save/load checkpoint
        checkpoint_every: Save checkpoint every N samples

    Returns:
        attributions: Attribution maps (N, L, 4)
    """
    print(f"Computing ISM attributions for task_index: {task_index}")

    # -------------------------------------------------------------------------
    # CHECKPOINT RECOVERY
    # -------------------------------------------------------------------------
    if checkpoint_path and os.path.exists(checkpoint_path):
        with h5py.File(checkpoint_path, 'r') as f:
            start_idx = f.attrs['last_completed_idx'] + 1
            attributions_partial = f['attributions'][:start_idx]
        print(f"Resuming from checkpoint at index {start_idx}")
    else:
        start_idx = 0
        attributions_partial = None

    if start_idx >= len(x_mut):
        print("Attributions already complete, loading from checkpoint")
        with h5py.File(checkpoint_path, 'r') as f:
            return f['attributions'][:]

    # -------------------------------------------------------------------------
    # CREATE ATTRIBUTER
    # ISM requires pred_fun for forward-pass predictions
    # -------------------------------------------------------------------------
    attributer = Attributer(
        model,
        method='ism',
        task_index=task_index,
        compress_fun=tf.math.reduce_mean,
        pred_fun=model.predict_on_batch,  # Required for ISM
        gpu=True
    )

    # -------------------------------------------------------------------------
    # COMPUTE ATTRIBUTIONS WITH CHECKPOINTING
    # -------------------------------------------------------------------------
    t1 = time.time()

    n_samples = len(x_mut)
    all_attributions = []

    if attributions_partial is not None:
        all_attributions.append(attributions_partial)

    for chunk_start in range(start_idx, n_samples, checkpoint_every):
        chunk_end = min(chunk_start + checkpoint_every, n_samples)
        print(f"\nProcessing samples {chunk_start} to {chunk_end} of {n_samples}")

        x_chunk = x_mut[chunk_start:chunk_end]

        # Compute ISM attributions for this chunk
        # ISM uses smaller batch size due to memory requirements
        chunk_attributions = attributer.compute(
            x=x_chunk,
            batch_size=32,
            gpu=GPU_DEVICE,
            log2fc=False,  # Use difference, not log2 fold change
        )

        all_attributions.append(chunk_attributions)

        # Save checkpoint
        if checkpoint_path:
            attributions_so_far = np.concatenate(all_attributions, axis=0)
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            with h5py.File(checkpoint_path, 'w') as f:
                f.create_dataset('attributions', data=attributions_so_far,
                               compression='gzip', compression_opts=4)
                f.attrs['last_completed_idx'] = chunk_end - 1
                f.attrs['n_samples'] = n_samples
            print(f"Checkpoint saved at index {chunk_end - 1}")

    attributions = np.concatenate(all_attributions, axis=0)

    t2 = time.time() - t1
    print(f'ISM attribution time: {t2/60:.2f} minutes')

    return attributions


# =============================================================================
# INTEGRATED GRADIENTS ATTRIBUTION FUNCTION
# =============================================================================

def compute_intgrad(x_mut, task_index, checkpoint_path=None, checkpoint_every=5000):
    """
    Compute Integrated Gradients attributions with checkpointing.

    Integrated Gradients interpolates between a baseline (zeros) and the input,
    computing gradients at each step and averaging them.

    Args:
        x_mut: Input sequences (N, L, 4)
        task_index: 0 for Dev, 1 for Hk
        checkpoint_path: Path to save/load checkpoint
        checkpoint_every: Save checkpoint every N samples

    Returns:
        attributions: Attribution maps (N, L, 4)
    """
    print(f"Computing Integrated Gradients attributions for task_index: {task_index}")

    # -------------------------------------------------------------------------
    # CHECKPOINT RECOVERY
    # -------------------------------------------------------------------------
    if checkpoint_path and os.path.exists(checkpoint_path):
        with h5py.File(checkpoint_path, 'r') as f:
            start_idx = f.attrs['last_completed_idx'] + 1
            attributions_partial = f['attributions'][:start_idx]
        print(f"Resuming from checkpoint at index {start_idx}")
    else:
        start_idx = 0
        attributions_partial = None

    if start_idx >= len(x_mut):
        print("Attributions already complete, loading from checkpoint")
        with h5py.File(checkpoint_path, 'r') as f:
            return f['attributions'][:]

    # -------------------------------------------------------------------------
    # CREATE ATTRIBUTER
    # -------------------------------------------------------------------------
    attributer = Attributer(
        model,
        method='intgrad',
        task_index=task_index,
        compress_fun=tf.math.reduce_mean,
        gpu=True
    )

    # -------------------------------------------------------------------------
    # COMPUTE ATTRIBUTIONS WITH CHECKPOINTING
    # -------------------------------------------------------------------------
    t1 = time.time()

    n_samples = len(x_mut)
    all_attributions = []

    if attributions_partial is not None:
        all_attributions.append(attributions_partial)

    for chunk_start in range(start_idx, n_samples, checkpoint_every):
        chunk_end = min(chunk_start + checkpoint_every, n_samples)
        print(f"\nProcessing samples {chunk_start} to {chunk_end} of {n_samples}")

        x_chunk = x_mut[chunk_start:chunk_end]

        # Compute integrated gradients for this chunk
        chunk_attributions = attributer.compute(
            x=x_chunk,
            batch_size=256,
            gpu=GPU_DEVICE,
            num_steps=20,           # Number of integration steps
            baseline_type='zeros',  # Use zeros as baseline
            multiply_by_inputs=False,
        )

        all_attributions.append(chunk_attributions)

        # Save checkpoint
        if checkpoint_path:
            attributions_so_far = np.concatenate(all_attributions, axis=0)
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            with h5py.File(checkpoint_path, 'w') as f:
                f.create_dataset('attributions', data=attributions_so_far,
                               compression='gzip', compression_opts=4)
                f.attrs['last_completed_idx'] = chunk_end - 1
                f.attrs['n_samples'] = n_samples
            print(f"Checkpoint saved at index {chunk_end - 1}")

    attributions = np.concatenate(all_attributions, axis=0)

    t2 = time.time() - t1
    print(f'Integrated Gradients attribution time: {t2/60:.2f} minutes')

    return attributions


# =============================================================================
# MAIN PROCESSING LOOP
# =============================================================================

# Map method names to their compute functions
COMPUTE_FUNCTIONS = {
    'deepshap': compute_deepshap,
    'saliency': compute_saliency,
    'ism': compute_ism,
    'intgrad': compute_intgrad,
}

# Process each attribution method in order
for method in METHODS:
    print(f"\n{'#'*60}")
    print(f"# PROCESSING METHOD: {method.upper()}")
    print(f"{'#'*60}")

    # Get the appropriate compute function
    compute_func = COMPUTE_FUNCTIONS[method]

    # Process each task (Dev and Hk)
    for task in TASKS:
        # Determine task_index (0 for Dev, 1 for Hk)
        task_index = 0 if task == 'Dev' else 1

        # Find all sequence folders for this task
        task_dir = f'mutagenisis_library/{task}'
        seq_folders = [f for f in os.listdir(task_dir) if f.startswith('seq_')]

        # Process each sequence
        for seq_folder in seq_folders:
            seq_idx = int(seq_folder.split('_')[1])
            print(f"\n{'='*50}")
            print(f"Processing {method} - {task} seq_{seq_idx}")
            print(f"{'='*50}")

            # -----------------------------------------------------------------
            # QUICK SKIP CHECK
            # If ALL attribution files already exist for this method/task/seq,
            # skip to the next sequence
            # -----------------------------------------------------------------
            if all_attributions_exist(method, task, seq_idx, SUBSET_SIZES):
                print(f"All attribution files exist for {method}/{task}/seq_{seq_idx} - skipping")
                continue

            # -----------------------------------------------------------------
            # LOAD FULL 100K LIBRARY
            # We need the sequences to compute attributions
            # -----------------------------------------------------------------
            seqs_100k, preds_100k, orig_idx = load_library_100k(task, seq_idx)

            # -----------------------------------------------------------------
            # COMPUTE OR LOAD 100K ATTRIBUTIONS
            # -----------------------------------------------------------------
            attr_100k_path = f'attribution_map_libraries/{method}/{task}/seq_{seq_idx}/100K.h5'

            if attributions_exist(method, task, seq_idx):
                # 100K attributions already exist - load them
                print(f"Loading existing 100K attributions...")
                attributions_100k = load_attributions(attr_100k_path)
            else:
                # Compute 100K attributions with checkpointing
                print(f"Computing attributions for 100K samples...")
                checkpoint_path = f'attribution_map_libraries/{method}/{task}/seq_{seq_idx}/checkpoint.h5'

                attributions_100k = compute_func(
                    seqs_100k,
                    task_index,
                    checkpoint_path=checkpoint_path,
                    checkpoint_every=CHECKPOINT_EVERY
                )

                # Save the 100K attributions
                save_attributions(attr_100k_path, attributions_100k, orig_idx, subset_idx=None)

                # Remove checkpoint after successful completion
                if os.path.exists(checkpoint_path):
                    os.remove(checkpoint_path)
                print(f"Saved 100K attributions")

            # -----------------------------------------------------------------
            # PROCESS EACH SUBSET SIZE
            # Extract subsets using the EXACT same subset_idx from mutagenesis library
            # This ensures the subset content matches library_size_optimization.py
            # -----------------------------------------------------------------
            for size_label, size in SUBSET_SIZES.items():
                print(f"\n--- {size_label} ---")

                # Check if this attribution file already exists
                attr_path = f'attribution_map_libraries/{method}/{task}/seq_{seq_idx}/{size_label}.h5'
                if os.path.exists(attr_path):
                    print(f"Skipping {size_label} - attribution file already exists")
                    continue

                # Load subset_idx from the mutagenesis library (this is the source of truth)
                # The subset_idx tells us exactly which samples from 100K are in this subset
                _, _, _, subset_idx = load_library(task, seq_idx, size_label)

                if size_label == '100K':
                    # For 100K, use all samples (no subsetting needed)
                    seqs = seqs_100k
                    preds = preds_100k
                    attrs = attributions_100k
                else:
                    # For smaller subsets, use subset_idx to extract EXACT same samples
                    # as the mutagenesis library - this is CRITICAL for consistency
                    seqs = seqs_100k[subset_idx]
                    preds = preds_100k[subset_idx]
                    attrs = attributions_100k[subset_idx]

                # Save the attribution subset with subset_idx preserved
                save_attributions(attr_path, attrs, orig_idx, subset_idx=subset_idx)

                print(f"Seqs: {seqs.shape}, Preds: {preds.shape}, Attrs: {attrs.shape}")

print("\n" + "="*60)
print("ATTRIBUTION MAP GENERATION COMPLETE")
print("="*60)
