# Core imports
import os
import subprocess
import numpy as np
import pandas as pd
import h5py
import random
from urllib.request import urlretrieve
import matplotlib.pyplot as plt


# TensorFlow/Keras imports for model loading
import tensorflow as tf
from keras.models import model_from_json

# SEAM imports
import seam
from seam import Compiler, Attributer, Clusterer, MetaExplainer
from seam.logomaker_batch.batch_logo import BatchLogo

## This script will sweep through the library sizes with baseline hyperparameters

library_size_sweep = [100000, 75000, 50000, 25000, 10000, 5000, 1000]

# baseline hyperparameters
attribution_method = 'deepshap'
cluster_method = 'hiearchical'
cluster_number = 30
mutation_rate = .10


## load test library

#open libraries from library_selection.ipynb
import pickle

with open('../libraries/hyperparam_libraries.pkl', 'rb') as f:
    libraries = pickle.load(f)
    dev_loci = libraries['dev']
    hk_loci = libraries['hk']

if len(dev_loci)==8 and len(hk_loci) ==8:
    print("Size of test library validated.")
else:
    print("Wrong Library!!!")

## load DeepSTARR model
# import model
import os
from urllib.request import urlretrieve
from keras.models import model_from_json

model_dir = '../models/deepstarr'
os.makedirs(model_dir, exist_ok=True)

model_json_file = os.path.join(model_dir, 'deepstarr.model.json')
model_weights_file = os.path.join(model_dir, 'deepstarr.model.h5')

with open(model_json_file, 'r') as f:
    model_json = f.read()

model = model_from_json(model_json, custom_objects={'Functional': tf.keras.Model})
model.load_weights(model_weights_file)

print("Model loaded successfully!")

## load DeepSHAP
## load deepSHAP
def seam_deepshap(x_mut, task_index, checkpoint_path=None, checkpoint_every=5000):
    """Compute DeepSHAP attributions with optional checkpointing."""
    x_ref = x_mut
    print(f"Computing attributions for task_index: {task_index}")
    import time
    import tensorflow as tf
    from keras.models import model_from_json
    import numpy as np
    import random

    # Check for existing checkpoint
    if checkpoint_path and os.path.exists(checkpoint_path):
        with h5py.File(checkpoint_path, 'r') as f:
            start_idx = f.attrs['last_completed_idx'] + 1
            attributions_partial = f['attributions'][:start_idx]
        print(f"Resuming from checkpoint at index {start_idx}")
    else:
        start_idx = 0
        attributions_partial = None

    # If already complete, return
    if start_idx >= len(x_mut):
        print("Attributions already complete, loading from checkpoint")
        with h5py.File(checkpoint_path, 'r') as f:
            return f['attributions'][:]

    # Configuration
    attribution_method = 'deepshap'
    gpu = 0
    
    # Model paths
    keras_model_json = '../models/deepstarr/deepstarr.model.json'
    keras_model_weights = '../models/deepstarr/deepstarr.model.h5'

    if attribution_method == 'deepshap':
        try:
            tf.compat.v1.disable_eager_execution()
            tf.compat.v1.disable_v2_behavior()
            print("TensorFlow eager execution disabled for DeepSHAP compatibility")
            
            try:
                import shap
            except ImportError:
                raise ImportError("SHAP package required for DeepSHAP attribution")
            
            shap.explainers.deep.deep_tf.op_handlers["AddV2"] = shap.explainers.deep.deep_tf.passthrough

            keras_model = model_from_json(open(keras_model_json).read(), custom_objects={'Functional': tf.keras.Model})
            np.random.seed(113)
            random.seed(0)
            keras_model.load_weights(keras_model_weights)
            model = keras_model
            
            _ = model(tf.keras.Input(shape=model.input_shape[1:]))
            
        except ImportError:
            raise
        except Exception as e:
            print(f"Warning: Could not setup TensorFlow for DeepSHAP. Error: {e}")
            print("DeepSHAP may not work properly.")
        
        def deepstarr_compress(x):
            if hasattr(x, 'outputs'):
                return tf.reduce_sum(x.outputs[task_index], axis=-1)
            else:
                return x

        attributer = Attributer(
            model,
            method=attribution_method,
            task_index=task_index,
            compress_fun=deepstarr_compress
        )

        attributer.show_params(attribution_method)

        t1 = time.time()
        
        # Process in chunks with checkpointing
        n_samples = len(x_mut)
        all_attributions = []
        
        # Add previously computed attributions if resuming
        if attributions_partial is not None:
            all_attributions.append(attributions_partial)
        
        for chunk_start in range(start_idx, n_samples, checkpoint_every):
            chunk_end = min(chunk_start + checkpoint_every, n_samples)
            print(f"\nProcessing samples {chunk_start} to {chunk_end} of {n_samples}")
            
            x_chunk = x_mut[chunk_start:chunk_end]
            x_ref_chunk = x_chunk
            
            chunk_attributions = attributer.compute(
                x_ref=x_ref_chunk,
                x=x_chunk,
                save_window=None,
                batch_size=64,
                gpu=gpu,
            )
            
            all_attributions.append(chunk_attributions)
            
            # Save checkpoint
            if checkpoint_path:
                attributions_so_far = np.concatenate(all_attributions, axis=0)
                os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
                with h5py.File(checkpoint_path, 'w') as f:
                    f.create_dataset('attributions', data=attributions_so_far, compression='gzip', compression_opts=4)
                    f.attrs['last_completed_idx'] = chunk_end - 1
                    f.attrs['n_samples'] = n_samples
                print(f"Checkpoint saved at index {chunk_end - 1}")
        
        attributions = np.concatenate(all_attributions, axis=0)
        
        t2 = time.time() - t1
        print(f'Attribution time: {t2/60:.2f} minutes')
        
        return attributions


### START SEAM

## Helper functions

def load_library_100k(task, seq_idx):
    """Load the full 100K library."""
    filepath = f'mutagenisis_library/{task}/seq_{seq_idx}/100K.h5'
    with h5py.File(filepath, 'r') as f:
        sequences = f['sequences'][:]
        predictions = f['predictions'][:]
        original_idx = f.attrs['original_idx']
    return sequences, predictions, original_idx


def load_library(task, seq_idx, size_label):
    """Load a specific size library including subset_idx."""
    filepath = f'mutagenisis_library/{task}/seq_{seq_idx}/{size_label}.h5'
    with h5py.File(filepath, 'r') as f:
        sequences = f['sequences'][:]
        predictions = f['predictions'][:]
        original_idx = f.attrs['original_idx']
        subset_idx = f['subset_idx'][:] if 'subset_idx' in f else None
    return sequences, predictions, original_idx, subset_idx

def save_attributions(filepath, attributions, original_idx, subset_idx=None):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with h5py.File(filepath, 'w') as f:
        f.create_dataset('attributions', data=attributions, compression='gzip', compression_opts=4)
        if subset_idx is not None:
            f.create_dataset('subset_idx', data=subset_idx, compression='gzip', compression_opts=4)
        f.attrs['original_idx'] = original_idx
        f.attrs['n_samples'] = len(attributions)

def load_attributions(filepath):
    with h5py.File(filepath, 'r') as f:
        return f['attributions'][:]

def attributions_exist(task, seq_idx):
    """Check if 100K attributions already exist."""
    filepath = f'attribution_map_libraries/deepSHAP/{task}/seq_{seq_idx}/100K.h5'
    return os.path.exists(filepath)

# Subset sizes and labels
subset_sizes = {
    '100K': 100000,
    '75K': 75000,
    '50K': 50000,
    '25K': 25000,
    '10K': 10000,
    '5K': 5000,
    '1K': 1000
}

tasks = ['Dev', 'Hk']

def all_attributions_exist(task, seq_idx, subset_sizes):
    """Check if ALL attribution files exist for a given task/seq."""
    for size_label in subset_sizes.keys():
        attr_path = f'attribution_map_libraries/deepSHAP/{task}/seq_{seq_idx}/{size_label}.h5'
        if not os.path.exists(attr_path):
            return False
    return True

for task in tasks:
    task_index = 0 if task == 'Dev' else 1

    task_dir = f'mutagenisis_library/{task}'
    seq_folders = [f for f in os.listdir(task_dir) if f.startswith('seq_')]

    for seq_folder in seq_folders:
        seq_idx = int(seq_folder.split('_')[1])
        print(f"\n{'='*50}")
        print(f"Processing {task} seq_{seq_idx}")
        print(f"{'='*50}")

        # Quick check: skip entirely if ALL attribution files already exist
        if all_attributions_exist(task, seq_idx, subset_sizes):
            print(f"All attribution files exist for {task} seq_{seq_idx} - skipping")
            continue

        # Load full 100K library
        seqs_100k, preds_100k, orig_idx = load_library_100k(task, seq_idx)
        
        # Compute or load 100K attributions
        attr_100k_path = f'attribution_map_libraries/deepSHAP/{task}/seq_{seq_idx}/100K.h5'
        if attributions_exist(task, seq_idx):
            print(f"Loading existing 100K attributions...")
            attributions_100k = load_attributions(attr_100k_path)
        else:
                    
            print(f"Computing attributions for 100K samples...")
            checkpoint_path = f'attribution_map_libraries/deepSHAP/{task}/seq_{seq_idx}/checkpoint.h5'
            attributions_100k = seam_deepshap(seqs_100k, task_index, 
                                            checkpoint_path=checkpoint_path,
                                            checkpoint_every=5000)
            save_attributions(attr_100k_path, attributions_100k, orig_idx, subset_idx=None)
            # Optionally remove checkpoint after successful completion
            if os.path.exists(checkpoint_path):
                os.remove(checkpoint_path)
            print(f"Saved 100K attributions")
        
        # Process each size
        for size_label, size in subset_sizes.items():
            print(f"\n--- {size_label} ---")

            # Check if attribution file already exists - skip if so
            attr_path = f'attribution_map_libraries/deepSHAP/{task}/seq_{seq_idx}/{size_label}.h5'
            if os.path.exists(attr_path):
                print(f"Skipping {size_label} - attribution file already exists")
                continue

            # Load the subset_idx from the mutagenesis library (source of truth)
            _, _, _, subset_idx = load_library(task, seq_idx, size_label)

            if size_label == '100K':
                seqs = seqs_100k
                preds = preds_100k
                attrs = attributions_100k
            else:
                # Use the EXACT subset_idx from the mutagenesis library
                seqs = seqs_100k[subset_idx]
                preds = preds_100k[subset_idx]
                attrs = attributions_100k[subset_idx]

            # Save attribution subset
            save_attributions(attr_path, attrs, orig_idx, subset_idx=subset_idx)

            print(f"Seqs: {seqs.shape}, Preds: {preds.shape}, Attrs: {attrs.shape}")

## Cluster with Hierarchical 

def cluster_and_save(attrs, task, seq_idx, size_label, n_clusters=30):
    """Cluster attributions and save results. Skip if already exists."""

    cluster_dir = f'results/library_size_sweep/cluster_metadata/{task}/seq_{seq_idx}/{size_label}'
    linkage_path = os.path.join(cluster_dir, 'hierarchical_linkage_ward.npy')
    labels_path = os.path.join(cluster_dir, 'cluster_labels.npy')
    
    # Skip if already exists
    if os.path.exists(linkage_path) and os.path.exists(labels_path):
        print(f"Skipping clustering for {size_label} - already exists")
        return np.load(linkage_path), np.load(labels_path)
    
    os.makedirs(cluster_dir, exist_ok=True)

    import time
    print(f"Clustering {len(attrs)} samples...")
    t_start = time.time()

    clusterer = Clusterer(attrs, gpu=True)

    linkage = clusterer.cluster(
        method='hierarchical',
        link_method='ward',
        batch_size=20000
    )
    print(f"Clustering completed in {(time.time() - t_start)/60:.1f} min")
    
    labels, cut_level = clusterer.get_cluster_labels(
        linkage,
        criterion='maxclust',
        n_clusters=n_clusters
    )
    
    np.save(linkage_path, linkage)
    np.save(labels_path, labels)
    print(f"Saved clustering for {size_label}")
    
    return linkage, labels


# Run clustering for all sizes
for task in tasks:
    task_dir = f'mutagenisis_library/{task}'
    seq_folders = [f for f in os.listdir(task_dir) if f.startswith('seq_')]
    
    for seq_folder in seq_folders:
        seq_idx = int(seq_folder.split('_')[1])
        print(f"\n{'='*50}")
        print(f"Clustering {task} seq_{seq_idx}")
        print(f"{'='*50}")
        
        for size_label in reversed(list(subset_sizes.keys())):
            print(f"\n--- {size_label} ---")
            
            # Load attributions
            attr_path = f'attribution_map_libraries/deepSHAP/{task}/seq_{seq_idx}/{size_label}.h5'
            if not os.path.exists(attr_path):
                print(f"Skipping {size_label} - no attributions found")
                continue
            
            attrs = load_attributions(attr_path)
            linkage, labels = cluster_and_save(attrs, task, seq_idx, size_label, n_clusters=cluster_number)


## MetaExplainer - Generate MSM from cluster metadata

def generate_and_save_msm(task, seq_idx, size_label, n_clusters=30, gpu=True):
    """Load cluster metadata and generate MSM."""

    # Define paths
    cluster_dir = f'results/library_size_sweep/cluster_metadata/{task}/seq_{seq_idx}/{size_label}'
    linkage_path = os.path.join(cluster_dir, 'hierarchical_linkage_ward.npy')
    labels_path = os.path.join(cluster_dir, 'cluster_labels.npy')

    csm_dir = f'results/library_size_sweep/CSM/{task}/seq_{seq_idx}/{size_label}'
    msm_path = os.path.join(csm_dir, 'msm.csv')

    # Skip if MSM already exists
    if os.path.exists(msm_path):
        print(f"Skipping {task}/seq_{seq_idx}/{size_label} - MSM already exists")
        return None

    # Check if cluster metadata exists
    if not os.path.exists(linkage_path) or not os.path.exists(labels_path):
        print(f"Skipping {task}/seq_{seq_idx}/{size_label} - cluster metadata not found")
        return None

    # Load cluster metadata
    linkage = np.load(linkage_path)
    labels = np.load(labels_path)

    # Load attributions
    attr_path = f'attribution_map_libraries/deepSHAP/{task}/seq_{seq_idx}/{size_label}.h5'
    if not os.path.exists(attr_path):
        print(f"Skipping {task}/seq_{seq_idx}/{size_label} - attributions not found")
        return None
    attributions = load_attributions(attr_path)

    # Load sequences and predictions
    seqs, preds, orig_idx, subset_idx = load_library(task, seq_idx, size_label)

    # Get reference sequence (first sequence, index 0)
    x_ref = seqs[0:1]

    # Create Clusterer and set cluster_labels
    clusterer = Clusterer(attributions, gpu=gpu)
    clusterer.cluster_labels = labels

    # Create mave_df using Compiler
    compiler = Compiler(
        x=seqs,
        y=preds,
        x_ref=x_ref,
        alphabet=['A', 'C', 'G', 'T'],
        gpu=gpu
    )
    mave_df = compiler.compile()

    # Initialize MetaExplainer
    meta = MetaExplainer(
        clusterer=clusterer,
        mave_df=mave_df,
        attributions=attributions,
        sort_method='median',
        ref_idx=0,
        mut_rate=mutation_rate
    )

    # Generate MSM
    msm = meta.generate_msm(gpu=gpu)

    # Save outputs
    os.makedirs(csm_dir, exist_ok=True)

    # Save MSM
    msm.to_csv(msm_path, index=False)
    print(f"Saved MSM to {msm_path}")

    # Compute and save cluster statistics
    cluster_stats = []
    for k in meta.cluster_indices:
        k_mask = meta.mave['Cluster'] == k
        k_scores = meta.mave.loc[k_mask, 'DNN']
        cluster_stats.append({
            'Cluster': k,
            'Occupancy': k_mask.sum(),
            'Median_DNN': k_scores.median(),
            'Mean_DNN': k_scores.mean(),
            'Std_DNN': k_scores.std()
        })
    cluster_stats_df = pd.DataFrame(cluster_stats)
    cluster_stats_df.to_csv(os.path.join(csm_dir, 'cluster_stats.csv'), index=False)
    print(f"Saved cluster stats")

    # Add Cluster_Sorted column to membership_df (same logic as plot_cluster_stats)
    if meta.cluster_order is not None:
        mapping_dict = {old_k: new_k for new_k, old_k in enumerate(meta.cluster_order)}
        meta.membership_df['Cluster_Sorted'] = meta.membership_df['Cluster'].map(mapping_dict)

    # Save membership dataframe
    meta.membership_df.to_csv(os.path.join(csm_dir, 'membership_df.csv'), index=False)
    print(f"Saved membership dataframe")

    # Save WT cluster info (ref_idx=0 is the WT sequence)
    ref_cluster = meta.membership_df.loc[0, 'Cluster']
    ref_cluster_sorted = meta.membership_df.loc[0, 'Cluster_Sorted'] if 'Cluster_Sorted' in meta.membership_df.columns else ref_cluster
    wt_info = pd.DataFrame({
        'ref_idx': [0],
        'WT_Cluster': [ref_cluster],
        'WT_Cluster_Sorted': [ref_cluster_sorted]
    })
    wt_info.to_csv(os.path.join(csm_dir, 'wt_cluster_info.csv'), index=False)
    print(f"Saved WT cluster info")

    return msm


# Run MSM generation for all available cluster metadata
for task in tasks:
    cluster_task_dir = f'results/library_size_sweep/cluster_metadata/{task}'

    if not os.path.exists(cluster_task_dir):
        print(f"Skipping {task} - no cluster metadata directory")
        continue

    seq_folders = [f for f in os.listdir(cluster_task_dir) if f.startswith('seq_')]

    for seq_folder in seq_folders:
        seq_idx = int(seq_folder.split('_')[1])
        print(f"\n{'='*50}")
        print(f"Generating MSM for {task} seq_{seq_idx}")
        print(f"{'='*50}")

        for size_label in reversed(list(subset_sizes.keys())):
            print(f"\n--- {size_label} ---")
            generate_and_save_msm(task, seq_idx, size_label, n_clusters=cluster_number, gpu=True)




