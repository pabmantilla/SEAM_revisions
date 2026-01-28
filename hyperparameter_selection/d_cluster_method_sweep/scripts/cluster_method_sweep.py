#######
#Script for sweeping through cluster method (Hiearchical, K-means, PCA+K-means, tSNE+K-means)
#######


# Core imports
import os
import numpy as np
import pandas as pd
import h5py
import pickle
import random
import time

# TensorFlow/Keras imports for model loading
import tensorflow as tf


tf.compat.v1.disable_eager_execution()
tf.compat.v1.disable_v2_behavior()
from keras.models import model_from_json
import shap
shap.explainers.deep.deep_tf.op_handlers["AddV2"] = shap.explainers.deep.deep_tf.passthrough

# SEAM imports
import seam
from seam import Compiler, Attributer, Clusterer, MetaExplainer

# SQUID imports for mutagenesis
import squid

# Scipy for correlation analysis
from scipy.stats import spearmanr, pearsonr

# Matplotlib for plotting
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# =============================================================================
# Configuration
# =============================================================================
cluster_number = 30
cluster_method = ['hierarchical', 'kmeans', 'pca+kmeans']
sweep_values = cluster_method
mutation_rate = .10 
lib_size = 25000

task_index = 0  # Dev task

# Base paths
BASE_DIR = '/grid/wsbs/home_norepl/pmantill/SEAM_revisions/SEAM_revisions/hyperparameter_selection'
DEV20_LIBRARY_PATH = f'{BASE_DIR}/data_and_models/dev_20_library/dev_20_library.pkl'
MODEL_DIR = f'{BASE_DIR}/data_and_models/models'
MUTAGENESIS_LIBRARY_DIR = f'{BASE_DIR}/data_and_models/cluster_method_sweep'
ATTRIBUTION_DIR = f'{BASE_DIR}/d_cluster_method_sweep/seq_libraries/cluster_method_sweep/deepshap'
RESULTS_DIR = f'{BASE_DIR}/d_cluster_method_sweep/results'

alphabet = ['A', 'C', 'G', 'T']


############################
# Helper functions
############################

def check_gpu():
    """Check and print GPU availability."""
    print("\n=== GPU Check ===")
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"GPU available: {len(gpus)} GPU(s) detected")
        for gpu in gpus:
            print(f"  - {gpu.name}")
    else:
        print("WARNING: No GPU detected - running on CPU")
    print("=" * 17 + "\n")




###########################
# Loading and completion check functions
###########################

def check_libraries_complete(seq_indices):
    """Check if all mutagenesis libraries exist for all sequences."""
    for seq_idx in seq_indices:
        filepath = f'{MUTAGENESIS_LIBRARY_DIR}/Dev/seq_{seq_idx}/25K.h5'
        if not os.path.exists(filepath):
            return False
    return True

def check_attributions_complete(seq_indices):
    """Check if all attributions exist for all sequences."""
    for seq_idx in seq_indices:
        filepath = f'{ATTRIBUTION_DIR}/Dev/seq_{seq_idx}/25K.h5'
        if not os.path.exists(filepath):
            return False
    return True

def load_library_25k(seq_idx):
    """Load the 25K library for a sequence."""
    filepath = f'{MUTAGENESIS_LIBRARY_DIR}/Dev/seq_{seq_idx}/25K.h5'
    with h5py.File(filepath, 'r') as f:
        sequences = f['sequences'][:]
        predictions = f['predictions'][:]
        original_idx = f.attrs['original_idx']
        library_index = f['library_index'][:] if 'library_index' in f else np.arange(len(sequences))
    return sequences, predictions, original_idx, library_index

def check_clustering_complete(seq_indices, sweep_values):
    """Check if all clustering results exist for all sequences and cluster methods."""
    for seq_idx in seq_indices:
        for sweep_value in sweep_values:
            cluster_method = sweep_value
            cluster_dir = f'{RESULTS_DIR}/clustering/seq_{seq_idx}/{cluster_method}_clustering'
            labels_path = os.path.join(cluster_dir, 'cluster_labels.npy')
            if not os.path.exists(labels_path):
                return False
            # Only check linkage for hierarchical
            if cluster_method == 'hierarchical':
                linkage_path = os.path.join(cluster_dir, 'hierarchical_linkage_ward.npy')
                if not os.path.exists(linkage_path):
                    return False
    return True

def check_msms_complete(seq_indices, sweep_values):
    """Check if all MSMs and variance summaries exist for all sequences and cluster numbers."""
    for seq_idx in seq_indices:
        for sweep_value in sweep_values:
            cluster_method = sweep_value
            msm_dir = f'{RESULTS_DIR}/msms/seq_{seq_idx}/{cluster_method}_clustering'
            msm_path = os.path.join(msm_dir, 'msm.csv')
            variance_path = os.path.join(msm_dir, 'variance_summary.csv')
            if not os.path.exists(msm_path) or not os.path.exists(variance_path):
                return False
    return True

def load_attributions(seq_idx):
    """Load attributions for a sequence."""
    filepath = f'{ATTRIBUTION_DIR}/Dev/seq_{seq_idx}/25K.h5'
    with h5py.File(filepath, 'r') as f:
        return f['attributions'][:]

def check_correlations_complete(seq_indices, reference_method):
    """Check if all correlation files exist for all sequences."""
    for seq_idx in seq_indices:
        corr_path = f'{RESULTS_DIR}/correlations/seq_{seq_idx}/correlation_with_{reference_method}_method.csv'
        if not os.path.exists(corr_path):
            return False
    return True

def check_correlation_plots_complete(seq_indices, reference_method):
    """Check if all individual correlation plots exist for all sequences."""
    for seq_idx in seq_indices:
        plot_path = f'{RESULTS_DIR}/correlations/seq_{seq_idx}/correlation_with_{reference_method}_method.png'
        if not os.path.exists(plot_path):
            return False
    return True


def check_summary_plot_complete():
    """Check if the summary correlation plot exists."""
    plot_path = f'{RESULTS_DIR}/results_final/correlation_summary.png'
    return os.path.exists(plot_path)


def check_summary_plots_complete(sweep_values):
    """Check if all summary correlation plots exist (one per cluster method as reference)."""
    for reference_method in sweep_values:
        plot_path = f'{RESULTS_DIR}/results_final/correlation_summary_with_{reference_method}_method.png'
        if not os.path.exists(plot_path):
            return False
    return True


###########################
# Saving functions
###########################

def save_library(filepath, sequences, predictions, original_idx):
    """Save mutagenesis library to HDF5 file."""
    n_samples = len(sequences)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with h5py.File(filepath, 'w') as f:
        f.create_dataset('sequences', data=sequences, compression='gzip', compression_opts=4)
        f.create_dataset('predictions', data=predictions, compression='gzip', compression_opts=4)
        f.create_dataset('library_index', data=np.arange(n_samples), compression='gzip', compression_opts=4)
        f.attrs['original_idx'] = original_idx
        f.attrs['n_samples'] = n_samples

def save_attributions(filepath, attributions, original_idx, subset_idx=None):
    """Save attributions to HDF5 file with optional subset indices."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with h5py.File(filepath, 'w') as f:
        f.create_dataset('attributions', data=attributions, compression='gzip', compression_opts=4)
        if subset_idx is not None:
            f.create_dataset('subset_idx', data=subset_idx, compression='gzip', compression_opts=4)
        f.attrs['original_idx'] = original_idx
        f.attrs['n_samples'] = len(attributions)

def cluster_and_save_hierarchical(attrs, seq_idx, gpu=True):
    """Cluster attributions using hierarchical clustering and save results."""
    cluster_dir = f'{RESULTS_DIR}/clustering/seq_{seq_idx}/hierarchical_clustering'
    linkage_path = os.path.join(cluster_dir, 'hierarchical_linkage_ward.npy')
    labels_path = os.path.join(cluster_dir, 'cluster_labels.npy')

    # Skip if already exists
    if os.path.exists(linkage_path) and os.path.exists(labels_path):
        print(f"Skipping clustering for hierarchical_clustering - already exists")
        return np.load(linkage_path), np.load(labels_path)

    os.makedirs(cluster_dir, exist_ok=True)

    print(f"Clustering {len(attrs)} samples...")
    t_start = time.time()

    clusterer = Clusterer(attrs, gpu=gpu)

    linkage = clusterer.cluster(
        method='hierarchical',
        link_method='ward',
        batch_size=20000
    )
    print(f"Clustering completed in {(time.time() - t_start)/60:.1f} min")

    labels, cut_level = clusterer.get_cluster_labels(
        linkage,
        criterion='maxclust',
        n_clusters=cluster_number  # Uses global cluster_number (30)
    )

    np.save(linkage_path, linkage)
    np.save(labels_path, labels)
    print(f"Saved clustering for hierarchical_clustering")

    return linkage, labels

def cluster_and_save_kmeans(attrs, seq_idx, cluster_method, gpu=True):
    """Cluster attributions using kmeans or pca+kmeans and save results."""
    cluster_dir = f'{RESULTS_DIR}/clustering/seq_{seq_idx}/{cluster_method}_clustering'
    labels_path = os.path.join(cluster_dir, 'cluster_labels.npy')

    # Skip if already exists
    if os.path.exists(labels_path):
        print(f"Skipping clustering for {cluster_method}_clustering - already exists")
        return np.load(labels_path)

    os.makedirs(cluster_dir, exist_ok=True)
    t_start = time.time()

    if cluster_method == 'kmeans':
        # Direct kmeans on flattened attribution maps
        clusterer = Clusterer(attrs, gpu=gpu)
        labels = clusterer.cluster(
            embedding=None,  # Uses flattened maps directly
            method='kmeans',
            n_clusters=cluster_number,
            random_state=42,
            n_init=10,
            max_iter=300
        )
        print(f'KMeans clustering time: {(time.time() - t_start)/60:.2f} minutes')

    elif cluster_method == 'pca+kmeans':
        # PCA embedding followed by kmeans
        clusterer = Clusterer(attrs, method='pca', gpu=gpu)

        t_embed = time.time()
        pca_embedding = clusterer.embed(
            n_components=20,
            plot_eigenvalues=False,
            save_path=None
        )
        print(f'Embedding (PCA) time: {(time.time() - t_embed)/60:.2f} minutes')

        t_kmeans = time.time()
        labels = clusterer.cluster(
            embedding=pca_embedding,
            method='kmeans',
            n_clusters=cluster_number,
            random_state=42,
            n_init=10,
            max_iter=300
        )
        print(f'KMeans clustering time: {(time.time() - t_kmeans)/60:.2f} minutes')

    else:
        raise ValueError(f"Unsupported cluster method: {cluster_method}")

    # Save labels
    np.save(labels_path, labels)
    print(f"Saved clustering for {cluster_method}_clustering")

    return labels

def generate_and_save_msm(seq_idx, sweep_value, gpu=True):
    """Load cluster metadata and generate MSM."""
    cluster_method = sweep_value

    # Define paths
    cluster_dir = f'{RESULTS_DIR}/clustering/seq_{seq_idx}/{cluster_method}_clustering'
    labels_path = os.path.join(cluster_dir, 'cluster_labels.npy')

    msm_dir = f'{RESULTS_DIR}/msms/seq_{seq_idx}/{cluster_method}_clustering'
    msm_path = os.path.join(msm_dir, 'msm.csv')

    # Skip if MSM already exists
    if os.path.exists(msm_path):
        print(f"Skipping seq_{seq_idx}/{cluster_method}_clustering - MSM already exists")
        return None

    # Check if cluster labels exist
    if not os.path.exists(labels_path):
        print(f"Skipping seq_{seq_idx}/{cluster_method}_clustering - cluster labels not found")
        return None

    # Load cluster labels
    labels = np.load(labels_path)

    # Load attributions
    try:
        attributions = load_attributions(seq_idx)
    except:
        print(f"Skipping seq_{seq_idx}/{cluster_method}_clustering - attributions not found")
        return None

    # Load sequences and predictions
    seqs, preds, orig_idx, library_index = load_library_25k(seq_idx)

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
        alphabet=alphabet,
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
    os.makedirs(msm_dir, exist_ok=True)

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
    cluster_stats_df.to_csv(os.path.join(msm_dir, 'cluster_stats.csv'), index=False)
    print(f"Saved cluster stats")

    # Add Cluster_Sorted column to membership_df
    if meta.cluster_order is not None:
        mapping_dict = {old_k: new_k for new_k, old_k in enumerate(meta.cluster_order)}
        meta.membership_df['Cluster_Sorted'] = meta.membership_df['Cluster'].map(mapping_dict)

    # Save membership dataframe
    meta.membership_df.to_csv(os.path.join(msm_dir, 'membership_df.csv'), index=False)
    print(f"Saved membership dataframe")

    # Save WT cluster info (ref_idx=0 is the WT sequence)
    ref_cluster = meta.membership_df.loc[0, 'Cluster']
    ref_cluster_sorted = meta.membership_df.loc[0, 'Cluster_Sorted'] if 'Cluster_Sorted' in meta.membership_df.columns else ref_cluster
    wt_info = pd.DataFrame({
        'ref_idx': [0],
        'WT_Cluster': [ref_cluster],
        'WT_Cluster_Sorted': [ref_cluster_sorted]
    })
    wt_info.to_csv(os.path.join(msm_dir, 'wt_cluster_info.csv'), index=False)
    print(f"Saved WT cluster info")

    return msm

def compute_and_save_variance_summary(seq_idx, sweep_value):
    """Load MSM and compute variance of entropy across clusters for each position."""
    cluster_method = sweep_value

    msm_dir = f'{RESULTS_DIR}/msms/seq_{seq_idx}/{cluster_method}_clustering'
    msm_path = os.path.join(msm_dir, 'msm.csv')
    variance_path = os.path.join(msm_dir, 'variance_summary.csv')

    # Skip if variance summary already exists
    if os.path.exists(variance_path):
        print(f"Skipping seq_{seq_idx}/{cluster_method}_clustering - variance summary already exists")
        return None

    # Check if MSM exists
    if not os.path.exists(msm_path):
        print(f"Skipping seq_{seq_idx}/{cluster_method}_clustering - MSM not found")
        return None

    # Load MSM
    msm = pd.read_csv(msm_path)

    # Pivot to get Cluster x Position matrix of entropy values
    entropy_matrix = msm.pivot(index='Cluster', columns='Position', values='Entropy')

    # Compute variance across clusters for each position
    variance_per_position = entropy_matrix.var(axis=0)

    # Create DataFrame and save
    variance_df = pd.DataFrame({
        'Position': variance_per_position.index,
        'Variance': variance_per_position.values
    })

    variance_df.to_csv(variance_path, index=False)
    print(f"Saved variance summary to {variance_path}")

    return variance_df

def check_variance_combined_plots_complete(seq_indices):
    """Check if all combined variance summary plots exist."""
    for seq_idx in seq_indices:
        plot_path = f'{RESULTS_DIR}/msms/seq_{seq_idx}/variance_summary_combined.png'
        if not os.path.exists(plot_path):
            return False
    return True

def compute_cluster_method_correlations(seq_idx, sweep_values, reference_method):
    """Compute Pearson and Spearman correlations with reference cluster method on variance summaries."""
    output_dir = f'{RESULTS_DIR}/correlations/seq_{seq_idx}'
    corr_path = os.path.join(output_dir, f'correlation_with_{reference_method}_method.csv')

    # Skip if correlations already exist
    if os.path.exists(corr_path):
        print(f"Skipping seq_{seq_idx} - correlations already exist")
        return None

    # Load reference variance summary
    variance_ref_path = f'{RESULTS_DIR}/msms/seq_{seq_idx}/{reference_method}_clustering/variance_summary.csv'
    if not os.path.exists(variance_ref_path):
        print(f"Skipping seq_{seq_idx} - {reference_method}_clustering variance summary not found")
        return None

    variance_ref = pd.read_csv(variance_ref_path)['Variance'].values

    # Compute correlations for each cluster method
    results = []
    for sweep_value in sweep_values:
        variance_path = f'{RESULTS_DIR}/msms/seq_{seq_idx}/{sweep_value}_clustering/variance_summary.csv'
        if not os.path.exists(variance_path):
            continue

        variance_values = pd.read_csv(variance_path)['Variance'].values
        pearson_corr, _ = pearsonr(variance_ref, variance_values)
        spearman_corr, _ = spearmanr(variance_ref, variance_values)

        results.append({
            'Cluster_Method': sweep_value,
            'Pearson': pearson_corr,
            'Spearman': spearman_corr
        })

    if len(results) < 2:
        print(f"Skipping seq_{seq_idx} - need at least 2 cluster methods")
        return None

    corr_df = pd.DataFrame(results)
    os.makedirs(output_dir, exist_ok=True)
    corr_df.to_csv(corr_path, index=False)
    print(f"Saved correlations to {corr_path}")
    return corr_df

###########################
# Attribution functions
###########################

def seam_deepshap(x_mut, task_index, checkpoint_path=None, checkpoint_every=5000):
    """Compute DeepSHAP attributions with optional checkpointing."""
    print(f"Computing attributions for task_index: {task_index}")

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
    gpu = 0

    # Model paths
    keras_model_json = os.path.join(MODEL_DIR, 'deepstarr.model.json')
    keras_model_weights = os.path.join(MODEL_DIR, 'deepstarr.model.h5')

    try:
        keras_model = model_from_json(open(keras_model_json).read(), custom_objects={'Functional': tf.keras.Model})
        np.random.seed(113)
        random.seed(0)
        keras_model.load_weights(keras_model_weights)
        model_local = keras_model

        _ = model_local(tf.keras.Input(shape=model_local.input_shape[1:]))

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
        model_local,
        method='deepshap',
        task_index=task_index,
        compress_fun=deepstarr_compress
    )

    attributer.show_params('deepshap')

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

###########################
# Plotting functions
###########################

def plot_correlation_for_seq(seq_idx, reference_method):
    """Plot correlation for each cluster method for a single sequence."""
    output_dir = f'{RESULTS_DIR}/correlations/seq_{seq_idx}'
    corr_path = os.path.join(output_dir, f'correlation_with_{reference_method}_method.csv')
    plot_path = os.path.join(output_dir, f'correlation_with_{reference_method}_method.png')

    if not os.path.exists(corr_path):
        print(f"Skipping plot for seq_{seq_idx} - correlation file not found")
        return None

    corr_df = pd.read_csv(corr_path)

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot Pearson correlation as line plot
    x_positions = range(len(corr_df))
    ax.plot(x_positions, corr_df['Pearson'], 'o-',
            color='steelblue', markersize=8, linewidth=2, label='Pearson')

    # Mark reference method
    method_list = corr_df['Cluster_Method'].tolist()
    if reference_method in method_list:
        ref_idx = method_list.index(reference_method)
        ax.axvline(x=ref_idx, color='gray', linestyle='--', alpha=0.7, label=f'{reference_method} reference')

    # Set x-axis labels
    ax.set_xticks(x_positions)
    ax.set_xticklabels(corr_df['Cluster_Method'], rotation=45, ha='right')

    # Add horizontal line at correlation = 1
    ax.axhline(y=1, color='lightgray', linestyle=':', alpha=0.7)

    ax.set_xlabel('Cluster Method', fontsize=12)
    ax.set_ylabel('Pearson Correlation', fontsize=12)
    ax.set_title(f'Correlation with {reference_method} Reference - Seq {seq_idx}', fontsize=14)
    ax.legend(loc='lower right')
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved correlation plot to {plot_path}")
    return corr_df


def plot_summary_correlations(seq_indices, reference_method):
    """Plot summary of correlations for all sequences on a single figure."""
    # Create output directory
    output_dir = f'{RESULTS_DIR}/results_final'
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, 'correlation_summary.png')

    # Collect data from all sequences
    all_data = []
    valid_seq_indices = []

    for seq_idx in seq_indices:
        corr_path = f'{RESULTS_DIR}/correlations/seq_{seq_idx}/correlation_with_{reference_method}_method.csv'
        if os.path.exists(corr_path):
            corr_df = pd.read_csv(corr_path)
            corr_df['seq_idx'] = seq_idx
            all_data.append(corr_df)
            valid_seq_indices.append(seq_idx)

    if not all_data:
        print("No correlation data found for any sequence")
        return None

    # Get method names from first dataframe
    method_names = all_data[0]['Cluster_Method'].tolist()
    x_positions = list(range(len(method_names)))

    # Create colormap for sequences
    n_seqs = len(valid_seq_indices)
    colors = cm.viridis(np.linspace(0, 1, n_seqs))

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 7))

    # Plot each sequence as line plot
    for i, (seq_idx, corr_df) in enumerate(zip(valid_seq_indices, all_data)):
        ax.plot(x_positions, corr_df['Pearson'], 'o-',
                color=colors[i], markersize=6, linewidth=1.5, alpha=0.8,
                label=f'Seq {seq_idx}')

    # Mark reference method with vertical line
    if reference_method in method_names:
        ref_idx = method_names.index(reference_method)
        ax.axvline(x=ref_idx, color='black', linestyle='--', linewidth=1.5, alpha=0.7, label=f'{reference_method} reference')

    # Set x-axis labels
    ax.set_xticks(x_positions)
    ax.set_xticklabels(method_names, rotation=45, ha='right')

    # Add horizontal line at correlation = 1
    ax.axhline(y=1, color='gray', linestyle=':', alpha=0.5)

    ax.set_xlabel('Cluster Method', fontsize=12)
    ax.set_ylabel('Pearson Correlation', fontsize=12)
    ax.set_title(f'Correlation with {reference_method} Reference', fontsize=14)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8, ncol=2)

    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved summary correlation plot to {plot_path}")
    return plot_path


def compute_correlations_for_reference(seq_idx, reference_method, sweep_values):
    """Compute correlations with a given reference cluster number using variance summaries."""

    # Load reference variance summary
    variance_ref_path = f'{RESULTS_DIR}/msms/seq_{seq_idx}/{reference_method}_clustering/variance_summary.csv'
    if not os.path.exists(variance_ref_path):
        return None

    variance_ref = pd.read_csv(variance_ref_path)['Variance'].values

    # Compute correlations for each cluster number
    results = []
    for sweep_value in sweep_values:
        variance_path = f'{RESULTS_DIR}/msms/seq_{seq_idx}/{sweep_value}_clustering/variance_summary.csv'
        if not os.path.exists(variance_path):
            continue

        variance_values = pd.read_csv(variance_path)['Variance'].values
        pearson_corr, _ = pearsonr(variance_ref, variance_values)
        spearman_corr, _ = spearmanr(variance_ref, variance_values)

        results.append({
            'Cluster_Method': sweep_value,
            'Pearson': pearson_corr,
            'Spearman': spearman_corr
        })

    return pd.DataFrame(results) if results else None


def plot_summary_correlations_for_reference(seq_indices, reference_method, sweep_values):
    """Plot summary of correlations for all sequences on a single figure for a given reference."""
    # Create output directory
    output_dir = f'{RESULTS_DIR}/results_final'
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, f'correlation_summary_with_{reference_method}_method.png')

    # Collect data from all sequences (compute on the fly)
    all_data = []
    valid_seq_indices = []

    for seq_idx in seq_indices:
        corr_df = compute_correlations_for_reference(seq_idx, reference_method, sweep_values)
        if corr_df is not None and len(corr_df) > 1:
            corr_df['seq_idx'] = seq_idx
            all_data.append(corr_df)
            valid_seq_indices.append(seq_idx)

    if not all_data:
        print(f"No correlation data found for reference {reference_method}")
        return None

    # Get method names from first dataframe
    method_names = all_data[0]['Cluster_Method'].tolist()
    x_positions = list(range(len(method_names)))

    # Create colormap for sequences
    n_seqs = len(valid_seq_indices)
    colors = cm.viridis(np.linspace(0, 1, n_seqs))

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 7))

    # Plot each sequence as line plot
    for i, (seq_idx, corr_df) in enumerate(zip(valid_seq_indices, all_data)):
        ax.plot(x_positions, corr_df['Pearson'], 'o-',
                color=colors[i], markersize=6, linewidth=1.5, alpha=0.8,
                label=f'Seq {seq_idx}')

    # Mark reference method with vertical line
    if reference_method in method_names:
        ref_idx = method_names.index(reference_method)
        ax.axvline(x=ref_idx, color='black', linestyle='--', linewidth=1.5, alpha=0.7, label=f'{reference_method} reference')

    # Set x-axis labels
    ax.set_xticks(x_positions)
    ax.set_xticklabels(method_names, rotation=45, ha='right')

    # Add horizontal line at correlation = 1
    ax.axhline(y=1, color='gray', linestyle=':', alpha=0.5)

    ax.set_xlabel('Cluster Method', fontsize=12)
    ax.set_ylabel('Pearson Correlation', fontsize=12)
    ax.set_title(f'Correlation with {reference_method} Method Reference', fontsize=14)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8, ncol=2)

    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved summary correlation plot to {plot_path}")
    return plot_path


def plot_all_summary_correlation_plots(seq_indices, sweep_values):
    """Generate individual summary plots for each cluster method as reference."""
    for reference_method in sweep_values:
        plot_path = f'{RESULTS_DIR}/results_final/correlation_summary_with_{reference_method}_method.png'
        if not os.path.exists(plot_path):
            print(f"Plot for {reference_method} method is missing, generating...")
            plot_summary_correlations_for_reference(seq_indices, reference_method, sweep_values)
        else:
            print(f"Plot for {reference_method} method already exists")

def plot_combined_variance_summary(seq_idx):
    """Plot all variance summaries stacked vertically for all cluster methods."""
    msm_base_dir = f'{RESULTS_DIR}/msms/seq_{seq_idx}'
    plot_path = os.path.join(msm_base_dir, 'variance_summary_combined.png')

    # Skip if combined plot already exists
    if os.path.exists(plot_path):
        print(f"Skipping combined variance plot for seq_{seq_idx} - already exists")
        return

    # Load all variance summaries
    variance_data = {}
    for method in sweep_values:
        variance_path = os.path.join(msm_base_dir, f'{method}_clustering', 'variance_summary.csv')
        if os.path.exists(variance_path):
            variance_data[method] = pd.read_csv(variance_path)

    if len(variance_data) == 0:
        print(f"Skipping combined variance plot for seq_{seq_idx} - no variance summaries found")
        return

    # Create figure with subplots
    n_plots = len(variance_data)
    fig, axes = plt.subplots(n_plots, 1, figsize=(12, 2.5 * n_plots), sharex=True)

    if n_plots == 1:
        axes = [axes]

    # Get global y-axis limits
    all_variances = [df['Variance'].values for df in variance_data.values()]
    y_max = max(v.max() for v in all_variances)
    y_min = min(v.min() for v in all_variances)

    for ax, (label, df) in zip(axes, variance_data.items()):
        ax.plot(df['Position'], df['Variance'], linewidth=0.8)
        ax.set_ylabel(f'{label}')
        ax.set_ylim(y_min, y_max * 1.05)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Position')
    fig.suptitle(f'Dev seq_{seq_idx} - Entropy Variance by Position', fontsize=12)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved combined variance plot to {plot_path}")



### 
# Main loop
###
if __name__ == "__main__":
    print("=" * 60)
    print("Cluster Method Sweep")
    print("=" * 60)

    reference_method = 'hierarchical'

    # 1. Load Dev_20 library
    print("\nLoading Dev_20 library...")
    dev_pkl = pd.read_pickle(DEV20_LIBRARY_PATH)
    dev_pkl = dev_pkl["dev"]
    print(f"Loaded {len(dev_pkl)} sequences")
    seq_indices = dev_pkl["test_idx"].tolist()
    x_seqs = dev_pkl["ohe_seq"]

    # 2. Load DeepSTARR model
    print("\nLoading DeepSTARR model...")
    model_json_file = os.path.join(MODEL_DIR, 'deepstarr.model.json')
    model_weights_file = os.path.join(MODEL_DIR, 'deepstarr.model.h5')

    with open(model_json_file, 'r') as f:
        model_json = f.read()

    model = model_from_json(model_json, custom_objects={'Functional': tf.keras.Model})
    np.random.seed(113)
    random.seed(0)
    model.load_weights(model_weights_file)
    print("Model loaded successfully!")
    check_gpu()
    # 3. Generate ONE 25K library and compute ONE set of DeepSHAP attributions per sequence
    # (cluster method only affects clustering, not the library or attributions)
    libraries_complete = check_libraries_complete(seq_indices)
    attributions_complete = check_attributions_complete(seq_indices)

    if libraries_complete and attributions_complete:
        print("\n" + "=" * 60)
        print("SKIPPING: All libraries and attributions already exist")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("Generating 25K libraries and attributions (one per sequence)")
        print("=" * 60)

        # Generate libraries for all sequences (ONE per sequence)
        for i, (x_seq, idx) in enumerate(zip(x_seqs, seq_indices)):
            output_dir = f'{MUTAGENESIS_LIBRARY_DIR}/Dev/seq_{idx}'
            output_file = f'{output_dir}/25K.h5'

            # Check if library already exists
            if os.path.exists(output_file):
                print(f"Skipping seq_{idx} - library already exists")
                continue

            os.makedirs(output_dir, exist_ok=True)

            x_seq = np.array(x_seq)

            # Create predictor
            pred_generator = squid.predictor.ScalarPredictor(
                pred_fun=model.predict_on_batch,
                task_idx=task_index,
                batch_size=512
            )

            # Create mutagenizer
            mut_generator = squid.mutagenizer.RandomMutagenesis(
                mut_rate=mutation_rate,
                seed=42
            )

            # Create MAVE
            mave = squid.mave.InSilicoMAVE(
                mut_generator,
                pred_generator,
                seq_length=249,
                mut_window=[0, 249]
            )

            # Generate 25k mutant sequences
            x_mut, y_mut = mave.generate(x_seq, num_sim=lib_size)

            # Save library
            save_library(output_file, x_mut, y_mut, idx)
            print(f"[{i+1}/{len(x_seqs)}] Created seq_{idx}/25K.h5")

        # Compute DeepSHAP attributions (ONE per sequence)
        for idx in seq_indices:
            output_dir = f'{ATTRIBUTION_DIR}/Dev/seq_{idx}'
            output_file = f'{output_dir}/25K.h5'

            if os.path.exists(output_file):
                print(f"Skipping seq_{idx} - attributions already exist")
                continue

            # Load library
            x_mut, y_mut, original_idx, library_index = load_library_25k(idx)

            # Get DeepSHAP attributions
            checkpoint_path = f'{output_dir}/checkpoint.h5'
            os.makedirs(output_dir, exist_ok=True)
            attributions = seam_deepshap(x_mut, task_index, checkpoint_path=checkpoint_path)

            # Save attributions
            save_attributions(output_file, attributions, original_idx)

            # Remove checkpoint after successful completion
            if os.path.exists(checkpoint_path):
                os.remove(checkpoint_path)

            print(f"Saved attributions for seq_{idx}")

    # 4. Clustering and MSM generation for all sequences and cluster numbers
    clustering_complete = check_clustering_complete(seq_indices, sweep_values)
    msms_complete = check_msms_complete(seq_indices, sweep_values)

    if clustering_complete and msms_complete:
        print("\n" + "=" * 60)
        print("SKIPPING: All clustering and MSM results already exist")
        print("=" * 60)
    else:
        for seq_idx in seq_indices:
            print("\n" + "=" * 60)
            print(f"Clustering and MSM generation for seq_{seq_idx}")
            print("=" * 60)

            # Load attributions ONCE per sequence
            try:
                attrs = load_attributions(seq_idx)
            except Exception as e:
                print(f"Skipping seq_{seq_idx} - no attributions found: {e}")
                continue

            for main_sweep_value in sweep_values:
                cluster_method = main_sweep_value
                print(f"\n--- {cluster_method} clustering ---")

                # Cluster with different cluster methods

                if cluster_method == 'hierarchical':
                    linkage, labels = cluster_and_save_hierarchical(attrs, seq_idx, gpu=True)
                else:
                    labels = cluster_and_save_kmeans(attrs, seq_idx, cluster_method, gpu=True)
                   

                # Generate MSM
                generate_and_save_msm(seq_idx, main_sweep_value, gpu=True)

                # Compute variance summary
                compute_and_save_variance_summary(seq_idx, main_sweep_value)

    # 5. Generate combined variance summary plots
    if check_variance_combined_plots_complete(seq_indices):
        print("\n" + "=" * 60)
        print("SKIPPING: All combined variance plots already exist")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("Generating combined variance summary plots")
        print("=" * 60)

        for seq_idx in seq_indices:
            plot_combined_variance_summary(seq_idx)

    # 6. Compute correlations with reference cluster number
    if check_correlations_complete(seq_indices, reference_method):
        print("\n" + "=" * 60)
        print("SKIPPING: All correlation files already exist")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("Computing correlations with reference cluster method")
        print("=" * 60)

        for seq_idx in seq_indices:
            print(f"\nComputing correlations for seq_{seq_idx}")
            compute_cluster_method_correlations(seq_idx, sweep_values, reference_method=reference_method)

    # 7. Generate individual correlation plots for each sequence
    if check_correlation_plots_complete(seq_indices, reference_method):
        print("\n" + "=" * 60)
        print("SKIPPING: All individual correlation plots already exist")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("Generating correlation plots for each sequence")
        print("=" * 60)

        for seq_idx in seq_indices:
            plot_correlation_for_seq(seq_idx, reference_method=reference_method)

    # 8. Generate summary plot with all sequences
    if check_summary_plot_complete():
        print("\n" + "=" * 60)
        print("SKIPPING: Summary correlation plot already exists")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("Generating summary correlation plot")
        print("=" * 60)

        plot_summary_correlations(seq_indices, reference_method=reference_method)

    # 9. Generate per-reference summary plots (one for each cluster number as reference)
    if check_summary_plots_complete(sweep_values):
        print("\n" + "=" * 60)
        print("SKIPPING: All per-reference summary plots already exist")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("Generating per-reference summary plots")
        print("=" * 60)

        plot_all_summary_correlation_plots(seq_indices, sweep_values)

