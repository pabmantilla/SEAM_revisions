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
mutation_rates = [.75, .50, .25, 0.20, .15, .12, .10, .08, .05, .03, .01]   # High to low sweep
lib_size = 25000
cluster_number = 30
task_index = 0  # Dev task

# Base paths
BASE_DIR = '/grid/wsbs/home_norepl/pmantill/SEAM_revisions/SEAM_revisions/hyperparameter_selection'
DEV20_LIBRARY_PATH = f'{BASE_DIR}/data_and_models/dev_20_library/dev_20_library.pkl'
MODEL_DIR = f'{BASE_DIR}/data_and_models/models'
MUTAGENESIS_LIBRARY_DIR = f'{BASE_DIR}/data_and_models/mut_sweep_libraries'
ATTRIBUTION_DIR = f'{BASE_DIR}/b_mutation_rate_sweep/seq_libraries/mut_sweep/deepshap'
RESULTS_DIR = f'{BASE_DIR}/b_mutation_rate_sweep/results'

alphabet = ['A', 'C', 'G', 'T']

# =============================================================================
# Helper Functions
# =============================================================================

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
        print(gpus)
    print("=" * 17 + "\n")


def get_mut_label(mut_rate):
    """Convert mutation rate to label string (e.g., 0.75 -> '75.0%')"""
    return f"{mut_rate*100}%"


# =============================================================================
# Completion Check Functions
# =============================================================================

def check_libraries_complete(seq_indices):
    """Check if all mutagenesis libraries exist for all sequences and mutation rates."""
    for seq_idx in seq_indices:
        for mut_rate in mutation_rates:
            mut_label = get_mut_label(mut_rate)
            filepath = f'{MUTAGENESIS_LIBRARY_DIR}/Dev/seq_{seq_idx}/{mut_label}/25K.h5'
            if not os.path.exists(filepath):
                return False
    return True


def check_attributions_complete(seq_indices):
    """Check if all attributions exist for all sequences and mutation rates."""
    for seq_idx in seq_indices:
        for mut_rate in mutation_rates:
            mut_label = get_mut_label(mut_rate)
            filepath = f'{ATTRIBUTION_DIR}/Dev/seq_{seq_idx}/{mut_label}/25K.h5'
            if not os.path.exists(filepath):
                return False
    return True


def check_clustering_complete(seq_indices):
    """Check if all clustering results exist for all sequences and mutation rates."""
    for seq_idx in seq_indices:
        for mut_rate in mutation_rates:
            mut_label = get_mut_label(mut_rate)
            cluster_dir = f'{RESULTS_DIR}/clustering/seq_{seq_idx}/{mut_label}'
            linkage_path = os.path.join(cluster_dir, 'hierarchical_linkage_ward.npy')
            labels_path = os.path.join(cluster_dir, 'cluster_labels.npy')
            if not os.path.exists(linkage_path) or not os.path.exists(labels_path):
                return False
    return True


def check_msms_complete(seq_indices):
    """Check if all MSMs and variance summaries exist for all sequences and mutation rates."""
    for seq_idx in seq_indices:
        for mut_rate in mutation_rates:
            mut_label = get_mut_label(mut_rate)
            msm_dir = f'{RESULTS_DIR}/msms/seq_{seq_idx}/{mut_label}'
            msm_path = os.path.join(msm_dir, 'msm.csv')
            variance_path = os.path.join(msm_dir, 'variance_summary.csv')
            if not os.path.exists(msm_path) or not os.path.exists(variance_path):
                return False
    return True


def check_correlations_complete(seq_indices):
    """Check if all correlation files exist and contain all mutation rates."""
    ref_label = get_mut_label(0.10)
    expected_rates = len(mutation_rates)
    for seq_idx in seq_indices:
        corr_path = f'{RESULTS_DIR}/correlations/seq_{seq_idx}/correlation_with_{ref_label}.csv'
        if not os.path.exists(corr_path):
            return False
        # Check if all mutation rates are present
        corr_df = pd.read_csv(corr_path)
        if len(corr_df) < expected_rates:
            return False
    return True


def check_correlation_plots_complete(seq_indices):
    """Check if all individual correlation plots exist for all sequences."""
    ref_label = get_mut_label(0.10)
    for seq_idx in seq_indices:
        plot_path = f'{RESULTS_DIR}/correlations/seq_{seq_idx}/correlation_with_{ref_label}.png'
        if not os.path.exists(plot_path):
            return False
    return True


def check_summary_plot_complete():
    """Check if the summary correlation plot exists."""
    plot_path = f'{RESULTS_DIR}/results_final/correlation_summary.png'
    return os.path.exists(plot_path)


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


def load_library_25k(seq_idx, mut_rate):
    """Load the 25K library for a sequence and mutation rate."""
    mut_label = get_mut_label(mut_rate)
    filepath = f'{MUTAGENESIS_LIBRARY_DIR}/Dev/seq_{seq_idx}/{mut_label}/25K.h5'
    with h5py.File(filepath, 'r') as f:
        sequences = f['sequences'][:]
        predictions = f['predictions'][:]
        original_idx = f.attrs['original_idx']
        library_index = f['library_index'][:] if 'library_index' in f else np.arange(len(sequences))
    return sequences, predictions, original_idx, library_index


def save_attributions(filepath, attributions, original_idx, subset_idx=None):
    """Save attributions to HDF5 file with optional subset indices."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with h5py.File(filepath, 'w') as f:
        f.create_dataset('attributions', data=attributions, compression='gzip', compression_opts=4)
        if subset_idx is not None:
            f.create_dataset('subset_idx', data=subset_idx, compression='gzip', compression_opts=4)
        f.attrs['original_idx'] = original_idx
        f.attrs['n_samples'] = len(attributions)


def load_attributions_mut(seq_idx, mut_rate):
    """Load attributions for a sequence and mutation rate."""
    mut_label = get_mut_label(mut_rate)
    filepath = f'{ATTRIBUTION_DIR}/Dev/seq_{seq_idx}/{mut_label}/25K.h5'
    with h5py.File(filepath, 'r') as f:
        return f['attributions'][:]


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


def cluster_and_save(attrs, seq_idx, mut_rate, n_clusters=30):
    """Cluster attributions and save results. Skip if already exists."""
    mut_label = get_mut_label(mut_rate)
    cluster_dir = f'{RESULTS_DIR}/clustering/seq_{seq_idx}/{mut_label}'
    linkage_path = os.path.join(cluster_dir, 'hierarchical_linkage_ward.npy')
    labels_path = os.path.join(cluster_dir, 'cluster_labels.npy')

    # Skip if already exists
    if os.path.exists(linkage_path) and os.path.exists(labels_path):
        print(f"Skipping clustering for {mut_label} - already exists")
        return np.load(linkage_path), np.load(labels_path)

    os.makedirs(cluster_dir, exist_ok=True)

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
    print(f"Saved clustering for {mut_label}")

    return linkage, labels


def generate_and_save_msm(seq_idx, mut_rate, n_clusters=30, gpu=True):
    """Load cluster metadata and generate MSM."""
    mut_label = get_mut_label(mut_rate)

    # Define paths
    cluster_dir = f'{RESULTS_DIR}/clustering/seq_{seq_idx}/{mut_label}'
    linkage_path = os.path.join(cluster_dir, 'hierarchical_linkage_ward.npy')
    labels_path = os.path.join(cluster_dir, 'cluster_labels.npy')

    msm_dir = f'{RESULTS_DIR}/msms/seq_{seq_idx}/{mut_label}'
    msm_path = os.path.join(msm_dir, 'msm.csv')

    # Skip if MSM already exists
    if os.path.exists(msm_path):
        print(f"Skipping seq_{seq_idx}/{mut_label} - MSM already exists")
        return None

    # Check if cluster metadata exists
    if not os.path.exists(linkage_path) or not os.path.exists(labels_path):
        print(f"Skipping seq_{seq_idx}/{mut_label} - cluster metadata not found")
        return None

    # Load cluster metadata
    linkage = np.load(linkage_path)
    labels = np.load(labels_path)

    # Load attributions
    try:
        attributions = load_attributions_mut(seq_idx, mut_rate)
    except:
        print(f"Skipping seq_{seq_idx}/{mut_label} - attributions not found")
        return None

    # Load sequences and predictions
    seqs, preds, orig_idx, library_index = load_library_25k(seq_idx, mut_rate)

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
        mut_rate=mut_rate
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


def compute_and_save_variance_summary(seq_idx, mut_rate):
    """Load MSM and compute variance of entropy across clusters for each position."""
    mut_label = get_mut_label(mut_rate)

    msm_dir = f'{RESULTS_DIR}/msms/seq_{seq_idx}/{mut_label}'
    msm_path = os.path.join(msm_dir, 'msm.csv')
    variance_path = os.path.join(msm_dir, 'variance_summary.csv')

    # Skip if variance summary already exists
    if os.path.exists(variance_path):
        print(f"Skipping seq_{seq_idx}/{mut_label} - variance summary already exists")
        return None

    # Check if MSM exists
    if not os.path.exists(msm_path):
        print(f"Skipping seq_{seq_idx}/{mut_label} - MSM not found")
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


def plot_combined_variance_summary(seq_idx):
    """Plot all variance summaries stacked vertically for all mutation rates."""
    msm_base_dir = f'{RESULTS_DIR}/msms/seq_{seq_idx}'
    plot_path = os.path.join(msm_base_dir, 'variance_summary_combined.png')

    # Skip if combined plot already exists
    if os.path.exists(plot_path):
        print(f"Skipping combined variance plot for seq_{seq_idx} - already exists")
        return

    # Load all variance summaries (sorted by mutation rate: high to low)
    variance_data = {}
    for mut_rate in mutation_rates:
        mut_label = get_mut_label(mut_rate)
        variance_path = os.path.join(msm_base_dir, mut_label, 'variance_summary.csv')
        if os.path.exists(variance_path):
            variance_data[mut_label] = pd.read_csv(variance_path)

    if len(variance_data) == 0:
        print(f"Skipping combined variance plot for seq_{seq_idx} - no variance summaries found")
        return

    # Create figure with subplots
    n_plots = len(variance_data)
    fig, axes = plt.subplots(n_plots, 1, figsize=(12, 2.5 * n_plots), sharex=True)

    # Handle case of single plot
    if n_plots == 1:
        axes = [axes]

    # Get global y-axis limits for consistent scaling
    all_variances = [df['Variance'].values for df in variance_data.values()]
    y_max = max(v.max() for v in all_variances)
    y_min = min(v.min() for v in all_variances)

    # Plot each variance summary
    for ax, (mut_label, df) in zip(axes, variance_data.items()):
        ax.plot(df['Position'], df['Variance'], linewidth=0.8)
        ax.set_ylabel(f'{mut_label}')
        ax.set_ylim(y_min, y_max * 1.05)
        ax.grid(True, alpha=0.3)

    # Set common labels
    axes[-1].set_xlabel('Position')
    fig.suptitle(f'Dev seq_{seq_idx} - Entropy Variance by Position', fontsize=12)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved combined variance plot to {plot_path}")


def check_variance_combined_plots_complete(seq_indices):
    """Check if all combined variance summary plots exist for all sequences."""
    for seq_idx in seq_indices:
        plot_path = f'{RESULTS_DIR}/msms/seq_{seq_idx}/variance_summary_combined.png'
        if not os.path.exists(plot_path):
            return False
    return True


def compute_mut_rate_correlations(seq_idx, reference_mut_rate):
    """Compute Pearson and Spearman correlations with 10% reference on variance summaries."""
    reference_mut_rate = reference_mut_rate
    ref_label = get_mut_label(reference_mut_rate)
    output_dir = f'{RESULTS_DIR}/correlations/seq_{seq_idx}'
    corr_path = os.path.join(output_dir, f'correlation_with_{ref_label}.csv')

    # Skip if correlations already exist
    if os.path.exists(corr_path):
        print(f"Skipping seq_{seq_idx} - correlations already exist")
        return None

    # Load reference variance summary
    variance_ref_path = f'{RESULTS_DIR}/msms/seq_{seq_idx}/{ref_label}/variance_summary.csv'
    if not os.path.exists(variance_ref_path):
        print(f"Skipping seq_{seq_idx} - {ref_label} variance summary not found")
        return None

    variance_ref = pd.read_csv(variance_ref_path)['Variance'].values

    # Compute correlations for each mutation rate
    results = []
    for mut_rate in mutation_rates:
        mut_label = get_mut_label(mut_rate)
        variance_path = f'{RESULTS_DIR}/msms/seq_{seq_idx}/{mut_label}/variance_summary.csv'
        if not os.path.exists(variance_path):
            continue

        variance_values = pd.read_csv(variance_path)['Variance'].values
        pearson_corr, _ = pearsonr(variance_ref, variance_values)
        spearman_corr, _ = spearmanr(variance_ref, variance_values)

        results.append({
            'Mut_Rate': mut_label,
            'Mut_Rate_Numeric': mut_rate,
            'Pearson': pearson_corr,
            'Spearman': spearman_corr
        })

    if len(results) < 2:
        print(f"Skipping seq_{seq_idx} - need at least 2 mutation rates")
        return None

    corr_df = pd.DataFrame(results)
    os.makedirs(output_dir, exist_ok=True)
    corr_df.to_csv(corr_path, index=False)
    print(f"Saved correlations to {corr_path}")
    return corr_df


def plot_correlation_for_seq(seq_idx, reference_mut_rate):
    """Plot correlation vs distance from 10% for a single sequence."""
    reference_mut_rate = reference_mut_rate
    ref_label = get_mut_label(reference_mut_rate)
    output_dir = f'{RESULTS_DIR}/correlations/seq_{seq_idx}'
    corr_path = os.path.join(output_dir, f'correlation_with_{ref_label}.csv')
    plot_path = os.path.join(output_dir, f'correlation_with_{ref_label}.png')

    if not os.path.exists(corr_path):
        print(f"Skipping plot for seq_{seq_idx} - correlation file not found")
        return None

    corr_df = pd.read_csv(corr_path)

    # Calculate distance from 10% (centered at 0)
    corr_df['Distance_from_10%'] = (corr_df['Mut_Rate_Numeric'] - reference_mut_rate)*100

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot Pearson correlation
    ax.plot(corr_df['Distance_from_10%'], corr_df['Pearson'], 'o-',
            color='steelblue', markersize=8, linewidth=2, label='Pearson')

    # Add vertical line at 0 (the 10% reference point)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.7, label='10% reference')

    # Add horizontal line at correlation = 1
    ax.axhline(y=1, color='lightgray', linestyle=':', alpha=0.7)

    ax.set_xlabel('Distance from 10% Mutation Rate', fontsize=12)
    ax.set_ylabel('Pearson Correlation', fontsize=12)
    ax.set_title(f'Correlation with 10% Reference - Seq {seq_idx}', fontsize=14)
    ax.legend(loc='lower right')
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved correlation plot to {plot_path}")
    return corr_df


def plot_summary_correlations(seq_indices, reference_mut_rate):
    """Plot summary of correlations for all sequences on a single figure."""
    reference_mut_rate = reference_mut_rate
    ref_label = get_mut_label(reference_mut_rate)

    # Create output directory
    output_dir = f'{RESULTS_DIR}/results_final'
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, 'correlation_summary.png')

    # Collect data from all sequences
    all_data = []
    valid_seq_indices = []

    for seq_idx in seq_indices:
        corr_path = f'{RESULTS_DIR}/correlations/seq_{seq_idx}/correlation_with_{ref_label}.csv'
        if os.path.exists(corr_path):
            corr_df = pd.read_csv(corr_path)
            corr_df['seq_idx'] = seq_idx
            corr_df['Distance_from_10%'] = (corr_df['Mut_Rate_Numeric'] - reference_mut_rate)*100
            all_data.append(corr_df)
            valid_seq_indices.append(seq_idx)

    if not all_data:
        print("No correlation data found for any sequence")
        return None

    # Create colormap for sequences
    n_seqs = len(valid_seq_indices)
    colors = cm.viridis(np.linspace(0, 1, n_seqs))

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 7))

    for i, (seq_idx, corr_df) in enumerate(zip(valid_seq_indices, all_data)):
        ax.plot(corr_df['Distance_from_10%'], corr_df['Pearson'], 'o-',
                color=colors[i], markersize=6, linewidth=1.5, alpha=0.8,
                label=f'Seq {seq_idx}')

    # Add vertical line at 0 (the 10% reference point)
    ax.axvline(x=0, color='black', linestyle='--', linewidth=1.5, alpha=0.7, label='10% reference')

    # Add horizontal line at correlation = 1
    ax.axhline(y=1, color='gray', linestyle=':', alpha=0.5)

    ax.set_xlabel('Distance from 10% Mutation Rate', fontsize=12)
    ax.set_ylabel('Pearson Correlation', fontsize=12)
    ax.set_title('Correlation with 10% Reference', fontsize=14)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8, ncol=2)
    

    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved summary correlation plot to {plot_path}")
    return plot_path


# =============================================================================
# Main Execution
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Mutation Rate Sweep")
    print("=" * 60)


    reference_mut_rate = 0.10

    # Check GPU availability
    check_gpu()

    # 1. Load Dev_20 library (20 sequences, 3 removed)
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

    # 3. Generate 25K libraries and compute DeepSHAP attributions for each mutation rate
    libraries_complete = check_libraries_complete(seq_indices)
    attributions_complete = check_attributions_complete(seq_indices)

    if libraries_complete and attributions_complete:
        print("\n" + "=" * 60)
        print("SKIPPING: All libraries and attributions already exist")
        print("=" * 60)
    else:
        for mut_rate in mutation_rates:
            mut_label = get_mut_label(mut_rate)
            print("\n" + "=" * 60)
            print(f"Processing mutation rate: {mut_label}")
            print("=" * 60)

            # Generate libraries for all sequences
            for i, (x_seq, idx) in enumerate(zip(x_seqs, seq_indices)):
                output_dir = f'{MUTAGENESIS_LIBRARY_DIR}/Dev/seq_{idx}/{mut_label}'
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
                    mut_rate=mut_rate,
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
                print(f"[{i+1}/{len(x_seqs)}] Created seq_{idx}/25K.h5 with Mutation Rate {mut_label}")

            # Compute DeepSHAP attributions for each library
            for idx in seq_indices:
                output_dir = f'{ATTRIBUTION_DIR}/Dev/seq_{idx}/{mut_label}'
                output_file = f'{output_dir}/25K.h5'

                if os.path.exists(output_file):
                    print(f"Skipping seq_{idx} - attributions already exist")
                    continue

                # Load library
                x_mut, y_mut, original_idx, library_index = load_library_25k(idx, mut_rate)

                # Get DeepSHAP attributions
                checkpoint_path = f'{output_dir}/checkpoint.h5'
                attributions = seam_deepshap(x_mut, task_index, checkpoint_path=checkpoint_path)

                # Save attributions
                save_attributions(output_file, attributions, original_idx)

                # Remove checkpoint after successful completion
                if os.path.exists(checkpoint_path):
                    os.remove(checkpoint_path)

                print(f"Saved attributions for seq_{idx} with Mutation Rate {mut_label}")

    # 4. Clustering and MSM generation for all sequences and mutation rates
    clustering_complete = check_clustering_complete(seq_indices)
    msms_complete = check_msms_complete(seq_indices)

    if clustering_complete and msms_complete:
        print("\n" + "=" * 60)
        print("SKIPPING: All clustering and MSM results already exist")
        print("=" * 60)
    else:
        for seq_idx in seq_indices:
            print("\n" + "=" * 60)
            print(f"Clustering and MSM generation for seq_{seq_idx}")
            print("=" * 60)

            for mut_rate in mutation_rates:
                mut_label = get_mut_label(mut_rate)
                print(f"\n--- {mut_label} ---")

                # Load attributions
                try:
                    attrs = load_attributions_mut(seq_idx, mut_rate)
                except Exception as e:
                    print(f"Skipping {mut_label} - no attributions found: {e}")
                    continue

                # Cluster
                linkage, labels = cluster_and_save(attrs, seq_idx, mut_rate, n_clusters=cluster_number)

                # Generate MSM
                generate_and_save_msm(seq_idx, mut_rate, n_clusters=cluster_number, gpu=True)

                # Compute variance summary
                compute_and_save_variance_summary(seq_idx, mut_rate)

    # 5. Generate combined variance summary plots for each sequence
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

    # 6. Compute correlations with 10% reference
    if check_correlations_complete(seq_indices):
        print("\n" + "=" * 60)
        print("SKIPPING: All correlation files already exist")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("Computing correlations with 10% reference")
        print("=" * 60)

        for seq_idx in seq_indices:
            print(f"\nComputing correlations for seq_{seq_idx}")
            compute_mut_rate_correlations(seq_idx, reference_mut_rate=reference_mut_rate)

    # 7. Generate individual correlation plots for each sequence
    if check_correlation_plots_complete(seq_indices):
        print("\n" + "=" * 60)
        print("SKIPPING: All individual correlation plots already exist")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("Generating correlation plots for each sequence")
        print("=" * 60)

        for seq_idx in seq_indices:
            plot_correlation_for_seq(seq_idx, reference_mut_rate=reference_mut_rate)

    # 8. Generate summary plot with all sequences
    if check_summary_plot_complete():
        print("\n" + "=" * 60)
        print("SKIPPING: Summary correlation plot already exists")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("Generating summary correlation plot")
        print("=" * 60)

        plot_summary_correlations(seq_indices, reference_mut_rate=reference_mut_rate)

    print("\n" + "=" * 60)
    print("Mutation rate sweep complete!")
    print("=" * 60)
