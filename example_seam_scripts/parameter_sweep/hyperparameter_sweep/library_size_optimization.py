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
from seam import Compiler, Attributer, Clusterer, MetaExplainer, Identifier
from seam.logomaker_batch.batch_logo import BatchLogo

## This script will sweep through the library sizes with baseline hyperparameters

library_size_sweep = [100000, 75000, 50000, 25000, 10000, 5000, 1000]

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

if len(dev_loci)==6 and len(hk_loci) ==8:
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
def seam_deepshap(x_mut, task_index):
    x_ref = x_mut
    import time
    import tensorflow as tf
    from keras.models import model_from_json
    import numpy as np
    import random

    # Configuration
    attribution_method = 'deepshap'  # or 'gradientshap', 'integratedgradients', etc.
    task_index = task_index  # 0 for Dev, 1 for Hk
    gpu = 0  # GPU device number
    save_data = True
    save_path = './attributions'  # Where to save results
    os.makedirs(save_path, exist_ok=True)

    # Model paths
    keras_model_json = 'models/deepstarr/deepstarr.model.json'
    keras_model_weights = 'models/deepstarr/deepstarr.model.h5'

    if attribution_method == 'deepshap':
        try:
            # Disable eager execution first
            tf.compat.v1.disable_eager_execution()
            tf.compat.v1.disable_v2_behavior()
            print("TensorFlow eager execution disabled for DeepSHAP compatibility")
            
            # Import SHAP to configure handlers
            try:
                import shap
            except ImportError:
                print("ERROR: SHAP package is not installed.")
                print("To install SHAP for DeepSHAP attribution, run:")
                print("pip install kundajelab-shap==1")
                raise ImportError("SHAP package required for DeepSHAP attribution")
            
            # Handle AddV2 operation (element-wise addition) as a linear operation
            shap.explainers.deep.deep_tf.op_handlers["AddV2"] = shap.explainers.deep.deep_tf.passthrough

            # Load the model after eager execution is disabled
            keras_model = model_from_json(open(keras_model_json).read(), custom_objects={'Functional': tf.keras.Model})
            np.random.seed(113)
            random.seed(0)
            keras_model.load_weights(keras_model_weights)
            model = keras_model
            
            # Rebuild model to ensure proper graph construction
            _ = model(tf.keras.Input(shape=model.input_shape[1:]))
            
        except ImportError:
            raise
        except Exception as e:
            print(f"Warning: Could not setup TensorFlow for DeepSHAP. Error: {e}")
            print("DeepSHAP may not work properly.")
        
        # Create attributer for DeepSHAP
        def deepstarr_compress(x):
            """DeepSTARR compression function for DeepSHAP."""
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
        attributions = attributer.compute(
            x_ref=x_ref,
            x=x_mut,
            save_window=None,
            batch_size=16,
            gpu=gpu,
        )
        t2 = time.time() - t1
        print(f'Attribution time: {t2/60:.2f} minutes')
    else:
        # Use unified Attributer for other methods
        attributer = Attributer(
            model,
            method=attribution_method,
            task_index=task_index,
            compress_fun=lambda x: x,
            pred_fun=model.predict_on_batch,
        )

        attributer.show_params(attribution_method)

        t1 = time.time()
        attributions = attributer.compute(
            x_ref=x_ref,
            x=x_mut,
            save_window=None,
            batch_size=256,
            gpu=gpu
        )
        t2 = time.time() - t1
        print(f'Attribution time: {t2/60:.2f} minutes')


## START SEAM

tasks = [0,1]

for task in tasks:

    # Squid library creation
    








