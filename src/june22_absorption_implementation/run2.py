import torch
import numpy as np
from sae_lens.saes import SAE
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import csv
import pandas as pd
import argparse
from transformer_lens import HookedTransformer
from sae_bench.evals.absorption.eval_config import AbsorptionEvalConfig
from sae_bench.evals.absorption.feature_absorption import run_feature_absortion_experiment
from sae_bench.evals.absorption.k_sparse_probing import run_k_sparse_probing_experiment
from sae_bench.sae_bench_utils import activation_collection, general_utils
import json
from collections import Counter
from pathlib import Path

device="cuda" if torch.cuda.is_available() else "cpu"

sae, cfg_dict, sparsity = SAE.from_pretrained_with_cfg_and_sparsity(
    # test for layer 12, relu, k=20
    release="sae_bench_gemma-2-2b_vanilla_width-2pow14_date-1109",
    sae_id="blocks.12.hook_resid_post__trainer_0",
    device=device
    )

model = HookedTransformer.from_pretrained_no_processing(
    default_prepend_bos=True, 
    model_name="gemma-2-2b"
    )

print(cfg_dict)
