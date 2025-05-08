import logging
from lib import evaluation

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Evaluate model ensemble
paths = [
    'runs/f30k_scan_matrix_i2t_dp_sim256/results_f30k.npy',
    'runs/f30k_scan_matrix_i2t_dp_sim256/results_f30k.npy',
]

# evaluation.eval_ensemble(results_paths=paths, fold5=True)
evaluation.eval_ensemble(results_paths=paths, fold5=False)
