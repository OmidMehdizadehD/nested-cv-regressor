import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.utils import resample

def get_bootstrap_ci(y_true, y_pred, n_bootstrap=$1000$):
    boot_mae, boot_r2 = [], []
    y_test_arr, y_pred_arr = np.array(y_true), np.array(y_pred)
    indices = np.arange(len(y_test_arr))

    for i in range(n_bootstrap):
        boot_idx = resample(indices, random_state=i)
        y_test_boot = y_test_arr[boot_idx]
        y_pred_boot = y_pred_arr[boot_idx]
        
        boot_mae.append(mean_absolute_error(y_test_boot, y_pred_boot))
        boot_r2.append(r2_score(y_test_boot, y_pred_boot))

    mae_ci = f"$[{np.percentile(boot_mae, 2.5):.3f}, {np.percentile(boot_mae, 97.5):.3f}]$"
    r2_ci = f"$[{np.percentile(boot_r2, 2.5):.3f}, {np.percentile(boot_r2, 97.5):.3f}]$"
    return mae_ci, r2_ci
