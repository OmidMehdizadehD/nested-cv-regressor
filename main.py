import argparse
import os
import pandas as pd
import numpy as np
import warnings
from scipy.stats import pearsonr
from sklearn.model_selection import KFold, cross_validate, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

# Using imblearn pipeline to allow dropping samples (outliers) during fit without leakage
from imblearn.pipeline import Pipeline
from imblearn.FunctionSampler import FunctionSampler

from src.data_loader import load_and_split_dataset
from src.preprocessing import get_scaler, osvm_resampler
from src.feature_selection import get_feature_selector
from src.models import MODELS
from src.evaluation import get_bootstrap_ci
from src.plotting import plot_actual_vs_predicted

warnings.filterwarnings('ignore')

def parse_args():
    parser = argparse.ArgumentParser(description="Biomedical Regression Pipeline")
    parser.add_argument('--data', type=str, required=True, help="Path to the dataset (.csv or .xlsx)")
    parser.add_argument('--target', type=str, default='RPImax', help="Target column name")
    parser.add_argument('--scaling', type=str, default='robust', choices=['standard', 'robust', 'minmax', 'maxabs', 'power', 'quantile'])
    parser.add_argument('--selection', type=str, default='rfecv_rf', choices=['mutual_info', 'rfe_rf', 'rfecv_rf', 'lasso'])
    parser.add_argument('--outdir', type=str, default='results', help="Directory to save outputs")
    
    # New Arguments for Outlier Detection
    parser.add_argument('--no-outliers', action='store_true', help="Disable OneClassSVM outlier removal during training.")
    parser.add_argument('--contamination', type=float, default=0.01, help="Expected proportion of outliers for OneClassSVM (default: 0.01).")
    
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    print("=====================================================")
    print(" Patient-Level ML Evaluation Framework")
    print("=====================================================\n")

    train_data, test_data, feature_cols = load_and_split_dataset(args.data, target_col=args.target)
    
    train_df_clean = train_data.dropna(subset=[args.target])
    test_df_clean = test_data.dropna(subset=[args.target])
    
    X_train, y_train = train_df_clean[feature_cols], train_df_clean[args.target]
    X_test, y_test = test_df_clean[feature_cols], test_df_clean[args.target]

    all_results = []
    
    for model_name, (estimator, param_grid) in MODELS.items():
        print(f"Evaluating Model: {model_name}...")
        
        # Dynamically construct pipeline steps based on user arguments
        pipeline_steps = []
        
        if not args.no_outliers:
            pipeline_steps.append(('outliers', FunctionSampler(func=osvm_resampler, kw_args={'contamination': args.contamination})))
            print(f" -> Outlier detection ENABLED (contamination: {args.contamination})")
        else:
            print(" -> Outlier detection DISABLED")
            
        pipeline_steps.extend([
            ('scaler', get_scaler(args.scaling)),
            ('selector', get_feature_selector(args.selection)),
            ('model', estimator)
        ])

        # Pipeline includes conditionally added outlier removal
        pipeline = Pipeline(pipeline_steps)

        search = RandomizedSearchCV(
            pipeline, param_distributions=param_grid, n_iter=15,
            cv=KFold(n_splits=3, shuffle=True, random_state=42),
            scoring='neg_mean_absolute_error', random_state=42, n_jobs=-1
        )

        # A. Nested Cross-Validation
        print(f" -> Running Nested CV for {model_name}...")
        outer_kf = KFold(n_splits=10, shuffle=True, random_state=42)
        cv_results = cross_validate(search, X_train, y_train, cv=outer_kf, scoring=['neg_mean_absolute_error', 'r2'], n_jobs=-1)
        
        cv_mae_mean, cv_mae_std = np.mean(-cv_results['test_neg_mean_absolute_error']), np.std(-cv_results['test_neg_mean_absolute_error'])
        cv_r2_mean, cv_r2_std = np.mean(cv_results['test_r2']), np.std(cv_results['test_r2'])

        # B. Final Model Training & Export Features
        search.fit(X_train, y_train)
        best_model = search.best_estimator_
        
        # Save selected features
        selector = best_model.named_steps['selector']
        try:
            mask = selector.get_support()
            selected_features = np.array(feature_cols)[mask]
            pd.DataFrame({'Selected_Features': selected_features}).to_csv(f"{args.outdir}/{model_name}_features.csv", index=False)
        except AttributeError:
            print(f"   (Feature export not supported for {args.selection})")

        # C. Test Set Evaluation
        y_pred = best_model.predict(X_test)
        test_mae = mean_absolute_error(y_test, y_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        test_r2 = r2_score(y_test, y_pred)
        test_r, _ = pearsonr(y_test, y_pred)
        mae_ci, r2_ci = get_bootstrap_ci(y_test, y_pred)

        all_results.append({
            'Model': model_name,
            'CV MAE': f"{cv_mae_mean:.3f} ± {cv_mae_std:.3f}",
            'CV R²': f"{cv_r2_mean:.3f} ± {cv_r2_std:.3f}",
            'Test MAE': f"{test_mae:.3f}",
            'Test R²': f"{test_r2:.3f}",
            'MAE 95% CI': mae_ci,
            'R² 95% CI': r2_ci
        })

        metrics_dict = {'MAE': test_mae, 'RMSE': test_rmse, 'R2': test_r2, 'R': test_r}
        plot_actual_vs_predicted(y_test, y_pred, metrics_dict, args.target, model_name, args.outdir)

    print("\n=====================================================")
    results_df = pd.DataFrame(all_results)
    print(results_df.to_markdown(index=False))
    results_df.to_csv(f"{args.outdir}/summary_metrics.csv", index=False)
    print(f"\nAll results, plots, and selected features saved to '{args.outdir}' directory.")

if __name__ == "__main__":
    main()
