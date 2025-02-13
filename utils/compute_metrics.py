import numpy as np
import pandas as pd
from typing import Tuple, Dict

def calculate_se_sp(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    """Calculate Sensitivity and Specificity for binary predictions"""
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))

    all_pos = np.sum(y_true == 1)
    all_neg = np.sum(y_true == 0)

    sensitivity = tp / all_pos if all_pos > 0 else 0
    specificity = tn / all_neg if all_neg > 0 else 0

    return sensitivity, specificity

def calculate_binary_kl_distances(sensitivity: float, specificity: float) -> Dict[str, float]:
    """Calculate KL distances and related metrics using sensitivity and specificity"""
    # Avoid log(0)
    eps = 1e-10
    Se = np.clip(sensitivity, eps, 1-eps)
    Sp = np.clip(specificity, eps, 1-eps)

    # Calculate likelihood ratios
    LR_plus = Se / (1 - Sp)  # LR for positive result
    LR_minus = (1 - Se) / Sp  # LR for negative result

    # Calculate D(f||g) using the paper's formula
    D_f_g = (1 - Se) * np.log((1 - Se) / Sp) + Se * np.log(Se / (1 - Sp))

    # Calculate D(g||f) using the paper's formula
    D_g_f = Sp * np.log(Sp / (1 - Se)) + (1 - Sp) * np.log((1 - Sp) / Se)

    # Calculate rule in/out potentials
    P_in = np.exp(D_f_g)
    P_out = np.exp(D_g_f)

    return {
        'sensitivity': Se,
        'specificity': Sp,
        'LR_plus': LR_plus,
        'LR_minus': LR_minus,
        'D_f_g': D_f_g,
        'D_g_f': D_g_f,
        'P_in': P_in,
        'P_out': P_out
    }

def analyze_thresholds(probabilities: np.ndarray,
                      true_labels: np.ndarray,
                      n_thresholds: int = 100) -> pd.DataFrame:
    """
    Analyze multiple thresholds and calculate metrics for each.
    Similar to Table 4 in the paper.

    Args:
        probabilities: Array of predicted probabilities
        true_labels: Array of true binary labels (0 or 1)
        n_thresholds: Number of thresholds to evaluate

    Returns:
        DataFrame with metrics for each threshold
    """
    # Generate thresholds using percentiles of the probability distribution
    percentiles = np.linspace(0, 100, n_thresholds)
    thresholds = np.percentile(probabilities, percentiles)

    results = []
    for threshold in thresholds:
        # Convert probabilities to binary predictions
        binary_predictions = (probabilities >= threshold).astype(int)

        # Calculate sensitivity and specificity
        sensitivity, specificity = calculate_se_sp(true_labels, binary_predictions)

        # Calculate all metrics
        metrics = calculate_binary_kl_distances(sensitivity, specificity)
        metrics['threshold'] = threshold

        results.append(metrics)

    # Convert to DataFrame
    df_results = pd.DataFrame(results)

    # Find optimal thresholds
    best_rule_in = df_results.loc[df_results['P_in'].idxmax()]
    best_rule_out = df_results.loc[df_results['P_out'].idxmax()]

    print("\nOptimal threshold for rule-in potential:")
    print(f"Threshold = {best_rule_in['threshold']:.3f}")
    print(f"Pin = {best_rule_in['P_in']:.3f}")
    print(f"Sensitivity = {best_rule_in['sensitivity']:.3f}")
    print(f"Specificity = {best_rule_in['specificity']:.3f}")

    print("\nOptimal threshold for rule-out potential:")
    print(f"Threshold = {best_rule_out['threshold']:.3f}")
    print(f"Pout = {best_rule_out['P_out']:.3f}")
    print(f"Sensitivity = {best_rule_out['sensitivity']:.3f}")
    print(f"Specificity = {best_rule_out['specificity']:.3f}")

    return df_results

def calculate_actual_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate metrics directly from the data (actual/empirical values)
    """
    # Counts for each category
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    # Derived counts
    actual_positives = tp + fn
    actual_negatives = tn + fp
    predicted_positives = tp + fp
    predicted_negatives = tn + fn
    total = len(y_true)

    # Calculate metrics
    metrics = {
        'actual_sensitivity': tp / actual_positives if actual_positives > 0 else 0,
        'actual_specificity': tn / actual_negatives if actual_negatives > 0 else 0,
        'actual_ppv': tp / predicted_positives if predicted_positives > 0 else 0,
        'actual_npv': tn / predicted_negatives if predicted_negatives > 0 else 0,
        'actual_prevalence': actual_positives / total,
        'confusion_matrix': {
            'tp': int(tp),
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn)
        }
    }

    return metrics

def calculate_theoretical_metrics(sensitivity: float,
                                specificity: float,
                                prevalence: float) -> Dict[str, float]:
    """
    Calculate theoretical metrics using Bayes' theorem
    """
    # Calculate theoretical PPV
    theoretical_ppv = (sensitivity * prevalence) / (
        sensitivity * prevalence + (1 - specificity) * (1 - prevalence)
    )

    # Calculate theoretical NPV
    theoretical_npv = (specificity * (1 - prevalence)) / (
        (1 - sensitivity) * prevalence + specificity * (1 - prevalence)
    )

    return {
        'theoretical_ppv': theoretical_ppv,
        'theoretical_npv': theoretical_npv
    }

def analyze_predictions(probabilities: np.ndarray,
                       true_labels: np.ndarray,
                       threshold: float = 0.5) -> pd.DataFrame:
    """
    Analyze predictions and return both theoretical and actual metrics
    """
    # Get binary predictions
    predictions = (probabilities >= threshold).astype(int)

    # Calculate actual metrics
    actual_metrics = calculate_actual_metrics(true_labels, predictions)

    # Calculate theoretical metrics using actual Se, Sp, and prevalence
    theoretical_metrics = calculate_theoretical_metrics(
        sensitivity=actual_metrics['actual_sensitivity'],
        specificity=actual_metrics['actual_specificity'],
        prevalence=actual_metrics['actual_prevalence']
    )

    # Combine metrics
    all_metrics = {**actual_metrics, **theoretical_metrics}

    # Create a comparison DataFrame
    comparison = pd.DataFrame({
        'Metric': ['PPV', 'NPV'],
        'Actual': [all_metrics['actual_ppv'], all_metrics['actual_npv']],
        'Theoretical': [all_metrics['theoretical_ppv'], all_metrics['theoretical_npv']],
        'Difference': [
            all_metrics['actual_ppv'] - all_metrics['theoretical_ppv'],
            all_metrics['actual_npv'] - all_metrics['theoretical_npv']
        ]
    })

    # Print detailed results
    print("\nConfusion Matrix:")
    print(f"True Positives: {all_metrics['confusion_matrix']['tp']}")
    print(f"True Negatives: {all_metrics['confusion_matrix']['tn']}")
    print(f"False Positives: {all_metrics['confusion_matrix']['fp']}")
    print(f"False Negatives: {all_metrics['confusion_matrix']['fn']}")

    print(f"\nPrevalence: {all_metrics['actual_prevalence']:.3f}")
    print(f"Sensitivity: {all_metrics['actual_sensitivity']:.3f}")
    print(f"Specificity: {all_metrics['actual_specificity']:.3f}")

    print("\nComparison of Actual vs Theoretical Values:")
    print(comparison.round(3))

    return all_metrics 

# Examples:
#results_df = analyze_thresholds(
#    df['class_1'].values,
#    df['True Label'].values,
#    n_thresholds=350
#)
#results_df[results_df['threshold'] >= 0.49]


#analyze_predictions(
#    df['aki_positive'].values,
#    df['True Label'].values,
#    threshold=0.50
#)


