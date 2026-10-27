"""Utility functions: feature selection, patient-level splitting and pair generation."""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, chi2_contingency
from sklearn.metrics import mutual_info_score
import warnings
import seaborn as sns
import itertools
import random
import os
warnings.filterwarnings('ignore')

def clean_features_by_type(df, target_col='OS_YEARS',
                          mi_threshold_numeric=0.2,
                          mi_threshold_categorical=0.3,
                          corr_threshold=0.1,
                          chi2_threshold=0.05,
                          n_bins=10):
    """
    Clean features, handling numeric and categorical variables separately.

    For numeric variables:
      - Uses Pearson correlation for association with the target
      - Uses MI to detect redundancy between numeric variables

    For categorical variables:
      - Uses Chi² for association with the target (discretized target)
      - Uses MI to detect redundancy between categorical variables

    Redundancy across different variable types is not considered.

    Parameters:
    -----------
    df : DataFrame
    target_col : str
        Continuous target variable (OS_YEARS)
    mi_threshold_numeric : float
        MI threshold for redundancy between numeric variables
    mi_threshold_categorical : float
        MI threshold for redundancy between categorical variables
    corr_threshold : float
        Minimum correlation required to keep a numeric variable
    chi2_threshold : float
        Maximum p-value allowed to keep a categorical variable
    n_bins : int
        Number of bins used to discretize continuous variables for MI

    Returns:
    --------
    tuple: (df_clean, numeric_selected, categorical_selected, results_info)
    """

    # 1. Data preparation
    features = [col for col in df.columns if col != target_col]

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found")

    # 2. Split variables by type
    numeric_features = []
    categorical_features = []

    for col in features:
        if pd.api.types.is_numeric_dtype(df[col]):
            # Numeric variable with more than 10 unique values = numeric
            if df[col].nunique() > 10:
                numeric_features.append(col)
            else:
                # Few unique values = treat as categorical
                categorical_features.append(col)
        else:
            # Non-numeric variable = categorical
            categorical_features.append(col)

    print("="*60)
    print("VARIABLE CLASSIFICATION")
    print("="*60)
    print(f"Numeric variables ({len(numeric_features)}):")
    print(f"  {', '.join(numeric_features[:10])}{'...' if len(numeric_features) > 10 else ''}")
    print(f"\nCategorical variables ({len(categorical_features)}):")
    print(f"  {', '.join(categorical_features[:10])}{'...' if len(categorical_features) > 10 else ''}")

    # 3. Discretize the target for Chi² (if needed)
    df_processed = df.copy()
    if df_processed[target_col].nunique() > 10:
        # Discretize into quartiles
        df_processed[f'{target_col}_binned'] = pd.qcut(
            df_processed[target_col],
            q=4,
            labels=['Q1', 'Q2', 'Q3', 'Q4']
        )
        target_binned = f'{target_col}_binned'
    else:
        target_binned = target_col

    # 4. Functions to compute association with the target

    # Numeric variables: Pearson correlation
    def compute_numeric_association(col):
        """Compute the absolute correlation with the target"""
        valid_data = df[[col, target_col]].dropna()
        if len(valid_data) < 10:
            return 0.0
        try:
            corr, p_value = pearsonr(valid_data[col], valid_data[target_col])
            return abs(corr)
        except:
            return 0.0

    # Categorical variables: Chi² test
    def compute_categorical_association(col):
        """Compute the Chi² p-value against the discretized target"""
        # Build contingency table
        contingency = pd.crosstab(df_processed[col], df_processed[target_binned])

        # Minimum conditions for the test
        if contingency.shape[0] < 2 or contingency.shape[1] < 2:
            return 1.0  # Maximum p-value = no association

        if contingency.sum().sum() < 20:  # Too few observations
            return 1.0

        # Run the Chi² test
        try:
            chi2, p, dof, expected = chi2_contingency(contingency)
            return p  # Return the p-value (smaller = better)
        except:
            return 1.0

    # 5. Compute associations
    print("\n" + "="*60)
    print("COMPUTING ASSOCIATIONS WITH THE TARGET")
    print("="*60)

    numeric_associations = {}
    for col in numeric_features:
        numeric_associations[col] = compute_numeric_association(col)

    categorical_associations = {}
    for col in categorical_features:
        categorical_associations[col] = compute_categorical_association(col)

    # 6. Prepare data for MI computation (discretization)
    df_for_mi = df.copy()

    # Discretize numeric variables
    for col in numeric_features:
        try:
            df_for_mi[col] = pd.qcut(df_for_mi[col], q=n_bins, labels=False, duplicates='drop')
        except:
            df_for_mi[col] = pd.cut(df_for_mi[col], bins=n_bins, labels=False)

    # Encode categorical variables
    for col in categorical_features:
        if not pd.api.types.is_numeric_dtype(df_for_mi[col]):
            df_for_mi[col] = pd.factorize(df_for_mi[col])[0]

    # Fill NaNs
    df_for_mi = df_for_mi.fillna(-1)

    # 7. Function to clean a group of variables
    def clean_feature_group(feature_list, is_numeric=True):
        """Clean a group of variables (numeric or categorical)"""
        if not feature_list:
            return [], {}

        # MI matrix for this group
        n_features = len(feature_list)
        mi_matrix = pd.DataFrame(0.0, index=feature_list, columns=feature_list)

        # Compute MI for each pair
        for i in range(n_features):
            for j in range(i+1, n_features):
                col1, col2 = feature_list[i], feature_list[j]
                mi = mutual_info_score(df_for_mi[col1], df_for_mi[col2])
                mi_matrix.loc[col1, col2] = mi
                mi_matrix.loc[col2, col1] = mi

        # MI threshold depending on the type
        mi_threshold = mi_threshold_numeric if is_numeric else mi_threshold_categorical

        # Identify redundant pairs
        to_keep = set(feature_list)
        removed = {}

        high_mi_pairs = []
        for i in range(n_features):
            for j in range(i+1, n_features):
                col1, col2 = feature_list[i], feature_list[j]
                mi_value = mi_matrix.loc[col1, col2]
                if mi_value > mi_threshold:
                    high_mi_pairs.append((mi_value, col1, col2))

        # Sort by decreasing MI
        high_mi_pairs.sort(reverse=True, key=lambda x: x[0])

        # Process each redundant pair
        for mi_value, col1, col2 in high_mi_pairs:
            if col1 in to_keep and col2 in to_keep:
                # Compare associations with the target
                if is_numeric:
                    assoc1 = numeric_associations[col1]
                    assoc2 = numeric_associations[col2]
                else:
                    # For categorical variables, we want the SMALLEST p-value
                    assoc1 = 1 - categorical_associations[col1]  # Transform so that larger = better
                    assoc2 = 1 - categorical_associations[col2]

                # Decide which one to keep
                if assoc1 > assoc2:
                    to_keep.remove(col2)
                    removed[col2] = {
                        'reason': f'Redundant with {col1} (MI={mi_value:.3f})',
                        'better': col1,
                        'type': 'numeric' if is_numeric else 'categorical'
                    }
                elif assoc2 > assoc1:
                    to_keep.remove(col1)
                    removed[col1] = {
                        'reason': f'Redundant with {col2} (MI={mi_value:.3f})',
                        'better': col2,
                        'type': 'numeric' if is_numeric else 'categorical'
                    }
                else:
                    # Tie: keep the one with fewer missing values
                    missing1 = df[col1].isna().sum()
                    missing2 = df[col2].isna().sum()
                    if missing1 <= missing2:
                        to_keep.remove(col2)
                        removed[col2] = {
                            'reason': f'Redundant with {col1} (MI={mi_value:.3f}, tied, less missing)',
                            'better': col1,
                            'type': 'numeric' if is_numeric else 'categorical'
                        }
                    else:
                        to_keep.remove(col1)
                        removed[col1] = {
                            'reason': f'Redundant with {col2} (MI={mi_value:.3f}, tied, less missing)',
                            'better': col2,
                            'type': 'numeric' if is_numeric else 'categorical'
                        }

        # Apply quality thresholds
        final_kept = []
        low_quality_removed = {}

        for col in list(to_keep):
            if is_numeric:
                # Numeric: minimum correlation
                if numeric_associations[col] < corr_threshold:
                    to_keep.remove(col)
                    low_quality_removed[col] = {
                        'reason': f'Low correlation (|r|={numeric_associations[col]:.3f} < {corr_threshold})',
                        'type': 'numeric'
                    }
                else:
                    final_kept.append(col)
            else:
                # Categorical: maximum p-value
                if categorical_associations[col] > chi2_threshold:
                    to_keep.remove(col)
                    low_quality_removed[col] = {
                        'reason': f'Weak Chi² association (p={categorical_associations[col]:.3f} > {chi2_threshold})',
                        'type': 'categorical'
                    }
                else:
                    final_kept.append(col)

        # Combine removals
        all_removed = {**removed, **low_quality_removed}

        return final_kept, all_removed, mi_matrix

    # 8. Clean both groups separately
    print("\n" + "="*60)
    print("CLEANING NUMERIC VARIABLES")
    print("="*60)
    numeric_selected, numeric_removed, mi_matrix_numeric = clean_feature_group(
        numeric_features, is_numeric=True
    )

    print(f"Initial: {len(numeric_features)} variables")
    print(f"Kept: {len(numeric_selected)} variables")
    print(f"Removed: {len(numeric_removed)} variables")

    if numeric_selected:
        print("\nSelected numeric variables (by correlation):")
        selected_with_corr = [(col, numeric_associations[col]) for col in numeric_selected]
        selected_with_corr.sort(key=lambda x: x[1], reverse=True)
        for col, corr in selected_with_corr:
            print(f"  - {col}: |r| = {corr:.3f}")

    print("\n" + "="*60)
    print("CLEANING CATEGORICAL VARIABLES")
    print("="*60)
    categorical_selected, categorical_removed, mi_matrix_categorical = clean_feature_group(
        categorical_features, is_numeric=False
    )

    print(f"Initial: {len(categorical_features)} variables")
    print(f"Kept: {len(categorical_selected)} variables")
    print(f"Removed: {len(categorical_removed)} variables")

    if categorical_selected:
        print("\nSelected categorical variables (by Chi² p-value):")
        selected_with_p = [(col, categorical_associations[col]) for col in categorical_selected]
        selected_with_p.sort(key=lambda x: x[1])  # Sort by increasing p-value
        for col, p_val in selected_with_p:
            print(f"  - {col}: p = {p_val:.4f}")

    # 9. Build the final DataFrame
    all_selected = numeric_selected + categorical_selected
    df_clean = df[all_selected + [target_col]].copy()

    # 10. Final summary
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    print(f"Total initial variables: {len(features)}")
    print(f"  - Numeric: {len(numeric_features)} → {len(numeric_selected)} kept")
    print(f"  - Categorical: {len(categorical_features)} → {len(categorical_selected)} kept")
    print(f"Total final variables: {len(all_selected)}")

    # Prepare the results information
    results_info = {
        'numeric': {
            'selected': numeric_selected,
            'removed': numeric_removed,
            'associations': numeric_associations,
            'mi_matrix': mi_matrix_numeric
        },
        'categorical': {
            'selected': categorical_selected,
            'removed': categorical_removed,
            'associations': categorical_associations,
            'mi_matrix': mi_matrix_categorical
        }
    }

    return df_clean, numeric_selected, categorical_selected, results_info

def quick_feature_cleaner(df, target='OS_YEARS'):
    """
    Simplified version with sensible default parameters for medical data
    """
    return clean_features_by_type(
        df=df,
        target_col=target,
        mi_threshold_numeric=0.25,  # MI threshold for numeric variables
        mi_threshold_categorical=0.3,  # MI threshold for categorical variables
        corr_threshold=0.05,  # Minimum correlation
        chi2_threshold=0.1,  # Maximum p-value for Chi²
        n_bins=8
    )

def visualize_cleaning_results(results_info):
    """
    Visualize the results of feature cleaning.

    :param results_info: Dictionary with 'numeric' and 'categorical' keys, or the 4th element of the tuple returned by clean_features_by_type
    """
    # Handle tuple input (if results_info is the full return tuple from clean_features_by_type)
    if isinstance(results_info, tuple):
        results_info = results_info[3]  # Extract results_info from the tuple

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 1. Correlations of the selected numeric variables
    numeric_data = results_info['numeric']
    if numeric_data['selected']:
        selected_numeric = [(col, numeric_data['associations'][col])
                           for col in numeric_data['selected']]
        selected_numeric.sort(key=lambda x: x[1], reverse=True)

        cols, corrs = zip(*selected_numeric[:15])  # Top 15

        axes[0, 0].barh(range(len(cols)), corrs)
        axes[0, 0].set_yticks(range(len(cols)))
        axes[0, 0].set_yticklabels(cols)
        axes[0, 0].set_xlabel('Absolute correlation')
        axes[0, 0].set_title('Top 15 numeric variables (by correlation)')
        axes[0, 0].invert_yaxis()

    # 2. p-values of the selected categorical variables
    categorical_data = results_info['categorical']
    if categorical_data['selected']:
        selected_categorical = [(col, categorical_data['associations'][col])
                               for col in categorical_data['selected']]
        selected_categorical.sort(key=lambda x: x[1])  # By increasing p-value

        cols, pvals = zip(*selected_categorical[:15])  # Top 15

        axes[0, 1].barh(range(len(cols)), pvals)
        axes[0, 1].axvline(x=0.05, color='r', linestyle='--', alpha=0.5, label='p=0.05')
        axes[0, 1].set_yticks(range(len(cols)))
        axes[0, 1].set_yticklabels(cols)
        axes[0, 1].set_xlabel('Chi² p-value')
        axes[0, 1].set_title('Top 15 categorical variables (by p-value)')
        axes[0, 1].invert_yaxis()
        axes[0, 1].legend()

    # 3. Distribution of removal reasons
    all_removed = list(numeric_data['removed'].items()) + list(categorical_data['removed'].items())

    if all_removed:
        reasons = []
        for col, info in all_removed:
            reason = info['reason']
            if 'Redundant' in reason:
                reasons.append('Redundancy (high MI)')
            elif 'Low correlation' in reason or 'Weak Chi²' in reason:
                reasons.append('Weak association with target')
            else:
                reasons.append('Other')

        reason_counts = pd.Series(reasons).value_counts()
        axes[1, 0].pie(reason_counts.values, labels=reason_counts.index, autopct='%1.1f%%')
        axes[1, 0].set_title('Reasons for variable removal')

    # 4. MI matrix of the numeric variables (heatmap)
    if not numeric_data['mi_matrix'].empty and len(numeric_data['selected']) > 1:
        # Extract the sub-matrix for the selected variables
        selected_mi = numeric_data['mi_matrix'].loc[
            numeric_data['selected'], numeric_data['selected']
        ]
        sns.heatmap(selected_mi, ax=axes[1, 1], cmap='YlOrRd',
                   cbar_kws={'label': 'Mutual information'})
        axes[1, 1].set_title('MI matrix of the selected numeric variables')

    plt.tight_layout()
    plt.show()

def variable_frequency_table(df):
    """
    Returns summary statistics for each variable in the dataframe.

    Parameters:
    df (pd.DataFrame): Input dataframe

    Returns:
    pd.DataFrame: Table with columns: Variable, Unique_Values, Non_Missing_Count, Missing_Count, Non_Missing_%
    """
    results = []

    for column in df.columns:
        total = len(df)
        non_missing = df[column].notna().sum()
        missing = df[column].isna().sum()
        unique_count = df[column].nunique()
        non_missing_pct = (non_missing / total) * 100

        results.append({
            'Variable': column,
            'Unique_Values': unique_count,
            'Non_Missing_Count': non_missing,
            'Missing_Count': missing,
            'Non_Missing_%': f"{non_missing_pct:.2f}%"
        })

    return pd.DataFrame(results)


"""
Util functions for training
"""

def split_patients_no_leakage(X, Y, train_size=0.6, val_size=0.2, test_size=0.2, random_state=42, id_column='ID'):
    """
    Split the data into train/val/test, ensuring that no patient appears in more than one set.
    This prevents data leakage when generating pairs.

    :param X: Feature DataFrame (must contain id_column)
    :param Y: DataFrame with columns [OS_STATUS, OS_YEARS]
    :param train_size: Proportion for training (default 0.6)
    :param val_size: Proportion for validation (default 0.2)
    :param test_size: Proportion for testing (default 0.2)
    :param random_state: Seed for reproducibility
    :param id_column: Name of the ID column
    :return: Tuple (X_train, Y_train, X_val, Y_val, X_test, Y_test)
    """
    from sklearn.model_selection import train_test_split

    # Check that the proportions sum to 1
    assert abs(train_size + val_size + test_size - 1.0) < 1e-6, "Proportions must sum to 1"

    n_total = len(X)

    # First split: train vs (val + test)
    X_train, X_temp, Y_train, Y_temp = train_test_split(
        X, Y,
        test_size=(val_size + test_size),
        random_state=random_state,
        stratify=Y.iloc[:, 0]  # Stratify on OS_STATUS
    )

    # Second split: val vs test
    val_ratio = val_size / (val_size + test_size)
    X_val, X_test, Y_val, Y_test = train_test_split(
        X_temp, Y_temp,
        test_size=(1 - val_ratio),
        random_state=random_state,
        stratify=Y_temp.iloc[:, 0]
    )

    # Reset index to avoid issues
    X_train = X_train.reset_index(drop=True)
    Y_train = Y_train.reset_index(drop=True)
    X_val = X_val.reset_index(drop=True)
    Y_val = Y_val.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    Y_test = Y_test.reset_index(drop=True)

    print(f"Split done with no patient leakage:")
    print(f"  Train: {len(X_train)} patients ({len(X_train)/n_total*100:.1f}%)")
    print(f"  Val:   {len(X_val)} patients ({len(X_val)/n_total*100:.1f}%)")
    print(f"  Test:  {len(X_test)} patients ({len(X_test)/n_total*100:.1f}%)")

    # Check that there is no overlap of IDs
    train_ids = set(X_train[id_column])
    val_ids = set(X_val[id_column])
    test_ids = set(X_test[id_column])

    assert len(train_ids & val_ids) == 0, "Leakage detected between train and val!"
    assert len(train_ids & test_ids) == 0, "Leakage detected between train and test!"
    assert len(val_ids & test_ids) == 0, "Leakage detected between val and test!"
    print("  ✓ No patient overlap detected")

    return X_train, Y_train, X_val, Y_val, X_test, Y_test

def generate_pairs_concat(X, Y, max_pairs=None, include_abs_diff=False, random_state=42, id_column='ID'):
    """
    Generate pairs by concatenation, with an option to add absolute differences.
    Returns the pairs of IDs (not row indices).

    :param X: Feature DataFrame (must contain id_column)
    :param Y: DataFrame with columns [OS_STATUS, OS_YEARS]
    :param max_pairs: Maximum number of pairs to generate
    :param include_abs_diff: If True, add absolute differences as features
    :param random_state: Seed for reproducibility
    :param id_column: Name of the ID column in X (default: 'ID')
    :return: Tuple (X_pairs_df, y_pairs_array, pair_ids)
    """


    # Check that id_column exists
    if id_column not in X.columns:
        raise ValueError(f"Column '{id_column}' not found in X")

    # Exclude the ID column from the features used for pair generation
    X_features = X.drop(columns=[id_column])

    n = len(X)
    indices = list(itertools.combinations(range(n), 2))

    if max_pairs and max_pairs < len(indices):
        random.seed(random_state)
        indices = random.sample(indices, max_pairs)

    X_pairs = []
    y_pairs = []
    pair_ids = []

    for i, j in indices:
        status_i, time_i = Y.iloc[i, 0], Y.iloc[i, 1]
        status_j, time_j = Y.iloc[j, 0], Y.iloc[j, 1]

        # Convert boolean to int if needed
        status_i = int(status_i)
        status_j = int(status_j)

        # Check comparability
        comparable = False
        label = None

        if status_i == 1 and status_j == 1:
            comparable = True
            label = 1 if time_i < time_j else 0
        elif status_i == 1 and status_j == 0 and time_i <= time_j:
            comparable = True
            label = 1
        elif status_i == 0 and status_j == 1 and time_i >= time_j:
            comparable = True
            label = 0

        if comparable:
            # Use X_features (without ID) for the pairs
            xi, xj = X_features.iloc[i].values, X_features.iloc[j].values

            # Basic concatenation
            feat = np.concatenate([xi, xj])

            # Option: add absolute differences
            if include_abs_diff:
                abs_diff = np.abs(xi - xj)
                feat = np.concatenate([feat, abs_diff])

            X_pairs.append(feat)
            y_pairs.append(label)

            # Store the actual IDs (not the indices)
            id_i = X.iloc[i][id_column]
            id_j = X.iloc[j][id_column]
            pair_ids.append((id_i, id_j))

    if len(X_pairs) == 0:
        raise ValueError("No comparable pair was found.")

    # Build DataFrame with pair_ids as the first column
    X_pairs_array = np.array(X_pairs)
    n_features = X_pairs_array.shape[1]
    feature_names = [f'feat_{i}' for i in range(n_features)]

    result_df = pd.DataFrame(X_pairs_array, columns=feature_names)
    result_df.insert(0, 'pair_ids', pair_ids)

    print(f"CONCAT: {len(X_pairs)} pairs generated, {n_features} features per pair")

    return result_df, np.array(y_pairs), pair_ids



"""
Others
"""
def save_pairs_to_excel(
    X_pairs_train, y_pairs_train, ids_train,
    X_pairs_val, y_pairs_val, ids_val,
    X_pairs_test, y_pairs_test, ids_test,
    output_path='data/processed_data.xlsx',
    target_col_name='target',
    id_col_name='patient_ids',
    sheet_names=None
):
    """
    Save the train/val/test pairs to an Excel file with 3 sheets.

    :param X_pairs_train: Feature DataFrame for train
    :param y_pairs_train: Label array for train
    :param ids_train: List of ID tuples for train
    :param X_pairs_val: Feature DataFrame for validation
    :param y_pairs_val: Label array for validation
    :param ids_val: List of ID tuples for validation
    :param X_pairs_test: Feature DataFrame for test
    :param y_pairs_test: Label array for test
    :param ids_test: List of ID tuples for test
    :param output_path: Path of the output Excel file
    :param target_col_name: Name of the target column (default: 'target')
    :param id_col_name: Name of the IDs column (default: 'patient_ids')
    :param sheet_names: Dict with keys 'train', 'val', 'test' to customize the sheet names
    :return: Dict with the created DataFrames
    """

    # Define the sheet names
    if sheet_names is None:
        sheet_names = {'train': 'train', 'val': 'calibration', 'test': 'test'}

    # Build the Train DataFrame (pair_ids is already present)
    train_df = X_pairs_train.copy()
    train_df[target_col_name] = y_pairs_train
    # Rename pair_ids to patient_ids if it exists, otherwise add it
    if 'pair_ids' in train_df.columns:
        train_df = train_df.rename(columns={'pair_ids': id_col_name})
    else:
        train_df.insert(0, id_col_name, [str(pair) for pair in ids_train])

    # Build the Validation DataFrame
    val_df = X_pairs_val.copy()
    val_df[target_col_name] = y_pairs_val
    if 'pair_ids' in val_df.columns:
        val_df = val_df.rename(columns={'pair_ids': id_col_name})
    else:
        val_df.insert(0, id_col_name, [str(pair) for pair in ids_val])

    # Build the Test DataFrame
    test_df = X_pairs_test.copy()
    test_df[target_col_name] = y_pairs_test
    if 'pair_ids' in test_df.columns:
        test_df = test_df.rename(columns={'pair_ids': id_col_name})
    else:
        test_df.insert(0, id_col_name, [str(pair) for pair in ids_test])

    # Create the directory if needed
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"✓ Directory created: {output_dir}")

    # Save to an Excel file with 3 sheets
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        train_df.to_excel(writer, sheet_name=sheet_names['train'], index=False)
        val_df.to_excel(writer, sheet_name=sheet_names['val'], index=False)
        test_df.to_excel(writer, sheet_name=sheet_names['test'], index=False)

    print(f"✓ Data saved to '{output_path}'")
    print(f"  - Sheet '{sheet_names['train']}': {len(train_df)} pairs")
    print(f"  - Sheet '{sheet_names['val']}': {len(val_df)} pairs")
    print(f"  - Sheet '{sheet_names['test']}': {len(test_df)} pairs")

    return {
        'train': train_df,
        'val': val_df,
        'test': test_df
    }
