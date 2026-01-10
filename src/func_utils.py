import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, chi2_contingency, pointbiserialr
from sklearn.metrics import mutual_info_score
import warnings
import seaborn as sns
import itertools
import random
warnings.filterwarnings('ignore')

"""
Util functions for data cleaning and feature selection.
"""
def clean_features_by_type(df, target_col='OS.YEARS', 
                          mi_threshold_numeric=0.2, 
                          mi_threshold_categorical=0.3,
                          corr_threshold=0.1,
                          chi2_threshold=0.05,
                          n_bins=10):
    """
    Nettoie les variables en traitant séparément quantitatives et qualitatives.
    
    Pour les variables quantitatives:
      - Utilise Pearson pour l'association avec la cible
      - Utilise MI pour détecter la redondance entre variables quantitatives
    
    Pour les variables qualitatives:
      - Utilise Chi² pour l'association avec la cible (cible discrétisée)
      - Utilise MI pour détecter la redondance entre variables qualitatives
    
    Les redondances entre types différents ne sont pas considérées.
    
    Paramètres:
    -----------
    df : DataFrame
    target_col : str
        Variable cible continue (OS.YEARS)
    mi_threshold_numeric : float
        Seuil MI pour redondance entre variables quantitatives
    mi_threshold_categorical : float
        Seuil MI pour redondance entre variables qualitatives
    corr_threshold : float
        Seuil de corrélation minimale pour conserver une variable quantitative
    chi2_threshold : float
        Seuil de p-value maximale pour conserver une variable qualitative
    n_bins : int
        Nombre de bins pour discrétiser les variables continues pour MI
    
    Retourne:
    --------
    tuple: (df_clean, numeric_selected, categorical_selected, results_info)
    """
    
    # 1. Préparation des données
    features = [col for col in df.columns if col != target_col]
    
    if target_col not in df.columns:
        raise ValueError(f"Colonne cible '{target_col}' non trouvée")
    
    # 2. Séparation des variables
    numeric_features = []
    categorical_features = []
    
    for col in features:
        if pd.api.types.is_numeric_dtype(df[col]):
            # Variable numérique avec plus de 10 valeurs uniques = quantitative
            if df[col].nunique() > 10:
                numeric_features.append(col)
            else:
                # Peu de valeurs = traiter comme catégorielle
                categorical_features.append(col)
        else:
            # Variable non-numérique = catégorielle
            categorical_features.append(col)
    
    print("="*60)
    print("CLASSIFICATION DES VARIABLES")
    print("="*60)
    print(f"Variables quantitatives ({len(numeric_features)}):")
    print(f"  {', '.join(numeric_features[:10])}{'...' if len(numeric_features) > 10 else ''}")
    print(f"\nVariables qualitatives ({len(categorical_features)}):")
    print(f"  {', '.join(categorical_features[:10])}{'...' if len(categorical_features) > 10 else ''}")
    
    # 3. Discrétiser la cible pour Chi² (si nécessaire)
    df_processed = df.copy()
    if df_processed[target_col].nunique() > 10:
        # Discrétiser en quartiles
        df_processed[f'{target_col}_binned'] = pd.qcut(
            df_processed[target_col], 
            q=4, 
            labels=['Q1', 'Q2', 'Q3', 'Q4']
        )
        target_binned = f'{target_col}_binned'
    else:
        target_binned = target_col
    
    # 4. Fonction pour calculer l'association avec la cible
    
    # Pour variables quantitatives: corrélation de Pearson
    def compute_numeric_association(col):
        """Calcule la corrélation absolue avec la cible"""
        valid_data = df[[col, target_col]].dropna()
        if len(valid_data) < 10:
            return 0.0
        try:
            corr, p_value = pearsonr(valid_data[col], valid_data[target_col])
            return abs(corr)
        except:
            return 0.0
    
    # Pour variables qualitatives: test du Chi²
    def compute_categorical_association(col):
        """Calcule la p-value du Chi² avec la cible discrétisée"""
        # Créer table de contingence
        contingency = pd.crosstab(df_processed[col], df_processed[target_binned])
        
        # Conditions minimales pour le test
        if contingency.shape[0] < 2 or contingency.shape[1] < 2:
            return 1.0  # p-value maximale = pas d'association
        
        if contingency.sum().sum() < 20:  # Trop peu d'observations
            return 1.0
        
        # Appliquer le test Chi²
        try:
            chi2, p, dof, expected = chi2_contingency(contingency)
            return p  # Retourne la p-value (plus petite = meilleure)
        except:
            return 1.0
    
    # 5. Calculer les associations
    print("\n" + "="*60)
    print("CALCUL DES ASSOCIATIONS AVEC LA CIBLE")
    print("="*60)
    
    numeric_associations = {}
    for col in numeric_features:
        numeric_associations[col] = compute_numeric_association(col)
    
    categorical_associations = {}
    for col in categorical_features:
        categorical_associations[col] = compute_categorical_association(col)
    
    # 6. Préparer les données pour le calcul de MI (discrétisation)
    df_for_mi = df.copy()
    
    # Discrétiser les variables quantitatives
    for col in numeric_features:
        try:
            df_for_mi[col] = pd.qcut(df_for_mi[col], q=n_bins, labels=False, duplicates='drop')
        except:
            df_for_mi[col] = pd.cut(df_for_mi[col], bins=n_bins, labels=False)
    
    # Encoder les variables qualitatives
    for col in categorical_features:
        if not pd.api.types.is_numeric_dtype(df_for_mi[col]):
            df_for_mi[col] = pd.factorize(df_for_mi[col])[0]
    
    # Remplir les NaN
    df_for_mi = df_for_mi.fillna(-1)
    
    # 7. Fonction pour nettoyer un groupe de variables
    def clean_feature_group(feature_list, is_numeric=True):
        """Nettoie un groupe de variables (quantitatives ou qualitatives)"""
        if not feature_list:
            return [], {}
        
        # Matrice MI pour ce groupe
        n_features = len(feature_list)
        mi_matrix = pd.DataFrame(0.0, index=feature_list, columns=feature_list)
        
        # Calculer MI pour chaque paire
        for i in range(n_features):
            for j in range(i+1, n_features):
                col1, col2 = feature_list[i], feature_list[j]
                mi = mutual_info_score(df_for_mi[col1], df_for_mi[col2])
                mi_matrix.loc[col1, col2] = mi
                mi_matrix.loc[col2, col1] = mi
        
        # Seuil MI selon le type
        mi_threshold = mi_threshold_numeric if is_numeric else mi_threshold_categorical
        
        # Identifier les paires redondantes
        to_keep = set(feature_list)
        removed = {}
        
        high_mi_pairs = []
        for i in range(n_features):
            for j in range(i+1, n_features):
                col1, col2 = feature_list[i], feature_list[j]
                mi_value = mi_matrix.loc[col1, col2]
                if mi_value > mi_threshold:
                    high_mi_pairs.append((mi_value, col1, col2))
        
        # Trier par MI décroissant
        high_mi_pairs.sort(reverse=True, key=lambda x: x[0])
        
        # Traiter chaque paire redondante
        for mi_value, col1, col2 in high_mi_pairs:
            if col1 in to_keep and col2 in to_keep:
                # Comparer les associations avec la cible
                if is_numeric:
                    assoc1 = numeric_associations[col1]
                    assoc2 = numeric_associations[col2]
                else:
                    # Pour catégorielles, on veut la PLUS PETITE p-value
                    assoc1 = 1 - categorical_associations[col1]  # Transformer pour que plus grand = mieux
                    assoc2 = 1 - categorical_associations[col2]
                
                # Décider laquelle garder
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
                    # Égalité: garder celle avec moins de valeurs manquantes
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
        
        # Appliquer les seuils de qualité
        final_kept = []
        low_quality_removed = {}
        
        for col in list(to_keep):
            if is_numeric:
                # Pour quantitatives: corrélation minimale
                if numeric_associations[col] < corr_threshold:
                    to_keep.remove(col)
                    low_quality_removed[col] = {
                        'reason': f'Low correlation (|r|={numeric_associations[col]:.3f} < {corr_threshold})',
                        'type': 'numeric'
                    }
                else:
                    final_kept.append(col)
            else:
                # Pour qualitatives: p-value maximale
                if categorical_associations[col] > chi2_threshold:
                    to_keep.remove(col)
                    low_quality_removed[col] = {
                        'reason': f'Weak Chi² association (p={categorical_associations[col]:.3f} > {chi2_threshold})',
                        'type': 'categorical'
                    }
                else:
                    final_kept.append(col)
        
        # Combiner les suppressions
        all_removed = {**removed, **low_quality_removed}
        
        return final_kept, all_removed, mi_matrix
    
    # 8. Nettoyer les deux groupes séparément
    print("\n" + "="*60)
    print("NETTOYAGE DES VARIABLES QUANTITATIVES")
    print("="*60)
    numeric_selected, numeric_removed, mi_matrix_numeric = clean_feature_group(
        numeric_features, is_numeric=True
    )
    
    print(f"Initial: {len(numeric_features)} variables")
    print(f"Conservées: {len(numeric_selected)} variables")
    print(f"Supprimées: {len(numeric_removed)} variables")
    
    if numeric_selected:
        print("\nVariables quantitatives sélectionnées (par corrélation):")
        selected_with_corr = [(col, numeric_associations[col]) for col in numeric_selected]
        selected_with_corr.sort(key=lambda x: x[1], reverse=True)
        for col, corr in selected_with_corr:
            print(f"  - {col}: |r| = {corr:.3f}")
    
    print("\n" + "="*60)
    print("NETTOYAGE DES VARIABLES QUALITATIVES")
    print("="*60)
    categorical_selected, categorical_removed, mi_matrix_categorical = clean_feature_group(
        categorical_features, is_numeric=False
    )
    
    print(f"Initial: {len(categorical_features)} variables")
    print(f"Conservées: {len(categorical_selected)} variables")
    print(f"Supprimées: {len(categorical_removed)} variables")
    
    if categorical_selected:
        print("\nVariables qualitatives sélectionnées (par p-value Chi²):")
        selected_with_p = [(col, categorical_associations[col]) for col in categorical_selected]
        selected_with_p.sort(key=lambda x: x[1])  # Trier par p-value croissante
        for col, p_val in selected_with_p:
            print(f"  - {col}: p = {p_val:.4f}")
    
    # 9. Créer le DataFrame final
    all_selected = numeric_selected + categorical_selected
    df_clean = df[all_selected + [target_col]].copy()
    
    # 10. Résumé final
    print("\n" + "="*60)
    print("RÉSUMÉ FINAL")
    print("="*60)
    print(f"Total variables initiales: {len(features)}")
    print(f"  - Quantitatives: {len(numeric_features)} → {len(numeric_selected)} conservées")
    print(f"  - Qualitatives: {len(categorical_features)} → {len(categorical_selected)} conservées")
    print(f"Total variables finales: {len(all_selected)}")
    
    # Préparer les informations de résultats
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

def quick_feature_cleaner(df, target='OS.YEARS'):
    """
    Version simplifiée avec paramètres par défaut raisonnables pour données médicales
    """
    return clean_features_by_type(
        df=df,
        target_col=target,
        mi_threshold_numeric=0.25,  # MI seuil pour variables quantitatives
        mi_threshold_categorical=0.3,  # MI seuil pour variables qualitatives
        corr_threshold=0.05,  # Corrélation minimale
        chi2_threshold=0.1,  # p-value maximale pour Chi²
        n_bins=8
    )

def visualize_cleaning_results(results_info):
    """
    Visualise les résultats du nettoyage des variables.
    
    :param results_info: Dictionary with 'numeric' and 'categorical' keys, or the 4th element of the tuple returned by clean_features_by_type
    """
    # Handle tuple input (if results_info is the full return tuple from clean_features_by_type)
    if isinstance(results_info, tuple):
        results_info = results_info[3]  # Extract results_info from the tuple
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Corrélations des variables quantitatives sélectionnées
    numeric_data = results_info['numeric']
    if numeric_data['selected']:
        selected_numeric = [(col, numeric_data['associations'][col]) 
                           for col in numeric_data['selected']]
        selected_numeric.sort(key=lambda x: x[1], reverse=True)
        
        cols, corrs = zip(*selected_numeric[:15])  # Top 15
        
        axes[0, 0].barh(range(len(cols)), corrs)
        axes[0, 0].set_yticks(range(len(cols)))
        axes[0, 0].set_yticklabels(cols)
        axes[0, 0].set_xlabel('Corrélation absolue')
        axes[0, 0].set_title('Top 15 variables quantitatives (par corrélation)')
        axes[0, 0].invert_yaxis()
    
    # 2. p-values des variables qualitatives sélectionnées
    categorical_data = results_info['categorical']
    if categorical_data['selected']:
        selected_categorical = [(col, categorical_data['associations'][col]) 
                               for col in categorical_data['selected']]
        selected_categorical.sort(key=lambda x: x[1])  # Par p-value croissante
        
        cols, pvals = zip(*selected_categorical[:15])  # Top 15
        
        axes[0, 1].barh(range(len(cols)), pvals)
        axes[0, 1].axvline(x=0.05, color='r', linestyle='--', alpha=0.5, label='p=0.05')
        axes[0, 1].set_yticks(range(len(cols)))
        axes[0, 1].set_yticklabels(cols)
        axes[0, 1].set_xlabel('p-value Chi²')
        axes[0, 1].set_title('Top 15 variables qualitatives (par p-value)')
        axes[0, 1].invert_yaxis()
        axes[0, 1].legend()
    
    # 3. Distribution des raisons de suppression
    all_removed = list(numeric_data['removed'].items()) + list(categorical_data['removed'].items())
    
    if all_removed:
        reasons = []
        for col, info in all_removed:
            reason = info['reason']
            if 'Redundant' in reason:
                reasons.append('Redondance (MI élevé)')
            elif 'Low correlation' in reason or 'Weak Chi²' in reason:
                reasons.append('Faible association avec cible')
            else:
                reasons.append('Autre')
        
        reason_counts = pd.Series(reasons).value_counts()
        axes[1, 0].pie(reason_counts.values, labels=reason_counts.index, autopct='%1.1f%%')
        axes[1, 0].set_title('Raisons de suppression des variables')
    
    # 4. Matrice MI des variables quantitatives (heatmap)
    if not numeric_data['mi_matrix'].empty and len(numeric_data['selected']) > 1:
        # Extraire la sous-matrice pour les variables sélectionnées
        selected_mi = numeric_data['mi_matrix'].loc[
            numeric_data['selected'], numeric_data['selected']
        ]
        sns.heatmap(selected_mi, ax=axes[1, 1], cmap='YlOrRd', 
                   cbar_kws={'label': 'Information Mutuelle'})
        axes[1, 1].set_title('Matrice MI des variables quantitatives sélectionnées')
    
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
    Split les données en train/val/test en s'assurant qu'aucun patient n'apparaît dans plusieurs ensembles.
    Ceci évite le data leakage lors de la génération de paires.
    
    : param X: DataFrame des features (doit contenir id_column)
    :param Y: DataFrame avec colonnes [OS_STATUS, OS_YEARS]
    :param train_size:  Proportion pour l'entraînement (défaut 0.6)
    :param val_size: Proportion pour la validation (défaut 0.2)
    :param test_size: Proportion pour le test (défaut 0.2)
    :param random_state:  Seed pour reproductibilité
    :param id_column: Nom de la colonne ID
    :return: Tuple (X_train, Y_train, X_val, Y_val, X_test, Y_test)
    """
    from sklearn.model_selection import train_test_split
    
    # Vérifier que les proportions somment à 1
    assert abs(train_size + val_size + test_size - 1.0) < 1e-6, "Les proportions doivent sommer à 1"
    
    n_total = len(X)
    
    # Premier split:  train vs (val + test)
    X_train, X_temp, Y_train, Y_temp = train_test_split(
        X, Y, 
        test_size=(val_size + test_size), 
        random_state=random_state,
        stratify=Y. iloc[:, 0]  # Stratifier sur OS_STATUS
    )
    
    # Deuxième split: val vs test
    val_ratio = val_size / (val_size + test_size)
    X_val, X_test, Y_val, Y_test = train_test_split(
        X_temp, Y_temp,
        test_size=(1 - val_ratio),
        random_state=random_state,
        stratify=Y_temp.iloc[:, 0]
    )
    
    # Reset index pour éviter les problèmes
    X_train = X_train.reset_index(drop=True)
    Y_train = Y_train.reset_index(drop=True)
    X_val = X_val.reset_index(drop=True)
    Y_val = Y_val.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    Y_test = Y_test.reset_index(drop=True)
    
    print(f"Split réalisé sans leakage de patients:")
    print(f"  Train: {len(X_train)} patients ({len(X_train)/n_total*100:.1f}%)")
    print(f"  Val:    {len(X_val)} patients ({len(X_val)/n_total*100:.1f}%)")
    print(f"  Test:  {len(X_test)} patients ({len(X_test)/n_total*100:.1f}%)")
    
    # Vérifier qu'il n'y a pas de chevauchement d'IDs
    train_ids = set(X_train[id_column])
    val_ids = set(X_val[id_column])
    test_ids = set(X_test[id_column])
    
    assert len(train_ids & val_ids) == 0, "Leakage détecté entre train et val!"
    assert len(train_ids & test_ids) == 0, "Leakage détecté entre train et test!"
    assert len(val_ids & test_ids) == 0, "Leakage détecté entre val et test!"
    print("  ✓ Aucun chevauchement de patients détecté")
    
    return X_train, Y_train, X_val, Y_val, X_test, Y_test

def generate_pairs_concat(X, Y, max_pairs=None, include_abs_diff=False, random_state=42, id_column='ID'):
    """
    Génère des paires en concaténation, avec option d'ajouter les différences absolues. 
    Retourne les paires d'IDs (pas les indices de ligne).
    
    :param X:  DataFrame des features (doit contenir id_column)
    :param Y: DataFrame avec colonnes [OS_STATUS, OS_YEARS]
    :param max_pairs: Nombre maximum de paires à générer
    :param include_abs_diff: Si True, ajoute les différences absolues comme features
    :param random_state:  Seed pour reproductibilité
    :param id_column: Nom de la colonne ID dans X (défaut:  'ID')
    :return: Tuple (X_pairs_df, y_pairs_array, pair_ids)
    """

    
    # Vérifier que id_column existe
    if id_column not in X.columns:
        raise ValueError(f"Colonne '{id_column}' non trouvée dans X")
    
    # Exclure la colonne ID des features pour la génération de paires
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
        
        # Vérifier comparabilité
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
            # Utiliser X_features (sans ID) pour les paires
            xi, xj = X_features.iloc[i].values, X_features.iloc[j].values
            
            # Concaténation de base
            feat = np. concatenate([xi, xj])
            
            # Option: ajouter les différences absolues
            if include_abs_diff: 
                abs_diff = np. abs(xi - xj)
                feat = np.concatenate([feat, abs_diff])
            
            X_pairs.append(feat)
            y_pairs. append(label)
            
            # Stocker les IDs réels (pas les indices)
            id_i = X.iloc[i][id_column]
            id_j = X.iloc[j][id_column]
            pair_ids.append((id_i, id_j))
    
    if len(X_pairs) == 0:
        raise ValueError("Aucune paire comparable n'a été trouvée.")
    
    # Créer DataFrame avec pair_ids en première colonne
    X_pairs_array = np.array(X_pairs)
    n_features = X_pairs_array.shape[1]
    feature_names = [f'feat_{i}' for i in range(n_features)]
    
    result_df = pd.DataFrame(X_pairs_array, columns=feature_names)
    result_df.insert(0, 'pair_ids', pair_ids)
    
    print(f"CONCAT: {len(X_pairs)} paires générées, {n_features} features par paire")
    
    return result_df, np. array(y_pairs), pair_ids



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
    Sauvegarde les paires train/val/test dans un fichier Excel avec 3 sheets.
    
    : param X_pairs_train: DataFrame des features pour train
    :param y_pairs_train: Array des labels pour train
    :param ids_train: Liste des tuples d'IDs pour train
    :param X_pairs_val: DataFrame des features pour validation
    :param y_pairs_val: Array des labels pour validation
    :param ids_val: Liste des tuples d'IDs pour validation
    :param X_pairs_test: DataFrame des features pour test
    :param y_pairs_test: Array des labels pour test
    :param ids_test: Liste des tuples d'IDs pour test
    :param output_path: Chemin du fichier Excel de sortie
    :param target_col_name: Nom de la colonne cible (défaut: 'target')
    :param id_col_name:  Nom de la colonne IDs (défaut: 'patient_ids')
    :param sheet_names: Dict avec clés 'train', 'val', 'test' pour personnaliser les noms des sheets
    : return: Dict avec les DataFrames créés
    """

    # Définir les noms des sheets
    if sheet_names is None:
        sheet_names = {'train': 'train', 'val': 'calibration', 'test': 'test'}
    
    # Créer le DataFrame pour Train (pair_ids est déjà présent)
    train_df = X_pairs_train.copy()
    train_df[target_col_name] = y_pairs_train
    # Renommer pair_ids en patient_ids si elle existe, sinon l'ajouter
    if 'pair_ids' in train_df.columns:
        train_df = train_df.rename(columns={'pair_ids': id_col_name})
    else:
        train_df.insert(0, id_col_name, [str(pair) for pair in ids_train])
    
    # Créer le DataFrame pour Validation
    val_df = X_pairs_val.copy()
    val_df[target_col_name] = y_pairs_val
    if 'pair_ids' in val_df.columns:
        val_df = val_df.rename(columns={'pair_ids': id_col_name})
    else:
        val_df.insert(0, id_col_name, [str(pair) for pair in ids_val])
    
    # Créer le DataFrame pour Test
    test_df = X_pairs_test.copy()
    test_df[target_col_name] = y_pairs_test
    if 'pair_ids' in test_df.columns:
        test_df = test_df.rename(columns={'pair_ids': id_col_name})
    else:
        test_df.insert(0, id_col_name, [str(pair) for pair in ids_test])
    
    # Créer le répertoire si nécessaire
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"✓ Répertoire créé:  {output_dir}")
    
    # Sauvegarder dans un fichier Excel avec 3 sheets
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        train_df.to_excel(writer, sheet_name=sheet_names['train'], index=False)
        val_df.to_excel(writer, sheet_name=sheet_names['val'], index=False)
        test_df.to_excel(writer, sheet_name=sheet_names['test'], index=False)
    
    print(f"✓ Données sauvegardées dans '{output_path}'")
    print(f"  - Sheet '{sheet_names['train']}': {len(train_df)} paires")
    print(f"  - Sheet '{sheet_names['val']}': {len(val_df)} paires")
    print(f"  - Sheet '{sheet_names['test']}': {len(test_df)} paires")
    
    return {
        'train': train_df,
        'val': val_df,
        'test':  test_df
    }






if __name__ == "__main__":

        # Charger vos données
    df = pd.read_csv('vos_donnees_leucemie.csv')

    # Option 1: Utilisation avec paramètres par défaut
    df_clean, numeric_selected, categorical_selected, results = quick_feature_cleaner(
        df, target='OS.YEARS'
    )

    # Option 2: Utilisation avec paramètres personnalisés
    df_clean, numeric_selected, categorical_selected, results = clean_features_by_type(
        df=df,
        target_col='OS.YEARS',
        mi_threshold_numeric=0.2,  # MI > 0.2 = redondance forte
        mi_threshold_categorical=0.25,
        corr_threshold=0.05,  # Garder si |r| > 0.05
        chi2_threshold=0.05,  # Garder si p-value Chi² < 0.05
        n_bins=8
    )

    # Visualiser les résultats
    visualize_cleaning_results(results)

    # Sauvegarder les données nettoyées
    df_clean.to_csv('donnees_nettoyees.csv', index=False)

    # Afficher un résumé textuel
    print("\n" + "="*60)
    print("VARIABLES SÉLECTIONNÉES POUR LA MODÉLISATION")
    print("="*60)
    print(f"\nVariables quantitatives ({len(numeric_selected)}):")
    for col in numeric_selected:
        corr = results['numeric']['associations'][col]
        print(f"  ✓ {col} (|r| = {corr:.3f})")

    print(f"\nVariables qualitatives ({len(categorical_selected)}):")
    for col in categorical_selected:
        pval = results['categorical']['associations'][col]
        print(f"  ✓ {col} (p = {pval:.4f})")

        