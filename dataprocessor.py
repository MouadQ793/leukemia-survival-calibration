import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from typing import Optional, Dict, Any
import warnings
import itertools
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import random

# =============================================================
# Module for Data Processing Class 
# =============================================================

class DataProcessor:
    
    def __init__(self, molecular_data: pd.DataFrame, clinical_data: pd.DataFrame, target_data: pd.DataFrame) -> None:
        """
        Initialize the DataProcessor with molecular, clinical, and target data.
        """
        self.molecular_data : pd.DataFrame = molecular_data
        self.clinical_data : pd.DataFrame = clinical_data
        self.target_data : pd.DataFrame = target_data
        self.merged_table : pd.DataFrame = None
        self.treated_data : pd.DataFrame = None
        self.temporary : pd.DataFrame = None
        self.temporary_annotation : str = None
    
    def set_clinical_data(self, clinical_data: pd.DataFrame) -> None:
        """
        Sets the clinical dataset.
        """
        self.clinical_data = clinical_data
        
    def merge_tables(self, cli = None, mol = None) -> None:
        self.merged_table = self.molecular_data.merge(self.clinical_data,on="ID",how="left"
                                                                ).merge(self.target_data,on="ID",how="left")    
    
    def set_molecular_data(self, molecular_data: pd.DataFrame) -> None:
        """
        Sets the molecular dataset.
        """
        self.molecular_data = molecular_data
        
    def save_data(self):
        self.treated_data = self.temporary
        self.temporary = None
        warnings.warn("The temporary attribute dataframe has been deleted, refer to treated data ", FutureWarning)
        
    def return_data(self):
        """
        Saves the treated dataset after processing.
        Useful for saving data after merging and preprocessing.
        """
        return self.treated_data

    def get_data_shapes(self) -> tuple:
        """
        Returns the shapes of the molecular, clinical, and target datasets.
        """
        return (self.molecular_data.shape, self.clinical_data.shape, self.target_data.shape)

    def value_types(self) -> dict:
        """
        Returns the data types of each column in the datasets.
        """
        return {
            'molecular_data_types': self.molecular_data.dtypes.to_dict(),
            'clinical_data_types': self.clinical_data.dtypes.to_dict(),
            'target_data_types': self.target_data.dtypes.to_dict()
        }
        
    def show(self) -> None:
        """
        Display the current format of the Data
        """
        if self.merged_table is None : 
            print("Molecular Data :")
            print(self.molecular_data.head())
            print("\nClinical Data :")
            print(self.clinical_data.head())    
            print("\nTarget Data :")
            print(self.target_data.head())
        else :
            print("Merged Data table : ")
            print(self.merged_table.head())
        
    def mergedatabis(self) -> pd.DataFrame:
        """
        Merges Molecular, clinical and target data on ID. 
        - Keeps multiple rows per patient. These rows are treated in different ways by another method
        """
        merged : pd.DataFrame = self.clinical_data.merge(self.molecular_data, on = "ID", how="left").merge(self.target_data, on="ID", how="left")
        self.merged_table = merged
        return merged

    
    def preprocess(self) -> tuple:
        # Impute missing values
        imputer = SimpleImputer(strategy='mean')
        self.molecular_data.iloc[:, 1:] = imputer.fit_transform(self.molecular_data.iloc[:, 1:])
        self.clinical_data.iloc[:, 1:] = imputer.fit_transform(self.clinical_data.iloc[:, 1:])

        # Standardize features
        scaler = StandardScaler()
        self.molecular_data.iloc[:, 1:] = scaler.fit_transform(self.molecular_data.iloc[:, 1:])
        self.clinical_data.iloc[:, 1:] = scaler.fit_transform(self.clinical_data.iloc[:, 1:])

        return self.molecular_data, self.clinical_data, self.target_data
    
    def plot_quant_var(self) -> None:
        """
        Plots quantitative variables from molecular and clinical datasets.
        """
        # Plotting molecular data
        quant_cols = self.molecular_data.select_dtypes(include=[np.number]).columns.tolist()

        self.molecular_data[quant_cols].hist(bins=30, figsize=(15, 10))
        plt.suptitle("Molecular Data Distributions")
        plt.show()
        
        # Plotting clinical data
        quant_cols = self.clinical_data.select_dtypes(include=[np.number]).columns.tolist()

        self.clinical_data[quant_cols].hist(bins=30, figsize=(15, 10))
        plt.suptitle("Clinical Data Distributions")
        plt.show()
        
    def missing_quant_var(self) -> Dict[str, Any]:
        """
        Analyzes missing quantitative variables in molecular and clinical datasets.
        """
        missing_info = {}

        for dataset_name, dataset in [('molecular_data', self.molecular_data), ('clinical_data', self.clinical_data)]:
            quant_cols = dataset.select_dtypes(include=[np.number]).columns.tolist()
            missing_counts = dataset[quant_cols].isnull().sum()
            missing_percent = (missing_counts / len(dataset)) * 100

            missing_info[dataset_name] = pd.DataFrame({
                'missing_count': missing_counts,
                'missing_percent': missing_percent
            }).sort_values(by='missing_percent', ascending=False)

        return missing_info
    
    def test_gaussian(self) -> Dict[str, Any]:
        """
        Tests for Gaussian distribution in quantitative variables of molecular and clinical datasets.
        """
        from scipy.stats import shapiro

        gaussian_info = {}

        for dataset_name, dataset in [('molecular_data', self.molecular_data), ('clinical_data', self.clinical_data)]:
            quant_cols = dataset.select_dtypes(include=[np.number]).columns.tolist()
            gaussian_results = {}

            for col in quant_cols:
                stat, p_value = shapiro(dataset[col].dropna())
                gaussian_results[col] = {
                    'statistic': stat,
                    'p_value': p_value,
                    'is_gaussian': p_value > 0.05
                }

            gaussian_info[dataset_name] = pd.DataFrame.from_dict(gaussian_results, orient='index')

        return gaussian_info
    
    def qqplot_quant_var(self, vars:list[str]) -> None:
        """
        Generates Q-Q plots for quantitative variables in molecular and clinical datasets to evaluate normal distributions.
        """
        import scipy.stats as stats
        molecular_col = self.molecular_data.columns
        clinical_col = self.clinical_data.columns
        for var in vars:
            if var in molecular_col:
                plt.figure(figsize=(6, 6))
                stats.probplot(self.molecular_data[var].dropna(), dist="norm", plot=plt)
                plt.title(f"Q-Q Plot for {var} in Molecular data")
                plt.show()
            else :
                plt.figure(figsize=(6, 6))
                stats.probplot(self.clinical_data[var].dropna(), dist="norm", plot=plt)
                plt.title(f"Q-Q Plot for {var} in Clinical data")
                plt.show()
                
    def kernel_density_plot(self,var: str) -> None:
        """
        Plots Kernel Density Estimation for a specific quantitative variable in molecular or clinical datasets.
        """
        for dataset_name, dataset in [('molecular_data', self.molecular_data), ('clinical_data', self.clinical_data)]:
            if var in dataset.columns:
                plt.figure(figsize=(8, 5))
                dataset[var].plot(kind='kde')
                plt.title(f"Kernel Density Estimation for {var} in {dataset_name}")
                plt.xlabel(var)
                plt.ylabel('Density')
                plt.show()
                
    def estimate_gaussian_params(self, var: list[str]) -> Dict[str, any]:
        """
        Estimates mean and standard deviation for specified quantitative variables in molecular and clinical datasets.
        """
        params: dict[str, any] = {}

        for dataset_name, dataset in [('molecular_data', self.molecular_data), ('clinical_data', self.clinical_data)]:
            for col in var:
                if col in dataset.columns:
                    mean: np.float64 = dataset[col].mean()
                    std: np.float64 = dataset[col].std()
                    params[f"{dataset_name}_{col}"] = {'mean': mean, 'std': std}

        return params

    def simulate_values(self, var: str, n: int) -> pd.Series:
        """
        Simulates new values for a specified quantitative variable based on estimated Gaussian parameters.
        """
        params = self.estimate_gaussian_params([var])
        simulated_values = {}

        for key, value in params.items():
            mean = value['mean']
            std = value['std']
            key = key.replace("clinical_data_", "").replace("molecular_data_", "")
            simulated_values[key] = np.random.normal(loc=mean, scale=std, size=n)

        return simulated_values
    
    def add_danger(self):
        self.molecular_data["DANGER"] = self.molecular_data["VAF"] * self.molecular_data["DEPTH"]

    def aggregate_genes(self):
        """
        Groups genes  by summing danger 
        
        """
        self.merge_tables()
        
        merged_data = self.merged_table[["ID","CHR","GENE","DANGER","EFFECT"]].copy()
        gene_train_dummies = pd.get_dummies(merged_data, columns=['GENE','CHR','EFFECT'])

        gene_train_F = gene_train_dummies.groupby(['ID']).agg({
        'DANGER': 'sum',  # Sum DEPTH values
        **{col: 'max' for col in gene_train_dummies.columns if col not in ['ID', 'OS_YEARS', 'OS_STATUS', 'DANGER']}
        }).reset_index()
        self.temporary_annotation = "Aggregation gene"
        self.temporary = gene_train_F 
        
    def get_frequency(self):
        if self.temporary_annotation != "Aggregation gene":
            raise ValueError("La table de traitement temporaire n'est plus celle d'aggregations des genes")
        
        data_without_id = self.temporary.drop(columns=["ID"])
        
        # Calculer la fréquence des valeurs positives (non-nulles) pour chaque variable
        positive_frequencies = ((data_without_id > 0).sum() / data_without_id.shape[0]) * 100
    
        # Filtrer les variables avec fréquence > 10% ET qui ont des valeurs positives
        frequent_positive_vars = positive_frequencies[positive_frequencies > 10].sort_values(ascending=False)
        
        return frequent_positive_vars

def show_frequency_table(data,frequent_dummies):
    
        data_without_id = data.temporary.drop(columns=["ID"])
        print(f"Variables avec valeurs positives > 10% des cas: {len(frequent_dummies)}")
        print(f"Total de variables (avec ID): {len(data_without_id.columns)}")
        print(f"Réduction: {len(data_without_id.columns) - len(frequent_dummies)} variables supprimées")
        
        # Analyse détaillée
        stats_df = pd.DataFrame({
            'Variable': frequent_dummies.index,
            'Positive_Frequency_%': frequent_dummies.values,
            'Total_Sum': data_without_id[frequent_dummies.index].sum().values,
            'Max_Value': data_without_id[frequent_dummies.index].max().values
        }).round(2)
        
        print(f"\nTop 20 variables les plus fréquentes:")
        print(stats_df.head(20).to_string())
        
def correlation(data,interesting_vars):
    liste = interesting_vars.index.to_list()
    liste.append("OS_YEARS")
    return data.temporary.merge(data.target_data, on = "ID", how="left")[liste].corr()

def plot_correlation(correlation,target):

    # Get correlations with OS_YEARS (excluding OS_YEARS itself)
    correlation_target = correlation[target].drop(target).sort_values(key=abs, ascending=False)
    
    # Take top 20 strongest correlations
    top_correlations = correlation_target.head(20)
    
    # Create the plot
    plt.figure(figsize=(15, 10))
    colors = ['red' if x < 0 else 'blue' for x in top_correlations.values]
    bars = plt.barh(range(len(top_correlations)), top_correlations.values, color=colors, alpha=0.7)
    
    # Customize the plot
    plt.yticks(range(len(top_correlations)), top_correlations.index, fontsize=10)
    plt.xlabel(f'Corrélation avec {target}', fontsize=12)
    plt.title(f'Top 20 des Corrélations avec {target}', fontsize=14, fontweight='bold')
    plt.grid(axis='x', alpha=0.3)
    plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    
    # Add values on bars
    for i, (bar, value) in enumerate(zip(bars, top_correlations.values)):
        plt.text(value + (0.01 if value > 0 else -0.01), i, f'{value:.3f}', 
                va='center', ha='left' if value > 0 else 'right', fontsize=9)
    
    plt.tight_layout()
    plt.show()
    
    # Print statistics
    print(f"=== Corrélations avec {target} ===")
    print(f"Nombre total de variables: {len(correlation_target)}")
    print(f"Corrélation moyenne: {correlation_target.mean():.3f}")
    print(f"Corrélation médiane: {correlation_target.median():.3f}")
    print(f"Corrélation max (absolue): {correlation_target.abs().max():.3f}")
    
    strong_correlations = correlation_target[abs(correlation_target) > 0.1]
    print(f"\nCorrélations fortes (|r| > 0.1): {len(strong_correlations)} variables")
    
    if len(strong_correlations) > 0:
        for var, corr in strong_correlations.head(10).items():
            print(f"  {var}: {corr:.3f}")
    
    return correlation_target

def generate_pairs(X, Y, max_pairs=None, mode="diff", random_state=42):
    n = len(X)
    indices = list(itertools.combinations(range(n), 2))

    if max_pairs and max_pairs < len(indices):
        random.seed(random_state)
        indices = random.sample(indices, max_pairs)

    X_pairs = []
    y_pairs = []
    non_comparable = []
    for i, j in indices:
        if Y.iloc[i][1] >= Y.iloc[j][1]:
            if Y.iloc[i][0] < Y.iloc[j][0] or (Y.iloc[i][0] == Y.iloc[j][0] == 1):
                label = int(Y.iloc[i][1] > Y.iloc[j][1])
                xi, xj = X.iloc[i].values, X.iloc[j].values
                if mode == "diff":
                    feat = xi - xj
                elif mode == "concat":
                    feat = np.concatenate([xi, xj])
                label = int(Y.iloc[i][1] < Y.iloc[j][1])
                xi, xj = X.iloc[i].values, X.iloc[j].values
                if mode == "diff":
                    feat = xj - xi
                elif mode == "concat":
                    feat = np.concatenate([xj, xi])
                else:
                    raise ValueError("mode must be 'diff' or 'concat'")

                X_pairs.append(feat)
                y_pairs.append(label)
        if Y.iloc[i][1] <= Y.iloc[j][1]:
            if Y.iloc[i][0] > Y.iloc[j][0] or (Y.iloc[i][0] == Y.iloc[j][0] == 1):
                label = int(Y.iloc[i][1] > Y.iloc[j][1])
                xi, xj = X.iloc[i].values, X.iloc[j].values
                if mode == "diff":
                    feat = xi - xj
                elif mode == "concat":
                    feat = np.concatenate([xi, xj])
                label = int(Y.iloc[i][1] < Y.iloc[j][1])
                xi, xj = X.iloc[i].values, X.iloc[j].values
                if mode == "diff":
                    feat = xj - xi
                elif mode == "concat":
                    feat = np.concatenate([xj, xi])
                else:
                    raise ValueError("mode must be 'diff' or 'concat'")

                X_pairs.append(feat)
                y_pairs.append(label)
        else : 
            non_comparable.append((i,j))
    return np.array(X_pairs), np.array(y_pairs)

# =============================================================
# Implementation example
# =============================================================

if __name__ == "__main__":
    
    Xtrain_molecular: str = r"C:\Users\mouad\Desktop\ISFA\M2\QRT\Initial data\X_train\molecular_train.csv"
    Xtrain_clinical: str = r"C:\Users\mouad\Desktop\ISFA\M2\QRT\Initial data\X_train\clinical_train.csv"
    target_train: str = r"C:\Users\mouad\Desktop\ISFA\M2\QRT\Initial data\target_train.csv"

    data_train_mol: pd.DataFrame = pd.read_csv(Xtrain_molecular)
    data_train_cli: pd.DataFrame = pd.read_csv(Xtrain_clinical)
    target_df: pd.DataFrame = pd.read_csv(target_train)

    # Convert 'OS_YEARS' to numeric, forcing errors to NaN 

    target_df['OS_YEARS'] = pd.to_numeric(target_df['OS_YEARS'], errors='coerce')

    # Ensure 'OS_STATUS' is boolean
    target_df['OS_STATUS'] = target_df['OS_STATUS'].astype(bool)
    print(f" Shape des données moléculaires {data_train_mol.shape} \n shape des données cliniques {data_train_cli.shape} \n shape des données cibles {target_df.shape}")

    b = DataProcessor(data_train_mol, data_train_cli, target_df)
    
    gaussian_var = ["HB","DEPTH"]
    params = b.estimate_gaussian_params(gaussian_var)
    print("Estimated Gaussian Parameters:")
    for key, value in params.items():
        print(f"  {key}: Mean = {value['mean']}, Std = {value['std']}")

    valeurs_manquantes = b.simulate_values("HB",100)
    
    b.add_danger()
    b.aggregate_genes()


