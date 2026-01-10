import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from typing import Optional, Dict, Any
import warnings
import itertools
import seaborn as sns
import random
import re

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
    
    """
    Data handling methods
    """
    def merge_tables(self, cli = None, mol = None) -> None:
        """
        Merges molecular, clinical, and target data on the 'ID' column.
        """
        self.merged_table = self.molecular_data.merge(self.clinical_data,on="ID",how="left"
                                                                ).merge(self.target_data,on="ID",how="left")    
    
    def remove_columns(self, columns: list, data_type: str = "molecular") -> None:
        """
        Removes specified columns from either molecular or clinical data.

        :param columns: List of column names to remove.
        :param data_type: The type of data to remove columns from ('molecular' or 'clinical').
        :raises ValueError: If data_type is not 'molecular' or 'clinical'.
        :raises ValueError: If any column doesn't exist in the specified data.
        """
        if data_type.lower() == "molecular":
            for col in columns:
                if col not in self.molecular_data.columns:
                    raise ValueError(f"Column '{col}' not found in molecular data.")
            self.molecular_data = self.molecular_data.drop(columns=columns)
        elif data_type.lower() == "clinical":
            for col in columns:
                if col not in self.clinical_data.columns:
                    raise ValueError(f"Column '{col}' not found in clinical data.")
            self.clinical_data = self.clinical_data.drop(columns=columns)
        else:
            raise ValueError("data_type must be 'molecular' or 'clinical'.")
    
    def save_data(self, file_path: str = "data/treated_data.csv") -> pd.DataFrame:
        """
        Saves the treated data to a CSV file in the 'data' directory and returns it.

        :param file_path: The file path where the treated data will be saved.
        :return: The treated data as a DataFrame.
        """
        if self.temporary is None:
            raise ValueError("No temporary data to save. Process data before saving.")

        self.treated_data = self.temporary
        self.temporary = None
        self.treated_data.to_csv(file_path, index=False)
        warnings.warn("The temporary attribute dataframe has been deleted, refer to treated data.", FutureWarning)
        return self.treated_data
    
    
    """ 
    Statistical overview methods
    """

    def show(self) -> None:
        """
        Display the current format of the Data, if merged table is not created, show individual dataframes.
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

    def get_data_shapes(self) -> tuple:
            """
            Returns the shapes of the molecular, clinical, and target datasets.
            """
            return (f"the molecular data shape: {self.molecular_data.shape}, clinical data shape: {self.clinical_data.shape}, target data shape: {self.target_data.shape}")
    
    def get_summary_statistics(self) -> Dict[str, pd.DataFrame]:
        """
        Returns summary statistics for molecular, clinical, and target datasets.
        """
        return {
            "molecular_data": self.molecular_data.describe(),
            "clinical_data": self.clinical_data.describe(),
            "target_data": self.target_data.describe()
        }


    def get_missing_values(self) -> Dict[str, pd.DataFrame]:
        """
        Returns the percentage of missing values for each column in the datasets.
        """
        return {
            "molecular_data": self.molecular_data.isnull().mean() * 100,
            "clinical_data": self.clinical_data.isnull().mean() * 100,
            "target_data": self.target_data.isnull().mean() * 100
        }

    def get_column_types(self) -> Dict[str, pd.Series]:
        """
        Returns the data types of columns in each dataset.
        """
        return {
            "molecular_data": self.molecular_data.dtypes,
            "clinical_data": self.clinical_data.dtypes,
            "target_data": self.target_data.dtypes
        } 
    def get_unique_values(self) -> Dict[str, pd.Series]:
        """
        Returns the number of unique values for each column in the datasets.
        """
        return {
            "molecular_data": self.molecular_data.nunique(),
            "clinical_data": self.clinical_data.nunique(),
            "target_data": self.target_data.nunique()
        }


    """
    Feature engineering methods
    """
    def add_danger(self):
        """
        Adds a 'DANGER' column to the molecular data by multiplying 'VAF' and 'DEPTH'.
        """
        self.molecular_data["DANGER"] = self.molecular_data["VAF"] * self.molecular_data["DEPTH"]

    def aggregate_genes(self):

        """
        Groups genes  by summing danger and creating dummies for categorical variables.
        """
        self.merge_tables()
        
        merged_data = self.merged_table[["ID","CHR","GENE","DANGER","EFFECT"]].copy()
        gene_train_dummies = pd.get_dummies(merged_data, columns=['GENE','CHR','EFFECT'])

        gene_train_F = gene_train_dummies.groupby(['ID']).agg({
        'DANGER': 'sum',  # Sum DEPTH values
        **{col: 'max' for col in gene_train_dummies.columns if col not in ['ID', 'OS_YEARS', 'OS_STATUS', 'DANGER']}
        }).reset_index()
        self.temporary_annotation = "Aggregation gene"
        self.temporary = gene_train_F.merge(self.clinical_data,on="ID",how="left"
                                                                ).merge(self.target_data,on="ID",how="left")
 

    def classify_cytogenetics(self, text: str) -> str:
        """
        Classifies cytogenetics labels into categories: normal, defavorable, or intermediaire.

        :param text: The cytogenetics text to classify.
        :return: The classification category.
        """
        t = str(text).lower().replace(" ", "")

        # ---------- 1) CARYOTYPE NORMAL ----------
        if ("46,xx" in t or "46,xy" in t) and not re.search(r"[\+\-]|del|t\(|inv\(", t):
            return "normal"

        # ---------- 2) DÉFAVORABLE (ELN 2022) ----------
        unfavorable_patterns = [
            r"-5", r"del\(5", r"5q",     # chromosome 5
            r"-7", r"del\(7", r"7q",     # chromosome 7
            r"t\(6;9",                   # DEK-NUP214
            r"t\(9;22",                  # BCR-ABL1
            r"inv\(3", r"t\(3;3",        # MECOM/EVI1
            r"t\(8;16",                  # KAT6A-CREBBP
            r"11q23", r"kmt2a", r"mll",  # KMT2A rearrangements
            r"i\(17q", r"del\(17p",      # chromosome 17 anomalies
        ]
        if any(re.search(p, t) for p in unfavorable_patterns):
            return "defavorable"

        # Caryotype complexe (>= 3 anomalies)
        if len(re.findall(r"[\+\-]", t)) >= 3:
            return "defavorable"

        # ---------- 3) INTERMÉDIAIRE ----------
        intermediate_patterns = [
            r"t\(9;11",      # KMT2A::MLLT3
            r"\+8", r"trisomy8",
        ]
        if any(re.search(p, t) for p in intermediate_patterns):
            return "intermediaire"

        # Par défaut (non normal, non défavorable)
        return "intermediaire"

    def apply_cytogenetics_classification(self, column_name: str = "CYTOGENETICS") -> pd.DataFrame:
        """
        Applies the cytogenetics classification to the specified column in the clinical data.

        :param column_name: The name of the column containing cytogenetics data.
        :return: The updated clinical DataFrame with a new 'CYTO_CATEGORY' column.
        """
        if column_name not in self.clinical_data.columns:
            raise ValueError(f"Column '{column_name}' not found in clinical data.")

        self.clinical_data["CYTO_CATEGORY"] = self.clinical_data[column_name].apply(self.classify_cytogenetics)
        self.clinical_data.drop("CYTOGENETICS", axis=1, inplace=True)
        return self.clinical_data

    def create_cytogenetics_dummies(self, column_name: str = "CYTO_CATEGORY", drop_original: bool = True) -> pd.DataFrame:
        """
        Creates dummy variables from the cytogenetics classification column.

        :param column_name: The name of the cytogenetics classification column (default: 'CYTO_CATEGORY').
        :param drop_original: Whether to drop the original classification column after creating dummies (default: True).
        :return: The updated clinical DataFrame with dummy variables.
        """
        if column_name not in self.clinical_data.columns:
            raise ValueError(f"Column '{column_name}' not found in clinical data.")

        # Create dummy variables with prefix to avoid column name conflicts
        cyto_dummies = pd.get_dummies(self.clinical_data[column_name], prefix="CYTO")
        
        # Concatenate the dummy variables to the clinical data
        self.clinical_data = pd.concat([self.clinical_data, cyto_dummies], axis=1)
        
        # Drop the original column if specified
        if drop_original:
            self.clinical_data.drop(column_name, axis=1, inplace=True)
        
        return self.clinical_data

