import pandas as pd
from typing import Dict
import warnings
import re

# =============================================================
# Module for Data Processing Class 
# =============================================================


def _format_chromosome(value) -> str:
    """Return a chromosome label as a clean string ("1".."22", "X"), or "NA" when missing."""
    if pd.isna(value):
        return "NA"
    try:
        return str(int(float(value)))
    except (TypeError, ValueError):
        return str(value).upper()


class DataProcessor:
  
    def __init__(self, molecular_data: pd.DataFrame, clinical_data: pd.DataFrame, target_data: pd.DataFrame = None) -> None:
        """
        Initialize the DataProcessor with molecular, clinical, and (optionally) target data.
        The target is optional so that the same processing can be applied to the test set.
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
        self.merged_table = self.molecular_data.merge(self.clinical_data, on="ID", how="left")
        if self.target_data is not None:
            self.merged_table = self.merged_table.merge(self.target_data, on="ID", how="left")
    
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
    
    def save_data(self, file_path: str = "data/treated_data.csv", cols=None) -> pd.DataFrame:
        """
        Saves the treated data to a CSV file in the 'data' directory and returns it.

        :param file_path: The file path where the treated data will be saved.
        :return: The treated data as a DataFrame.
        """
        if self.temporary is None:
            raise ValueError("No temporary data to save. Process data before saving.")

        if cols is not None:
            self.treated_data = self.temporary[cols]
        else:
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
        Adds a 'DANGER' column to the molecular data: the variant allele fraction (VAF) of each
        mutation, summed per patient in aggregate_genes (a clonal-burden score).

        An earlier version used VAF x DEPTH. DEPTH is the sequencing depth, a technical quantity
        that depends on the centre's sequencing panel (median 621 to 1,496 reads across the
        training centres, 493 in the challenge test centre), so VAF x DEPTH mostly encoded the
        centre and shifted for patients from a new centre.
        """
        self.molecular_data["DANGER"] = self.molecular_data["VAF"].fillna(0.0)

    def aggregate_genes(self):
        """
        Aggregate the mutations per patient (sum of DANGER, binary indicators for genes,
        chromosomes and effects) and join them to the clinical table.

        The join starts from the clinical table, so patients without any recorded mutation are
        kept (DANGER = 0 and every indicator False) instead of being silently dropped.
        """
        mol = self.molecular_data[["ID", "CHR", "GENE", "DANGER", "EFFECT"]].copy()
        # CHR is read as a mix of numbers and strings ("X"); use one string format
        mol["CHR"] = mol["CHR"].map(_format_chromosome)
        dummies = pd.get_dummies(mol, columns=["GENE", "CHR", "EFFECT"])

        indicator_cols = [c for c in dummies.columns if c not in ("ID", "DANGER")]
        grouped = dummies.groupby("ID")
        per_patient = pd.concat(
            [grouped["DANGER"].sum(), grouped[indicator_cols].max()], axis=1
        ).reset_index()

        features = self.clinical_data.merge(per_patient, on="ID", how="left")
        features["DANGER"] = features["DANGER"].fillna(0.0)
        features[indicator_cols] = features[indicator_cols].fillna(False).astype(bool)

        if self.target_data is not None:
            features = features.merge(self.target_data, on="ID", how="left")

        self.temporary_annotation = "Aggregation gene"
        self.temporary = features

    @staticmethod
    def classify_cytogenetics(text) -> str:
        """
        Classifies a karyotype (ISCN string) into a risk category: normal, intermediate,
        adverse, or missing when no karyotype is available.

        :param text: The cytogenetics text to classify.
        :return: The classification category.
        """
        if pd.isna(text) or str(text).strip() == "":
            # A missing karyotype is not evidence of an intermediate-risk karyotype
            return "missing"

        t = str(text).lower().replace(" ", "")

        # ---------- 0) FREE-TEXT ENTRIES (some centres do not use ISCN) ----------
        if not re.search(r"\d", t):
            if t.startswith("normal"):
                return "normal"
            if t.startswith("complex"):
                return "adverse"            # complex karyotype is adverse in ELN 2022
            if "notexamined" in t or "notevaluated" in t or "nomitos" in t or "failed" in t:
                return "missing"

        # ---------- 1) NORMAL KARYOTYPE ----------
        # Every clone must be a plain 46,XX or 46,XY (e.g. "46,xy[20]" or "46,xx[18]/46,xx[2]");
        # any other notation (i(17q), der(), add(), +8, ...) is an abnormality.
        clones = [c for c in re.split(r"/+", t) if c]
        if clones and all(re.fullmatch(r"46,x[xy](\[\d+\])?", c) for c in clones):
            return "normal"

        # ---------- 2) ADVERSE (ELN 2022) ----------
        # Look-arounds stop e.g. "5q" from matching "15q" or "-7" from matching "-17"
        unfavorable_patterns = [
            r"-5(?!\d)", r"del\(5\)", r"(?<!\d)5q",   # chromosome 5
            r"-7(?!\d)", r"del\(7\)", r"(?<!\d)7q",   # chromosome 7
            r"t\(6;9\)",                             # DEK-NUP214
            r"t\(9;22\)",                            # BCR-ABL1
            r"inv\(3\)", r"t\(3;3\)",                 # MECOM/EVI1
            r"t\(8;16\)",                            # KAT6A-CREBBP
            r"11q23", r"kmt2a", r"mll",              # KMT2A rearrangements
            r"i\(17\)\(q", r"i\(17q", r"del\(17\)\(p", r"-17(?!\d)",  # chromosome 17 anomalies
        ]
        if any(re.search(p, t) for p in unfavorable_patterns):
            return "adverse"

        # Complex karyotype (>= 3 abnormalities)
        if len(re.findall(r"[\+\-]", t)) >= 3:
            return "adverse"

        # ---------- 3) INTERMEDIATE ----------
        # Everything else, including t(9;11) and trisomy 8
        return "intermediate"

    def apply_cytogenetics_classification(self, column_name: str = "CYTOGENETICS") -> pd.DataFrame:
        """
        Applies the cytogenetics classification to the specified column in the clinical data.

        :param column_name: The name of the column containing cytogenetics data.
        :return: The updated clinical DataFrame with a new 'CYTO_CATEGORY' column.
        """
        if column_name not in self.clinical_data.columns:
            raise ValueError(f"Column '{column_name}' not found in clinical data.")

        self.clinical_data["CYTO_CATEGORY"] = self.clinical_data[column_name].apply(self.classify_cytogenetics)
        self.clinical_data = self.clinical_data.drop(columns=[column_name])
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
            self.clinical_data = self.clinical_data.drop(columns=[column_name])
        
        return self.clinical_data



MOLECULAR_COLS_TO_REMOVE = ["START", "END", "REF", "ALT", "PROTEIN_CHANGE"]


def build_patient_features(clinical: pd.DataFrame, molecular: pd.DataFrame,
                           target: pd.DataFrame = None,
                           feature_columns: list = None) -> pd.DataFrame:
    """
    Apply the whole feature-engineering pipeline of 00_data_processing.ipynb to one dataset
    (one row per patient in `clinical`).

    :param feature_columns: if given (the features selected on the training patients), the
        result is restricted to ['ID'] + feature_columns in that order. Indicator columns that
        do not occur in this dataset are added with value 0, so the columns always match the
        ones the models were trained on.
    """
    proc = DataProcessor(molecular.copy(), clinical.copy(), target)
    proc.remove_columns([c for c in MOLECULAR_COLS_TO_REMOVE if c in proc.molecular_data.columns])
    proc.apply_cytogenetics_classification()
    proc.create_cytogenetics_dummies()
    proc.add_danger()
    proc.aggregate_genes()
    features = proc.temporary

    if feature_columns is not None:
        numeric = {"BM_BLAST", "WBC", "ANC", "MONOCYTES", "HB", "PLT", "DANGER"}
        missing = [c for c in feature_columns if c not in features.columns]
        for c in missing:
            if c in numeric:
                raise ValueError(f"Numeric feature '{c}' is missing from the data.")
        features = features.reindex(columns=["ID"] + list(feature_columns), fill_value=False)
    return features
