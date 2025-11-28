"""
Data Loading and Validation for Wearable Health Equity Analysis
Written by Cazzy Aporbo

This module handles loading real or synthetic datasets, validating schemas,
and preparing data for analysis and visualization.

Disclaimer:
    This code is for research and exploratory purposes only.
    It does not perform clinical diagnosis or personalize medical advice.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass


@dataclass
class SchemaDefinition:
    """
    Defines expected columns and types for each data bucket.
    """
    required_columns: List[str]
    optional_columns: List[str]
    column_types: Dict[str, str]
    categorical_values: Dict[str, List[str]]


# Schema definitions for each bucket
USAGE_SCHEMA = SchemaDefinition(
    required_columns=[
        "user_id",
        "age",
        "race_ethnicity",
        "income_bracket",
        "education_level",
        "geography",
        "insurance_type",
        "owns_wearable",
        "uses_wearable_for_health",
        "shares_data_with_provider"
    ],
    optional_columns=[],
    column_types={
        "user_id": "string",
        "age": "numeric",
        "race_ethnicity": "categorical",
        "income_bracket": "categorical",
        "education_level": "categorical",
        "geography": "categorical",
        "insurance_type": "categorical",
        "owns_wearable": "boolean",
        "uses_wearable_for_health": "boolean",
        "shares_data_with_provider": "boolean"
    },
    categorical_values={
        "race_ethnicity": ["white", "black", "hispanic_latino", "asian", "multiracial", "other"],
        "income_bracket": ["low", "lower_middle", "middle", "upper_middle", "high"],
        "education_level": ["less_than_hs", "high_school", "some_college", "bachelors", "graduate"],
        "geography": ["urban", "suburban", "rural"],
        "insurance_type": ["private", "medicare", "medicaid", "uninsured", "military_va"]
    }
)

DEVICE_SCHEMA = SchemaDefinition(
    required_columns=[
        "user_id",
        "device_brand",
        "device_model",
        "data_format",
        "available_metrics",
        "api_available",
        "ehr_export_supported",
        "platforms_integrated",
        "fragmentation_index"
    ],
    optional_columns=[],
    column_types={
        "user_id": "string",
        "device_brand": "categorical",
        "device_model": "string",
        "data_format": "categorical",
        "available_metrics": "string",
        "api_available": "boolean",
        "ehr_export_supported": "boolean",
        "platforms_integrated": "categorical",
        "fragmentation_index": "numeric"
    },
    categorical_values={
        "data_format": ["proprietary_json", "open_fhir", "csv_export"],
        "platforms_integrated": ["vendor_portal_only", "ehr_system", "third_party_aggregator"]
    }
)

QUALITY_SCHEMA = SchemaDefinition(
    required_columns=[
        "user_id",
        "skin_tone_category",
        "sensor_accuracy_score",
        "motion_context",
        "error_rate",
        "dropout_rate",
        "wear_time_hours_per_day",
        "willingness_to_share_score",
        "actually_shared_data"
    ],
    optional_columns=[],
    column_types={
        "user_id": "string",
        "skin_tone_category": "categorical",
        "sensor_accuracy_score": "numeric",
        "motion_context": "categorical",
        "error_rate": "numeric",
        "dropout_rate": "numeric",
        "wear_time_hours_per_day": "numeric",
        "willingness_to_share_score": "numeric",
        "actually_shared_data": "boolean"
    },
    categorical_values={
        "skin_tone_category": ["type_i_ii", "type_iii_iv", "type_v_vi"],
        "motion_context": ["rest", "walking", "running"]
    }
)


class DataValidationError(Exception):
    """Raised when data fails schema validation."""
    pass


class DataLoader:
    """
    Handles loading and validation of wearable health equity datasets.
    
    Supports loading from CSV, Parquet, or synthetic generation.
    Validates schema compliance and reports data quality issues.
    """
    
    def __init__(self, verbose: bool = True):
        """
        Initialize the data loader.
        
        Args:
            verbose: If True, print validation messages
        """
        self.verbose = verbose
        self.validation_warnings: List[str] = []
    
    def _log(self, message: str):
        """Print message if verbose mode is enabled."""
        if self.verbose:
            print(message)
    
    def validate_schema(
        self,
        df: pd.DataFrame,
        schema: SchemaDefinition,
        bucket_name: str
    ) -> bool:
        """
        Validate a DataFrame against its expected schema.
        
        Args:
            df: DataFrame to validate
            schema: SchemaDefinition for the bucket
            bucket_name: Human readable name for error messages
        
        Returns:
            True if validation passes
        
        Raises:
            DataValidationError: If required columns are missing
        """
        self.validation_warnings = []
        
        # Check required columns
        missing = set(schema.required_columns) - set(df.columns)
        if missing:
            raise DataValidationError(
                f"Missing required columns in {bucket_name}: {missing}"
            )
        
        # Check data types
        for col, expected_type in schema.column_types.items():
            if col not in df.columns:
                continue
            
            if expected_type == "numeric":
                if not pd.api.types.is_numeric_dtype(df[col]):
                    self.validation_warnings.append(
                        f"Column {col} expected numeric, found {df[col].dtype}"
                    )
            
            elif expected_type == "boolean":
                if not pd.api.types.is_bool_dtype(df[col]):
                    # Check if can be coerced
                    unique_vals = df[col].dropna().unique()
                    if not all(v in [True, False, 0, 1, "True", "False"] for v in unique_vals):
                        self.validation_warnings.append(
                            f"Column {col} expected boolean, found non-boolean values"
                        )
            
            elif expected_type == "categorical":
                if col in schema.categorical_values:
                    valid_values = set(schema.categorical_values[col])
                    actual_values = set(df[col].dropna().unique())
                    invalid = actual_values - valid_values
                    if invalid:
                        self.validation_warnings.append(
                            f"Column {col} has unexpected values: {invalid}"
                        )
        
        # Check for excessive missing values
        for col in schema.required_columns:
            missing_pct = df[col].isna().mean()
            if missing_pct > 0.1:
                self.validation_warnings.append(
                    f"Column {col} has {missing_pct:.1%} missing values"
                )
        
        # Report warnings
        for warning in self.validation_warnings:
            self._log(f"  Warning: {warning}")
        
        return True
    
    def load_csv(
        self,
        path: Union[str, Path],
        bucket_type: str
    ) -> pd.DataFrame:
        """
        Load data from a CSV file and validate schema.
        
        Args:
            path: Path to CSV file
            bucket_type: One of 'usage', 'device', 'quality'
        
        Returns:
            Validated DataFrame
        """
        self._log(f"Loading {bucket_type} data from {path}")
        
        df = pd.read_csv(path)
        
        schema_map = {
            "usage": USAGE_SCHEMA,
            "device": DEVICE_SCHEMA,
            "quality": QUALITY_SCHEMA
        }
        
        if bucket_type not in schema_map:
            raise ValueError(f"Unknown bucket type: {bucket_type}")
        
        self.validate_schema(df, schema_map[bucket_type], bucket_type)
        self._log(f"  Loaded {len(df)} rows")
        
        return df
    
    def load_parquet(
        self,
        path: Union[str, Path],
        bucket_type: str
    ) -> pd.DataFrame:
        """
        Load data from a Parquet file and validate schema.
        
        Args:
            path: Path to Parquet file
            bucket_type: One of 'usage', 'device', 'quality'
        
        Returns:
            Validated DataFrame
        """
        self._log(f"Loading {bucket_type} data from {path}")
        
        df = pd.read_parquet(path)
        
        schema_map = {
            "usage": USAGE_SCHEMA,
            "device": DEVICE_SCHEMA,
            "quality": QUALITY_SCHEMA
        }
        
        if bucket_type not in schema_map:
            raise ValueError(f"Unknown bucket type: {bucket_type}")
        
        self.validate_schema(df, schema_map[bucket_type], bucket_type)
        self._log(f"  Loaded {len(df)} rows")
        
        return df
    
    def load_all_from_directory(
        self,
        directory: Union[str, Path],
        file_format: str = "csv"
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load all three buckets from a directory.
        
        Expects files named:
            individual_population_usage.{format}
            device_stream_fragmentation.{format}
            equity_bias_quality.{format}
        
        Args:
            directory: Path to directory containing data files
            file_format: Either 'csv' or 'parquet'
        
        Returns:
            Tuple of (usage_df, device_df, quality_df)
        """
        directory = Path(directory)
        
        expected_files = {
            "usage": f"individual_population_usage.{file_format}",
            "device": f"device_stream_fragmentation.{file_format}",
            "quality": f"equity_bias_quality.{file_format}"
        }
        
        load_func = self.load_csv if file_format == "csv" else self.load_parquet
        
        dfs = {}
        for bucket, filename in expected_files.items():
            filepath = directory / filename
            if not filepath.exists():
                raise FileNotFoundError(f"Expected file not found: {filepath}")
            dfs[bucket] = load_func(filepath, bucket)
        
        return dfs["usage"], dfs["device"], dfs["quality"]


def compute_fragmentation_index(
    api_available: bool,
    ehr_export_supported: bool,
    data_format: str,
    platforms_integrated: str,
    available_metrics: str
) -> float:
    """
    Compute a fragmentation index from raw device characteristics.
    
    This function can be used when a fragmentation_index is not
    provided in the source data. Higher values indicate more
    fragmentation and less interoperability.
    
    Args:
        api_available: Whether an API exists for data access
        ehr_export_supported: Whether EHR export is supported
        data_format: Data format type
        platforms_integrated: Integration status
        available_metrics: Comma separated list of available metrics
    
    Returns:
        Fragmentation index between 0 and 1
    """
    score = 0.5  # Start at neutral
    
    # API availability reduces fragmentation
    if api_available:
        score -= 0.15
    else:
        score += 0.10
    
    # EHR export reduces fragmentation significantly
    if ehr_export_supported:
        score -= 0.20
    else:
        score += 0.10
    
    # Data format impact
    format_impact = {
        "open_fhir": -0.15,
        "csv_export": -0.05,
        "proprietary_json": 0.10
    }
    score += format_impact.get(data_format, 0.05)
    
    # Platform integration impact
    platform_impact = {
        "ehr_system": -0.15,
        "third_party_aggregator": -0.05,
        "vendor_portal_only": 0.15
    }
    score += platform_impact.get(platforms_integrated, 0.05)
    
    # More metrics generally means more complex data (slight increase)
    n_metrics = len(available_metrics.split(",")) if available_metrics else 0
    if n_metrics > 5:
        score += 0.05
    
    return max(0.0, min(1.0, score))


def merge_all_buckets(
    usage_df: pd.DataFrame,
    device_df: pd.DataFrame,
    quality_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge all three data buckets into a single analysis DataFrame.
    
    Only includes users who own wearables and appear in all buckets.
    
    Args:
        usage_df: Individual and population usage data
        device_df: Device stream and fragmentation data
        quality_df: Equity, bias, and quality data
    
    Returns:
        Merged DataFrame with all columns
    """
    merged = usage_df.merge(device_df, on="user_id", how="inner")
    merged = merged.merge(quality_df, on="user_id", how="inner")
    
    return merged


def get_data_summary(
    usage_df: pd.DataFrame,
    device_df: pd.DataFrame,
    quality_df: pd.DataFrame
) -> Dict:
    """
    Generate a summary dictionary of the datasets.
    
    Useful for report generation and dashboard metadata.
    
    Args:
        usage_df: Individual and population usage data
        device_df: Device stream and fragmentation data
        quality_df: Equity, bias, and quality data
    
    Returns:
        Dictionary with summary statistics
    """
    summary = {
        "total_population": len(usage_df),
        "wearable_owners": usage_df["owns_wearable"].sum(),
        "ownership_rate": usage_df["owns_wearable"].mean(),
        "health_usage_rate_among_owners": (
            usage_df.loc[usage_df["owns_wearable"], "uses_wearable_for_health"].mean()
        ),
        "data_sharing_rate_among_health_users": (
            usage_df.loc[usage_df["uses_wearable_for_health"], "shares_data_with_provider"].mean()
        ),
        "unique_device_brands": device_df["device_brand"].nunique(),
        "mean_fragmentation_index": device_df["fragmentation_index"].mean(),
        "ehr_export_rate": device_df["ehr_export_supported"].mean(),
        "mean_sensor_accuracy": quality_df["sensor_accuracy_score"].mean(),
        "mean_dropout_rate": quality_df["dropout_rate"].mean(),
        "demographic_breakdown": {
            "income_brackets": usage_df["income_bracket"].value_counts().to_dict(),
            "race_ethnicity": usage_df["race_ethnicity"].value_counts().to_dict(),
            "geography": usage_df["geography"].value_counts().to_dict()
        }
    }
    
    return summary
