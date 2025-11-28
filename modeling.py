"""
Modeling and Analytics for Wearable Health Equity Analysis
Written by Cazzy Aporbo

This module implements statistical models and fairness metrics for
analyzing equity, bias, and data quality in wearable health data.

Disclaimer:
    This code is for research and exploratory purposes only.
    It does not perform clinical diagnosis or personalize medical advice.
    Model outputs should be interpreted with appropriate caution and
    validated against domain expertise.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import cross_val_predict
from sklearn.calibration import calibration_curve
import warnings


@dataclass
class ModelResults:
    """
    Container for model outputs and diagnostics.
    """
    model_name: str
    coefficients: Dict[str, float]
    predicted_probabilities: np.ndarray
    feature_names: List[str]
    accuracy: float
    auc_roc: Optional[float] = None
    calibration_data: Optional[Dict] = None


@dataclass
class FairnessMetrics:
    """
    Container for fairness and disparity metrics.
    """
    metric_name: str
    group_rates: Dict[str, float]
    disparity_ratio: float
    absolute_gap: float
    demographic_parity_difference: float
    reference_group: str
    disadvantaged_group: str


class EquityModelingPipeline:
    """
    Pipeline for building and evaluating equity-focused models.
    
    Implements logistic regression models for ownership, health usage,
    and EHR linkability prediction, along with fairness diagnostics.
    """
    
    def __init__(self, random_seed: int = 42):
        """
        Initialize the modeling pipeline.
        
        Args:
            random_seed: Seed for reproducibility
        """
        self.random_seed = random_seed
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.scaler = StandardScaler()
        self.models: Dict[str, LogisticRegression] = {}
        self.results: Dict[str, ModelResults] = {}
    
    def _encode_categorical(
        self,
        df: pd.DataFrame,
        columns: List[str],
        fit: bool = True
    ) -> pd.DataFrame:
        """
        Encode categorical columns as numeric values.
        
        Args:
            df: Input DataFrame
            columns: List of columns to encode
            fit: If True, fit new encoders. If False, use existing.
        
        Returns:
            DataFrame with encoded columns
        """
        result = df.copy()
        
        for col in columns:
            if col not in result.columns:
                continue
            
            if fit or col not in self.label_encoders:
                # Fit a new encoder if requested or if column not seen before
                self.label_encoders[col] = LabelEncoder()
                result[col] = self.label_encoders[col].fit_transform(result[col].astype(str))
            else:
                result[col] = self.label_encoders[col].transform(result[col].astype(str))
        
        return result
    
    def prepare_ownership_features(
        self,
        usage_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Prepare features for wearable ownership prediction.
        
        Args:
            usage_df: Individual and population usage DataFrame
        
        Returns:
            Tuple of (feature_matrix, target_array)
        """
        feature_cols = [
            "age",
            "race_ethnicity",
            "income_bracket",
            "education_level",
            "geography",
            "insurance_type"
        ]
        
        X = usage_df[feature_cols].copy()
        y = usage_df["owns_wearable"].astype(int).values
        
        categorical_cols = [
            "race_ethnicity",
            "income_bracket",
            "education_level",
            "geography",
            "insurance_type"
        ]
        
        X = self._encode_categorical(X, categorical_cols, fit=True)
        
        # Scale numeric features
        X["age"] = self.scaler.fit_transform(X[["age"]])
        
        return X, y
    
    def fit_ownership_model(
        self,
        usage_df: pd.DataFrame
    ) -> ModelResults:
        """
        Fit logistic regression model for wearable ownership.
        
        Args:
            usage_df: Individual and population usage DataFrame
        
        Returns:
            ModelResults with coefficients and predictions
        """
        X, y = self.prepare_ownership_features(usage_df)
        
        model = LogisticRegression(
            random_state=self.random_seed,
            max_iter=500,
            solver="lbfgs"
        )
        model.fit(X, y)
        
        self.models["ownership"] = model
        
        # Get predicted probabilities using cross-validation for honest estimates
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            probs = cross_val_predict(
                model, X, y,
                cv=5,
                method="predict_proba"
            )[:, 1]
        
        # Compute accuracy
        accuracy = (model.predict(X) == y).mean()
        
        # Extract coefficients
        coef_dict = dict(zip(X.columns, model.coef_[0]))
        
        results = ModelResults(
            model_name="Wearable Ownership",
            coefficients=coef_dict,
            predicted_probabilities=probs,
            feature_names=list(X.columns),
            accuracy=accuracy
        )
        
        self.results["ownership"] = results
        return results
    
    def fit_health_usage_model(
        self,
        usage_df: pd.DataFrame
    ) -> ModelResults:
        """
        Fit logistic regression for health-focused wearable usage.
        
        Conditional on ownership, predicts health usage.
        
        Args:
            usage_df: Individual and population usage DataFrame
        
        Returns:
            ModelResults with coefficients and predictions
        """
        # Filter to owners only
        owners = usage_df[usage_df["owns_wearable"]].copy()
        
        feature_cols = [
            "age",
            "race_ethnicity",
            "income_bracket",
            "education_level",
            "geography",
            "insurance_type"
        ]
        
        X = owners[feature_cols].copy()
        y = owners["uses_wearable_for_health"].astype(int).values
        
        categorical_cols = [
            "race_ethnicity",
            "income_bracket",
            "education_level",
            "geography",
            "insurance_type"
        ]
        
        X = self._encode_categorical(X, categorical_cols, fit=False)
        X["age"] = self.scaler.transform(X[["age"]])
        
        model = LogisticRegression(
            random_state=self.random_seed,
            max_iter=500,
            solver="lbfgs"
        )
        model.fit(X, y)
        
        self.models["health_usage"] = model
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            probs = cross_val_predict(
                model, X, y,
                cv=5,
                method="predict_proba"
            )[:, 1]
        
        accuracy = (model.predict(X) == y).mean()
        coef_dict = dict(zip(X.columns, model.coef_[0]))
        
        results = ModelResults(
            model_name="Health Usage (among owners)",
            coefficients=coef_dict,
            predicted_probabilities=probs,
            feature_names=list(X.columns),
            accuracy=accuracy
        )
        
        self.results["health_usage"] = results
        return results
    
    def fit_ehr_linkability_model(
        self,
        merged_df: pd.DataFrame
    ) -> ModelResults:
        """
        Fit model predicting effective EHR linkability.
        
        Combines device features (API, export support) with user
        characteristics to estimate the probability that wearable
        data can be meaningfully connected to clinical records.
        
        Args:
            merged_df: Merged DataFrame with all three buckets
        
        Returns:
            ModelResults with coefficients and predictions
        """
        # Define EHR linkability as EHR export support AND actually sharing
        y = (
            merged_df["ehr_export_supported"] &
            merged_df["actually_shared_data"]
        ).astype(int).values
        
        feature_cols = [
            "age",
            "income_bracket",
            "race_ethnicity",
            "insurance_type",
            "geography",
            "fragmentation_index",
            "api_available",
            "platforms_integrated",
            "willingness_to_share_score"
        ]
        
        X = merged_df[feature_cols].copy()
        
        categorical_cols = [
            "income_bracket",
            "race_ethnicity",
            "insurance_type",
            "geography",
            "platforms_integrated"
        ]
        
        X = self._encode_categorical(X, categorical_cols, fit=False)
        X["api_available"] = X["api_available"].astype(int)
        
        # Scale numeric features
        numeric_cols = ["age", "fragmentation_index", "willingness_to_share_score"]
        X[numeric_cols] = StandardScaler().fit_transform(X[numeric_cols])
        
        model = LogisticRegression(
            random_state=self.random_seed,
            max_iter=500,
            solver="lbfgs"
        )
        model.fit(X, y)
        
        self.models["ehr_linkability"] = model
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            probs = cross_val_predict(
                model, X, y,
                cv=5,
                method="predict_proba"
            )[:, 1]
        
        accuracy = (model.predict(X) == y).mean()
        coef_dict = dict(zip(X.columns, model.coef_[0]))
        
        results = ModelResults(
            model_name="EHR Linkability",
            coefficients=coef_dict,
            predicted_probabilities=probs,
            feature_names=list(X.columns),
            accuracy=accuracy
        )
        
        self.results["ehr_linkability"] = results
        return results
    
    def compute_group_rates(
        self,
        df: pd.DataFrame,
        outcome_col: str,
        group_col: str
    ) -> Dict[str, float]:
        """
        Compute outcome rates by group.
        
        Args:
            df: DataFrame with outcome and group columns
            outcome_col: Name of the outcome column
            group_col: Name of the grouping column
        
        Returns:
            Dictionary mapping group names to rates
        """
        rates = df.groupby(group_col)[outcome_col].mean().to_dict()
        return rates
    
    def compute_fairness_metrics(
        self,
        df: pd.DataFrame,
        outcome_col: str,
        group_col: str,
        reference_group: Optional[str] = None
    ) -> FairnessMetrics:
        """
        Compute fairness metrics comparing outcome rates across groups.
        
        Args:
            df: DataFrame with outcome and group columns
            outcome_col: Name of the outcome column
            group_col: Name of the grouping column
            reference_group: Group to use as reference (defaults to highest rate)
        
        Returns:
            FairnessMetrics with disparity calculations
        """
        rates = self.compute_group_rates(df, outcome_col, group_col)
        
        if reference_group is None:
            reference_group = max(rates, key=rates.get)
        
        disadvantaged_group = min(rates, key=rates.get)
        
        ref_rate = rates[reference_group]
        dis_rate = rates[disadvantaged_group]
        
        # Avoid division by zero
        disparity_ratio = dis_rate / ref_rate if ref_rate > 0 else 0.0
        absolute_gap = ref_rate - dis_rate
        
        # Demographic parity difference: max rate - min rate
        all_rates = list(rates.values())
        dpd = max(all_rates) - min(all_rates)
        
        return FairnessMetrics(
            metric_name=f"{outcome_col} by {group_col}",
            group_rates=rates,
            disparity_ratio=disparity_ratio,
            absolute_gap=absolute_gap,
            demographic_parity_difference=dpd,
            reference_group=reference_group,
            disadvantaged_group=disadvantaged_group
        )
    
    def compute_accuracy_disparity(
        self,
        quality_df: pd.DataFrame,
        device_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Compute sensor accuracy disparities by skin tone and device.
        
        Args:
            quality_df: Equity, bias, and quality DataFrame
            device_df: Device stream DataFrame
        
        Returns:
            Dictionary with disparity statistics
        """
        merged = quality_df.merge(
            device_df[["user_id", "device_brand"]],
            on="user_id"
        )
        
        # By skin tone
        skin_tone_accuracy = merged.groupby("skin_tone_category")["sensor_accuracy_score"].agg([
            "mean", "std", "count"
        ]).to_dict("index")
        
        # By brand
        brand_accuracy = merged.groupby("device_brand")["sensor_accuracy_score"].agg([
            "mean", "std", "count"
        ]).to_dict("index")
        
        # Cross tabulation: skin tone by brand
        cross_accuracy = merged.groupby(
            ["skin_tone_category", "device_brand"]
        )["sensor_accuracy_score"].mean().unstack().to_dict()
        
        # Calculate the accuracy gap
        tone_means = merged.groupby("skin_tone_category")["sensor_accuracy_score"].mean()
        accuracy_gap = tone_means.max() - tone_means.min()
        
        return {
            "by_skin_tone": skin_tone_accuracy,
            "by_brand": brand_accuracy,
            "cross_tabulation": cross_accuracy,
            "accuracy_gap": accuracy_gap,
            "highest_accuracy_group": tone_means.idxmax(),
            "lowest_accuracy_group": tone_means.idxmin()
        }
    
    def compute_fragmentation_disparity(
        self,
        device_df: pd.DataFrame,
        usage_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Compute fragmentation index disparities by demographics.
        
        Args:
            device_df: Device stream DataFrame
            usage_df: Usage DataFrame for demographics
        
        Returns:
            Dictionary with fragmentation statistics
        """
        merged = device_df.merge(
            usage_df[["user_id", "income_bracket", "race_ethnicity"]],
            on="user_id"
        )
        
        # By income
        income_frag = merged.groupby("income_bracket")["fragmentation_index"].agg([
            "mean", "std", "median"
        ]).to_dict("index")
        
        # By race/ethnicity
        race_frag = merged.groupby("race_ethnicity")["fragmentation_index"].agg([
            "mean", "std", "median"
        ]).to_dict("index")
        
        # By brand
        brand_frag = merged.groupby("device_brand")["fragmentation_index"].agg([
            "mean", "std", "median"
        ]).to_dict("index")
        
        # Calculate income based fragmentation gap
        income_means = merged.groupby("income_bracket")["fragmentation_index"].mean()
        frag_gap = income_means.max() - income_means.min()
        
        return {
            "by_income": income_frag,
            "by_race_ethnicity": race_frag,
            "by_brand": brand_frag,
            "income_fragmentation_gap": frag_gap
        }
    
    def get_calibration_data(
        self,
        model_name: str,
        df: pd.DataFrame,
        group_col: str,
        n_bins: int = 5
    ) -> Dict[str, Dict]:
        """
        Compute calibration data for a fitted model by group.
        
        Args:
            model_name: Name of the model in self.results
            df: DataFrame with the group column
            group_col: Column to stratify calibration by
            n_bins: Number of bins for calibration curve
        
        Returns:
            Dictionary mapping groups to calibration data
        """
        if model_name not in self.results:
            raise ValueError(f"Model {model_name} not found")
        
        results = self.results[model_name]
        probs = results.predicted_probabilities
        
        # Get the actual outcome based on model
        if model_name == "ownership":
            y_true = df["owns_wearable"].astype(int).values
        elif model_name == "health_usage":
            df_subset = df[df["owns_wearable"]]
            y_true = df_subset["uses_wearable_for_health"].astype(int).values
        elif model_name == "ehr_linkability":
            y_true = (
                df["ehr_export_supported"] & df["actually_shared_data"]
            ).astype(int).values
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        calibration_by_group = {}
        
        for group in df[group_col].unique():
            if model_name == "health_usage":
                mask = (df["owns_wearable"]) & (df[group_col] == group)
            else:
                mask = df[group_col] == group
            
            group_probs = probs[mask.values] if hasattr(mask, "values") else probs[mask]
            group_y = y_true[mask.values] if hasattr(mask, "values") else y_true[mask]
            
            if len(group_y) < n_bins * 2:
                continue
            
            try:
                prob_true, prob_pred = calibration_curve(
                    group_y, group_probs, n_bins=n_bins, strategy="uniform"
                )
                calibration_by_group[group] = {
                    "predicted_probability": prob_pred.tolist(),
                    "observed_frequency": prob_true.tolist(),
                    "n_samples": len(group_y),
                    "mean_predicted": float(group_probs.mean()),
                    "mean_observed": float(group_y.mean())
                }
            except ValueError:
                # Not enough data for calibration
                continue
        
        return calibration_by_group


def generate_disparity_summary(
    usage_df: pd.DataFrame,
    device_df: pd.DataFrame,
    quality_df: pd.DataFrame,
    pipeline: EquityModelingPipeline
) -> Dict[str, Any]:
    """
    Generate a comprehensive summary of disparities across all dimensions.
    
    Args:
        usage_df: Usage DataFrame
        device_df: Device DataFrame
        quality_df: Quality DataFrame
        pipeline: Fitted EquityModelingPipeline
    
    Returns:
        Dictionary with all disparity summaries
    """
    summary = {}
    
    # Ownership disparities
    summary["ownership_by_income"] = pipeline.compute_fairness_metrics(
        usage_df, "owns_wearable", "income_bracket"
    )
    
    summary["ownership_by_race"] = pipeline.compute_fairness_metrics(
        usage_df, "owns_wearable", "race_ethnicity"
    )
    
    # Health usage disparities
    owners = usage_df[usage_df["owns_wearable"]]
    summary["health_usage_by_income"] = pipeline.compute_fairness_metrics(
        owners, "uses_wearable_for_health", "income_bracket"
    )
    
    # Accuracy disparities
    summary["accuracy_disparities"] = pipeline.compute_accuracy_disparity(
        quality_df, device_df
    )
    
    # Fragmentation disparities
    summary["fragmentation_disparities"] = pipeline.compute_fragmentation_disparity(
        device_df, usage_df
    )
    
    # Model performance summary
    summary["model_accuracies"] = {
        name: results.accuracy
        for name, results in pipeline.results.items()
    }
    
    return summary
