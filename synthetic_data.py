"""
Synthetic Data Generator for Wearable Health Equity Analysis
Written by Cazzy Aporbo

This module generates realistic synthetic datasets for exploring equity,
data quality, interoperability, and fragmentation in wearable health data.
Distributions are informed by published literature on digital health disparities.

Disclaimer:
    This generator is for research and exploratory purposes only.
    It does not perform clinical diagnosis or personalize medical advice.
    Real world patterns may differ substantially from synthetic approximations.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class SyntheticDataConfig:
    """
    Configuration parameters for synthetic data generation.
    
    Adjust these values to explore different equity scenarios.
    Default values reflect patterns observed in digital health literature
    regarding ownership disparities, sensor accuracy variations, and
    data sharing behaviors across demographic groups.
    """
    
    n_users: int = 2000
    random_seed: int = 42
    
    # Age distribution parameters
    age_mean: float = 45.0
    age_std: float = 15.0
    age_min: int = 18
    age_max: int = 85
    
    # Ownership base rates by income (reflecting digital divide patterns)
    ownership_base_by_income: Dict[str, float] = None
    
    # Ownership modifiers by race/ethnicity (multiplicative adjustments)
    ownership_modifier_by_race: Dict[str, float] = None
    
    # Sensor accuracy base by skin tone category
    accuracy_base_by_skin_tone: Dict[str, float] = None
    
    # Fragmentation propensity by device brand
    fragmentation_by_brand: Dict[str, Tuple[float, float]] = None
    
    # Data sharing propensity by insurance type
    sharing_by_insurance: Dict[str, float] = None
    
    def __post_init__(self):
        """Set default distributions if not provided."""
        if self.ownership_base_by_income is None:
            self.ownership_base_by_income = {
                "low": 0.25,
                "lower_middle": 0.40,
                "middle": 0.55,
                "upper_middle": 0.72,
                "high": 0.85
            }
        
        if self.ownership_modifier_by_race is None:
            self.ownership_modifier_by_race = {
                "white": 1.0,
                "black": 0.85,
                "hispanic_latino": 0.88,
                "asian": 1.05,
                "multiracial": 0.92,
                "other": 0.90
            }
        
        if self.accuracy_base_by_skin_tone is None:
            # Values reflect literature on PPG sensor bias
            self.accuracy_base_by_skin_tone = {
                "type_i_ii": 0.95,
                "type_iii_iv": 0.91,
                "type_v_vi": 0.84
            }
        
        if self.fragmentation_by_brand is None:
            # (mean, std) for fragmentation index by brand
            self.fragmentation_by_brand = {
                "Apple": (0.25, 0.10),
                "Fitbit": (0.35, 0.12),
                "Garmin": (0.40, 0.15),
                "Samsung": (0.38, 0.14),
                "Whoop": (0.55, 0.18),
                "Oura": (0.50, 0.16),
                "Withings": (0.45, 0.14),
                "Other": (0.65, 0.20)
            }
        
        if self.sharing_by_insurance is None:
            self.sharing_by_insurance = {
                "private": 0.45,
                "medicare": 0.38,
                "medicaid": 0.28,
                "uninsured": 0.15,
                "military_va": 0.52
            }


class SyntheticDataGenerator:
    """
    Generates synthetic wearable health datasets across three data buckets.
    
    The generator produces correlated, realistic data that can be used
    to develop and test equity analysis pipelines before real data
    becomes available.
    """
    
    def __init__(self, config: Optional[SyntheticDataConfig] = None):
        """
        Initialize the generator with configuration.
        
        Args:
            config: SyntheticDataConfig instance. Uses defaults if None.
        """
        self.config = config or SyntheticDataConfig()
        self.rng = np.random.default_rng(self.config.random_seed)
    
    def _generate_demographics(self, n: int) -> pd.DataFrame:
        """
        Generate demographic variables for n users.
        
        Returns:
            DataFrame with user_id and demographic columns
        """
        # Age with truncated normal
        ages = self.rng.normal(self.config.age_mean, self.config.age_std, n)
        ages = np.clip(ages, self.config.age_min, self.config.age_max).astype(int)
        
        # Race/ethnicity with approximate US population distribution
        race_probs = [0.58, 0.13, 0.19, 0.06, 0.03, 0.01]
        race_labels = ["white", "black", "hispanic_latino", "asian", "multiracial", "other"]
        race_ethnicity = self.rng.choice(race_labels, n, p=race_probs)
        
        # Income bracket with slight correlation to age
        income_labels = ["low", "lower_middle", "middle", "upper_middle", "high"]
        income_base_probs = np.array([0.18, 0.22, 0.28, 0.20, 0.12])
        
        # Adjust income probabilities based on age (peak earning years 35 to 55)
        income_indices = np.zeros(n, dtype=int)
        for i in range(n):
            age = ages[i]
            age_factor = 1.0 + 0.3 * np.exp(-((age - 45) ** 2) / 200)
            adjusted_probs = income_base_probs.copy()
            adjusted_probs[3:] *= age_factor
            adjusted_probs[0:2] /= age_factor
            adjusted_probs /= adjusted_probs.sum()
            income_indices[i] = self.rng.choice(5, p=adjusted_probs)
        income_bracket = [income_labels[idx] for idx in income_indices]
        
        # Education level with correlation to income
        edu_labels = ["less_than_hs", "high_school", "some_college", "bachelors", "graduate"]
        education = []
        edu_by_income = {
            "low": [0.25, 0.35, 0.25, 0.12, 0.03],
            "lower_middle": [0.12, 0.30, 0.32, 0.20, 0.06],
            "middle": [0.05, 0.20, 0.32, 0.30, 0.13],
            "upper_middle": [0.02, 0.10, 0.22, 0.40, 0.26],
            "high": [0.01, 0.05, 0.12, 0.42, 0.40]
        }
        for inc in income_bracket:
            education.append(self.rng.choice(edu_labels, p=edu_by_income[inc]))
        
        # Geography
        geo_labels = ["urban", "suburban", "rural"]
        geo_probs = [0.31, 0.52, 0.17]
        geography = self.rng.choice(geo_labels, n, p=geo_probs)
        
        # Insurance type with income correlation
        ins_labels = ["private", "medicare", "medicaid", "uninsured", "military_va"]
        insurance = []
        ins_by_income = {
            "low": [0.15, 0.10, 0.45, 0.25, 0.05],
            "lower_middle": [0.35, 0.12, 0.28, 0.18, 0.07],
            "middle": [0.55, 0.15, 0.12, 0.10, 0.08],
            "upper_middle": [0.72, 0.12, 0.05, 0.04, 0.07],
            "high": [0.82, 0.10, 0.02, 0.02, 0.04]
        }
        for inc in income_bracket:
            insurance.append(self.rng.choice(ins_labels, p=ins_by_income[inc]))
        
        return pd.DataFrame({
            "user_id": [f"U{str(i).zfill(5)}" for i in range(n)],
            "age": ages,
            "race_ethnicity": race_ethnicity,
            "income_bracket": income_bracket,
            "education_level": education,
            "geography": geography,
            "insurance_type": insurance
        })
    
    def generate_individual_population_usage(self) -> pd.DataFrame:
        """
        Generate Bucket 1: Individual and Population Usage data.
        
        This dataset captures who owns wearables, who uses them for health
        purposes, and who shares data with healthcare providers.
        
        Returns:
            DataFrame with usage and demographic columns
        """
        n = self.config.n_users
        df = self._generate_demographics(n)
        
        # Calculate ownership probability based on demographics
        ownership_probs = np.zeros(n)
        for i in range(n):
            base_prob = self.config.ownership_base_by_income[df.loc[i, "income_bracket"]]
            race_mod = self.config.ownership_modifier_by_race[df.loc[i, "race_ethnicity"]]
            
            # Age modifier: younger people more likely to own
            age = df.loc[i, "age"]
            age_mod = 1.0 - 0.008 * max(0, age - 40)
            
            # Geography modifier
            geo = df.loc[i, "geography"]
            geo_mod = {"urban": 1.05, "suburban": 1.0, "rural": 0.85}[geo]
            
            ownership_probs[i] = np.clip(base_prob * race_mod * age_mod * geo_mod, 0, 1)
        
        owns_wearable = self.rng.random(n) < ownership_probs
        df["owns_wearable"] = owns_wearable
        
        # Uses wearable for health: conditional on ownership
        uses_for_health = np.zeros(n, dtype=bool)
        for i in range(n):
            if owns_wearable[i]:
                # Higher age and health consciousness increase health use
                age = df.loc[i, "age"]
                age_factor = 0.5 + 0.01 * min(age, 65)
                edu = df.loc[i, "education_level"]
                edu_factor = {"less_than_hs": 0.7, "high_school": 0.8, "some_college": 0.9,
                             "bachelors": 1.0, "graduate": 1.1}[edu]
                prob = 0.6 * age_factor * edu_factor
                uses_for_health[i] = self.rng.random() < np.clip(prob, 0, 1)
        
        df["uses_wearable_for_health"] = uses_for_health
        
        # Shares data with provider: conditional on usage
        shares_data = np.zeros(n, dtype=bool)
        for i in range(n):
            if uses_for_health[i]:
                ins = df.loc[i, "insurance_type"]
                base_share = self.config.sharing_by_insurance[ins]
                # Trust and access factors
                geo = df.loc[i, "geography"]
                geo_factor = {"urban": 1.1, "suburban": 1.0, "rural": 0.75}[geo]
                shares_data[i] = self.rng.random() < np.clip(base_share * geo_factor, 0, 1)
        
        df["shares_data_with_provider"] = shares_data
        
        return df
    
    def generate_device_stream_fragmentation(
        self,
        usage_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Generate Bucket 2: Device Stream and Fragmentation data.
        
        This dataset describes the devices owned, data formats, and
        interoperability characteristics for each wearable user.
        
        Args:
            usage_df: DataFrame from generate_individual_population_usage()
        
        Returns:
            DataFrame with device and fragmentation columns
        """
        owners = usage_df[usage_df["owns_wearable"]].copy()
        n = len(owners)
        
        # Device brand distribution varies by income
        brand_labels = list(self.config.fragmentation_by_brand.keys())
        brand_by_income = {
            "low": [0.10, 0.30, 0.05, 0.15, 0.02, 0.02, 0.06, 0.30],
            "lower_middle": [0.15, 0.28, 0.08, 0.18, 0.03, 0.03, 0.05, 0.20],
            "middle": [0.25, 0.25, 0.12, 0.18, 0.05, 0.05, 0.05, 0.05],
            "upper_middle": [0.38, 0.18, 0.15, 0.12, 0.07, 0.05, 0.03, 0.02],
            "high": [0.50, 0.12, 0.15, 0.08, 0.08, 0.04, 0.02, 0.01]
        }
        
        device_brands = []
        for inc in owners["income_bracket"]:
            device_brands.append(self.rng.choice(brand_labels, p=brand_by_income[inc]))
        
        # Device models (simplified)
        model_by_brand = {
            "Apple": ["Watch Series 9", "Watch SE", "Watch Ultra"],
            "Fitbit": ["Charge 6", "Sense 2", "Versa 4"],
            "Garmin": ["Venu 3", "Forerunner 265", "Fenix 7"],
            "Samsung": ["Galaxy Watch 6", "Galaxy Fit 3"],
            "Whoop": ["Whoop 4.0"],
            "Oura": ["Oura Ring Gen 3"],
            "Withings": ["ScanWatch 2", "Steel HR"],
            "Other": ["Generic Tracker", "Budget Smartwatch"]
        }
        device_models = [self.rng.choice(model_by_brand[b]) for b in device_brands]
        
        # Data format
        format_by_brand = {
            "Apple": ["proprietary_json", "open_fhir"],
            "Fitbit": ["proprietary_json", "csv_export"],
            "Garmin": ["proprietary_json", "csv_export"],
            "Samsung": ["proprietary_json"],
            "Whoop": ["proprietary_json"],
            "Oura": ["proprietary_json", "csv_export"],
            "Withings": ["proprietary_json", "open_fhir"],
            "Other": ["csv_export", "proprietary_json"]
        }
        format_probs = {
            "Apple": [0.6, 0.4],
            "Fitbit": [0.7, 0.3],
            "Garmin": [0.65, 0.35],
            "Samsung": [1.0],
            "Whoop": [1.0],
            "Oura": [0.75, 0.25],
            "Withings": [0.5, 0.5],
            "Other": [0.6, 0.4]
        }
        data_formats = [self.rng.choice(format_by_brand[b], p=format_probs[b]) 
                       for b in device_brands]
        
        # Available metrics
        metrics_by_brand = {
            "Apple": ["steps,heart_rate,spo2,hrv,sleep_stages,ecg"],
            "Fitbit": ["steps,heart_rate,spo2,sleep_stages"],
            "Garmin": ["steps,heart_rate,hrv,sleep_stages,stress"],
            "Samsung": ["steps,heart_rate,spo2,sleep_stages"],
            "Whoop": ["heart_rate,hrv,sleep_stages,strain,recovery"],
            "Oura": ["heart_rate,hrv,sleep_stages,temperature,activity"],
            "Withings": ["steps,heart_rate,spo2,ecg,temperature"],
            "Other": ["steps,heart_rate"]
        }
        available_metrics = [metrics_by_brand[b][0] for b in device_brands]
        
        # API availability
        api_by_brand = {
            "Apple": 0.85,
            "Fitbit": 0.80,
            "Garmin": 0.75,
            "Samsung": 0.60,
            "Whoop": 0.70,
            "Oura": 0.75,
            "Withings": 0.80,
            "Other": 0.20
        }
        api_available = [self.rng.random() < api_by_brand[b] for b in device_brands]
        
        # EHR export support
        ehr_by_brand = {
            "Apple": 0.70,
            "Fitbit": 0.45,
            "Garmin": 0.25,
            "Samsung": 0.30,
            "Whoop": 0.15,
            "Oura": 0.20,
            "Withings": 0.55,
            "Other": 0.05
        }
        ehr_export = [self.rng.random() < ehr_by_brand[b] for b in device_brands]
        
        # Platforms integrated
        platform_options = ["vendor_portal_only", "ehr_system", "third_party_aggregator"]
        platforms = []
        for i, brand in enumerate(device_brands):
            if ehr_export[i]:
                platforms.append(self.rng.choice(
                    ["ehr_system", "third_party_aggregator"],
                    p=[0.6, 0.4]
                ))
            elif api_available[i]:
                platforms.append(self.rng.choice(
                    ["vendor_portal_only", "third_party_aggregator"],
                    p=[0.5, 0.5]
                ))
            else:
                platforms.append("vendor_portal_only")
        
        # Fragmentation index
        frag_indices = []
        for brand in device_brands:
            mean, std = self.config.fragmentation_by_brand[brand]
            frag = self.rng.normal(mean, std)
            frag_indices.append(np.clip(frag, 0, 1))
        
        return pd.DataFrame({
            "user_id": owners["user_id"].values,
            "device_brand": device_brands,
            "device_model": device_models,
            "data_format": data_formats,
            "available_metrics": available_metrics,
            "api_available": api_available,
            "ehr_export_supported": ehr_export,
            "platforms_integrated": platforms,
            "fragmentation_index": frag_indices
        })
    
    def generate_equity_bias_quality(
        self,
        usage_df: pd.DataFrame,
        device_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Generate Bucket 3: Equity, Bias, and Quality data.
        
        This dataset captures sensor accuracy, data completeness, and
        sharing behavior patterns that may vary by demographic group.
        
        Args:
            usage_df: DataFrame from generate_individual_population_usage()
            device_df: DataFrame from generate_device_stream_fragmentation()
        
        Returns:
            DataFrame with quality and equity columns
        """
        merged = usage_df[usage_df["owns_wearable"]].merge(device_df, on="user_id")
        n = len(merged)
        
        # Assign skin tone categories with correlation to race/ethnicity
        skin_tone_by_race = {
            "white": [0.60, 0.35, 0.05],
            "black": [0.02, 0.18, 0.80],
            "hispanic_latino": [0.15, 0.55, 0.30],
            "asian": [0.25, 0.55, 0.20],
            "multiracial": [0.25, 0.45, 0.30],
            "other": [0.30, 0.40, 0.30]
        }
        tone_labels = ["type_i_ii", "type_iii_iv", "type_v_vi"]
        skin_tones = []
        for race in merged["race_ethnicity"]:
            skin_tones.append(self.rng.choice(tone_labels, p=skin_tone_by_race[race]))
        
        # Sensor accuracy score
        accuracy_scores = []
        for i in range(n):
            base = self.config.accuracy_base_by_skin_tone[skin_tones[i]]
            # Brand modifier
            brand = merged.iloc[i]["device_brand"]
            brand_mod = {
                "Apple": 1.02,
                "Fitbit": 0.98,
                "Garmin": 1.00,
                "Samsung": 0.97,
                "Whoop": 1.01,
                "Oura": 1.00,
                "Withings": 1.01,
                "Other": 0.92
            }[brand]
            score = base * brand_mod + self.rng.normal(0, 0.03)
            accuracy_scores.append(np.clip(score, 0.5, 1.0))
        
        # Motion context
        motion_labels = ["rest", "walking", "running"]
        motion_probs = [0.40, 0.40, 0.20]
        motion_context = self.rng.choice(motion_labels, n, p=motion_probs)
        
        # Error rate (inversely related to accuracy, affected by motion)
        error_rates = []
        for i in range(n):
            base_error = 1.0 - accuracy_scores[i]
            motion = motion_context[i]
            motion_mult = {"rest": 0.7, "walking": 1.0, "running": 1.5}[motion]
            error = base_error * motion_mult + self.rng.normal(0, 0.02)
            error_rates.append(np.clip(error, 0, 0.5))
        
        # Dropout rate (missing data proportion)
        dropout_rates = []
        for i in range(n):
            inc = merged.iloc[i]["income_bracket"]
            inc_base = {"low": 0.25, "lower_middle": 0.20, "middle": 0.15,
                       "upper_middle": 0.10, "high": 0.08}[inc]
            dropout = inc_base + self.rng.normal(0, 0.05)
            dropout_rates.append(np.clip(dropout, 0, 0.6))
        
        # Wear time hours per day
        wear_times = []
        for i in range(n):
            uses_health = merged.iloc[i]["uses_wearable_for_health"]
            base = 14.0 if uses_health else 8.0
            wear = base + self.rng.normal(0, 3)
            wear_times.append(np.clip(wear, 2, 24))
        
        # Willingness to share score
        willingness_scores = []
        for i in range(n):
            ins = merged.iloc[i]["insurance_type"]
            base = self.config.sharing_by_insurance[ins]
            edu = merged.iloc[i]["education_level"]
            edu_mod = {"less_than_hs": 0.8, "high_school": 0.9, "some_college": 1.0,
                      "bachelors": 1.1, "graduate": 1.15}[edu]
            score = base * edu_mod + self.rng.normal(0, 0.1)
            willingness_scores.append(np.clip(score, 0, 1))
        
        # Actually shared data
        actually_shared = merged["shares_data_with_provider"].values.copy()
        
        return pd.DataFrame({
            "user_id": merged["user_id"].values,
            "skin_tone_category": skin_tones,
            "sensor_accuracy_score": accuracy_scores,
            "motion_context": motion_context,
            "error_rate": error_rates,
            "dropout_rate": dropout_rates,
            "wear_time_hours_per_day": wear_times,
            "willingness_to_share_score": willingness_scores,
            "actually_shared_data": actually_shared
        })
    
    def generate_all_buckets(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Generate all three data buckets in sequence.
        
        Returns:
            Tuple of (usage_df, device_df, quality_df)
        """
        usage_df = self.generate_individual_population_usage()
        device_df = self.generate_device_stream_fragmentation(usage_df)
        quality_df = self.generate_equity_bias_quality(usage_df, device_df)
        
        return usage_df, device_df, quality_df


def generate_synthetic_scenario(
    scenario_name: str = "baseline",
    n_users: int = 2000,
    random_seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Generate a named scenario with preconfigured parameters.
    
    Available scenarios:
        baseline: Default distributions based on literature
        high_disparity: Exaggerated ownership and accuracy gaps
        improved_equity: Reduced disparities (optimistic scenario)
        fragmented_ecosystem: Higher fragmentation across all brands
    
    Args:
        scenario_name: One of the available scenario names
        n_users: Number of synthetic users to generate
        random_seed: Random seed for reproducibility
    
    Returns:
        Tuple of (usage_df, device_df, quality_df)
    """
    config = SyntheticDataConfig(n_users=n_users, random_seed=random_seed)
    
    if scenario_name == "high_disparity":
        config.ownership_base_by_income = {
            "low": 0.15,
            "lower_middle": 0.30,
            "middle": 0.50,
            "upper_middle": 0.75,
            "high": 0.92
        }
        config.accuracy_base_by_skin_tone = {
            "type_i_ii": 0.96,
            "type_iii_iv": 0.88,
            "type_v_vi": 0.75
        }
    
    elif scenario_name == "improved_equity":
        config.ownership_base_by_income = {
            "low": 0.45,
            "lower_middle": 0.55,
            "middle": 0.62,
            "upper_middle": 0.70,
            "high": 0.78
        }
        config.ownership_modifier_by_race = {
            "white": 1.0,
            "black": 0.96,
            "hispanic_latino": 0.97,
            "asian": 1.02,
            "multiracial": 0.98,
            "other": 0.97
        }
        config.accuracy_base_by_skin_tone = {
            "type_i_ii": 0.94,
            "type_iii_iv": 0.92,
            "type_v_vi": 0.90
        }
    
    elif scenario_name == "fragmented_ecosystem":
        config.fragmentation_by_brand = {
            "Apple": (0.40, 0.12),
            "Fitbit": (0.55, 0.15),
            "Garmin": (0.60, 0.18),
            "Samsung": (0.58, 0.16),
            "Whoop": (0.72, 0.15),
            "Oura": (0.68, 0.14),
            "Withings": (0.62, 0.16),
            "Other": (0.82, 0.12)
        }
    
    generator = SyntheticDataGenerator(config)
    return generator.generate_all_buckets()
