"""
Wearable Health Equity Analysis Toolkit
Written by Cazzy Aporbo

Main entry point for running complete equity analysis pipelines.
Supports both programmatic usage and command-line execution.

Disclaimer:
    This code is for research and exploratory purposes only.
    It does not perform clinical diagnosis or personalize medical advice.
"""

import argparse
from pathlib import Path
from typing import Optional, Tuple
import pandas as pd
import json

from synthetic_data import generate_synthetic_scenario, SyntheticDataGenerator
from data_loading import DataLoader, merge_all_buckets, get_data_summary
from modeling import EquityModelingPipeline, generate_disparity_summary
from visualization import EquityVisualizer
from report_generator import ReportGenerator, generate_dashboard_json


def run_full_analysis(
    scenario_name: str = "baseline",
    n_users: int = 2000,
    random_seed: int = 42,
    output_dir: str = "outputs",
    save_data: bool = True,
    verbose: bool = True
) -> dict:
    """
    Run a complete wearable health equity analysis pipeline.
    
    This function generates or loads data, fits models, computes
    fairness metrics, generates visualizations, and produces reports.
    
    Args:
        scenario_name: Name of the synthetic data scenario to use.
            Options: 'baseline', 'high_disparity', 'improved_equity', 'fragmented_ecosystem'
        n_users: Number of synthetic users to generate
        random_seed: Random seed for reproducibility
        output_dir: Directory for all outputs
        save_data: Whether to save intermediate data files
        verbose: Whether to print progress messages
    
    Returns:
        Dictionary with paths to all generated outputs
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    def log(message: str):
        if verbose:
            print(message)
    
    log(f"Starting wearable health equity analysis: {scenario_name}")
    log(f"Generating synthetic data with {n_users} users...")
    
    # Generate synthetic data
    usage_df, device_df, quality_df = generate_synthetic_scenario(
        scenario_name=scenario_name,
        n_users=n_users,
        random_seed=random_seed
    )
    
    if save_data:
        data_dir = output_path / "data"
        data_dir.mkdir(exist_ok=True)
        usage_df.to_csv(data_dir / "individual_population_usage.csv", index=False)
        device_df.to_csv(data_dir / "device_stream_fragmentation.csv", index=False)
        quality_df.to_csv(data_dir / "equity_bias_quality.csv", index=False)
        log(f"Saved data files to {data_dir}")
    
    # Compute data summary
    log("Computing data summary...")
    data_summary = get_data_summary(usage_df, device_df, quality_df)
    
    # Merge data for modeling
    log("Merging data buckets...")
    merged_df = merge_all_buckets(usage_df, device_df, quality_df)
    
    # Fit models
    log("Fitting equity models...")
    pipeline = EquityModelingPipeline(random_seed=random_seed)
    
    ownership_results = pipeline.fit_ownership_model(usage_df)
    log(f"  Ownership model accuracy: {ownership_results.accuracy:.3f}")
    
    health_results = pipeline.fit_health_usage_model(usage_df)
    log(f"  Health usage model accuracy: {health_results.accuracy:.3f}")
    
    ehr_results = pipeline.fit_ehr_linkability_model(merged_df)
    log(f"  EHR linkability model accuracy: {ehr_results.accuracy:.3f}")
    
    # Generate disparity summary
    log("Computing disparity metrics...")
    disparity_summary = generate_disparity_summary(
        usage_df, device_df, quality_df, pipeline
    )
    
    # Get calibration data
    calibration_data = pipeline.get_calibration_data(
        "ownership", usage_df, "income_bracket"
    )
    
    # Generate visualizations
    log("Generating visualizations...")
    viz = EquityVisualizer(
        output_dir=str(output_path / "visualizations"),
        save_html=True,
        save_png=True
    )
    
    # Population equity visuals
    viz.plot_ownership_by_income(usage_df)
    viz.plot_ownership_by_race_ethnicity(usage_df)
    viz.plot_ownership_heatmap(usage_df)
    viz.plot_usage_funnel(usage_df)
    
    # Fragmentation visuals
    viz.plot_fragmentation_by_brand(device_df)
    viz.plot_fragmentation_by_income(device_df, usage_df)
    viz.plot_interoperability_matrix(device_df)
    viz.plot_fragmentation_vs_linkability(
        merged_df,
        ehr_results.predicted_probabilities
    )
    
    # Sensor bias visuals
    viz.plot_accuracy_by_skin_tone(quality_df, device_df)
    viz.plot_accuracy_by_brand_and_tone(quality_df, device_df)
    viz.plot_dropout_by_demographics(quality_df, usage_df)
    viz.plot_data_quality_radar(quality_df, usage_df, device_df)
    
    # Model visuals
    viz.plot_predicted_ownership_by_group(
        usage_df,
        ownership_results.predicted_probabilities
    )
    viz.plot_fairness_dashboard(disparity_summary)
    viz.plot_calibration_by_group(calibration_data)
    
    # Export figure metadata
    figure_metadata = viz.export_all_figures_metadata()
    log(f"Generated {len(figure_metadata['figures'])} visualizations")
    
    # Generate reports
    log("Generating analysis report...")
    report_gen = ReportGenerator(output_dir=str(output_path / "reports"))
    
    report_content = report_gen.generate_full_report(
        data_summary=data_summary,
        disparity_summary=disparity_summary,
        scenario_name=scenario_name,
        is_synthetic=True,
        figure_metadata=figure_metadata
    )
    
    # Generate dashboard JSON
    dashboard_data = generate_dashboard_json(
        data_summary=data_summary,
        disparity_summary=disparity_summary,
        figure_metadata=figure_metadata,
        output_path=str(output_path / "dashboard_data.json")
    )
    
    log(f"Analysis complete. Outputs saved to {output_path}")
    
    # Return summary of outputs
    return {
        "output_directory": str(output_path),
        "data_files": {
            "usage": str(output_path / "data" / "individual_population_usage.csv"),
            "device": str(output_path / "data" / "device_stream_fragmentation.csv"),
            "quality": str(output_path / "data" / "equity_bias_quality.csv")
        } if save_data else None,
        "visualizations_directory": str(output_path / "visualizations"),
        "report_markdown": str(output_path / "reports" / "analysis_report.md"),
        "report_html": str(output_path / "reports" / "analysis_report.html"),
        "dashboard_json": str(output_path / "dashboard_data.json"),
        "data_summary": data_summary,
        "model_accuracies": {
            "ownership": ownership_results.accuracy,
            "health_usage": health_results.accuracy,
            "ehr_linkability": ehr_results.accuracy
        }
    }


def run_analysis_from_files(
    data_directory: str,
    output_dir: str = "outputs",
    file_format: str = "csv",
    verbose: bool = True
) -> dict:
    """
    Run analysis using existing data files.
    
    Args:
        data_directory: Directory containing the three data bucket files
        output_dir: Directory for outputs
        file_format: Format of input files ('csv' or 'parquet')
        verbose: Whether to print progress
    
    Returns:
        Dictionary with paths to generated outputs
    """
    def log(message: str):
        if verbose:
            print(message)
    
    log(f"Loading data from {data_directory}...")
    
    loader = DataLoader(verbose=verbose)
    usage_df, device_df, quality_df = loader.load_all_from_directory(
        data_directory,
        file_format=file_format
    )
    
    # Continue with same analysis pipeline
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    data_summary = get_data_summary(usage_df, device_df, quality_df)
    merged_df = merge_all_buckets(usage_df, device_df, quality_df)
    
    log("Fitting models...")
    pipeline = EquityModelingPipeline()
    pipeline.fit_ownership_model(usage_df)
    pipeline.fit_health_usage_model(usage_df)
    pipeline.fit_ehr_linkability_model(merged_df)
    
    disparity_summary = generate_disparity_summary(
        usage_df, device_df, quality_df, pipeline
    )
    
    log("Generating visualizations...")
    viz = EquityVisualizer(output_dir=str(output_path / "visualizations"))
    
    # Generate all standard visualizations
    viz.plot_ownership_by_income(usage_df)
    viz.plot_ownership_by_race_ethnicity(usage_df)
    viz.plot_ownership_heatmap(usage_df)
    viz.plot_usage_funnel(usage_df)
    viz.plot_fragmentation_by_brand(device_df)
    viz.plot_fragmentation_by_income(device_df, usage_df)
    viz.plot_accuracy_by_skin_tone(quality_df, device_df)
    viz.plot_data_quality_radar(quality_df, usage_df, device_df)
    viz.plot_fairness_dashboard(disparity_summary)
    
    figure_metadata = viz.export_all_figures_metadata()
    
    log("Generating report...")
    report_gen = ReportGenerator(output_dir=str(output_path / "reports"))
    report_gen.generate_full_report(
        data_summary=data_summary,
        disparity_summary=disparity_summary,
        scenario_name="custom_data",
        is_synthetic=False,
        figure_metadata=figure_metadata
    )
    
    generate_dashboard_json(
        data_summary, disparity_summary, figure_metadata,
        str(output_path / "dashboard_data.json")
    )
    
    log(f"Analysis complete. Outputs saved to {output_path}")
    
    return {
        "output_directory": str(output_path),
        "data_summary": data_summary
    }


def main():
    """Command-line interface for the analysis toolkit."""
    parser = argparse.ArgumentParser(
        description="Wearable Health Equity Analysis Toolkit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py --scenario baseline --n-users 2000
    python main.py --scenario high_disparity --output results/high_disparity
    python main.py --data-dir my_data/ --format csv
        """
    )
    
    parser.add_argument(
        "--scenario",
        type=str,
        choices=["baseline", "high_disparity", "improved_equity", "fragmented_ecosystem"],
        default="baseline",
        help="Synthetic data scenario to generate"
    )
    
    parser.add_argument(
        "--n-users",
        type=int,
        default=2000,
        help="Number of synthetic users to generate"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="outputs",
        help="Output directory for results"
    )
    
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Directory with existing data files (overrides synthetic generation)"
    )
    
    parser.add_argument(
        "--format",
        type=str,
        choices=["csv", "parquet"],
        default="csv",
        help="Format of input data files when using --data-dir"
    )
    
    parser.add_argument(
        "--no-save-data",
        action="store_true",
        help="Do not save intermediate data files"
    )
    
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress messages"
    )
    
    args = parser.parse_args()
    
    if args.data_dir:
        results = run_analysis_from_files(
            data_directory=args.data_dir,
            output_dir=args.output,
            file_format=args.format,
            verbose=not args.quiet
        )
    else:
        results = run_full_analysis(
            scenario_name=args.scenario,
            n_users=args.n_users,
            random_seed=args.seed,
            output_dir=args.output,
            save_data=not args.no_save_data,
            verbose=not args.quiet
        )
    
    print(f"\nAnalysis complete!")
    print(f"Results saved to: {results['output_directory']}")
    
    if "model_accuracies" in results:
        print("\nModel Performance:")
        for model, acc in results["model_accuracies"].items():
            print(f"  {model}: {acc:.1%}")


if __name__ == "__main__":
    main()
