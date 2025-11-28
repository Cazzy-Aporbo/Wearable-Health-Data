"""
Report Generator for Wearable Health Equity Analysis
Written by Cazzy Aporbo

This module generates markdown and HTML reports summarizing
key findings from equity, data quality, and interoperability analyses.

Disclaimer:
    This code is for research and exploratory purposes only.
    It does not perform clinical diagnosis or personalize medical advice.
    Findings should be validated with domain expertise.
"""

from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import json


class ReportGenerator:
    """
    Generates summary reports from wearable health equity analysis.
    
    Reports are designed to be human-readable summaries that connect
    observed patterns to broader themes in digital health equity.
    """
    
    def __init__(self, output_dir: str = "outputs"):
        """
        Initialize the report generator.
        
        Args:
            output_dir: Directory for saving report outputs
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def _format_percentage(self, value: float) -> str:
        """Format a decimal as a percentage string."""
        return f"{value * 100:.1f}%"
    
    def _format_rate_comparison(
        self,
        high_rate: float,
        low_rate: float,
        high_group: str,
        low_group: str
    ) -> str:
        """Format a rate comparison as readable text."""
        gap = high_rate - low_rate
        ratio = high_rate / low_rate if low_rate > 0 else float("inf")
        
        return (
            f"{high_group} ({self._format_percentage(high_rate)}) vs "
            f"{low_group} ({self._format_percentage(low_rate)}), "
            f"gap of {self._format_percentage(gap)}"
        )
    
    def generate_executive_summary(
        self,
        data_summary: Dict,
        disparity_summary: Dict,
        scenario_name: str = "baseline"
    ) -> str:
        """
        Generate a brief executive summary section.
        
        Args:
            data_summary: Summary statistics from data_loading.get_data_summary()
            disparity_summary: Summary from modeling.generate_disparity_summary()
            scenario_name: Name of the analysis scenario
        
        Returns:
            Markdown formatted summary text
        """
        total = data_summary["total_population"]
        ownership = data_summary["ownership_rate"]
        health_use = data_summary["health_usage_rate_among_owners"]
        sharing = data_summary["data_sharing_rate_among_health_users"]
        
        # Extract disparity metrics
        income_fairness = disparity_summary.get("ownership_by_income")
        income_gap = income_fairness.absolute_gap if income_fairness else 0.0
        
        accuracy_disp = disparity_summary.get("accuracy_disparities", {})
        accuracy_gap = accuracy_disp.get("accuracy_gap", 0.0)
        
        frag_disp = disparity_summary.get("fragmentation_disparities", {})
        frag_gap = frag_disp.get("income_fragmentation_gap", 0.0)
        
        summary = f"""
## Executive Summary

This analysis examined wearable health technology adoption and data quality 
across a population of {total:,} individuals using the {scenario_name} scenario.

**Key Findings:**

Overall wearable ownership stands at {self._format_percentage(ownership)}, with 
{self._format_percentage(health_use)} of owners actively using devices for health 
purposes. Among health-focused users, {self._format_percentage(sharing)} share 
data with healthcare providers.

Ownership disparities by income bracket show a gap of {self._format_percentage(income_gap)} 
between the highest and lowest income groups. Sensor accuracy varies by skin tone 
category, with a gap of {self._format_percentage(accuracy_gap)} between groups with 
highest and lowest measured accuracy.

Data fragmentation shows meaningful variation across income levels, with a 
{frag_gap:.2f} point difference in fragmentation index between income extremes. 
This suggests that lower-income individuals may face additional barriers to 
data interoperability and clinical integration.

"""
        return summary
    
    def generate_data_description(self, data_summary: Dict) -> str:
        """
        Generate the data description section.
        
        Args:
            data_summary: Summary statistics from data_loading.get_data_summary()
        
        Returns:
            Markdown formatted description
        """
        demo = data_summary.get("demographic_breakdown", {})
        income_dist = demo.get("income_brackets", {})
        race_dist = demo.get("race_ethnicity", {})
        geo_dist = demo.get("geography", {})
        
        description = f"""
## Data Description

The analysis uses three integrated data buckets:

**Bucket 1: Individual and Population Usage**
Contains demographic characteristics and wearable adoption behaviors including 
device ownership, health-focused usage patterns, and data sharing preferences.

**Bucket 2: Device Stream and Fragmentation**
Documents device brands, data formats, API availability, and EHR integration 
capabilities. Includes a computed fragmentation index ranging from 0 (fully 
interoperable) to 1 (highly fragmented).

**Bucket 3: Equity, Bias, and Quality**
Captures sensor accuracy measurements, data completeness metrics, and sharing 
behaviors stratified by demographic factors including skin tone category.

**Population Composition:**

The analyzed population includes {sum(income_dist.values()):,} individuals distributed 
across income brackets, with representation from multiple racial and ethnic groups 
and geographic settings.

"""
        return description
    
    def generate_equity_findings(self, disparity_summary: Dict) -> str:
        """
        Generate the equity and disparity findings section.
        
        Args:
            disparity_summary: Summary from modeling.generate_disparity_summary()
        
        Returns:
            Markdown formatted findings
        """
        income_fairness = disparity_summary.get("ownership_by_income")
        race_fairness = disparity_summary.get("ownership_by_race")
        accuracy_disp = disparity_summary.get("accuracy_disparities", {})
        
        income_text = ""
        if income_fairness:
            rates = income_fairness.group_rates
            high_income = rates.get("high", 0)
            low_income = rates.get("low", 0)
            income_text = f"""
Wearable ownership varies substantially by income. High income individuals 
own wearables at a rate of {self._format_percentage(high_income)}, compared 
to {self._format_percentage(low_income)} among low income individuals.
"""
        
        race_text = ""
        if race_fairness:
            rates = race_fairness.group_rates
            max_group = max(rates, key=rates.get)
            min_group = min(rates, key=rates.get)
            race_text = f"""
Ownership also varies by race and ethnicity. The highest ownership rate 
({self._format_percentage(rates[max_group])}) was observed among {max_group.replace("_", " ")} 
respondents, while the lowest ({self._format_percentage(rates[min_group])}) was among 
{min_group.replace("_", " ")} respondents.
"""
        
        accuracy_text = ""
        high_acc = accuracy_disp.get("highest_accuracy_group", "")
        low_acc = accuracy_disp.get("lowest_accuracy_group", "")
        acc_gap = accuracy_disp.get("accuracy_gap", 0)
        if high_acc and low_acc:
            accuracy_text = f"""
Sensor accuracy shows variation by skin tone category. The highest accuracy 
was measured among individuals with {high_acc.replace("_", " ")} skin tones, 
while the lowest was among those with {low_acc.replace("_", " ")} skin tones, 
representing a gap of {self._format_percentage(acc_gap)}.
"""
        
        findings = f"""
## Equity and Disparity Findings

This section summarizes observed patterns related to health equity and 
representativity in wearable health data.

### Ownership Disparities

{income_text}
{race_text}

These patterns reflect broader digital divides documented in research on 
health technology access and socioeconomic factors.

### Sensor Accuracy Disparities

{accuracy_text}

Accuracy variations by skin tone have implications for the reliability of 
health metrics derived from photoplethysmography sensors. Devices may 
systematically produce less accurate readings for certain populations, 
potentially affecting clinical decision-making when this data is used.

### Implications for Data Representativity

The observed disparities in ownership and accuracy suggest that wearable 
health datasets may underrepresent certain populations in both quantity 
and quality of data. This creates risks for:

1. Biased algorithm development when training on non-representative data
2. Differential quality of derived health insights across demographic groups
3. Potential for widening rather than narrowing health disparities

"""
        return findings
    
    def generate_interoperability_findings(self, disparity_summary: Dict) -> str:
        """
        Generate the interoperability and fragmentation findings section.
        
        Args:
            disparity_summary: Summary from modeling.generate_disparity_summary()
        
        Returns:
            Markdown formatted findings
        """
        frag_disp = disparity_summary.get("fragmentation_disparities", {})
        
        brand_frag = frag_disp.get("by_brand", {})
        income_frag = frag_disp.get("by_income", {})
        
        brand_text = ""
        if brand_frag:
            # Find highest and lowest fragmentation brands
            brand_means = {b: v.get("mean", 0) for b, v in brand_frag.items()}
            low_frag_brand = min(brand_means, key=brand_means.get)
            high_frag_brand = max(brand_means, key=brand_means.get)
            brand_text = f"""
Fragmentation varies substantially by device brand. {low_frag_brand} devices 
show the lowest fragmentation (mean index {brand_means[low_frag_brand]:.2f}), 
while {high_frag_brand} devices show the highest (mean index 
{brand_means[high_frag_brand]:.2f}).
"""
        
        income_text = ""
        if income_frag:
            income_means = {i: v.get("mean", 0) for i, v in income_frag.items()}
            low_frag_income = min(income_means, key=income_means.get)
            high_frag_income = max(income_means, key=income_means.get)
            income_text = f"""
Lower income individuals tend to use devices with higher fragmentation 
indices, reflecting both device choice patterns and the availability of 
interoperable options at different price points.
"""
        
        findings = f"""
## Interoperability and Fragmentation Findings

Data fragmentation affects the ability to integrate wearable health information 
with clinical systems and other data sources.

### Fragmentation by Device Ecosystem

{brand_text}

Devices with lower fragmentation typically offer better API access, standardized 
data formats, and established EHR integration pathways.

### Fragmentation and Socioeconomic Status

{income_text}

This pattern creates a compounding challenge: populations already facing barriers 
to healthcare access may also face barriers to having their wearable data 
meaningfully integrated into clinical care.

### Implications for Data Integration

The fragmented device ecosystem presents challenges for:

1. Building comprehensive longitudinal health records from multiple data sources
2. Developing interoperable clinical workflows that incorporate wearable data
3. Ensuring equitable access to data-driven health insights

Efforts to improve interoperability should consider how standardization 
benefits may be distributed across different device ecosystems and user 
populations.

"""
        return findings
    
    def generate_limitations_section(self, is_synthetic: bool = True) -> str:
        """
        Generate the limitations and considerations section.
        
        Args:
            is_synthetic: Whether the data is synthetic
        
        Returns:
            Markdown formatted limitations
        """
        data_note = ""
        if is_synthetic:
            data_note = """
**Synthetic Data Note:** This analysis uses synthetically generated data 
designed to reflect patterns documented in published research. While the 
synthetic data captures directional relationships, the specific magnitudes 
and distributions should not be interpreted as precise estimates of 
real-world prevalence or effect sizes.
"""
        
        limitations = f"""
## Limitations and Considerations

{data_note}

### Methodological Limitations

1. Cross-sectional analysis cannot establish causal relationships
2. Model predictions rely on available covariates and may not capture all 
   relevant factors
3. Group-level disparities do not predict individual experiences

### Data Quality Considerations

1. Sensor accuracy measurements may vary based on measurement protocol
2. Self-reported data sharing behavior may not match actual behavior
3. Device brand categorization simplifies a complex ecosystem

### Interpretation Guidance

This analysis is intended for research and exploratory purposes only. It does 
not perform clinical diagnosis or provide personalized medical advice. Findings 
should be validated with domain expertise and, where possible, replicated with 
real-world data from relevant populations.

"""
        return limitations
    
    def generate_full_report(
        self,
        data_summary: Dict,
        disparity_summary: Dict,
        scenario_name: str = "baseline",
        is_synthetic: bool = True,
        figure_metadata: Optional[Dict] = None
    ) -> str:
        """
        Generate the complete analysis report.
        
        Args:
            data_summary: Summary statistics from data_loading
            disparity_summary: Summary from modeling
            scenario_name: Name of the analysis scenario
            is_synthetic: Whether data is synthetic
            figure_metadata: Optional dictionary with figure file paths
        
        Returns:
            Complete markdown report
        """
        timestamp = datetime.now().strftime("%Y-%m-%d")
        
        header = f"""
# Wearable Health Equity Analysis Report

**Analysis Date:** {timestamp}  
**Scenario:** {scenario_name}  
**Author:** Cazzy Aporbo  

---

"""
        
        sections = [
            header,
            self.generate_executive_summary(data_summary, disparity_summary, scenario_name),
            self.generate_data_description(data_summary),
            self.generate_equity_findings(disparity_summary),
            self.generate_interoperability_findings(disparity_summary),
            self.generate_limitations_section(is_synthetic)
        ]
        
        # Add figure references if provided
        if figure_metadata:
            figures_section = self._generate_figures_section(figure_metadata)
            sections.append(figures_section)
        
        full_report = "\n".join(sections)
        
        # Save markdown report
        report_path = self.output_dir / "analysis_report.md"
        with open(report_path, "w") as f:
            f.write(full_report)
        
        # Also generate simple HTML version
        self._save_html_report(full_report)
        
        return full_report
    
    def _generate_figures_section(self, figure_metadata: Dict) -> str:
        """Generate a section listing all figures."""
        figures = figure_metadata.get("figures", {})
        
        lines = ["\n## Visualizations\n"]
        lines.append("The following visualizations are available for this analysis:\n")
        
        for name, info in figures.items():
            title = info.get("title", name.replace("_", " ").title())
            png_path = info.get("png_path", "")
            html_path = info.get("html_path", "")
            
            lines.append(f"**{title}**")
            if png_path:
                lines.append(f"  Static: {png_path}")
            if html_path:
                lines.append(f"  Interactive: {html_path}")
            lines.append("")
        
        return "\n".join(lines)
    
    def _save_html_report(self, markdown_content: str):
        """Save a simple HTML version of the report."""
        # Basic HTML wrapper
        html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Wearable Health Equity Analysis Report</title>
    <style>
        body {{
            font-family: 'Source Sans Pro', 'Segoe UI', sans-serif;
            max-width: 900px;
            margin: 0 auto;
            padding: 40px 20px;
            line-height: 1.6;
            color: #3D3D3D;
            background-color: #FAFAFA;
        }}
        h1 {{
            color: #2D5A5A;
            border-bottom: 2px solid #7FBFBF;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #4A8B8B;
            margin-top: 40px;
        }}
        h3 {{
            color: #5AA3A3;
        }}
        pre {{
            background-color: #E8E4E1;
            padding: 15px;
            border-radius: 4px;
            overflow-x: auto;
        }}
        code {{
            font-family: 'Source Code Pro', Consolas, monospace;
        }}
        strong {{
            color: #2D5A5A;
        }}
        hr {{
            border: none;
            border-top: 1px solid #A8A8A8;
            margin: 30px 0;
        }}
    </style>
</head>
<body>
{content}
</body>
</html>
"""
        
        # Simple markdown to HTML conversion
        # (In production, use a proper markdown library)
        content = markdown_content
        
        # Basic conversions
        content = content.replace("# ", "<h1>").replace("\n## ", "\n</h1><h2>")
        content = content.replace("\n### ", "\n</h2><h3>")
        content = content.replace("\n---\n", "\n<hr>\n")
        
        # Wrap paragraphs
        paragraphs = content.split("\n\n")
        wrapped = []
        for p in paragraphs:
            if not p.strip().startswith("<"):
                p = f"<p>{p}</p>"
            wrapped.append(p)
        content = "\n".join(wrapped)
        
        html = html_template.format(content=content)
        
        html_path = self.output_dir / "analysis_report.html"
        with open(html_path, "w") as f:
            f.write(html)


def generate_dashboard_json(
    data_summary: Dict,
    disparity_summary: Dict,
    figure_metadata: Dict,
    output_path: str = "outputs/dashboard_data.json"
) -> Dict:
    """
    Generate JSON data suitable for dashboard consumption.
    
    Args:
        data_summary: Summary statistics from data_loading
        disparity_summary: Summary from modeling
        figure_metadata: Dictionary with figure file paths
        output_path: Path to save JSON output
    
    Returns:
        Dictionary with dashboard-ready data
    """
    # Extract key metrics for dashboard display
    dashboard_data = {
        "generated_at": datetime.now().isoformat(),
        "overview": {
            "total_population": data_summary["total_population"],
            "wearable_owners": data_summary["wearable_owners"],
            "ownership_rate": data_summary["ownership_rate"],
            "health_usage_rate": data_summary["health_usage_rate_among_owners"],
            "sharing_rate": data_summary["data_sharing_rate_among_health_users"]
        },
        "device_metrics": {
            "unique_brands": data_summary["unique_device_brands"],
            "mean_fragmentation": data_summary["mean_fragmentation_index"],
            "ehr_export_rate": data_summary["ehr_export_rate"]
        },
        "quality_metrics": {
            "mean_accuracy": data_summary["mean_sensor_accuracy"],
            "mean_dropout": data_summary["mean_dropout_rate"]
        },
        "disparities": {},
        "figures": figure_metadata.get("figures", {})
    }
    
    # Add disparity metrics
    if "ownership_by_income" in disparity_summary:
        income_fm = disparity_summary["ownership_by_income"]
        dashboard_data["disparities"]["income_ownership_gap"] = income_fm.absolute_gap
        dashboard_data["disparities"]["income_ownership_rates"] = income_fm.group_rates
    
    if "accuracy_disparities" in disparity_summary:
        acc_disp = disparity_summary["accuracy_disparities"]
        dashboard_data["disparities"]["accuracy_gap"] = acc_disp.get("accuracy_gap", 0)
        dashboard_data["disparities"]["accuracy_by_skin_tone"] = acc_disp.get("by_skin_tone", {})
    
    # Save to file
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(dashboard_data, f, indent=2, default=str)
    
    return dashboard_data
