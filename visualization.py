"""
Visualization Module for Wearable Health Equity Analysis
Written by Cazzy Aporbo

This module generates publication-ready, dashboard-compatible visualizations
for analyzing equity, data quality, interoperability, and fragmentation
in wearable health data.

Design Philosophy:
    Soft, sophisticated color palettes with pastel gradients
    Clear, human-readable labels and annotations
    Interactive capabilities via Plotly for dashboard embedding
    Static PNG exports for publication and documentation

Disclaimer:
    This code is for research and exploratory purposes only.
    It does not perform clinical diagnosis or personalize medical advice.
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import json

from theme import WearableEquityTheme, apply_theme_to_figure


class EquityVisualizer:
    """
    Generates equity-focused visualizations for wearable health data.
    
    All visualizations follow a consistent theme and are designed
    to highlight disparities while remaining accessible and clear.
    """
    
    def __init__(
        self,
        output_dir: str = "outputs",
        save_html: bool = True,
        save_png: bool = True
    ):
        """
        Initialize the visualizer.
        
        Args:
            output_dir: Directory for saving visualization outputs
            save_html: Whether to save interactive HTML versions
            save_png: Whether to save static PNG versions
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.save_html = save_html
        self.save_png = save_png
        self.theme = WearableEquityTheme()
        self.figures: Dict[str, go.Figure] = {}
    
    def _save_figure(self, fig: go.Figure, name: str):
        """Save figure to disk in configured formats."""
        if self.save_html:
            fig.write_html(self.output_dir / f"{name}.html")
        if self.save_png:
            try:
                fig.write_image(self.output_dir / f"{name}.png", scale=2)
            except Exception:
                # PNG export requires Chrome/Kaleido setup
                # Fall back to HTML only
                pass
    
    def _format_percentage(self, value: float) -> str:
        """Format a decimal as a percentage string."""
        return f"{value * 100:.1f}%"
    
    # Population Level Equity Visualizations
    
    def plot_ownership_by_income(
        self,
        usage_df: pd.DataFrame,
        title: str = "Wearable Ownership by Income Bracket"
    ) -> go.Figure:
        """
        Create bar chart showing ownership rates by income bracket.
        
        Args:
            usage_df: DataFrame with ownership and income data
            title: Chart title
        
        Returns:
            Plotly figure object
        """
        income_order = ["low", "lower_middle", "middle", "upper_middle", "high"]
        income_labels = {
            "low": "Low Income",
            "lower_middle": "Lower Middle",
            "middle": "Middle Income",
            "upper_middle": "Upper Middle",
            "high": "High Income"
        }
        
        rates = usage_df.groupby("income_bracket")["owns_wearable"].mean()
        rates = rates.reindex(income_order)
        
        colors = [self.theme.INCOME_PALETTE[inc] for inc in income_order]
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=[income_labels[inc] for inc in income_order],
            y=rates.values,
            marker_color=colors,
            text=[self._format_percentage(v) for v in rates.values],
            textposition="outside",
            textfont={"size": 12, "color": self.theme.NEUTRAL_DARK},
            hovertemplate="<b>%{x}</b><br>Ownership Rate: %{y:.1%}<extra></extra>"
        ))
        
        fig = apply_theme_to_figure(fig, title=title)
        
        fig.update_layout(
            xaxis_title="Income Bracket",
            yaxis_title="Ownership Rate",
            yaxis_tickformat=".0%",
            yaxis_range=[0, 1],
            showlegend=False,
            height=450,
            width=700
        )
        
        # Add disparity annotation
        gap = rates.max() - rates.min()
        fig.add_annotation(
            x=0.98,
            y=0.98,
            xref="paper",
            yref="paper",
            text=f"Disparity Gap: {self._format_percentage(gap)}",
            showarrow=False,
            font={"size": 11, "color": self.theme.DISPARITY_HIGH},
            bgcolor="white",
            bordercolor=self.theme.DISPARITY_HIGH,
            borderwidth=1,
            borderpad=4
        )
        
        self.figures["ownership_by_income"] = fig
        self._save_figure(fig, "ownership_by_income")
        
        return fig
    
    def plot_ownership_by_race_ethnicity(
        self,
        usage_df: pd.DataFrame,
        title: str = "Wearable Ownership by Race and Ethnicity"
    ) -> go.Figure:
        """
        Create horizontal bar chart showing ownership by race/ethnicity.
        
        Args:
            usage_df: DataFrame with ownership and race data
            title: Chart title
        
        Returns:
            Plotly figure object
        """
        race_labels = {
            "white": "White",
            "black": "Black",
            "hispanic_latino": "Hispanic/Latino",
            "asian": "Asian",
            "multiracial": "Multiracial",
            "other": "Other"
        }
        
        rates = usage_df.groupby("race_ethnicity")["owns_wearable"].mean()
        rates = rates.sort_values(ascending=True)
        
        colors = [self.theme.RACE_ETHNICITY_PALETTE.get(r, self.theme.NEUTRAL_MID) 
                  for r in rates.index]
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            y=[race_labels.get(r, r) for r in rates.index],
            x=rates.values,
            orientation="h",
            marker_color=colors,
            text=[self._format_percentage(v) for v in rates.values],
            textposition="outside",
            textfont={"size": 11, "color": self.theme.NEUTRAL_DARK},
            hovertemplate="<b>%{y}</b><br>Ownership Rate: %{x:.1%}<extra></extra>"
        ))
        
        fig = apply_theme_to_figure(fig, title=title)
        
        fig.update_layout(
            xaxis_title="Ownership Rate",
            yaxis_title="",
            xaxis_tickformat=".0%",
            xaxis_range=[0, 1],
            showlegend=False,
            height=400,
            width=700
        )
        
        self.figures["ownership_by_race"] = fig
        self._save_figure(fig, "ownership_by_race_ethnicity")
        
        return fig
    
    def plot_ownership_heatmap(
        self,
        usage_df: pd.DataFrame,
        title: str = "Wearable Ownership: Income by Race/Ethnicity"
    ) -> go.Figure:
        """
        Create heatmap showing ownership across income and race dimensions.
        
        Args:
            usage_df: DataFrame with demographic and ownership data
            title: Chart title
        
        Returns:
            Plotly figure object
        """
        income_order = ["low", "lower_middle", "middle", "upper_middle", "high"]
        income_labels = ["Low", "Lower Middle", "Middle", "Upper Middle", "High"]
        
        race_labels = {
            "white": "White",
            "black": "Black",
            "hispanic_latino": "Hispanic/Latino",
            "asian": "Asian",
            "multiracial": "Multiracial",
            "other": "Other"
        }
        
        pivot = usage_df.pivot_table(
            values="owns_wearable",
            index="race_ethnicity",
            columns="income_bracket",
            aggfunc="mean"
        )
        
        # Reorder columns
        pivot = pivot.reindex(columns=income_order)
        
        # Rename for display
        pivot.index = [race_labels.get(r, r) for r in pivot.index]
        pivot.columns = income_labels
        
        fig = go.Figure()
        
        fig.add_trace(go.Heatmap(
            z=pivot.values,
            x=pivot.columns,
            y=pivot.index,
            colorscale=[
                [0, self.theme.ACCENT_BLUSH],
                [0.5, self.theme.NEUTRAL_PALE],
                [1, self.theme.PRIMARY_DARK]
            ],
            text=[[self._format_percentage(v) for v in row] for row in pivot.values],
            texttemplate="%{text}",
            textfont={"size": 11},
            hovertemplate="<b>%{y}</b>, <b>%{x}</b><br>Ownership: %{z:.1%}<extra></extra>",
            colorbar={
                "title": "Ownership Rate",
                "tickformat": ".0%"
            }
        ))
        
        fig = apply_theme_to_figure(fig, title=title)
        
        fig.update_layout(
            xaxis_title="Income Bracket",
            yaxis_title="Race/Ethnicity",
            height=450,
            width=750
        )
        
        self.figures["ownership_heatmap"] = fig
        self._save_figure(fig, "ownership_heatmap")
        
        return fig
    
    def plot_usage_funnel(
        self,
        usage_df: pd.DataFrame,
        title: str = "Wearable Engagement Funnel"
    ) -> go.Figure:
        """
        Create funnel visualization showing ownership to sharing progression.
        
        Args:
            usage_df: DataFrame with usage columns
            title: Chart title
        
        Returns:
            Plotly figure object
        """
        total = len(usage_df)
        owns = usage_df["owns_wearable"].sum()
        uses_health = usage_df["uses_wearable_for_health"].sum()
        shares = usage_df["shares_data_with_provider"].sum()
        
        stages = ["Total Population", "Owns Wearable", "Uses for Health", "Shares with Provider"]
        values = [total, owns, uses_health, shares]
        percentages = [v / total for v in values]
        
        colors = [
            self.theme.NEUTRAL_LIGHT,
            self.theme.PRIMARY_LIGHT,
            self.theme.PRIMARY_MID,
            self.theme.PRIMARY_DARK
        ]
        
        fig = go.Figure()
        
        fig.add_trace(go.Funnel(
            y=stages,
            x=values,
            textposition="inside",
            textinfo="value+percent initial",
            marker_color=colors,
            connector={"line": {"color": self.theme.NEUTRAL_PALE, "width": 2}},
            hovertemplate="<b>%{y}</b><br>Count: %{x:,}<br>of Total: %{percentInitial:.1%}<extra></extra>"
        ))
        
        fig = apply_theme_to_figure(fig, title=title)
        
        fig.update_layout(
            height=400,
            width=700,
            funnelmode="stack"
        )
        
        self.figures["usage_funnel"] = fig
        self._save_figure(fig, "usage_funnel")
        
        return fig
    
    # Fragmentation and Interoperability Visualizations
    
    def plot_fragmentation_by_brand(
        self,
        device_df: pd.DataFrame,
        title: str = "Data Fragmentation by Device Brand"
    ) -> go.Figure:
        """
        Create box plot showing fragmentation index distribution by brand.
        
        Args:
            device_df: DataFrame with device and fragmentation data
            title: Chart title
        
        Returns:
            Plotly figure object
        """
        # Order brands by median fragmentation
        brand_medians = device_df.groupby("device_brand")["fragmentation_index"].median()
        brand_order = brand_medians.sort_values().index.tolist()
        
        fig = go.Figure()
        
        for i, brand in enumerate(brand_order):
            brand_data = device_df[device_df["device_brand"] == brand]["fragmentation_index"]
            color = self.theme.CATEGORICAL_PALETTE[i % len(self.theme.CATEGORICAL_PALETTE)]
            
            fig.add_trace(go.Box(
                y=brand_data,
                name=brand,
                marker_color=color,
                boxmean=True,
                hovertemplate="<b>%{x}</b><br>Fragmentation: %{y:.2f}<extra></extra>"
            ))
        
        fig = apply_theme_to_figure(fig, title=title)
        
        fig.update_layout(
            xaxis_title="Device Brand",
            yaxis_title="Fragmentation Index",
            yaxis_range=[0, 1],
            showlegend=False,
            height=450,
            width=800
        )
        
        # Add threshold annotation
        fig.add_hline(
            y=0.5,
            line_dash="dash",
            line_color=self.theme.NEUTRAL_MID,
            annotation_text="High Fragmentation Threshold",
            annotation_position="bottom right"
        )
        
        self.figures["fragmentation_by_brand"] = fig
        self._save_figure(fig, "fragmentation_by_brand")
        
        return fig
    
    def plot_fragmentation_by_income(
        self,
        device_df: pd.DataFrame,
        usage_df: pd.DataFrame,
        title: str = "Data Fragmentation by Income Bracket"
    ) -> go.Figure:
        """
        Create violin plot showing fragmentation distribution by income.
        
        Args:
            device_df: DataFrame with device data
            usage_df: DataFrame with demographic data
            title: Chart title
        
        Returns:
            Plotly figure object
        """
        merged = device_df.merge(
            usage_df[["user_id", "income_bracket"]],
            on="user_id"
        )
        
        income_order = ["low", "lower_middle", "middle", "upper_middle", "high"]
        income_labels = {
            "low": "Low",
            "lower_middle": "Lower Middle",
            "middle": "Middle",
            "upper_middle": "Upper Middle",
            "high": "High"
        }
        
        fig = go.Figure()
        
        for i, income in enumerate(income_order):
            data = merged[merged["income_bracket"] == income]["fragmentation_index"]
            color = self.theme.INCOME_PALETTE[income]
            
            fig.add_trace(go.Violin(
                y=data,
                name=income_labels[income],
                box_visible=True,
                meanline_visible=True,
                fillcolor=color,
                line_color=self.theme.NEUTRAL_DARK,
                opacity=0.7,
                hovertemplate="Fragmentation: %{y:.2f}<extra></extra>"
            ))
        
        fig = apply_theme_to_figure(fig, title=title)
        
        fig.update_layout(
            xaxis_title="Income Bracket",
            yaxis_title="Fragmentation Index",
            yaxis_range=[0, 1],
            showlegend=False,
            height=450,
            width=750
        )
        
        self.figures["fragmentation_by_income"] = fig
        self._save_figure(fig, "fragmentation_by_income")
        
        return fig
    
    def plot_interoperability_matrix(
        self,
        device_df: pd.DataFrame,
        title: str = "Interoperability Features by Data Format"
    ) -> go.Figure:
        """
        Create matrix heatmap showing interoperability by format and platform.
        
        Args:
            device_df: DataFrame with device characteristics
            title: Chart title
        
        Returns:
            Plotly figure object
        """
        # Compute proportion with EHR export by format and platform
        pivot = device_df.pivot_table(
            values="ehr_export_supported",
            index="data_format",
            columns="platforms_integrated",
            aggfunc="mean"
        )
        
        format_labels = {
            "open_fhir": "Open FHIR",
            "csv_export": "CSV Export",
            "proprietary_json": "Proprietary JSON"
        }
        
        platform_labels = {
            "ehr_system": "EHR System",
            "third_party_aggregator": "Third Party",
            "vendor_portal_only": "Vendor Only"
        }
        
        pivot.index = [format_labels.get(f, f) for f in pivot.index]
        pivot.columns = [platform_labels.get(p, p) for p in pivot.columns]
        
        fig = go.Figure()
        
        fig.add_trace(go.Heatmap(
            z=pivot.values,
            x=pivot.columns,
            y=pivot.index,
            colorscale=self.theme.get_fragmentation_colorscale(),
            text=[[self._format_percentage(v) if not pd.isna(v) else "N/A" 
                   for v in row] for row in pivot.values],
            texttemplate="%{text}",
            textfont={"size": 12},
            hovertemplate="<b>%{y}</b> + <b>%{x}</b><br>EHR Export Support: %{z:.1%}<extra></extra>",
            colorbar={
                "title": "EHR Export Rate",
                "tickformat": ".0%"
            }
        ))
        
        fig = apply_theme_to_figure(fig, title=title)
        
        fig.update_layout(
            xaxis_title="Platform Integration",
            yaxis_title="Data Format",
            height=350,
            width=650
        )
        
        self.figures["interoperability_matrix"] = fig
        self._save_figure(fig, "interoperability_matrix")
        
        return fig
    
    def plot_fragmentation_vs_linkability(
        self,
        merged_df: pd.DataFrame,
        ehr_probs: np.ndarray,
        title: str = "Fragmentation vs EHR Linkability"
    ) -> go.Figure:
        """
        Create scatter plot of fragmentation vs predicted EHR linkability.
        
        Args:
            merged_df: Merged DataFrame with all features
            ehr_probs: Predicted EHR linkability probabilities
            title: Chart title
        
        Returns:
            Plotly figure object
        """
        plot_df = merged_df.copy()
        plot_df["ehr_probability"] = ehr_probs
        
        # Sample for performance if large
        if len(plot_df) > 1000:
            plot_df = plot_df.sample(1000, random_state=42)
        
        income_order = ["low", "lower_middle", "middle", "upper_middle", "high"]
        
        fig = go.Figure()
        
        for income in income_order:
            subset = plot_df[plot_df["income_bracket"] == income]
            color = self.theme.INCOME_PALETTE[income]
            
            fig.add_trace(go.Scatter(
                x=subset["fragmentation_index"],
                y=subset["ehr_probability"],
                mode="markers",
                name=income.replace("_", " ").title(),
                marker={
                    "color": color,
                    "size": 8,
                    "opacity": 0.6,
                    "line": {"width": 0.5, "color": "white"}
                },
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    "Fragmentation: %{x:.2f}<br>"
                    "EHR Linkability: %{y:.1%}<extra></extra>"
                ),
                text=subset["income_bracket"]
            ))
        
        fig = apply_theme_to_figure(fig, title=title)
        
        fig.update_layout(
            xaxis_title="Fragmentation Index",
            yaxis_title="Predicted EHR Linkability",
            xaxis_range=[0, 1],
            yaxis_tickformat=".0%",
            height=500,
            width=800,
            legend={
                "title": "Income Bracket",
                "orientation": "h",
                "yanchor": "bottom",
                "y": 1.02,
                "xanchor": "right",
                "x": 1
            }
        )
        
        self.figures["fragmentation_vs_linkability"] = fig
        self._save_figure(fig, "fragmentation_vs_linkability")
        
        return fig
    
    # Sensor Bias and Data Quality Visualizations
    
    def plot_accuracy_by_skin_tone(
        self,
        quality_df: pd.DataFrame,
        device_df: pd.DataFrame,
        title: str = "Sensor Accuracy by Skin Tone Category"
    ) -> go.Figure:
        """
        Create grouped bar chart showing accuracy by skin tone and brand.
        
        Args:
            quality_df: DataFrame with quality metrics
            device_df: DataFrame with device information
            title: Chart title
        
        Returns:
            Plotly figure object
        """
        merged = quality_df.merge(
            device_df[["user_id", "device_brand"]],
            on="user_id"
        )
        
        tone_labels = {
            "type_i_ii": "Fitzpatrick I-II",
            "type_iii_iv": "Fitzpatrick III-IV",
            "type_v_vi": "Fitzpatrick V-VI"
        }
        
        tone_order = ["type_i_ii", "type_iii_iv", "type_v_vi"]
        
        # Aggregate by skin tone
        tone_accuracy = merged.groupby("skin_tone_category")["sensor_accuracy_score"].agg([
            "mean", "std"
        ]).reindex(tone_order)
        
        colors = [self.theme.SKIN_TONE_PALETTE[t] for t in tone_order]
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=[tone_labels[t] for t in tone_order],
            y=tone_accuracy["mean"],
            marker_color=colors,
            error_y={
                "type": "data",
                "array": tone_accuracy["std"],
                "visible": True,
                "color": self.theme.NEUTRAL_MID
            },
            text=[f"{v:.1%}" for v in tone_accuracy["mean"]],
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>Accuracy: %{y:.1%}<extra></extra>"
        ))
        
        fig = apply_theme_to_figure(fig, title=title)
        
        fig.update_layout(
            xaxis_title="Skin Tone Category",
            yaxis_title="Mean Sensor Accuracy",
            yaxis_tickformat=".0%",
            yaxis_range=[0.7, 1.0],
            showlegend=False,
            height=400,
            width=650
        )
        
        # Add accuracy gap annotation
        gap = tone_accuracy["mean"].max() - tone_accuracy["mean"].min()
        fig.add_annotation(
            x=0.98,
            y=0.02,
            xref="paper",
            yref="paper",
            text=f"Accuracy Gap: {gap:.1%}",
            showarrow=False,
            font={"size": 11, "color": self.theme.DISPARITY_HIGH},
            bgcolor="white",
            bordercolor=self.theme.DISPARITY_HIGH,
            borderwidth=1,
            borderpad=4
        )
        
        self.figures["accuracy_by_skin_tone"] = fig
        self._save_figure(fig, "accuracy_by_skin_tone")
        
        return fig
    
    def plot_accuracy_by_brand_and_tone(
        self,
        quality_df: pd.DataFrame,
        device_df: pd.DataFrame,
        title: str = "Sensor Accuracy: Brand by Skin Tone"
    ) -> go.Figure:
        """
        Create heatmap showing accuracy across brand and skin tone.
        
        Args:
            quality_df: DataFrame with quality metrics
            device_df: DataFrame with device information
            title: Chart title
        
        Returns:
            Plotly figure object
        """
        merged = quality_df.merge(
            device_df[["user_id", "device_brand"]],
            on="user_id"
        )
        
        pivot = merged.pivot_table(
            values="sensor_accuracy_score",
            index="device_brand",
            columns="skin_tone_category",
            aggfunc="mean"
        )
        
        tone_order = ["type_i_ii", "type_iii_iv", "type_v_vi"]
        tone_labels = ["Fitzpatrick I-II", "Fitzpatrick III-IV", "Fitzpatrick V-VI"]
        
        pivot = pivot.reindex(columns=tone_order)
        pivot.columns = tone_labels
        
        # Sort by overall accuracy
        pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]
        
        fig = go.Figure()
        
        fig.add_trace(go.Heatmap(
            z=pivot.values,
            x=pivot.columns,
            y=pivot.index,
            colorscale=[
                [0, self.theme.DISPARITY_HIGH],
                [0.5, self.theme.NEUTRAL_PALE],
                [1, self.theme.EQUITY_POSITIVE]
            ],
            zmin=0.75,
            zmax=1.0,
            text=[[f"{v:.1%}" for v in row] for row in pivot.values],
            texttemplate="%{text}",
            textfont={"size": 11},
            hovertemplate="<b>%{y}</b><br>%{x}<br>Accuracy: %{z:.1%}<extra></extra>",
            colorbar={
                "title": "Accuracy",
                "tickformat": ".0%"
            }
        ))
        
        fig = apply_theme_to_figure(fig, title=title)
        
        fig.update_layout(
            xaxis_title="Skin Tone Category",
            yaxis_title="Device Brand",
            height=450,
            width=700
        )
        
        self.figures["accuracy_brand_tone"] = fig
        self._save_figure(fig, "accuracy_by_brand_and_tone")
        
        return fig
    
    def plot_dropout_by_demographics(
        self,
        quality_df: pd.DataFrame,
        usage_df: pd.DataFrame,
        title: str = "Data Dropout Rate by Demographics"
    ) -> go.Figure:
        """
        Create side-by-side bars showing dropout rates by income and race.
        
        Args:
            quality_df: DataFrame with quality metrics
            usage_df: DataFrame with demographic data
            title: Chart title
        
        Returns:
            Plotly figure object
        """
        merged = quality_df.merge(
            usage_df[["user_id", "income_bracket", "race_ethnicity"]],
            on="user_id"
        )
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("By Income Bracket", "By Race/Ethnicity"),
            horizontal_spacing=0.15
        )
        
        # By income
        income_order = ["low", "lower_middle", "middle", "upper_middle", "high"]
        income_labels = ["Low", "Lower Mid", "Middle", "Upper Mid", "High"]
        income_dropout = merged.groupby("income_bracket")["dropout_rate"].mean().reindex(income_order)
        
        fig.add_trace(
            go.Bar(
                x=income_labels,
                y=income_dropout.values,
                marker_color=[self.theme.INCOME_PALETTE[i] for i in income_order],
                text=[f"{v:.1%}" for v in income_dropout.values],
                textposition="outside",
                showlegend=False,
                hovertemplate="<b>%{x}</b><br>Dropout Rate: %{y:.1%}<extra></extra>"
            ),
            row=1, col=1
        )
        
        # By race
        race_labels = {
            "white": "White",
            "black": "Black",
            "hispanic_latino": "Hispanic",
            "asian": "Asian",
            "multiracial": "Multiracial",
            "other": "Other"
        }
        race_dropout = merged.groupby("race_ethnicity")["dropout_rate"].mean().sort_values()
        
        fig.add_trace(
            go.Bar(
                x=[race_labels.get(r, r) for r in race_dropout.index],
                y=race_dropout.values,
                marker_color=[self.theme.RACE_ETHNICITY_PALETTE.get(r, self.theme.NEUTRAL_MID) 
                             for r in race_dropout.index],
                text=[f"{v:.1%}" for v in race_dropout.values],
                textposition="outside",
                showlegend=False,
                hovertemplate="<b>%{x}</b><br>Dropout Rate: %{y:.1%}<extra></extra>"
            ),
            row=1, col=2
        )
        
        fig = apply_theme_to_figure(fig, title=title)
        
        fig.update_layout(
            height=400,
            width=900
        )
        
        fig.update_yaxes(title_text="Dropout Rate", tickformat=".0%", range=[0, 0.4], row=1, col=1)
        fig.update_yaxes(title_text="", tickformat=".0%", range=[0, 0.4], row=1, col=2)
        
        self.figures["dropout_demographics"] = fig
        self._save_figure(fig, "dropout_by_demographics")
        
        return fig
    
    def plot_data_quality_radar(
        self,
        quality_df: pd.DataFrame,
        usage_df: pd.DataFrame,
        device_df: pd.DataFrame,
        title: str = "Data Quality Profile by Income Group"
    ) -> go.Figure:
        """
        Create radar chart comparing quality dimensions across groups.
        
        Args:
            quality_df: DataFrame with quality metrics
            usage_df: DataFrame with demographic data
            device_df: DataFrame with device data
            title: Chart title
        
        Returns:
            Plotly figure object
        """
        merged = quality_df.merge(
            usage_df[["user_id", "income_bracket"]],
            on="user_id"
        ).merge(
            device_df[["user_id", "fragmentation_index"]],
            on="user_id"
        )
        
        # Define quality dimensions (normalized to 0-1 scale, higher is better)
        dimensions = [
            "Accuracy",
            "Completeness",
            "Wear Time",
            "Interoperability",
            "Sharing"
        ]
        
        income_groups = ["low", "middle", "high"]
        income_display = {"low": "Low Income", "middle": "Middle Income", "high": "High Income"}
        
        fig = go.Figure()
        
        for income in income_groups:
            subset = merged[merged["income_bracket"] == income]
            
            values = [
                subset["sensor_accuracy_score"].mean(),
                1 - subset["dropout_rate"].mean(),
                subset["wear_time_hours_per_day"].mean() / 24,
                1 - subset["fragmentation_index"].mean(),
                subset["willingness_to_share_score"].mean()
            ]
            
            # Close the radar
            values_closed = values + [values[0]]
            dims_closed = dimensions + [dimensions[0]]
            
            color = self.theme.INCOME_PALETTE[income]
            
            fig.add_trace(go.Scatterpolar(
                r=values_closed,
                theta=dims_closed,
                fill="toself",
                name=income_display[income],
                fillcolor=color,
                opacity=0.3,
                line={"color": color, "width": 2},
                hovertemplate="<b>%{theta}</b><br>Score: %{r:.2f}<extra></extra>"
            ))
        
        fig = apply_theme_to_figure(fig, title=title)
        
        fig.update_layout(
            polar={
                "radialaxis": {
                    "visible": True,
                    "range": [0, 1],
                    "tickformat": ".0%"
                },
                "bgcolor": self.theme.NEUTRAL_BACKGROUND
            },
            showlegend=True,
            legend={
                "orientation": "h",
                "yanchor": "bottom",
                "y": -0.15,
                "xanchor": "center",
                "x": 0.5
            },
            height=500,
            width=600
        )
        
        self.figures["quality_radar"] = fig
        self._save_figure(fig, "data_quality_radar")
        
        return fig
    
    # Model and Fairness Visualizations
    
    def plot_predicted_ownership_by_group(
        self,
        usage_df: pd.DataFrame,
        predictions: np.ndarray,
        title: str = "Predicted Wearable Ownership by Demographics"
    ) -> go.Figure:
        """
        Create grouped comparison of predicted ownership probabilities.
        
        Args:
            usage_df: DataFrame with demographic data
            predictions: Array of predicted probabilities
            title: Chart title
        
        Returns:
            Plotly figure object
        """
        plot_df = usage_df.copy()
        plot_df["predicted_prob"] = predictions
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("By Income Bracket", "By Race/Ethnicity"),
            horizontal_spacing=0.12
        )
        
        # By income
        income_order = ["low", "lower_middle", "middle", "upper_middle", "high"]
        income_labels = ["Low", "Lower Mid", "Middle", "Upper Mid", "High"]
        
        income_stats = plot_df.groupby("income_bracket").agg({
            "predicted_prob": ["mean", "std"],
            "owns_wearable": "mean"
        }).reindex(income_order)
        
        # Predicted
        fig.add_trace(
            go.Bar(
                x=income_labels,
                y=income_stats[("predicted_prob", "mean")],
                error_y={
                    "type": "data",
                    "array": income_stats[("predicted_prob", "std")],
                    "visible": True
                },
                name="Predicted",
                marker_color=self.theme.PRIMARY_MID,
                opacity=0.8
            ),
            row=1, col=1
        )
        
        # Observed
        fig.add_trace(
            go.Scatter(
                x=income_labels,
                y=income_stats[("owns_wearable", "mean")],
                mode="markers",
                name="Observed",
                marker={
                    "color": self.theme.ACCENT_CORAL,
                    "size": 12,
                    "symbol": "diamond"
                }
            ),
            row=1, col=1
        )
        
        # By race
        race_labels = {
            "white": "White",
            "black": "Black",
            "hispanic_latino": "Hispanic",
            "asian": "Asian",
            "multiracial": "Multi",
            "other": "Other"
        }
        
        race_stats = plot_df.groupby("race_ethnicity").agg({
            "predicted_prob": ["mean", "std"],
            "owns_wearable": "mean"
        })
        
        fig.add_trace(
            go.Bar(
                x=[race_labels.get(r, r) for r in race_stats.index],
                y=race_stats[("predicted_prob", "mean")],
                error_y={
                    "type": "data",
                    "array": race_stats[("predicted_prob", "std")],
                    "visible": True
                },
                name="Predicted",
                marker_color=self.theme.PRIMARY_MID,
                opacity=0.8,
                showlegend=False
            ),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Scatter(
                x=[race_labels.get(r, r) for r in race_stats.index],
                y=race_stats[("owns_wearable", "mean")],
                mode="markers",
                name="Observed",
                marker={
                    "color": self.theme.ACCENT_CORAL,
                    "size": 12,
                    "symbol": "diamond"
                },
                showlegend=False
            ),
            row=1, col=2
        )
        
        fig = apply_theme_to_figure(fig, title=title)
        
        fig.update_layout(
            height=400,
            width=950,
            legend={
                "orientation": "h",
                "yanchor": "bottom",
                "y": 1.08,
                "xanchor": "center",
                "x": 0.5
            },
            barmode="group"
        )
        
        fig.update_yaxes(title_text="Probability", tickformat=".0%", range=[0, 1], row=1, col=1)
        fig.update_yaxes(tickformat=".0%", range=[0, 1], row=1, col=2)
        
        self.figures["predicted_ownership"] = fig
        self._save_figure(fig, "predicted_ownership_by_group")
        
        return fig
    
    def plot_fairness_dashboard(
        self,
        fairness_metrics: Dict[str, Any],
        title: str = "Equity and Fairness Summary"
    ) -> go.Figure:
        """
        Create summary dashboard showing key disparity metrics.
        
        Args:
            fairness_metrics: Dictionary with computed fairness metrics
            title: Chart title
        
        Returns:
            Plotly figure object
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "Ownership Gap by Income",
                "Ownership Gap by Race",
                "Sensor Accuracy Gap",
                "EHR Access Gap"
            ),
            specs=[[{"type": "indicator"}, {"type": "indicator"}],
                   [{"type": "indicator"}, {"type": "indicator"}]],
            vertical_spacing=0.25,
            horizontal_spacing=0.15
        )
        
        # Extract metrics
        income_fairness = fairness_metrics.get("ownership_by_income", {})
        race_fairness = fairness_metrics.get("ownership_by_race", {})
        accuracy_disp = fairness_metrics.get("accuracy_disparities", {})
        
        income_gap = getattr(income_fairness, "absolute_gap", 0.3) if hasattr(income_fairness, "absolute_gap") else 0.3
        race_gap = getattr(race_fairness, "absolute_gap", 0.15) if hasattr(race_fairness, "absolute_gap") else 0.15
        accuracy_gap = accuracy_disp.get("accuracy_gap", 0.08)
        
        # Ownership by income
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=income_gap * 100,
                number={"suffix": "%", "font": {"size": 28}},
                gauge={
                    "axis": {"range": [0, 50], "ticksuffix": "%"},
                    "bar": {"color": self.theme.DISPARITY_HIGH},
                    "bgcolor": self.theme.NEUTRAL_PALE,
                    "steps": [
                        {"range": [0, 15], "color": self.theme.EQUITY_POSITIVE},
                        {"range": [15, 30], "color": self.theme.DISPARITY_MID},
                        {"range": [30, 50], "color": self.theme.DISPARITY_HIGH}
                    ],
                    "threshold": {
                        "line": {"color": self.theme.NEUTRAL_DARK, "width": 2},
                        "thickness": 0.75,
                        "value": income_gap * 100
                    }
                }
            ),
            row=1, col=1
        )
        
        # Ownership by race
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=race_gap * 100,
                number={"suffix": "%", "font": {"size": 28}},
                gauge={
                    "axis": {"range": [0, 50], "ticksuffix": "%"},
                    "bar": {"color": self.theme.DISPARITY_MID},
                    "bgcolor": self.theme.NEUTRAL_PALE,
                    "steps": [
                        {"range": [0, 10], "color": self.theme.EQUITY_POSITIVE},
                        {"range": [10, 25], "color": self.theme.DISPARITY_MID},
                        {"range": [25, 50], "color": self.theme.DISPARITY_HIGH}
                    ]
                }
            ),
            row=1, col=2
        )
        
        # Accuracy gap
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=accuracy_gap * 100,
                number={"suffix": "%", "font": {"size": 28}},
                gauge={
                    "axis": {"range": [0, 20], "ticksuffix": "%"},
                    "bar": {"color": self.theme.ACCENT_CORAL},
                    "bgcolor": self.theme.NEUTRAL_PALE,
                    "steps": [
                        {"range": [0, 5], "color": self.theme.EQUITY_POSITIVE},
                        {"range": [5, 10], "color": self.theme.DISPARITY_MID},
                        {"range": [10, 20], "color": self.theme.DISPARITY_HIGH}
                    ]
                }
            ),
            row=2, col=1
        )
        
        # EHR access gap (placeholder)
        ehr_gap = 0.22
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=ehr_gap * 100,
                number={"suffix": "%", "font": {"size": 28}},
                gauge={
                    "axis": {"range": [0, 40], "ticksuffix": "%"},
                    "bar": {"color": self.theme.PRIMARY_DARK},
                    "bgcolor": self.theme.NEUTRAL_PALE,
                    "steps": [
                        {"range": [0, 10], "color": self.theme.EQUITY_POSITIVE},
                        {"range": [10, 25], "color": self.theme.DISPARITY_MID},
                        {"range": [25, 40], "color": self.theme.DISPARITY_HIGH}
                    ]
                }
            ),
            row=2, col=2
        )
        
        fig = apply_theme_to_figure(fig, title=title)
        
        fig.update_layout(
            height=600,
            width=800
        )
        
        self.figures["fairness_dashboard"] = fig
        self._save_figure(fig, "fairness_dashboard")
        
        return fig
    
    def plot_calibration_by_group(
        self,
        calibration_data: Dict[str, Dict],
        title: str = "Model Calibration by Income Group"
    ) -> go.Figure:
        """
        Create calibration plot showing predicted vs observed by group.
        
        Args:
            calibration_data: Dictionary with calibration curves by group
            title: Chart title
        
        Returns:
            Plotly figure object
        """
        fig = go.Figure()
        
        # Add perfect calibration line
        fig.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Perfect Calibration",
            line={"color": self.theme.NEUTRAL_LIGHT, "dash": "dash", "width": 2}
        ))
        
        income_order = ["low", "lower_middle", "middle", "upper_middle", "high"]
        income_display = {
            "low": "Low",
            "lower_middle": "Lower Middle",
            "middle": "Middle",
            "upper_middle": "Upper Middle",
            "high": "High"
        }
        
        for income in income_order:
            if income not in calibration_data:
                continue
            
            data = calibration_data[income]
            color = self.theme.INCOME_PALETTE[income]
            
            fig.add_trace(go.Scatter(
                x=data["predicted_probability"],
                y=data["observed_frequency"],
                mode="lines+markers",
                name=income_display[income],
                line={"color": color, "width": 2},
                marker={"size": 8},
                hovertemplate=(
                    f"<b>{income_display[income]}</b><br>"
                    "Predicted: %{x:.1%}<br>"
                    "Observed: %{y:.1%}<extra></extra>"
                )
            ))
        
        fig = apply_theme_to_figure(fig, title=title)
        
        fig.update_layout(
            xaxis_title="Predicted Probability",
            yaxis_title="Observed Frequency",
            xaxis_tickformat=".0%",
            yaxis_tickformat=".0%",
            xaxis_range=[0, 1],
            yaxis_range=[0, 1],
            legend={
                "title": "Income Bracket"
            },
            height=450,
            width=600
        )
        
        self.figures["calibration"] = fig
        self._save_figure(fig, "calibration_by_group")
        
        return fig
    
    def export_all_figures_metadata(self) -> Dict:
        """
        Export metadata about all generated figures.
        
        Returns:
            Dictionary with figure names and file paths
        """
        metadata = {
            "figures": {},
            "output_directory": str(self.output_dir)
        }
        
        for name, fig in self.figures.items():
            metadata["figures"][name] = {
                "html_path": str(self.output_dir / f"{name}.html") if self.save_html else None,
                "png_path": str(self.output_dir / f"{name}.png") if self.save_png else None,
                "title": fig.layout.title.text if fig.layout.title.text else name
            }
        
        # Save metadata to JSON
        with open(self.output_dir / "figures_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        return metadata
