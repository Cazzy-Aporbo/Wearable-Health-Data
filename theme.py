"""
Theme and Color Palette for Wearable Health Equity Visualizations
Written by Cazzy Aporbo

This module defines the visual identity for all charts and dashboards.
Inspired by publication-grade aesthetics with pastel gradients and
sophisticated color harmonies suitable for healthcare research contexts.
"""

from typing import Dict, List, Tuple


class WearableEquityTheme:
    """
    A cohesive visual theme for wearable health equity analysis.
    
    Design Philosophy:
        Soft, accessible colors that communicate data clearly while
        maintaining visual elegance appropriate for academic and
        policy audiences. Avoids harsh contrasts and prioritizes
        readability across different display contexts.
    """
    
    # Primary palette: soft teals and sage greens
    PRIMARY_DARK = "#2D5A5A"
    PRIMARY_MID = "#4A8B8B"
    PRIMARY_LIGHT = "#7FBFBF"
    PRIMARY_PALE = "#B8E0E0"
    
    # Accent palette: warm coral and amber tones
    ACCENT_WARM = "#E8A87C"
    ACCENT_CORAL = "#E27D60"
    ACCENT_ROSE = "#C38D9E"
    ACCENT_BLUSH = "#F5D0C5"
    
    # Neutral palette: warm grays
    NEUTRAL_DARK = "#3D3D3D"
    NEUTRAL_MID = "#6B6B6B"
    NEUTRAL_LIGHT = "#A8A8A8"
    NEUTRAL_PALE = "#E8E4E1"
    NEUTRAL_BACKGROUND = "#FAFAFA"
    
    # Semantic colors for disparity highlighting
    DISPARITY_HIGH = "#C75146"
    DISPARITY_MID = "#E8A87C"
    DISPARITY_LOW = "#7FBFBF"
    EQUITY_POSITIVE = "#6BAF92"
    
    # Sequential palette for gradient visualizations
    SEQUENTIAL_PALETTE = [
        "#F5F0E8",
        "#E0D8CC",
        "#C5D1C5",
        "#A3C4BC",
        "#7FBFBF",
        "#5AA3A3",
        "#3D8686",
        "#2D5A5A"
    ]
    
    # Categorical palette for group comparisons
    CATEGORICAL_PALETTE = [
        "#4A8B8B",  # teal
        "#E27D60",  # coral
        "#C38D9E",  # mauve
        "#E8A87C",  # amber
        "#85A392",  # sage
        "#8B7355",  # taupe
        "#A4C3D2",  # sky
        "#D4A5A5"   # dusty rose
    ]
    
    # Income bracket specific colors (ordered low to high)
    INCOME_PALETTE = {
        "low": "#C75146",
        "lower_middle": "#E8A87C",
        "middle": "#E0D8CC",
        "upper_middle": "#7FBFBF",
        "high": "#2D5A5A"
    }
    
    # Race/ethnicity palette (alphabetical, no hierarchy implied)
    RACE_ETHNICITY_PALETTE = {
        "asian": "#4A8B8B",
        "black": "#E27D60",
        "hispanic_latino": "#C38D9E",
        "white": "#E8A87C",
        "multiracial": "#85A392",
        "other": "#A4C3D2"
    }
    
    # Skin tone accuracy palette (Fitzpatrick scale inspired)
    SKIN_TONE_PALETTE = {
        "type_i_ii": "#F5D0C5",
        "type_iii_iv": "#D4A574",
        "type_v_vi": "#8B5A3C"
    }
    
    # Typography settings
    FONT_FAMILY = "Source Sans Pro, Segoe UI, sans-serif"
    FONT_FAMILY_MONO = "Source Code Pro, Consolas, monospace"
    TITLE_SIZE = 18
    SUBTITLE_SIZE = 14
    AXIS_LABEL_SIZE = 12
    TICK_SIZE = 10
    ANNOTATION_SIZE = 10
    
    # Layout constants
    MARGIN_TOP = 80
    MARGIN_BOTTOM = 60
    MARGIN_LEFT = 80
    MARGIN_RIGHT = 40
    PLOT_HEIGHT = 500
    PLOT_WIDTH = 800
    
    @classmethod
    def get_plotly_template(cls) -> Dict:
        """
        Returns a Plotly template dictionary for consistent styling.
        
        Returns:
            Dictionary suitable for plotly.io.templates
        """
        return {
            "layout": {
                "font": {
                    "family": cls.FONT_FAMILY,
                    "size": cls.AXIS_LABEL_SIZE,
                    "color": cls.NEUTRAL_DARK
                },
                "title": {
                    "font": {
                        "size": cls.TITLE_SIZE,
                        "color": cls.NEUTRAL_DARK
                    },
                    "x": 0.02,
                    "xanchor": "left"
                },
                "paper_bgcolor": cls.NEUTRAL_BACKGROUND,
                "plot_bgcolor": cls.NEUTRAL_BACKGROUND,
                "colorway": cls.CATEGORICAL_PALETTE,
                "margin": {
                    "t": cls.MARGIN_TOP,
                    "b": cls.MARGIN_BOTTOM,
                    "l": cls.MARGIN_LEFT,
                    "r": cls.MARGIN_RIGHT
                },
                "xaxis": {
                    "gridcolor": cls.NEUTRAL_PALE,
                    "linecolor": cls.NEUTRAL_LIGHT,
                    "tickfont": {"size": cls.TICK_SIZE},
                    "title": {"font": {"size": cls.AXIS_LABEL_SIZE}}
                },
                "yaxis": {
                    "gridcolor": cls.NEUTRAL_PALE,
                    "linecolor": cls.NEUTRAL_LIGHT,
                    "tickfont": {"size": cls.TICK_SIZE},
                    "title": {"font": {"size": cls.AXIS_LABEL_SIZE}}
                },
                "legend": {
                    "bgcolor": "rgba(255,255,255,0.8)",
                    "bordercolor": cls.NEUTRAL_PALE,
                    "borderwidth": 1,
                    "font": {"size": cls.TICK_SIZE}
                },
                "hoverlabel": {
                    "bgcolor": "white",
                    "bordercolor": cls.NEUTRAL_LIGHT,
                    "font": {"family": cls.FONT_FAMILY, "size": cls.TICK_SIZE}
                }
            }
        }
    
    @classmethod
    def get_disparity_colorscale(cls) -> List[Tuple[float, str]]:
        """
        Returns a diverging colorscale for disparity visualizations.
        Center represents equity, extremes represent disparity.
        
        Returns:
            List of tuples for Plotly colorscale
        """
        return [
            (0.0, cls.DISPARITY_HIGH),
            (0.25, cls.DISPARITY_MID),
            (0.5, cls.NEUTRAL_PALE),
            (0.75, cls.PRIMARY_LIGHT),
            (1.0, cls.PRIMARY_DARK)
        ]
    
    @classmethod
    def get_fragmentation_colorscale(cls) -> List[Tuple[float, str]]:
        """
        Returns a sequential colorscale for fragmentation index.
        Lower values (more interoperable) are lighter.
        
        Returns:
            List of tuples for Plotly colorscale
        """
        return [
            (0.0, "#E8F4F4"),
            (0.33, "#A3C4BC"),
            (0.66, "#5AA3A3"),
            (1.0, "#2D5A5A")
        ]


def apply_theme_to_figure(fig, title: str = None, subtitle: str = None):
    """
    Apply the wearable equity theme to a Plotly figure.
    
    Args:
        fig: Plotly figure object
        title: Optional main title
        subtitle: Optional subtitle (rendered as annotation)
    
    Returns:
        Modified figure with theme applied
    """
    theme = WearableEquityTheme()
    template = theme.get_plotly_template()
    
    fig.update_layout(template["layout"])
    
    if title:
        full_title = title
        if subtitle:
            full_title = f"{title}<br><span style='font-size:{theme.SUBTITLE_SIZE}px;color:{theme.NEUTRAL_MID}'>{subtitle}</span>"
        fig.update_layout(title={"text": full_title, "x": 0.02, "xanchor": "left"})
    
    return fig
