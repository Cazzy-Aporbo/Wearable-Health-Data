
# Wearable Health Equity Analysis Report

**Analysis Date:** 2025-11-27  
**Scenario:** baseline  
**Author:** Cazzy Aporbo  

---



## Executive Summary

This analysis examined wearable health technology adoption and data quality 
across a population of 2,000 individuals using the baseline scenario.

**Key Findings:**

Overall wearable ownership stands at 49.1%, with 
51.3% of owners actively using devices for health 
purposes. Among health-focused users, 41.5% share 
data with healthcare providers.

Ownership disparities by income bracket show a gap of 49.8% 
between the highest and lowest income groups. Sensor accuracy varies by skin tone 
category, with a gap of 11.0% between groups with 
highest and lowest measured accuracy.

Data fragmentation shows meaningful variation across income levels, with a 
0.12 point difference in fragmentation index between income extremes. 
This suggests that lower-income individuals may face additional barriers to 
data interoperability and clinical integration.



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

The analyzed population includes 2,000 individuals distributed 
across income brackets, with representation from multiple racial and ethnic groups 
and geographic settings.



## Equity and Disparity Findings

This section summarizes observed patterns related to health equity and 
representativity in wearable health data.

### Ownership Disparities


Wearable ownership varies substantially by income. High income individuals 
own wearables at a rate of 73.8%, compared 
to 24.1% among low income individuals.


Ownership also varies by race and ethnicity. The highest ownership rate 
(51.2%) was observed among asian 
respondents, while the lowest (43.5%) was among 
other respondents.


These patterns reflect broader digital divides documented in research on 
health technology access and socioeconomic factors.

### Sensor Accuracy Disparities


Sensor accuracy shows variation by skin tone category. The highest accuracy 
was measured among individuals with type i ii skin tones, 
while the lowest was among those with type v vi skin tones, 
representing a gap of 11.0%.


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



## Interoperability and Fragmentation Findings

Data fragmentation affects the ability to integrate wearable health information 
with clinical systems and other data sources.

### Fragmentation by Device Ecosystem


Fragmentation varies substantially by device brand. Apple devices 
show the lowest fragmentation (mean index 0.25), 
while Other devices show the highest (mean index 
0.63).


Devices with lower fragmentation typically offer better API access, standardized 
data formats, and established EHR integration pathways.

### Fragmentation and Socioeconomic Status


Lower income individuals tend to use devices with higher fragmentation 
indices, reflecting both device choice patterns and the availability of 
interoperable options at different price points.


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



## Limitations and Considerations


**Synthetic Data Note:** This analysis uses synthetically generated data 
designed to reflect patterns documented in published research. While the 
synthetic data captures directional relationships, the specific magnitudes 
and distributions should not be interpreted as precise estimates of 
real-world prevalence or effect sizes.


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



## Visualizations

The following visualizations are available for this analysis:

**Wearable Ownership by Income Bracket**
  Static: my_results/visualizations/ownership_by_income.png
  Interactive: my_results/visualizations/ownership_by_income.html

**Wearable Ownership by Race and Ethnicity**
  Static: my_results/visualizations/ownership_by_race.png
  Interactive: my_results/visualizations/ownership_by_race.html

**Wearable Ownership: Income by Race/Ethnicity**
  Static: my_results/visualizations/ownership_heatmap.png
  Interactive: my_results/visualizations/ownership_heatmap.html

**Wearable Engagement Funnel**
  Static: my_results/visualizations/usage_funnel.png
  Interactive: my_results/visualizations/usage_funnel.html

**Data Fragmentation by Device Brand**
  Static: my_results/visualizations/fragmentation_by_brand.png
  Interactive: my_results/visualizations/fragmentation_by_brand.html

**Data Fragmentation by Income Bracket**
  Static: my_results/visualizations/fragmentation_by_income.png
  Interactive: my_results/visualizations/fragmentation_by_income.html

**Interoperability Features by Data Format**
  Static: my_results/visualizations/interoperability_matrix.png
  Interactive: my_results/visualizations/interoperability_matrix.html

**Fragmentation vs EHR Linkability**
  Static: my_results/visualizations/fragmentation_vs_linkability.png
  Interactive: my_results/visualizations/fragmentation_vs_linkability.html

**Sensor Accuracy by Skin Tone Category**
  Static: my_results/visualizations/accuracy_by_skin_tone.png
  Interactive: my_results/visualizations/accuracy_by_skin_tone.html

**Sensor Accuracy: Brand by Skin Tone**
  Static: my_results/visualizations/accuracy_brand_tone.png
  Interactive: my_results/visualizations/accuracy_brand_tone.html

**Data Dropout Rate by Demographics**
  Static: my_results/visualizations/dropout_demographics.png
  Interactive: my_results/visualizations/dropout_demographics.html

**Data Quality Profile by Income Group**
  Static: my_results/visualizations/quality_radar.png
  Interactive: my_results/visualizations/quality_radar.html

**Predicted Wearable Ownership by Demographics**
  Static: my_results/visualizations/predicted_ownership.png
  Interactive: my_results/visualizations/predicted_ownership.html

**Equity and Fairness Summary**
  Static: my_results/visualizations/fairness_dashboard.png
  Interactive: my_results/visualizations/fairness_dashboard.html

**Model Calibration by Income Group**
  Static: my_results/visualizations/calibration.png
  Interactive: my_results/visualizations/calibration.html
