# Data Dictionary

## cholera_cases.csv

| Column | Type | Description |
|---|---|---|
| date | date | Week start date (Monday) |
| district | string | Zimbabwe administrative district |
| province | string | Province name |
| cases | int | Weekly confirmed + suspected cholera cases |
| deaths | int | Weekly cholera-attributable deaths |
| attack_rate_per_100k | float | Weekly cases per 100,000 population |
| population | int | District population estimate |
| wash_coverage | float | WASH coverage ratio (0–1) at time of record |
| cumulative_cases | int | Running total cases per district since Jan 2018 |

## climate.csv

| Column | Type | Description |
|---|---|---|
| date | date | Week start date |
| district | string | District name |
| rainfall_mm | float | Total weekly precipitation (mm) |
| temperature_c | float | Mean weekly temperature (°C) |
| humidity_pct | float | Mean relative humidity (%) |
| rainfall_anomaly_mm | float | Deviation from 10-year climatological weekly mean (mm) |
| rainfall_anomaly_pct | float | Percentage deviation from climatological mean |

## demographics.csv

| Column | Type | Description |
|---|---|---|
| year | int | Calendar year |
| district | string | District name |
| province | string | Province |
| population | int | Estimated resident population |
| population_density_km2 | float | People per km² |
| wash_coverage_pct | float | % population with improved water/sanitation |
| poverty_index | float | Poverty headcount ratio (0–1) |
| open_defecation_pct | float | % population practising open defecation |
| piped_water_access_pct | float | % with piped water access |
| healthcare_facilities | int | Number of public health facilities |

## interventions.csv

| Column | Type | Description |
|---|---|---|
| start_date | date | Intervention start date |
| end_date | date | Intervention end date |
| district | string | Target district |
| intervention_type | string | OCV_Campaign or WASH_Project |
| beneficiaries | int | Estimated number of direct beneficiaries |
| source | string | Implementing agency (WHO/UNICEF/MoHCC) |
