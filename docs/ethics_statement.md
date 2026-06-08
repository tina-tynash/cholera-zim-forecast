# Ethics Statement

## Data Privacy and Minimization

This project uses only aggregated, district-level public health data. No individual-level
patient records are collected, stored, or processed at any point in the pipeline.
All synthetic data is generated algorithmically and contains no real patient information.

## Data Sources and Sovereignty

Data is sourced from publicly available repositories:
- WHO Global Health Observatory
- Humanitarian Data Exchange (HDX)
- Zimbabwe Ministry of Health and Child Care (MoHCC) public reports
- World Bank Open Data

We acknowledge Zimbabwe's right to data sovereignty and recommend that any operational
deployment be co-developed with MoHCC and local health authorities.

## Community Benefit vs Surveillance Risk

This system is designed for **epidemic preparedness and early warning**, not population
surveillance. Risk scores represent district-level aggregates and cannot be used to
identify or track individuals. Deployment should be governed by a community benefit
framework with oversight from district health authorities.

## Responsible Deployment in LMIC Settings

Before operational deployment:
1. Engage Zimbabwe MoHCC and district health officers.
2. Conduct community consultation in high-risk districts.
3. Ensure dashboard access is available to local health workers, not only central authorities.
4. Establish a feedback mechanism for community health workers to flag forecast errors.
5. Do not use forecasts as the sole basis for resource allocation decisions.

## Open Science Commitment

All code, synthetic data, and documentation are released under the MIT License.
We encourage researchers in Zimbabwe and other SADC countries to fork, adapt,
and improve this system for their specific contexts.
