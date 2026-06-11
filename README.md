# HABaware Mobile Demo

A mobile demo application that advises users of potential health and safety risks from harmful algal blooms (HABs) at a specific location and time.

## Goal

HABaware combines multiple data sources with generative AI to produce a personalized, location- and time-specific risk assessment for anyone planning to recreate on or near a California waterbody.

## How It Works

A user provides (or the app detects) their location and planned visit time. HABaware synthesizes the following inputs to generate a risk advisory:

### Data Sources

- **Field Observation Data** — Real-time and historical bloom observations from the California SWAMP Freshwater HABs (FHAB) Monitoring Program, including confirmed cyanobacteria detections, cyanotoxin measurements, and advisory/warning status.
- **Remote Sensing Data** — Satellite and aerial imagery-derived bloom indices from the CA SWAMP FHAB remote sensing program, capturing surface bloom extent, intensity, and spatial distribution.
- **Historical HAB Records** — Previous detections of cyanobacteria and cyanotoxins at the target waterbody and at nearby or hydrologically/ecologically similar waterbodies.

### AI-Assisted Risk Synthesis

A generative AI model integrates the above data with additional contextual factors, including:

- Seasonal and historical bloom patterns at the location
- Water body characteristics (size, depth, trophic status, flow regime)
- Meteorological conditions (wind, temperature, mixing) at the time of the visit
- Bloom persistence and trajectory trends
- Proximity to known bloom hotspots or upstream sources
- User activity type (swimming, kayaking, fishing, dog walking, etc.)

The model produces a plain-language risk summary, an overall risk level (Low / Moderate / High / Very High), and activity-specific guidance.

## Data Attribution

Field observation and remote sensing data are sourced from the [California State Water Resources Control Board SWAMP FHAB Monitoring Program](https://www.waterboards.ca.gov/water_issues/programs/swamp/cyanohab/).

## Status

This repository contains an early-stage mobile demo. It is intended for demonstration and research purposes.
