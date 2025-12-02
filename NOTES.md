# FastSLAM Project Notes

## Overview
This project implements **FastSLAM 1.0** to improve the quality of a **very noisy map sensor** and generate a reliable map for navigation toward a goal. The goal is to combine SLAM-based filtering with raw sensor map data to obtain better state and map estimates.

---

## Key Concepts

### FastSLAM Structure
- Robot pose represented with **particles**
- Each particle maintains its own map using **EKFs for landmarks**
- Sensor updates adjust landmark means and covariances
- Particle weights reflect measurement likelihoods
- Resampling keeps best-consistent particles

### Noisy Map Sensor
The map sensor is extremely noisy. FastSLAM helps:
- Reduce noise via filtering across particles
- Fuse map sensor readings with SLAM map estimates
- Smooth and stabilize the resulting map

### Navigation
To reach a target goal:
- The filtered SLAM pose is used as the robot's localization estimate
- The improved SLAM map supports path planning
- A controller drives the robot toward the goal

---

## Implementation Roadmap
See TODO and REQUIREMENTS documents for detailed task breakdown.

---

## Useful Reminders
- Keep noise models realistic
- Use covariance matrices to track uncertainty
- Visualization is essential for debugging particle behavior and landmark estimates
- Evaluate sensor fusion quality by comparing noisy vs filtered map

---

## Future Improvements
- Upgrade to FastSLAM 2.0
- Add better controllers for path following
- Implement dynamic landmark handling
- Integrate more advanced filters for map sensor fusion
