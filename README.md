# FastSLAM Algorithm with Particle Filter – Robotics Notes

This repository implements a **FastSLAM-style algorithm** using **particle filters** to solve the classic robotics problem:

> **Simultaneous Localization and Mapping (SLAM)**

The project is highly educational and ideal for **robotics students**, **control engineers**, and anyone learning **probabilistic robotics**.

---

## Repository Structure

```text
FastSLAM-algorithm-with-Particle-Filter-Implementation-main/
├── FastSLAM_like.py
├── particlesFilter.py
├── particle_robot.py
├── robot_simulator.py
├── REQUIREMENTS.txt
├── NOTES.md
├── TODO.txt
├── logs/
│   ├── Figure_1_no_implementations.png
│   ├── Figure_2_with_particle_filter.png
│   ├── Figure_3_with_better_weights.png
│   ├── Figure_4_Map_exploration.png
│   └── Figure_5_most_stable_so_far.png
└── LICENSE
```

Each file represents a **logical component of the SLAM pipeline**.

---

## What Problem This Solves (SLAM)

SLAM asks:

> *How can a robot know where it is while building a map of an unknown environment?*

Mathematically:

```
P(x_t, m | z_1:t, u_1:t)
```

Where:
- `x_t` = robot pose
- `m` = map
- `z` = sensor measurements
- `u` = control inputs

---

## Why FastSLAM?

Classic SLAM is computationally expensive.

**FastSLAM** factorizes the problem:

```
P(x, m | z, u) = P(x | z, u) · Π P(m_i | x, z)
```

Meaning:
- Robot pose is estimated using a **particle filter**
- Each landmark is estimated independently

This makes SLAM **scalable**.

---

## Core Algorithm Breakdown

### 1. Particle Filter (Localization)

Each particle represents a **possible robot pose**.

Motion update:

```
x_t = f(x_{t-1}, u_t) + noise
```

Analogy:  
Throwing many guesses of where the robot *might* be.

---

### 2. Weight Update (Sensor Model)

Particles are weighted by how well they explain sensor data:

```
w_t ∝ P(z_t | x_t, m)
```

Particles that better match observations survive.

Analogy:  
Good guesses get rewarded, bad ones fade away.

---

### 3. Resampling

Particles with high weights are duplicated.

This focuses computation on **likely states**.

---

### 4. Mapping

Each particle maintains its **own map**.

Landmark updates often use a **Kalman Filter**:

```
μ_new = μ + K(z − Hμ)
```

---

## File Responsibilities

### `robot_simulator.py`
- Simulates robot motion and sensors
- Generates ground truth

### `particle_robot.py`
- Defines particle behavior
- Applies motion model

### `particlesFilter.py`
- Implements resampling
- Manages particle weights

### `FastSLAM_like.py`
- Orchestrates SLAM loop
- Integrates localization + mapping

---

## Visual Results

Images in the `logs/` folder show:
- No filter → noisy trajectory
- Particle filter → improved localization
- Stable mapping over time

These illustrate **convergence behavior**.

---

## Using This Repository as Study Notes

### Recommended Study Path
1. Read `NOTES.md`
2. Follow execution flow in `FastSLAM_like.py`
3. Plot particle distributions
4. Change noise parameters

### Questions to Explore
- Effect of particle count
- Sensor noise sensitivity
- Resampling frequency

---

## How to Run

1. Install requirements:
```bash
pip install -r REQUIREMENTS.txt
```

2. Run:
```bash
python FastSLAM_like.py
```

---

## Learning Outcomes

You will understand:
- Probabilistic localization
- Particle filters
- SLAM decomposition
- Robotics uncertainty handling

---

## Final Note

FastSLAM shows how **uncertainty is not avoided in robotics — it is modeled**.
This repository turns probability theory into something you can *see* and *debug*.
