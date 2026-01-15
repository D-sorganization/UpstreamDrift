# Technical Patent & Legal Risk Assessment

**Date:** 2026-01-26
**Status:** CRITICAL REVIEW
**Scope:** `shared/python/`, `engines/`, `launchers/`

**DISCLAIMER: This document is a technical analysis of codebase components that implement specific algorithms or methodologies known to be active areas of intellectual property in the golf technology sector. It is NOT legal advice. A qualified patent attorney should review these findings.**

## Executive Summary

This review identifies **critical** legal risks within the current codebase. The most immediate concern is the active use of the trademarked term **"Swing DNA"** (Mizuno) in the user interface, despite previous remediation efforts. Additionally, several algorithms for swing scoring and biomechanical analysis directly implement methodologies covered by active patents held by major competitors (K-Motion, Zepp, TrackMan).

## 🚨 Critical Priority: Trademark Infringement

### 1. "Swing DNA" Trademark
**Risk Level:** **SEVERE / ACTIVE**
**Owner:** Mizuno Corporation

*   **Violation:** The term "Swing DNA" is explicitly used in the user interface of the Unified Dashboard.
*   **Location:** `shared/python/dashboard/window.py`
    *   **Line 140:** `"Swing DNA (Radar)"` (Dropdown menu item)
    *   **Line 349:** `elif plot_type == "Swing DNA (Radar)":` (Logic handler)
    *   **Line 379:** `raise ValueError("Could not compute Swing DNA")` (Error message)
*   **Context:** While the backend class was renamed to `SwingProfileMetrics`, the UI label was not updated, leaving the application exposed to trademark litigation.
*   **Recommendation:** Rename all UI instances to **"Swing Profile"** or **"Biometric Radar"** immediately.

## ⚠️ High Priority: Patent Conflicts

### 2. Biofeedback & Scoring Algorithms (DTW)
**Risk Level:** **HIGH**
**Potential Conflict:** K-Motion Interactive, Zepp Labs (Blast Motion)

*   **Description:** The codebase implements a scoring algorithm based on Dynamic Time Warping (DTW) distance, normalized to a 0-100 scale.
*   **Location:** `shared/python/swing_comparison.py`
    *   **Method:** `SwingComparator.compute_kinematic_similarity`
    *   **Snippet:** `score = 100.0 / (1.0 + norm_dist)`
*   **Analysis:** Patents for "evaluating athletic motion using time-warped comparison" (e.g., US Patent 8,xxx,xxx) often cover this exact logic of deriving a user score from a DTW path cost.
*   **Recommendation:** Review claim charts for K-Motion patents. Consider implementing a non-DTW scoring metric (e.g., discrete keyframe comparison) if licensing is not an option.

### 3. "Swing Profile" / Radar Metrics
**Risk Level:** **HIGH**
**Potential Conflict:** Mizuno (Performance Fitting System), TPI

*   **Description:** The `SwingProfileMetrics` class calculates a 5-axis score: Speed, Sequence, Stability, Efficiency, Power.
*   **Location:** `shared/python/statistical_analysis.py`
    *   **Method:** `StatisticalAnalyzer.compute_swing_profile`
*   **Analysis:** The *selection* of these specific five metrics to define a "swing signature" closely mimics Mizuno's "Swing DNA" fitting system (which uses Head Speed, Tempo, Toe Down, Kick Angle, Release Factor). Even with a name change, the *method* of fitting a shaft/club based on this specific vector of metrics may be patented.

### 4. Kinematic Sequence Scoring
**Risk Level:** **HIGH**
**Potential Conflict:** Titleist Performance Institute (TPI), K-Motion

*   **Description:** The system scores the "efficiency" of the swing based on the proximal-to-distal peak velocity order (Pelvis -> Thorax -> Arm -> Club).
*   **Location:** `shared/python/statistical_analysis.py`
    *   **Method:** `StatisticalAnalyzer.analyze_kinematic_sequence`
*   **Analysis:** The methodology of quantifying "Kinematic Sequence Efficiency" is a core claim in several biomechanics patents.

## 🔸 Medium Priority: Method Patents & Trade Secrets

### 5. Gear Effect Approximation
**Risk Level:** **MEDIUM**
**Potential Conflict:** Simulator Manufacturers (TrackMan, Foresight)

*   **Description:** A specific linear approximation is used to calculate spin induced by off-center hits.
*   **Location:** `shared/python/impact_model.py`
    *   **Method:** `compute_gear_effect_spin`
    *   **Snippet:** `horizontal_spin = -gear_factor * h_offset * speed * 100`
*   **Analysis:** The specific constants (`100`, `50`) and linear relationship imply a derived empirical model that may be proprietary or covered by a "method of simulating golf ball flight" patent.

### 6. Chaos Theory Metrics (Lyapunov, RQA)
**Risk Level:** **MEDIUM (Niche)**
**Potential Conflict:** Advanced Biomechanics Research / specialized startups

*   **Description:** Implementation of Largest Lyapunov Exponent, Fractal Dimension (Higuchi), and Recurrence Quantification Analysis (RQA) for swing analysis.
*   **Location:** `shared/python/statistical_analysis.py`
    *   **Methods:** `estimate_lyapunov_exponent`, `compute_fractal_dimension`, `compute_rqa_metrics`
*   **Analysis:** While the math is public domain, the *application* of these chaos metrics to quantify "swing consistency" or "talent identification" is a known area of patent activity.

## Action Plan

1.  **Immediate Remediation:**
    *   [ ] **Fix UI Labels:** Rename "Swing DNA (Radar)" to "Swing Profile (Radar)" in `shared/python/dashboard/window.py`.
    *   [ ] **Audit Strings:** Search codebase for any other "Swing DNA" remnants.

2.  **Legal Review:**
    *   [ ] Submit `shared/python/swing_comparison.py` (DTW logic) to patent counsel.
    *   [ ] Submit `shared/python/impact_model.py` (Gear Effect) to patent counsel.

3.  **Documentation:**
    *   [ ] Maintain this risk assessment as a living document in the repository.
