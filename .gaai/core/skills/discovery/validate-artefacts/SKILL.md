---
name: validate-artefacts
description: Validate that all Discovery artefacts (Epics, Stories) are clear, governed, complete, and safe to pass into Delivery. Activate after generating Epics or Stories and before any Delivery planning. This is the mandatory Discovery → Delivery gate.
license: MIT
compatibility: Works with any filesystem-based AI coding agent
metadata:
  author: gaai-framework
  version: "1.0"
  category: discovery
  track: discovery
  id: SKILL-VALIDATE-ARTEFACTS-001
  updated_at: 2026-02-26
  status: stable
inputs:
  - contexts/artefacts/epics/**
  - contexts/artefacts/stories/**
  - contexts/artefacts/prd/**  (optional)
  - contexts/artefacts/marketing/**  (optional — observation logs, validated hypotheses)
  - contexts/artefacts/strategy/**  (optional — GTM plans, positioning)
  - contexts/rules/**
  - contexts/memory/**  (selective)
outputs:
  - validation_report
  - updated artefact status (optional)
---

# Validate Artefacts

## Purpose / When to Activate

Activate:
