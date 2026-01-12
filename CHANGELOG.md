# Changelog

All notable changes to the Golf Modeling Suite will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Comprehensive assessment framework (A-O) with 15 quality categories
- MyoSuite integration for musculoskeletal modeling
- OpenSim tutorials and example scripts

### Changed

- Updated README status from BETA to STABLE
- Removed broken GolfingRobot.png reference
- Cleaned 30+ debris files from root directory
- Updated .gitignore to prevent future accumulation

### Fixed

- Mypy errors in plotting module
- Type annotations across physics engines

## [1.0.0] - 2026-01-10

### Added

- 5 Physics Engines: MuJoCo, Drake, Pinocchio, OpenSim, MyoSuite
- 1,563+ unit tests for comprehensive validation
- Professional PyQt6 GUI launcher
- Multi-engine comparison capabilities
- URDF generator with bundled assets

### Features

- Manipulability ellipsoid visualization
- Flexible shaft dynamics modeling
- Grip contact force analysis
- Ground reaction force processing

### Infrastructure

- Cross-engine validation framework
- Scientific plotting architecture
- Energy monitoring system
