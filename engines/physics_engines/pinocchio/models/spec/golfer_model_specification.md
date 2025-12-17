# Golfer Biomechanical Model Specification

## Overview

This document specifies a full-body biomechanical model of a human golfer for use with Pinocchio dynamics simulation. The model includes detailed joint structures from feet to hands, with a golf club attached via constraints.

## Model Assumptions

### General Assumptions
- **Reference Subject**: 75 kg male, 1.75 m height
- **Coordinate System**: 
  - X: Forward (target direction)
  - Y: Left
  - Z: Up
- **Units**: 
  - Mass: kilograms (kg)
  - Length: meters (m)
  - Inertia: kg·m²
  - Angles: radians

### Anthropometric Assumptions
- Body segment inertial properties based on de Leva (1996) adjustments of Zatsiorsky-Seluyanov data
- Segments modeled as rigid bodies with realistic mass distribution
- Center of mass locations relative to segment endpoints follow standard biomechanics literature

## Joint Structure Specification

### Lower Body

#### Foot (Right/Left)
- **Type**: Rigid body with contact geometry
- **Contact**: Planar contact with ground (friction: μ=1.0)
- **Toe Joint**: 
  - Type: Revolute
  - Axis: Y (lateral, perpendicular to foot long axis)
  - Range: [-0.35, 0.52] rad (toe down to toe up)
  - Purpose: Allows toe raise/lower for weight transfer

#### Ankle (Right/Left)
- **Type**: Universal Joint (2 DOF)
  - DOF 1: Plantarflexion/Dorsiflexion (X-axis rotation)
    - Range: [-0.70, 0.52] rad
  - DOF 2: Inversion/Eversion (Y-axis rotation)
    - Range: [-0.35, 0.35] rad
- **Purpose**: Allows foot to raise/lower and tilt sideways

#### Knee (Right/Left)
- **Type**: Revolute Joint
- **Axis**: Y (lateral, perpendicular to leg)
- **Range**: [0, 2.62] rad (0° to 150° flexion)
- **Damping**: 5.0 N·m·s/rad (physiological joint damping)

#### Hip (Right/Left)
- **Type**: Gimbal Joint (3 DOF)
  - DOF 1: Flexion/Extension (Y-axis)
    - Range: [-1.57, 1.57] rad
  - DOF 2: Abduction/Adduction (X-axis)
    - Range: [-0.87, 0.52] rad
  - DOF 3: Internal/External Rotation (Z-axis)
    - Range: [-0.70, 0.70] rad
- **Purpose**: Full 3D hip motion

### Spine

#### Lumbar Segment 1 (L1-L2)
- **Type**: Universal Joint (2 DOF)
  - DOF 1: Flexion/Extension (X-axis)
  - DOF 2: Lateral Bending (Y-axis)

#### Lumbar Segment 2 (L2-L3)
- **Type**: Universal Joint (2 DOF)
  - Same as L1-L2

#### Lumbar Segment 3 (L3-L4)
- **Type**: Universal Joint (2 DOF)
  - Same as L1-L2

#### Thoracic Segment 1 (T12-L1)
- **Type**: Revolute Joint (1 DOF)
  - Axis: Z (longitudinal)
  - Purpose: Axial rotation

#### Thoracic Segment 2 (T8-T12)
- **Type**: Revolute Joint (1 DOF)
  - Axis: Z (longitudinal)

#### Thoracic Segment 3 (T1-T8)
- **Type**: Revolute Joint (1 DOF)
  - Axis: Z (longitudinal)

**Spine Total**: 3 Universal + 3 Revolute = 9 DOF total spine motion

### Upper Body - Right Side

#### Scapula (Right)
- **Type**: Universal Joint (2 DOF)
  - Connection: Rod from upper torso to shoulder
  - DOF 1: Elevation/Depression (Z-axis rotation)
    - Range: [-0.35, 0.70] rad
  - DOF 2: Protraction/Retraction (Y-axis rotation)
    - Range: [-0.52, 0.52] rad
- **Purpose**: Models scapular motion independent of GH joint

#### Shoulder - Glenohumeral Joint (Right)
- **Type**: Gimbal Joint (3 DOF)
  - DOF 1: Flexion/Extension (Y-axis)
    - Range: [-1.57, 3.14] rad
  - DOF 2: Abduction/Adduction (X-axis)
    - Range: [-0.52, 2.09] rad
  - DOF 3: Internal/External Rotation (Z-axis)
    - Range: [-1.57, 1.57] rad
- **Purpose**: Full shoulder ball-and-socket motion

#### Elbow (Right)
- **Type**: Revolute Joint
- **Axis**: Y (lateral, perpendicular to arm)
- **Range**: [0, 2.62] rad (0° to 150° flexion)
- **Damping**: 3.0 N·m·s/rad

#### Forearm Rotation (Right)
- **Type**: Revolute Joint
- **Axis**: X (longitudinal along forearm)
- **Range**: [-1.57, 1.57] rad (pronation to supination)
- **Purpose**: Pronation/supination

#### Wrist (Right)
- **Type**: Universal Joint (2 DOF)
  - DOF 1: Flexion/Extension (Y-axis)
    - Range: [-1.22, 1.22] rad
  - DOF 2: Radial/Ulnar Deviation (Z-axis)
    - Range: [-0.52, 0.70] rad
- **Purpose**: Wrist motion for club control

#### Finger Joint (Right)
- **Type**: Revolute Joint
- **Axis**: Y (along base of fingers, perpendicular to hand)
- **Range**: [0, 1.57] rad (0° to 90° flexion)
- **Purpose**: Finger grip closure

### Upper Body - Left Side

Left side follows same structure as right side:
- Scapula (Left) - Universal Joint
- Shoulder GH Joint (Left) - Gimbal Joint
- Elbow (Left) - Revolute Joint
- Forearm Rotation (Left) - Revolute Joint
- Wrist (Left) - Universal Joint
- Finger Joint (Left) - Revolute Joint

### Club

#### Club Shaft
- **Type**: Flexible beam (initially modeled as rigid)
- **Length**: 1.0 m (driver length)
- **Mass**: 0.05 kg
- **Connection**: Constraint-based attachment to hands (not rigid joint)

#### Club Head
- **Type**: Rigid body
- **Mass**: 0.20 kg
- **Inertia**: High MOI design
- **Connection**: Fixed to shaft end

## Constraints

### Hand-to-Club Constraints

#### Right Hand to Club
- **Type**: Rigid constraint (6 DOF locked)
- **Frame A**: Right hand grip frame (at finger joint)
- **Frame B**: Club shaft grip frame
- **Implementation**: Pinocchio constraint (not URDF joint)
- **Purpose**: Allows constraint-based connection for loop closure

#### Left Hand to Club
- **Type**: Rigid constraint (6 DOF locked)
- **Frame A**: Left hand grip frame
- **Frame B**: Club shaft grip frame
- **Implementation**: Pinocchio constraint

**Note**: The right hand-to-club connection is "broken" in the URDF tree structure and modeled as a constraint in Pinocchio to enable proper loop closure handling.

## Inertial Properties

### Lower Body Segments

#### Foot (Right/Left)
- **Mass**: 1.37 kg (1.83% of total body mass)
- **Length**: 0.26 m
- **COM Location**: 0.429 × length from heel (0.111 m)
- **Inertia** (about COM):
  - Ixx: 0.0034 kg·m²
  - Iyy: 0.0090 kg·m²
  - Izz: 0.0090 kg·m²

#### Shank (Right/Left)
- **Mass**: 3.53 kg (4.71% of total body mass)
- **Length**: 0.43 m
- **COM Location**: 0.433 × length from knee (0.186 m)
- **Inertia**:
  - Ixx: 0.0401 kg·m²
  - Iyy: 0.0042 kg·m²
  - Izz: 0.0042 kg·m²

#### Thigh (Right/Left)
- **Mass**: 7.05 kg (9.40% of total body mass)
- **Length**: 0.43 m
- **COM Location**: 0.433 × length from hip (0.186 m)
- **Inertia**:
  - Ixx: 0.1058 kg·m²
  - Iyy: 0.1058 kg·m²
  - Izz: 0.0203 kg·m²

### Spine Segments

#### Pelvis
- **Mass**: 11.70 kg (15.60% of total body mass)
- **Inertia**:
  - Ixx: 0.1337 kg·m²
  - Iyy: 0.1337 kg·m²
  - Izz: 0.1337 kg·m²

#### Lumbar Segments (each)
- **Mass**: 2.0 kg (average)
- **Length**: 0.12 m
- **Inertia**:
  - Ixx: 0.015 kg·m²
  - Iyy: 0.015 kg·m²
  - Izz: 0.002 kg·m²

#### Thoracic Segments (each)
- **Mass**: 1.5 kg (average)
- **Length**: 0.10 m
- **Inertia**:
  - Ixx: 0.012 kg·m²
  - Iyy: 0.012 kg·m²
  - Izz: 0.0015 kg·m²

### Upper Body Segments

#### Scapula (Right/Left)
- **Mass**: 1.5 kg (2.0% of total body mass)
- **Inertia**:
  - Ixx: 0.008 kg·m²
  - Iyy: 0.008 kg·m²
  - Izz: 0.003 kg·m²

#### Upper Arm (Right/Left)
- **Mass**: 1.98 kg (2.64% of total body mass)
- **Length**: 0.28 m
- **COM Location**: 0.436 × length from shoulder (0.122 m)
- **Inertia**:
  - Ixx: 0.0138 kg·m²
  - Iyy: 0.0138 kg·m²
  - Izz: 0.0021 kg·m²

#### Forearm (Right/Left)
- **Mass**: 1.18 kg (1.57% of total body mass)
- **Length**: 0.26 m
- **COM Location**: 0.430 × length from elbow (0.112 m)
- **Inertia**:
  - Ixx: 0.0074 kg·m²
  - Iyy: 0.0074 kg·m²
  - Izz: 0.0009 kg·m²

#### Hand (Right/Left)
- **Mass**: 0.61 kg (0.81% of total body mass)
- **Length**: 0.19 m
- **COM Location**: 0.506 × length from wrist (0.096 m)
- **Inertia**:
  - Ixx: 0.0011 kg·m²
  - Iyy: 0.0011 kg·m²
  - Izz: 0.0004 kg·m²

### Club

#### Club Shaft
- **Mass**: 0.05 kg
- **Length**: 1.0 m
- **COM Location**: 0.5 × length (0.5 m)
- **Inertia**:
  - Ixx: 0.0042 kg·m²
  - Iyy: 0.0042 kg·m²
  - Izz: 0.0001 kg·m²

#### Club Head
- **Mass**: 0.20 kg
- **Inertia**:
  - Ixx: 0.0025 kg·m²
  - Iyy: 0.0025 kg·m²
  - Izz: 0.0005 kg·m²

## Geometry Specifications

### Body Segment Geometry
- **Foot**: Box (0.26 m × 0
