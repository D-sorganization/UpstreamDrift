
# Data Pipeline Architecture

This diagram shows how data flows from simulation to analysis and storage.

```mermaid
graph LR
    Sim[Simulation Engine] -->|State Data (qpos, qvel)| Buffer[RingBuffer]
    Buffer -->|Sample 60Hz| Pipeline[Data Pipeline]
    
    Pipeline -->|Real-time| Plot[Live Plotting]
    Pipeline -->|Batch| Analysis[Biomechanics Analysis]
    
    Analysis -->|Metrics| Dashboard[Dashboard UI]
    Analysis -->|files| Storage[Storage Layer]
    
    Storage --> HDF5[(HDF5)]
    Storage --> C3D[(C3D)]
    Storage --> Parquet[(Parquet)]
```
