# Assessment: Scalability (Category N)

## Executive Summary
**Grade: 8/10**

The architecture allows for scalability. The API is stateless (mostly), and background tasks allow offloading heavy work. However, the reliance on local file storage for models/uploads and the GIL (Global Interpreter Lock) for CPU-bound tasks are limits.

## Strengths
1.  **Async:** `FastAPI` handles high concurrency for I/O bound tasks.
2.  **Stateless:** Easy to replicate API containers.
3.  **Task Queue:** Background tasks pattern established.

## Weaknesses
1.  **CPU Bound:** Physics simulations are CPU heavy; Python's GIL limits single-node scaling.
2.  **Shared Storage:** Multiple instances would need shared storage (e.g., S3/EFS) for models/uploads, currently local-path based.

## Recommendations
1.  **Celery/Redis:** Move from `BackgroundTasks` (in-process) to `Celery` + `Redis` for distributed task processing.
2.  **Object Storage:** Abstract file access to support S3.

## Detailed Analysis
- **Horizontal:** Possible with load balancer.
- **Vertical:** Limited by GIL (for Python parts).
