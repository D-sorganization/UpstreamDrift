# Assessment: Scalability (Category N)

## Grade: 7/10

## Analysis
The system has some scalable characteristics but is limited by stateful components.
- **Async**: The API uses `async/await` and background tasks for long-running operations, allowing high concurrency for I/O bound tasks.
- **Statelessness**: The REST API is mostly stateless, but the `active_tasks` dictionary is in-memory, preventing simple load balancing across multiple worker processes without sticky sessions or an external store (Redis).
- **Physics**: Physics simulations are CPU-bound and run locally or in Docker containers. Scaling this requires an orchestration layer (like Kubernetes) to manage worker containers, which is partially implied by the Docker support but not fully realized.

## Recommendations
1. **Redis**: Replace in-memory `active_tasks` with Redis to allow multiple API workers.
2. **Task Queue**: Move simulation jobs to a proper task queue (Celery/RQ) instead of FastAPI `BackgroundTasks` for better reliability and scaling.
