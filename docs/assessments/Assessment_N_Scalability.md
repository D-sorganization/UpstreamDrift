# Assessment N: Scalability

## Grade: 7/10

## Summary
The system is designed with some scalability in mind (asynchronous API, batched physics queries). However, the monolithic nature of the physics engines and some code files limits horizontal scalability.

## Strengths
- **Async API**: Handles concurrent requests reasonably well.
- **Stateless Engines**: The `PhysicsEngine` protocol encourages statelessness (or at least manageable state), allowing for potential pool management.

## Weaknesses
- **Vertical Scaling**: Physics simulations are CPU-bound and often single-threaded per instance.
- **Monoliths**: Large files and tight coupling make it hard to split the service into microservices if needed.

## Recommendations
1. **Worker Queues**: Move simulation tasks to a robust job queue (e.g., Celery/Redis) instead of just in-memory `BackgroundTasks`.
2. **Horizontal Scaling**: Containerize the engine service to run multiple instances behind a load balancer.
