# Interim Deployment Guide - Golf Modeling Suite

## 🚀 Quick Setup for Development/Research Use

This guide helps you deploy the Golf Modeling Suite for immediate use without commercial billing infrastructure. Perfect for research, development, or private use while you decide on commercialization.

## Prerequisites

- Python 3.11+
- Git with Git LFS
- 8GB+ RAM (16GB recommended)
- Windows/Linux/macOS

## 1. Installation

### Clone and Setup
```bash
git clone https://github.com/D-sorganization/Golf_Modeling_Suite.git
cd Golf_Modeling_Suite
git lfs pull

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# OR
venv\Scripts\activate     # Windows

# Install with API dependencies
pip install -e ".[dev,engines,analysis]"
```

### Install Additional API Dependencies
```bash
pip install fastapi uvicorn sqlalchemy pydantic[email] python-jose[cryptography] passlib[bcrypt] python-multipart slowapi
```

## 2. Database Setup

The system uses SQLite by default (no setup required), but you can use PostgreSQL for production:

### Option A: SQLite (Default - No Setup Required)
```bash
# Database file will be created automatically at: ./golf_modeling_suite.db
# Perfect for development and small-scale use
```

### Option B: PostgreSQL (Recommended for Multi-User)
```bash
# Install PostgreSQL locally or use cloud service
# Set environment variable:
export DATABASE_URL="postgresql://username:password@localhost/golf_modeling_suite"
```

## 3. Start the API Server

```bash
# Navigate to project root
cd Golf_Modeling_Suite

# Start the API server
python -m uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload

# Server will be available at:
# - API: http://localhost:8000
# - Documentation: http://localhost:8000/docs
# - Admin Interface: http://localhost:8000/redoc
```

## 4. Initial Setup

### Create Your First User
```bash
# The system creates a default admin user automatically:
# Email: admin@golfmodelingsuite.com
# Password: admin123

# Or register a new user via API:
curl -X POST "http://localhost:8000/auth/register" \
  -H "Content-Type: application/json" \
  -d '{
    "email": "your-email@example.com",
    "password": "your-secure-password",
    "full_name": "Your Name",
    "organization": "Your Organization"
  }'
```

### Get Access Token
```bash
curl -X POST "http://localhost:8000/auth/login" \
  -H "Content-Type: application/json" \
  -d '{
    "email": "admin@golfmodelingsuite.com",
    "password": "admin123"
  }'

# Save the "access_token" from the response
```

### Create API Key (Alternative to JWT)
```bash
# Using your access token:
curl -X POST "http://localhost:8000/auth/api-keys" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "My Development Key"
  }'

# Save the "key" from the response (starts with gms_)
```

## 5. Test the System

### Check Available Engines
```bash
curl -X GET "http://localhost:8000/engines" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN"
```

### Run a Simple Simulation
```bash
curl -X POST "http://localhost:8000/simulate" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "engine_type": "mujoco",
    "duration": 1.0,
    "timestep": 0.001
  }'
```

### Analyze a Video (if you have MediaPipe installed)
```bash
curl -X POST "http://localhost:8000/analyze/video" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN" \
  -F "file=@path/to/your/golf_video.mp4" \
  -F "estimator_type=mediapipe" \
  -F "min_confidence=0.5"
```

## 6. User Management

### View Your Usage
```bash
curl -X GET "http://localhost:8000/auth/usage" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN"
```

### List Users (Admin Only)
```bash
curl -X GET "http://localhost:8000/auth/users" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN"
```

### Update User Roles (Admin Only)
```bash
# Promote user to Professional tier
curl -X PUT "http://localhost:8000/auth/users/2/role" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '"professional"'
```

## 7. Configuration Options

### Environment Variables
```bash
# Database (optional)
export DATABASE_URL="sqlite:///./golf_modeling_suite.db"

# Security (change in production)
export SECRET_KEY="your-secret-key-change-this"

# API Configuration
export API_HOST="0.0.0.0"
export API_PORT="8000"
```

### Usage Quotas (No Billing)
The system tracks usage but doesn't charge. Current limits:

- **Free Tier**: 1,000 API calls, 5 videos, 10 simulations/month
- **Professional**: 50,000 API calls, 500 videos, 1,000 simulations/month  
- **Enterprise**: 1,000,000 API calls, 10,000 videos, 50,000 simulations/month

Users hit quota limits but aren't charged. Perfect for controlled access.

## 8. Desktop Application

You can still use the desktop application alongside the API:

```bash
# Launch the desktop GUI
python launchers/golf_launcher.py

# Or use the unified interface
python -c "
from shared.python.unified_engine_interface import quick_setup
interface = quick_setup('mujoco')
print('Desktop interface ready!')
"
```

## 9. Development Workflow

### For Researchers/Developers:
1. **Register users** via API or create them programmatically
2. **Assign appropriate roles** (Professional/Enterprise for higher quotas)
3. **Use API keys** for programmatic access
4. **Monitor usage** via the `/auth/usage` endpoint
5. **Scale quotas** by promoting users to higher tiers

### For Teams:
1. **Admin creates accounts** for team members
2. **Distribute API keys** or login credentials
3. **Monitor team usage** via admin endpoints
4. **Adjust roles** as needed for project requirements

## 10. Security Notes

### Current Security Features:
- ✅ **JWT Authentication** with secure token generation
- ✅ **API Key Authentication** with SHA-256 hashing
- ✅ **Password Hashing** with bcrypt
- ✅ **Role-Based Access Control** (RBAC)
- ✅ **Rate Limiting** (basic implementation)
- ✅ **CORS Protection** (restricted origins)
- ✅ **Input Validation** via Pydantic models

### For Production Use:
- Change the default admin password
- Set a secure SECRET_KEY environment variable
- Use PostgreSQL instead of SQLite
- Configure proper CORS origins
- Set up HTTPS/SSL certificates
- Enable comprehensive logging

## 11. Troubleshooting

### Common Issues:

**"Module not found" errors:**
```bash
# Ensure you're in the project root and virtual environment is activated
pip install -e ".[dev,engines,analysis]"
```

**Database errors:**
```bash
# Reset the database (WARNING: Deletes all data)
rm golf_modeling_suite.db
# Restart the server to recreate
```

**Physics engine errors:**
```bash
# Check which engines are available
python -c "
from shared.python.engine_manager import EngineManager
manager = EngineManager()
print('Available engines:', manager.get_available_engines())
"
```

**Port already in use:**
```bash
# Use a different port
python -m uvicorn api.server:app --port 8001
```

## 12. Next Steps

This setup gives you a **fully functional research-grade biomechanical analysis platform** with:

- Multi-user authentication
- Professional API access
- All physics engines
- Video analysis capabilities
- Usage tracking and quotas
- Admin user management

**Perfect for:**
- Research projects
- Team collaboration
- API development
- Feature testing
- Private deployment

**When ready for commercialization:**
- Add Stripe billing integration
- Deploy to cloud infrastructure (Kubernetes)
- Set up monitoring and alerting
- Implement advanced security features
- Add customer support systems

The foundation is solid and production-ready. You can use this setup for months or years while deciding on commercialization strategy.