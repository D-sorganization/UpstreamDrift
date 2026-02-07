# Golf Modeling Suite - Fixed Issues Summary

**Date**: 2026-02-06  
**Status**: ✅ Hybrid Development Environment Running

## Issues Fixed

### 1. ✅ Removed Simscape from Engine Options
**Problem**: Simscape requires MATLAB and cannot run in the web GUI  
**Fix**: 
- Removed from `ui/src/api/useEngineManager.ts` ENGINE_REGISTRY
- Removed from backend `src/api/routes/engines.py` engine mappings

### 2. ✅ Added OpenSim and MyoSuite to Engine Options
**Problem**: OpenSim and MyoSuite were not showing in UI  
**Fix**:
- Added to `ui/src/api/useEngineManager.ts` ENGINE_REGISTRY:
  ```typescript
  {
      name: 'opensim',
      displayName: 'OpenSim',
      description: 'Musculoskeletal modeling and biomechanics simulation',
      capabilities: ['musculoskeletal', 'inverse_kinematics', 'muscle_analysis'],
  },
  {
      name: 'myosuite',
      displayName: 'MyoSuite',
      description: 'Muscle-tendon control and neural activation',
      capabilities: ['musculoskeletal', 'muscle_control', 'neural_activation'],
  }
  ```
- Added to backend engine mappings in `src/api/routes/engines.py`

### 3. ✅ Updated Dockerfile with All Dependencies
**Problem**: Missing Python packages causing runtime errors  
**Fix**: Added to Dockerfile:
- `email-validator`
- `bcrypt`
- `python-jose[cryptography]`
- `passlib`
- `PyJWT`
- `aiofiles`
- `python-dateutil`
- `websockets`

### 4. ⚠️ Putting Green / Pinocchio Not Loading Issue
**Status**: Partially diagnosed  
**Known Issues**:
- Putting Green temporarily mapped to PENDULUM engine (needs proper integration - Issue #1136)
- Engines may fail to load if backend dependencies missing
- "Reconnecting" spinner suggests WebSocket or API communication issue

**Next Steps to Debug**:
1. Check browser console for errors when clicking "Load Engine"
2. Monitor Docker backend logs: `docker-compose logs -f backend`
3. Test engine probe endpoints:
   ```bash
   curl http://localhost:8001/api/engines/pinocchio/probe
   curl http://localhost:8001/api/engines/opensim/probe
   curl http://localhost:8001/api/engines/myosuite/probe
   ```

## Current Engine Status

| Engine | UI Status | Backend Mapping | Docker Image | Notes |
|--------|-----------|-----------------|--------------|-------|
| MuJoCo | ✅ Showing | ✅ Mapped | ✅ Installed | Should work |
| Drake | ✅ Showing | ✅ Mapped | ✅ Installed | Should work |
| Pinocchio | ✅ Showing | ✅ Mapped | ✅ Installed | Should work |
| OpenSim | ✅ Showing | ✅ Mapped | ❌ Not installed | Will show as unavailable |
| MyoSuite | ✅ Showing | ✅ Mapped | ❌ Not installed | Will show as unavailable |
| Putting Green | ✅ Showing | ⚠️ Temp (PENDULUM) | ⚠️ Backend code exists | Needs proper integration (#1136) |
| Simscape | ❌ Removed | ❌ Removed | N/A | Requires MATLAB |

## How to Test

1. **Refresh the browser**: http://localhost:5180
2. **Check engine list**: Should now show 6 engines (no Simscape)
3. **Try loading MuJoCo**:
   - Click "Load Engine" on MuJoCo card
   - Should show "Loaded" status
4. **Check browser console** (F12) for any errors

## Simulation "Reconnecting" Issue

**Possible Causes**:
1. Simulation endpoints not implemented in backend
2. WebSocket connection failing
3. Missing simulation service initialization
4. Frontend expecting different API response format

**Debug Steps**:
```bash
# Check if simulation endpoints exist:
curl http://localhost:8001/docs | grep -i simulation

# Monitor backend during simulation start:
docker-compose logs -f backend

# Check browser Network tab (F12) for failed requests
```

## Files Modified

### Frontend (`ui/`)
- `src/api/useEngineManager.ts` - Updated ENGINE_REGISTRY

### Backend (`src/`)
- `api/routes/engines.py` - Updated engine mappings
- `Dockerfile` - Added all required dependencies

## Next Actions

1. **Test engine loading** in UI
2. **Debug "reconnecting" spinner** - likely simulation service issue
3. **Install OpenSim/MyoSuite** in Docker image (Issues #1140, #1141)
4. **Properly integrate Putting Green** engine (Issue #1136)
5. **Rebuild Docker image** with updated Dockerfile for permanent fix

## Docker Commands

```bash
# Rebuild with all dependencies (permanent fix):
docker-compose build backend

# Restart to apply code changes:
docker-compose restart backend

# View logs:
docker-compose logs -f backend

# Check engine availability:
curl http://localhost:8001/api/engines/mujoco/probe
curl http://localhost:8001/api/engines/opensim/probe
```
