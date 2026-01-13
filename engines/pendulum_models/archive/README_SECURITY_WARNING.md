# ⚠️ SECURITY WARNING - ARCHIVED LEGACY CODE ⚠️

## DO NOT USE THIS CODE IN PRODUCTION

This directory contains **archived legacy code** that is kept for **historical reference only**.

### CRITICAL SECURITY ISSUES

The code in this directory contains **known security vulnerabilities**:

1. **Unsafe eval() usage** - Code injection vulnerability
   - Files affected: `safe_eval.py`, `pendulum_pyqt_app.py`, `double_pendulum.py`
   - Risk: Arbitrary code execution if user input reaches these functions

2. **Deprecated patterns** - No longer following project security standards

3. **No maintenance** - This code is NOT maintained and NOT tested

### WHAT TO USE INSTEAD

**For pendulum simulations, use:**
- `/engines/pendulum_models/python/` - Modern, secure implementation
- MuJoCo, Drake, or Pinocchio engines - Production-ready physics engines

**For safe mathematical expression evaluation, use:**
- `simpleeval` library (already in project dependencies)
- See: `/shared/python/` for examples of safe evaluation

### WHY IS THIS HERE?

This directory is preserved for:
- Historical reference
- Understanding evolution of the codebase
- Comparison with modern implementations
- Educational purposes (what NOT to do)

### SECURITY POLICY

- ❌ DO NOT import code from this directory
- ❌ DO NOT copy patterns from this directory
- ❌ DO NOT expose this code to user input
- ✅ DO use modern implementations in `/engines/physics_engines/`
- ✅ DO use `simpleeval` for expression evaluation

### REMOVAL PLAN

This directory may be removed in future releases. If you depend on any functionality here, please:
1. File an issue explaining the use case
2. Use the modern equivalents instead
3. Help migrate to secure implementations

---

**Last Updated**: January 2026
**Status**: ARCHIVED - DO NOT USE
**Maintained**: NO
**Security**: VULNERABLE
