---
issue: 12965
summary: "Sanitize 400/403/404 error details in handle_api_errors to prevent path leaks"
title: "Infra: Handle_Api_Errors Leaks Absolute Server Paths and Internals via Str(E) in 400/403/404 Bodies"
owner: "antigravity"
branch: "fix/12965-sanitize-api-errors"
paths: "src/api/middleware/error_handler.py,src/api/utils/error_codes.py,tests/unit/api/test_error_handler.py"
---
