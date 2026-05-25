from src.shared.python.logging_pkg.logging_config import _redact_sensitive

print(_redact_sensitive('{"password": "my secret password"}'))
print(_redact_sensitive('{"password": "my,secret,password"}'))
print(_redact_sensitive('password="my secret password"'))
