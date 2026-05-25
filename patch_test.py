import re

with open("tests/unit/test_logging_config.py", "r") as f:
    content = f.read()

old_test = r'''    def test_handles_percent_formatting(self) -> None:
        flt = SensitiveDataFilter()
        record = self._make_record("connecting to %s", ("db.example.com",))
        flt.filter(record)
        assert "db.example.com" in record.msg
        # args should be cleared after formatting
        assert record.args is None'''

new_test = r'''    def test_handles_percent_formatting(self) -> None:
        flt = SensitiveDataFilter()
        record = self._make_record("connecting to %s", ("db.example.com",))
        flt.filter(record)
        assert "db.example.com" in record.msg
        # args should be cleared after formatting
        assert record.args is None

    def test_json_quoted_keys(self) -> None:
        flt = SensitiveDataFilter()
        record = self._make_record('{"password": "secret123"}')
        flt.filter(record)
        assert "secret123" not in record.msg
        assert "REDACTED" in record.msg
        assert record.msg == '{"password": "***REDACTED***"}'

    def test_json_quoted_keys_multiple(self) -> None:
        flt = SensitiveDataFilter()
        record = self._make_record('{"api_key": "abc123xyz", "other": "value"}')
        flt.filter(record)
        assert "abc123xyz" not in record.msg
        assert "REDACTED" in record.msg
        assert record.msg == '{"api_key": "***REDACTED***", "other": "value"}'

    def test_comma_suffix(self) -> None:
        flt = SensitiveDataFilter()
        record = self._make_record('api_key=abc123xyz, other=value')
        flt.filter(record)
        assert "abc123xyz" not in record.msg
        assert "REDACTED" in record.msg
        assert record.msg == 'api_key=***REDACTED***, other=value'

    def test_space_in_secret(self) -> None:
        flt = SensitiveDataFilter()
        record = self._make_record('{"password": "my secret password"}')
        flt.filter(record)
        assert "my secret password" not in record.msg
        assert "REDACTED" in record.msg
        assert record.msg == '{"password": "***REDACTED***"}'

    def test_comma_in_secret(self) -> None:
        flt = SensitiveDataFilter()
        record = self._make_record('{"password": "my,secret,password"}')
        flt.filter(record)
        assert "my,secret,password" not in record.msg
        assert "REDACTED" in record.msg
        assert record.msg == '{"password": "***REDACTED***"}'

'''

content = content.replace(old_test, new_test)

with open("tests/unit/test_logging_config.py", "w") as f:
    f.write(content)
