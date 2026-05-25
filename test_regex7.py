import re

pattern = re.compile(
    r"(?i)"
    r"(['\"]?)"
    r"("
    r"password|passwd|pwd"
    r"|api_key|apikey|api[-_]?secret"
    r"|secret_key|secret[-_]?token"
    r"|access_token|auth_token|bearer"
    r"|private_key"
    r")"
    r"\1"
    r"([\s]*[=:]\s*)"
    r"("
    r"(['\"])(.*?)\5"
    r"|"
    r"([^\s'\",]+)"
    r")"
)

text1 = '{"password": "secret123"}'
text2 = '{"api_key": "abc123xyz", "other": "value"}'
text3 = 'api_key=abc123xyz, other=value'
text4 = 'api_key="abc123xyz"'
text5 = "api_key='abc123xyz'"
text6 = "password=secret123"
text7 = "password = secret123"
text8 = '{"password": "my secret password"}'
text9 = '{"password": "my,secret,password"}'
text10 = 'password="my secret password"'

def repl(m):
    # m.group(1): quote around key
    # m.group(2): key
    # m.group(3): separator
    # m.group(4): entire matched value
    # m.group(5): quote around value (if quoted)
    # m.group(6): value string (if quoted)
    # m.group(7): value string (if unquoted)

    quote = m.group(5) if m.group(5) else ''
    return f"{m.group(1)}{m.group(2)}{m.group(1)}{m.group(3)}{quote}***REDACTED***{quote}"

print(pattern.sub(repl, text1))
print(pattern.sub(repl, text2))
print(pattern.sub(repl, text3))
print(pattern.sub(repl, text4))
print(pattern.sub(repl, text5))
print(pattern.sub(repl, text6))
print(pattern.sub(repl, text7))
print(pattern.sub(repl, text8))
print(pattern.sub(repl, text9))
print(pattern.sub(repl, text10))
