import re

with open("src/shared/python/signal_toolkit/io.py", "r") as f:
    content = f.read()

content = content.replace(
    "time = np.asarray(data[time_key], dtype=float)",
    "time = np.asarray(data[time_key], dtype=float)"
)

content = re.sub(
    r"time = np.asarray\(data\[time_key\]\)",
    "time = np.asarray(data[time_key], dtype=float)",
    content
)
content = re.sub(
    r"values = np.asarray\(data\[value_key\]\)",
    "values = np.asarray(data[value_key], dtype=float)",
    content
)

with open("src/shared/python/signal_toolkit/io.py", "w") as f:
    f.write(content)
