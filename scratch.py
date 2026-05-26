import subprocess
try:
    subprocess.run(["bash", "/app/install.sh"], capture_output=True, text=True, check=True)
except subprocess.CalledProcessError as e:
    print("STDOUT:")
    print(e.stdout)
    print("STDERR:")
    print(e.stderr)
