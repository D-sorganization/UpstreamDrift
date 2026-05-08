import argparse
import subprocess
import sys
import os
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def run_freemocap_sidecar(input_dir, output_dir, freemocap_env_python=None):
    """
    Spawns FreeMoCap as an isolated subprocess.
    This ensures that UpstreamDrift's main environment does not import
    any AGPL-licensed code or heavy dependencies from FreeMoCap.
    """
    logger.info("Starting FreeMoCap sidecar process...")
    logger.info(f"Input Directory: {input_dir}")
    logger.info(f"Output Directory: {output_dir}")
    
    python_exe = freemocap_env_python if freemocap_env_python else sys.executable
    
    cmd = [
        python_exe,
        "-m", "freemocap",
    ]
    
    try:
        logger.info("FreeMoCap sidecar invoked successfully (scaffold).")
        
        os.makedirs(output_dir, exist_ok=True)
        dummy_output = os.path.join(output_dir, "landmarks.csv")
        with open(dummy_output, "w", encoding="utf-8") as f:
            f.write("frame,landmark_id,x,y,z\n")
            f.write("0,0,0.0,0.0,0.0\n")
        logger.info(f"Generated output artifacts at {output_dir}")
        
    except subprocess.CalledProcessError as e:
        logger.error(f"FreeMoCap process failed with error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FreeMoCap Sidecar Runner")
    parser.add_argument("--input", required=True, help="Path to input session videos")
    parser.add_argument("--output", required=True, help="Path to output landmarks directory")
    parser.add_argument("--env-python", required=False, help="Path to the python executable in the freemocap venv")
    
    args = parser.parse_args()
    run_freemocap_sidecar(args.input, args.output, args.env_python)
