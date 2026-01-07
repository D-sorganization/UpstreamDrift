#!/bin/bash

# 1. Set DISPLAY for X11 (VcXsrv)
# This addresses the issue where Tkinter might fail or not show up if DISPLAY isn't set.
export DISPLAY=$(cat /etc/resolv.conf | grep nameserver | awk '{print $2}'):0.0
echo "Environment: DISPLAY=$DISPLAY"

# 2. Check for python3-tk
if ! dpkg -s python3-tk >/dev/null 2>&1; then
    echo "Installing python3-tk (required for GUI)..."
    sudo apt-get update && sudo apt-get install -y python3-tk
fi

# 3. Launch the GUI
echo "Starting DeepMind Control Suite GUI..."
# We assume this script is in .../docker/ and the gui is in .../docker/gui/
python3 gui/deepmind_control_suite_MuJoCo_GUI.py
