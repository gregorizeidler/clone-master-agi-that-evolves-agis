#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script to start the Clone Master web interface.
This script facilitates running the Streamlit application.
"""

import os
import sys
import argparse
import subprocess

def main():
    """Starts the Streamlit application."""
    parser = argparse.ArgumentParser(description='Start the Clone Master web interface')
    parser.add_argument('--port', '-p', type=int, default=8501, help='Port to start Streamlit on (default: 8501)')
    args = parser.parse_args()
    
    print("Starting the Clone Master web interface...")
    
    # Determine the path to app.py
    script_dir = os.path.dirname(os.path.abspath(__file__))
    app_path = os.path.join(script_dir, "app.py")
    
    # Check if app.py exists
    if not os.path.exists(app_path):
        print(f"Error: File {app_path} not found!")
        print("Make sure you are in the correct directory.")
        sys.exit(1)
    
    try:
        # Build the command with the specified port
        cmd = [sys.executable, "-m", "streamlit", "run", app_path, "--server.port", str(args.port)]
        print(f"Starting Streamlit app on port {args.port}...")
        subprocess.run(cmd)
    except FileNotFoundError:
        print("Error: Streamlit not found!")
        print("Make sure Streamlit is installed:")
        print("  pip install streamlit")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"Error running Streamlit: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nApplication terminated by user.")
        sys.exit(0)

if __name__ == "__main__":
    main()
