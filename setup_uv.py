"""Setup script to initialize UV virtual environment and install dependencies."""

import os
import subprocess
import sys


def run_command(command, description):
    """Run a shell command and handle errors."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {command}")
    print(f"{'='*60}\n")
    
    result = subprocess.run(command, shell=True, capture_output=False)
    
    if result.returncode != 0:
        print(f"Error: {description} failed!")
        sys.exit(1)
    
    return result


def main():
    """Main setup function."""
    print("Soccer Tracker - UV Setup Script")
    print("=" * 60)
    
    # Check if UV is installed
    print("\n1. Checking if UV is installed...")
    result = subprocess.run("uv --version", shell=True, capture_output=True)
    if result.returncode != 0:
        print("UV is not installed. Installing UV...")
        run_command("pip install uv", "Installing UV")
    else:
        print(f"UV is installed: {result.stdout.decode().strip()}")
    
    # Create virtual environment
    print("\n2. Creating virtual environment...")
    if os.path.exists(".venv"):
        print(".venv directory already exists. Skipping creation.")
    else:
        run_command("uv venv", "Creating virtual environment")
    
    # Determine activation command based on OS
    if sys.platform == "win32":
        activate_cmd = ".venv\\Scripts\\activate"
        pip_cmd = ".venv\\Scripts\\uv.exe pip install -e ."
    else:
        activate_cmd = "source .venv/bin/activate"
        pip_cmd = ".venv/bin/uv pip install -e ."
    
    # Install dependencies
    print("\n3. Installing project dependencies...")
    run_command(pip_cmd, "Installing dependencies with UV")
    
    print("\n" + "=" * 60)
    print("Setup completed successfully!")
    print("=" * 60)
    print("\nTo activate the virtual environment, run:")
    if sys.platform == "win32":
        print(f"  {activate_cmd}")
    else:
        print(f"  {activate_cmd}")
    print("\nThen you can run the tracker with:")
    print("  python main.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
