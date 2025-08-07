#!/usr/bin/env python3
"""
Financial RAG Assistant Setup Script
Helps users set up the environment and dependencies
"""

import os
import subprocess
import sys
from pathlib import Path

def print_header(title):
    """Print a formatted header"""
    print("\n" + "=" * 50)
    print(f"🔧 {title}")
    print("=" * 50)

def check_python_version():
    """Check if Python version is compatible"""
    print_header("Checking Python Version")
    
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ is required")
        print(f"Current version: {sys.version}")
        return False
    
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
    return True

def install_dependencies():
    """Install required Python packages"""
    print_header("Installing Dependencies")
    
    try:
        print("📦 Installing Python packages...")
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Dependencies installed successfully")
            return True
        else:
            print("❌ Failed to install dependencies")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ Error installing dependencies: {e}")
        return False

def setup_environment():
    """Set up environment variables"""
    print_header("Setting Up Environment")
    
    env_file = Path(".env")
    env_example = Path(".env.example")
    
    if env_file.exists():
        print("ℹ️ .env file already exists")
        return True
    
    if not env_example.exists():
        print("❌ .env.example not found")
        return False
    
    # Copy example to .env
    with open(env_example, 'r') as f:
        content = f.read()
    
    with open(env_file, 'w') as f:
        f.write(content)
    
    print("✅ Created .env file from template")
    print("⚠️ Please edit .env file with your API keys:")
    print("   - OPENAI_API_KEY (required)")
    print("   - QDRANT_API_KEY (optional)")
    print("   - Other optional keys")
    
    return True

def check_qdrant():
    """Check if Qdrant is available"""
    print_header("Checking Qdrant")
    
    try:
        import requests
        response = requests.get("http://localhost:6333/health", timeout=5)
        if response.status_code == 200:
            print("✅ Qdrant is running locally")
            return True
    except:
        pass
    
    print("⚠️ Qdrant not detected locally")
    print("To start Qdrant with Docker:")
    print("   docker run -p 6333:6333 qdrant/qdrant")
    print("Or install locally: https://qdrant.tech/documentation/quick-start/")
    
    return False

def create_directories():
    """Create necessary directories"""
    print_header("Creating Directories")
    
    directories = [
        "data",
        "data/sec_filings", 
        "processed_data"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {directory}")
    
    return True

def validate_setup():
    """Validate the setup"""
    print_header("Validating Setup")
    
    # Check if .env has required keys
    env_file = Path(".env")
    if not env_file.exists():
        print("❌ .env file not found")
        return False
    
    with open(env_file, 'r') as f:
        env_content = f.read()
    
    if "OPENAI_API_KEY=your_openai_api_key_here" in env_content:
        print("⚠️ Please set your OPENAI_API_KEY in .env file")
        return False
    
    print("✅ Environment configuration looks good")
    
    # Try importing main modules
    try:
        from src import FinancialRAGAssistant
        print("✅ All modules can be imported")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def main():
    """Main setup function"""
    print("🚀 Financial RAG Assistant Setup")
    print("This script will help you set up the environment")
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        print("\n❌ Setup failed during dependency installation")
        sys.exit(1)
    
    # Set up environment
    if not setup_environment():
        print("\n❌ Setup failed during environment setup")
        sys.exit(1)
    
    # Create directories
    if not create_directories():
        print("\n❌ Setup failed during directory creation")
        sys.exit(1)
    
    # Check Qdrant
    check_qdrant()
    
    # Validate setup
    if validate_setup():
        print_header("Setup Complete! 🎉")
        print("\nNext steps:")
        print("1. Edit .env file with your API keys")
        print("2. Start Qdrant (if not already running)")
        print("3. Run the demo: python demo.py")
        print("4. Or run interactively: python main.py")
        print("\nFor help, see README.md")
    else:
        print("\n⚠️ Setup completed with warnings")
        print("Please check the issues above and try again")

if __name__ == "__main__":
    main()