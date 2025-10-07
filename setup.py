#!/usr/bin/env python3
"""Setup script for 254Carbon ML Platform."""

import subprocess
import sys
import os
from pathlib import Path
import argparse
import structlog

logger = structlog.get_logger("setup")


def run_command(command: str, description: str, check: bool = True):
    """Run a shell command with logging."""
    logger.info(f"Running: {description}")
    print(f"🔄 {description}...")
    
    try:
        result = subprocess.run(
            command,
            shell=True,
            check=check,
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print(f"✅ {description} completed")
            if result.stdout:
                print(result.stdout)
        else:
            print(f"❌ {description} failed")
            if result.stderr:
                print(result.stderr)
            if check:
                sys.exit(1)
                
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        if e.stdout:
            print(e.stdout)
        if e.stderr:
            print(e.stderr)
        if check:
            sys.exit(1)


def setup_development_environment():
    """Setup development environment."""
    print("🚀 Setting up 254Carbon ML Platform development environment...\n")
    
    # Check Python version
    python_version = sys.version_info
    if python_version < (3, 9):
        print("❌ Python 3.9+ required")
        sys.exit(1)
    print(f"✅ Python {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # Check Docker
    run_command("docker --version", "Checking Docker installation")
    run_command("docker-compose --version", "Checking Docker Compose installation")
    
    # Install Python dependencies
    run_command("pip install -r requirements.txt", "Installing Python dependencies")
    
    # Install pre-commit hooks
    run_command("pre-commit install", "Installing pre-commit hooks")
    
    # Create data directories
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    print("✅ Created data directories")
    
    # Copy environment file
    if not os.path.exists(".env"):
        if os.path.exists("env.example"):
            run_command("cp env.example .env", "Creating .env file")
            print("⚠️  Please update .env file with your configuration")
        else:
            print("⚠️  No env.example file found")
    
    print("\n🎉 Development environment setup completed!")
    print("\nNext steps:")
    print("1. Update .env file with your configuration")
    print("2. Run 'make docker-up' to start all services")
    print("3. Run 'make test' to verify everything works")
    print("4. Visit http://localhost:5000 for MLflow UI")


def setup_production_environment():
    """Setup production environment."""
    print("🏭 Setting up production environment...\n")
    
    # Check Kubernetes
    run_command("kubectl version --client", "Checking kubectl installation")
    run_command("helm version", "Checking Helm installation")
    
    # Check Vault (optional)
    run_command("vault version", "Checking Vault installation", check=False)
    
    # Validate Kubernetes manifests
    run_command("kubectl apply --dry-run=client -f k8s/base/", "Validating Kubernetes manifests")
    
    print("\n✅ Production environment validation completed!")
    print("\nNext steps:")
    print("1. Configure Kubernetes cluster access")
    print("2. Setup Vault for secret management")
    print("3. Deploy infrastructure: 'kubectl apply -f k8s/base/'")
    print("4. Deploy services with appropriate image tags")


def run_tests():
    """Run all tests."""
    print("🧪 Running tests...\n")
    
    # Lint and format
    run_command("black --check .", "Checking code formatting")
    run_command("flake8 .", "Running linting")
    run_command("mypy libs/ --ignore-missing-imports", "Running type checking")
    
    # Unit tests
    run_command("pytest tests/test_*.py -v", "Running unit tests")
    
    # Integration tests (if services are running)
    run_command("pytest tests/integration/ -v", "Running integration tests", check=False)
    
    # Contract tests
    run_command("pytest tests/contract/ -v", "Running contract tests")
    
    print("\n✅ All tests completed!")


def generate_sample_data():
    """Generate sample training data."""
    print("📊 Generating sample data...\n")
    
    run_command(
        "python training/curve_forecaster/data_generator.py",
        "Generating synthetic curve data"
    )
    
    print("✅ Sample data generated!")


def train_sample_model():
    """Train a sample model."""
    print("🤖 Training sample model...\n")
    
    # Start MLflow if not running
    print("Starting MLflow server...")
    run_command("make docker-up", "Starting services", check=False)
    
    # Wait a bit for services to start
    import time
    time.sleep(10)
    
    # Train model
    run_command(
        "python training/curve_forecaster/train.py --model-type ensemble --lookback-days 30",
        "Training curve forecasting model"
    )
    
    print("✅ Sample model trained!")
    print("Visit http://localhost:5000 to see the experiment in MLflow")


def main():
    """Main setup function."""
    parser = argparse.ArgumentParser(description="254Carbon ML Platform Setup")
    parser.add_argument(
        "command",
        choices=["dev", "prod", "test", "data", "train", "all"],
        help="Setup command to run"
    )
    
    args = parser.parse_args()
    
    # Configure logging
    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer()
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    if args.command == "dev":
        setup_development_environment()
    elif args.command == "prod":
        setup_production_environment()
    elif args.command == "test":
        run_tests()
    elif args.command == "data":
        generate_sample_data()
    elif args.command == "train":
        train_sample_model()
    elif args.command == "all":
        setup_development_environment()
        generate_sample_data()
        train_sample_model()
        run_tests()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
