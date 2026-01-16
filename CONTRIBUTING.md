# Contributing to Soccer Player Tracking

Thank you for your interest in contributing! This document provides guidelines and instructions for contributing to this project.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/adnanSattar/soccer-player-tracking.git`
3. Create a virtual environment and install dependencies:

   ```bash
   uv venv
   source .venv/bin/activate  # or .venv\Scripts\activate on Windows
   uv pip install -e ".[dev]"
   ```

## Development Setup

1. Install development dependencies:

   ```bash
   uv pip install -e ".[dev]"
   ```

2. Run code formatting:

   ```bash
   black src/ main.py scripts/
   ```

3. Run linting:

   ```bash
   ruff check src/ main.py scripts/
   ```

4. Run type checking:

   ```bash
   mypy src/ main.py
   ```

## Making Changes

1. Create a new branch for your feature/fix:

   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes following the code style guidelines

3. Test your changes:

   ```bash
   python main.py  # Test with a sample video
   ```

4. Commit your changes with clear messages:

   ```bash
   git commit -m "Add feature: description of what you added"
   ```

## Code Style

- Follow PEP 8 style guidelines
- Use type hints where possible
- Add docstrings to functions and classes
- Keep functions focused and small
- Write clear, descriptive variable names

## Pull Request Process

1. Update the README.md if needed
2. Ensure all tests pass (if applicable)
3. Update documentation for new features
4. Submit a pull request with a clear description

## Areas for Contribution

- Performance optimizations
- Additional tracking features
- Better team assignment algorithms
- Improved camera movement estimation
- Documentation improvements
- Bug fixes
- Test coverage

## Questions?

Open an issue for discussion before making major changes.
