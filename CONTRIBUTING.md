# Contributing to OrbyGlasses

Thank you for your interest in contributing to OrbyGlasses! This project aims to help blind and visually impaired people navigate safely using AI-powered computer vision.

## Code of Conduct

This project adheres to a Code of Conduct that all contributors are expected to follow. Please read [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) before contributing.

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check existing issues to avoid duplicates. When creating a bug report, include:

- **Clear descriptive title**
- **Exact steps to reproduce** the issue
- **Expected vs. actual behavior**
- **System information** (OS, Python version, hardware)
- **Log output** (if applicable)
- **Screenshots/videos** (if relevant)

### Suggesting Enhancements

Enhancement suggestions are welcome! Please include:

- **Clear use case** - How does this help blind users?
- **Detailed description** of the proposed feature
- **Alternative approaches** you've considered
- **Impact on performance** and usability

### Pull Requests

1. **Fork the repository** and create your branch from `main`
2. **Follow the development setup** instructions below
3. **Write clear, descriptive commit messages**
4. **Add tests** for new functionality
5. **Update documentation** as needed
6. **Ensure all tests pass** before submitting
7. **Keep PRs focused** - one feature or fix per PR

## Development Setup

### Prerequisites

- Python 3.10-3.12
- macOS (primary development platform)
- Camera access for testing

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/OrbyGlasses.git
cd OrbyGlasses

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements-dev.txt

# Install pySLAM (for SLAM features)
cd third_party/pyslam
# Follow pySLAM installation instructions
cd ../..
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_detection.py

# Run tests in parallel
pytest -n auto
```

### Code Style

We use automated formatting and linting:

```bash
# Format code with black
black src/ tests/

# Sort imports with isort
isort src/ tests/

# Check linting with flake8
flake8 src/ tests/

# Type checking with mypy
mypy src/
```

**Install pre-commit hooks** to automate this:

```bash
pre-commit install
```

## Project Structure

```
OrbyGlasses/
├── src/
│   ├── core/          # Core detection, depth, audio systems
│   ├── features/      # Advanced features (scene understanding, haptic, etc.)
│   ├── navigation/    # SLAM and indoor navigation
│   └── visualization/ # UI and debugging visualizations
├── tests/             # Test suite
├── config/            # Configuration files
├── docs/              # Documentation
└── third_party/       # External dependencies (pySLAM)
```

## Development Guidelines

### Safety First

OrbyGlasses is designed for blind users - **safety is paramount**:

- Never introduce changes that could delay obstacle warnings
- Always test with real-world scenarios
- Consider failure modes and graceful degradation
- Document any safety-critical changes

### Performance Matters

The system runs real-time on resource-constrained devices:

- Profile performance-critical code
- Avoid blocking operations in the main loop
- Use threading/async for I/O operations
- Target <200ms latency for audio warnings

### Accessibility

Keep the end user in mind:

- Audio feedback must be clear and concise
- Support voice control for hands-free operation
- Avoid visual-only debugging output
- Test with eyes closed when possible

### Testing Requirements

- **Unit tests** for all new functions
- **Integration tests** for feature interactions
- **Performance tests** for latency-critical code
- **End-to-end tests** for user workflows

### Documentation

- **Docstrings** for all public functions (Google style)
- **Type hints** throughout
- **README updates** for new features
- **CHANGELOG entries** for user-facing changes

## Priority Areas for Contribution

### High Impact

1. **Stair detection** - Critical safety feature
2. **Audio latency reduction** - Performance improvement
3. **SLAM stability** - Fix loop closure crashes
4. **Voice control** - Accessibility enhancement

### Medium Impact

5. **Indoor navigation** - Waypoint-based guidance
6. **Map persistence** - SLAM map save/load
7. **Test coverage** - Increase to 70%+
8. **Documentation** - API docs, tutorials

### Nice to Have

9. **Haptic feedback** - Hardware integration
10. **Multi-language support** - Internationalization
11. **Cloud sync** - Map sharing
12. **Mobile support** - iOS/Android ports

## Commit Message Guidelines

Follow the conventional commits format:

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `test`: Adding/updating tests
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `chore`: Maintenance tasks

**Example:**

```
feat(detection): add stair detection using depth discontinuities

Implements depth-based stair detection to identify vertical drops >30cm
within 2 meters. Provides high-priority audio warnings to prevent falls.

Closes #123
```

## Review Process

1. **Automated checks** must pass (tests, linting, type checking)
2. **Code review** by at least one maintainer
3. **Testing** on real hardware (if available)
4. **Documentation review** for clarity
5. **Merge** once approved

## Getting Help

- **GitHub Issues**: For bugs and feature requests
- **Discussions**: For questions and general discussion
- **Email**: [Your contact email]

## Recognition

Contributors will be recognized in:

- `CONTRIBUTORS.md` file
- GitHub contributors page
- Release notes for significant contributions

## License

By contributing, you agree that your contributions will be licensed under the GPL-3.0 License.

---

**Thank you for helping make navigation safer and more accessible for blind and visually impaired people!**
