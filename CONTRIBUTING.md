# Contributing to OrbyGlasses

Thank you for your interest in contributing to OrbyGlasses! This project aims to help blind and visually impaired people navigate safely, and we welcome contributions from everyone.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
- [Development Setup](#development-setup)
- [Coding Standards](#coding-standards)
- [Pull Request Process](#pull-request-process)
- [Testing](#testing)
- [Documentation](#documentation)

## Code of Conduct

This project adheres to a Code of Conduct that all contributors are expected to follow. Please read [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) before contributing.

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check the [issue tracker](https://github.com/yourusername/OrbyGlasses/issues) to avoid duplicates.

**When filing a bug report, include:**
- OrbyGlasses version (`git rev-parse HEAD`)
- macOS version (`sw_vers`)
- Python version (`python3 --version`)
- Steps to reproduce
- Expected vs actual behavior
- Relevant logs from `data/logs/orbyglass.log`
- Screenshots/videos if applicable

**Example:**
```markdown
**Environment:**
- OrbyGlasses: commit abc123
- macOS: 14.2 (Sonoma)
- Python: 3.11.5

**Steps to Reproduce:**
1. Launch with `./run_orby.sh`
2. Point camera at blank wall
3. SLAM loses tracking after 5 seconds

**Expected:** SLAM should relocalize
**Actual:** System freezes

**Logs:** [attach orbyglass.log]
```

### Suggesting Enhancements

We welcome feature suggestions! Please:
- Check if the feature is already [requested](https://github.com/yourusername/OrbyGlasses/issues)
- Explain the use case (how it helps blind users)
- Provide examples or mockups if applicable
- Consider accessibility implications

### Contributing Code

We welcome pull requests for:
- Bug fixes
- Performance improvements
- New features (discuss in an issue first)
- Documentation improvements
- Test coverage

## Development Setup

### 1. Fork and Clone

```bash
# Fork the repository on GitHub
# Then clone your fork
git clone https://github.com/YOUR_USERNAME/OrbyGlasses.git
cd OrbyGlasses
```

### 2. Set Up Development Environment

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install pytest pytest-cov black ruff mypy

# Install pySLAM
./install_pyslam.sh
```

### 3. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/issue-number-description
```

## Coding Standards

### Python Style

We follow [PEP 8](https://peps.python.org/pep-0008/) with some modifications:

- **Line length:** 100 characters (not 79)
- **Formatter:** Black
- **Linter:** Ruff
- **Type hints:** Encouraged but not required

**Format your code:**
```bash
black src/ tests/
ruff check src/ tests/ --fix
```

### Code Organization

```python
# Good: Clear, focused functions
def calculate_distance(depth_map, bbox):
    """Calculate median distance to object."""
    roi = depth_map[bbox.y1:bbox.y2, bbox.x1:bbox.x2]
    return np.median(roi)

# Bad: Large, unfocused functions
def process_everything(frame):
    # 500 lines of mixed responsibilities
    ...
```

### Accessibility First

**Remember: This software helps blind people navigate safely.**

- Audio feedback must be clear and timely (<500ms latency critical)
- Error messages should be spoken, not just printed
- Fail-safes are mandatory for safety-critical features
- Test with screen readers if adding UI

### Documentation

All public functions should have docstrings:

```python
def detect_objects(frame: np.ndarray, confidence: float = 0.55) -> List[Detection]:
    """
    Detect objects in a camera frame using YOLOv11.

    Args:
        frame: RGB image as numpy array (HxWx3)
        confidence: Detection threshold (0-1), default 0.55

    Returns:
        List of Detection objects with class, bbox, and confidence

    Example:
        >>> frame = cv2.imread('test.jpg')
        >>> detections = detect_objects(frame, confidence=0.6)
        >>> print(f"Found {len(detections)} objects")
    """
    ...
```

### Comments

```python
# Good: Explain WHY, not WHAT
# Use median instead of mean to handle outliers from depth estimation errors
distance = np.median(roi)

# Bad: Obvious comments
# Calculate the median
distance = np.median(roi)
```

## Pull Request Process

### 1. Ensure Quality

Before submitting:

```bash
# Run tests
pytest tests/ -v

# Check formatting
black --check src/ tests/

# Check linting
ruff check src/ tests/

# Check types (optional)
mypy src/
```

### 2. Update Documentation

- Update README.md if adding features
- Add/update docstrings
- Update CHANGELOG.md
- Add examples if applicable

### 3. Write Tests

All new features should include tests:

```python
# tests/test_new_feature.py
def test_distance_calculation():
    """Test distance calculation with synthetic depth map."""
    depth_map = np.ones((480, 640)) * 2.5  # 2.5m uniform depth
    bbox = BoundingBox(100, 100, 200, 200)

    distance = calculate_distance(depth_map, bbox)

    assert 2.4 <= distance <= 2.6, "Distance should be ~2.5m"
```

### 4. Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```bash
# Format: <type>(<scope>): <description>

feat(slam): add relocalization fallback to visual odometry
fix(audio): reduce TTS latency from 2s to 500ms
docs(setup): add troubleshooting for camera permissions
test(detection): add edge case tests for low-light conditions
refactor(main): split OrbyGlasses class into smaller modules
perf(depth): optimize depth estimation with frame skipping
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only
- `test`: Adding/fixing tests
- `refactor`: Code restructuring
- `perf`: Performance improvement
- `chore`: Maintenance tasks

### 5. Submit Pull Request

**PR Title:**
```
feat(slam): Add relocalization fallback
```

**PR Description Template:**
```markdown
## Description
Brief explanation of changes

## Motivation
Why is this change needed? What problem does it solve?

## Changes
- Added X
- Modified Y
- Fixed Z

## Testing
How did you test this?
- [ ] Manual testing with camera
- [ ] Unit tests pass
- [ ] Integration tests pass

## Accessibility Impact
How does this affect blind users?

## Screenshots/Videos
[If applicable]

## Checklist
- [ ] Code follows style guidelines
- [ ] Tests added/updated
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
```

### 6. Review Process

Maintainers will:
1. Review code within 3-7 days
2. Request changes if needed
3. Approve and merge when ready

**Please be patient and respectful during reviews.**

## Testing

### Running Tests

```bash
# All tests
pytest tests/ -v

# Specific test file
pytest tests/test_detection.py -v

# With coverage
pytest tests/ --cov=src --cov-report=html

# Open coverage report
open htmlcov/index.html
```

### Writing Tests

**Test Structure:**
```python
# tests/test_module.py
import pytest
from src.module import function_to_test

class TestModuleName:
    def test_normal_case(self):
        """Test typical usage."""
        result = function_to_test(normal_input)
        assert result == expected_output

    def test_edge_case(self):
        """Test boundary conditions."""
        result = function_to_test(edge_input)
        assert result is handled_correctly

    def test_error_handling(self):
        """Test error conditions."""
        with pytest.raises(ValueError):
            function_to_test(invalid_input)
```

**Mocking Hardware:**
```python
# Mock camera for testing without hardware
@pytest.fixture
def mock_camera(monkeypatch):
    class FakeCamera:
        def read(self):
            # Return synthetic test frame
            return True, np.zeros((480, 640, 3), dtype=np.uint8)

    monkeypatch.setattr('cv2.VideoCapture', lambda x: FakeCamera())
```

## Documentation

### README Updates

When adding features, update README.md:
- Quick Start section if setup changes
- Features list
- Configuration examples
- Troubleshooting if needed

### Architecture Documentation

For significant changes, update ARCHITECTURE.md:
- Component diagrams
- Data flow
- Integration points

### Code Comments

Use docstrings for:
- All public functions/classes
- Complex algorithms
- Non-obvious design decisions

## Accessibility Testing

### Testing with Screen Readers

If you have access to blind users or screen readers:
1. Test audio clarity and timing
2. Verify warnings are understandable
3. Check for confusing terminology
4. Measure real-world reaction time

### Performance Requirements

- **FPS:** ≥15 FPS minimum (20-30 target)
- **Audio latency:** <500ms for safety warnings
- **SLAM tracking:** Recovery within 3 seconds
- **Memory:** <2GB peak usage

## Getting Help

**Questions?**
- Open a [Discussion](https://github.com/yourusername/OrbyGlasses/discussions)
- Ask in the [issue tracker](https://github.com/yourusername/OrbyGlasses/issues)
- Email maintainers (check AUTHORS file)

**First-time contributors:**
- Look for issues tagged `good-first-issue`
- Start with documentation improvements
- Ask questions in issues/discussions

## Recognition

Contributors are recognized in:
- AUTHORS file
- CHANGELOG.md (per release)
- GitHub contributors page

## License

By contributing, you agree that your contributions will be licensed under the GNU General Public License v3.0 or later.

---

**Thank you for making OrbyGlasses better and helping blind people navigate safely!**
