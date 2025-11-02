# Changelog

All notable changes to OrbyGlasses will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Health monitoring system with automatic recovery
- Fail-safe mechanisms for SLAM tracking loss
- Production-ready packaging (pyproject.toml)
- MIT License for open-source distribution
- Comprehensive documentation (SETUP.md, TROUBLESHOOTING.md, ARCHITECTURE.md)
- Contributing guidelines and Code of Conduct

### Changed
- Refactored main.py into modular components for better maintainability
- Optimized audio latency for faster response times
- Simplified configuration file (removed unused experimental features)
- Updated README with clearer project overview

### Removed
- 22 duplicate/unused SLAM implementations (~390KB dead code)
- RTAB-Map source directory (108MB, never used)
- 7 obsolete test files and stubs
- 18 redundant documentation files
- 3 unused installation scripts (install_orbslam3.sh, install_rtabmap.sh, install_slam.sh)
- Tests for disabled experimental features

## [0.9.0] - 2025-10-29

### Added
- Ultra-aggressive relocalization fix for pySLAM tracking
- Comprehensive SLAM tuning for real-world environments
- Dense reconstruction mode with camera configuration

### Changed
- Improved pySLAM integration with better error handling
- Enhanced tracking stability in normal rooms
- Optimized performance with frame skipping

### Fixed
- pySLAM relocalization issues in low-texture environments
- Tracking loss in standard indoor spaces
- Visual odometry integration bugs

## [0.8.0] - 2025-10-28

### Added
- OrbyGlasses camera feed integration into Rerun visualizer
- Migration from DBoW2 to DBoW3 for improved loop closure
- Live pySLAM integration with 3D visualization
- Visual odometry mode for trajectory-only tracking

### Changed
- Switched to DBOW3 for better compatibility with macOS
- Improved feature detection with configurable ORB parameters

## [0.7.0] - 2025-10-25

### Added
- Initial pySLAM integration as primary SLAM system
- Support for multiple SLAM backends (pySLAM, simple SLAM fallback)
- VO (Visual Odometry) mode for lightweight tracking

### Changed
- Refactored SLAM system architecture
- Improved indoor navigation with better path planning

## [0.6.0] - 2025-10-20

### Added
- YOLOv11 integration for object detection
- Depth Anything V2 for depth estimation
- Safety system with three-level warnings (danger, caution, safe)
- Audio guidance with text-to-speech

### Changed
- Upgraded from YOLOv8 to YOLOv11n for better performance
- Improved depth estimation accuracy

## [0.5.0] - 2025-10-15

### Added
- Initial release of OrbyGlasses
- Basic object detection and depth estimation
- Audio feedback system
- Indoor navigation prototype
- Test suite with pytest

### Changed
- Core architecture established
- Configuration system with YAML files

---

## Version History Summary

- **0.9.x**: SLAM optimization and bug fixes
- **0.8.x**: Camera integration and DBoW3 migration
- **0.7.x**: pySLAM integration
- **0.6.x**: AI model upgrades (YOLOv11, Depth Anything V2)
- **0.5.x**: Initial prototype release
