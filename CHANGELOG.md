# Changelog

All notable changes to OrbyGlasses will be documented in this file.

## [Unreleased]

### Added
- Production configuration file (`config/config_production.yaml`) with all critical features enabled
- Consolidated developer documentation (`docs/DEVELOPER.md`)
- Clear separation between production and development configs

### Changed
- Updated README.md with production configuration guidance
- Consolidated 18+ redundant markdown files into main documentation
- Improved documentation structure and clarity

### Removed
- Removed redundant documentation files:
  - `ANALYSIS.md`, `BUGFIX_DEPTH_NONE.md`, `DEPLOYMENT_SUMMARY.md`
  - `FEATURE_MATCHING_FINAL.md`, `FEATURE_MATCHING_MODE.md`, `FEATURE_VISUALIZATION_GUIDE.md`
  - `FINAL_SUMMARY.md`, `MODE_EXPLANATION.md`, `MODE_USAGE.md`
  - `PERFORMANCE_FIXES.md`, `QUICKFIX.md`, `QUICK_START.md`
  - `REALTIME_USAGE.md`, `SPEED_OPTIMIZATION_GUIDE.md`, `WHATS_NEW.md`
- Consolidated technical docs from `docs/` into `docs/DEVELOPER.md`:
  - `PYSLAM_MODIFICATIONS.md`
  - `RELOCALIZATION_TUNING.md`
  - `DENSE_RECONSTRUCTION.md`
  - `REALTIME_DENSE_MAPPING.md`

## [0.9.0] - 2025-11-01

### Added
- Production-ready configuration with all critical features enabled
- pySLAM integration for real-time SLAM tracking
- Stair and curb detection for fall prevention
- Voice control with wake word ("Hey Orby")
- Map save/load functionality
- Indoor navigation with path planning
- Crash recovery (auto-disables loop closure on crashes)
- Memory management to prevent leaks

### Changed
- Improved safety features with automatic emergency stop
- Faster audio warnings (0.5 seconds for danger warnings)
- Better error messages and recovery
- Optimized performance for M1/M2/M3/M4 Macs

### Fixed
- Fixed camera tracking in normal rooms
- Fixed audio delay problems
- Fixed memory leaks
- Fixed SLAM relocalization issues

### Removed
- Removed 22 old test files
- Removed 15 old setup scripts
- Cleaned up 108MB of unused code

## [0.8.0] - 2025-10-15

### Added
- Indoor position tracking with pySLAM
- Real-time camera visualization
- Improved position accuracy

## [0.7.0] - 2025-10-01

### Added
- First working version of indoor navigation
- Position tracking inside buildings
- Basic path planning

## [0.6.0] - 2025-09-15

### Changed
- Updated object detection to YOLOv11
- Improved depth estimation accuracy
- Three-level warning system: danger, caution, safe

## [0.5.0] - 2025-09-01

### Added
- Initial release
- Object detection (YOLO)
- Depth estimation
- Audio guidance (TTS)
- Basic test suite

---

## Future Plans

### Priority 1
- Reduce audio latency to <500ms for danger warnings
- Add redundant audio (beeps + speech for urgent warnings)
- Improve stair detection accuracy (>90% target)

### Priority 2
- Stereo spatial audio positioning
- Haptic feedback integration
- Head-level hazard detection

### Future
- Mobile app (iOS/Android)
- Multi-language support
- Glass door detection
- Linux/Windows support
