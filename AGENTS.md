# Task Tracking

## Completed
- Design face recognition service architecture
- Create FaceService with embedding extraction and ChromaDB integration
- Integrate FaceService into CameraService for per-frame recognition
- Update BehaviorController to greet known persons and ask unknown
- Add insightface dependency and run test (all 91 pass)
- Fix missing `await` bug in `_register_name` (line 184)
- Add tests for known/unknown person greeting + name registration
- Implement audio echo suppression (mute mic during TTS playback)
- Add `Timer` utility and full-pipeline timing instrumentation (VAD, STT, LLM, RAG, TTS)
- Auto-select USB webcam over built-in camera (scans indices 0-3, prefers USB via sysfs)
- Webcam mic auto-selection (matches audio device to camera name via sounddevice)
- Fix VAD silence timeout latency: 1.0s → 0.5s
- AudioService accepts device_id parameter for mic selection
- Camera-to-audio device matching wired through BehaviorController
- Adaptive VAD noise floor: replaces hard-coded SILENCE_THRESHOLD with dynamic threshold (2.5x noise floor)
- Interruptible TTS: via barge-in (loud sound while muted) or open-palm gesture
- Barge-in detection: while muted, RMS checked against 3x noise-floor threshold to detect user interruption
- Gesture interruption: open palm detected by camera during TTS stops speech immediately
- Audio speech buffer cleared on mute to prevent residual echo processing

## Known Issues
- Keyboard interrupt (Ctrl+C) fails to stop process on first attempt; requires second kill
- Hand gesture service not working (no gesture detection)
