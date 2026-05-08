from __future__ import annotations

from virtual_assistant_be.services.camera_service import _classify_gesture


class MockLandmark:
    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0):
        self.x = x
        self.y = y
        self.z = z


def _make_landmarks() -> list:
    return [MockLandmark() for _ in range(21)]


class TestGestureClassification:
    def test_thumbs_up(self):
        lm = _make_landmarks()
        # Thumb extended: tip(4).x < ip(3).x (for right hand facing camera)
        lm[4].x = 0.3
        lm[3].x = 0.5
        # All other fingers curled: tip.y > pip.y
        for tip, pip in [(8, 6), (12, 10), (16, 14), (20, 18)]:
            lm[tip].y = 0.6
            lm[pip].y = 0.4
        assert _classify_gesture(lm) == "thumbs_up"

    def test_point(self):
        lm = _make_landmarks()
        # Index extended
        lm[8].y = 0.3
        lm[6].y = 0.5
        # Others curled
        lm[4].x = 0.6
        lm[3].x = 0.5
        for tip, pip in [(12, 10), (16, 14), (20, 18)]:
            lm[tip].y = 0.6
            lm[pip].y = 0.4
        assert _classify_gesture(lm) == "point"

    def test_peace(self):
        lm = _make_landmarks()
        # Index and middle extended
        lm[8].y = 0.3
        lm[6].y = 0.5
        lm[12].y = 0.3
        lm[10].y = 0.5
        # Ring and pinky curled
        lm[16].y = 0.6
        lm[14].y = 0.4
        lm[20].y = 0.6
        lm[18].y = 0.4
        # Thumb can be anything for peace
        assert _classify_gesture(lm) == "peace"

    def test_open_palm(self):
        lm = _make_landmarks()
        # All fingers extended
        lm[4].x = 0.3
        lm[3].x = 0.5
        for tip, pip in [(8, 6), (12, 10), (16, 14), (20, 18)]:
            lm[tip].y = 0.3
            lm[pip].y = 0.5
        assert _classify_gesture(lm) == "open_palm"

    def test_fist(self):
        lm = _make_landmarks()
        # All fingers curled
        lm[4].x = 0.6
        lm[3].x = 0.5
        for tip, pip in [(8, 6), (12, 10), (16, 14), (20, 18)]:
            lm[tip].y = 0.6
            lm[pip].y = 0.4
        assert _classify_gesture(lm) == "fist"

    def test_no_gesture_returns_none(self):
        lm = _make_landmarks()
        # Only ring finger extended — not a defined gesture
        lm[4].x = 0.6
        lm[3].x = 0.5
        for tip, pip in [(8, 6), (12, 10), (20, 18)]:
            lm[tip].y = 0.6
            lm[pip].y = 0.4
        lm[16].y = 0.3
        lm[14].y = 0.5
        assert _classify_gesture(lm) is None
