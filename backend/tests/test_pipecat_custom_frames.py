from __future__ import annotations

from virtual_assistant_be.pipecat.custom_frames import (
    GestureFrame,
    PersonAppearedFrame,
    PersonDisappearedFrame,
    TelegramMessageFrame,
)


class TestPersonAppearedFrame:
    def test_with_name(self):
        f = PersonAppearedFrame(person_name="Alice")
        assert f.person_name == "Alice"

    def test_without_name(self):
        f = PersonAppearedFrame(person_name=None)
        assert f.person_name is None

    def test_defaults(self):
        f = PersonAppearedFrame()
        assert f.person_name is None


class TestPersonDisappearedFrame:
    def test_create(self):
        f = PersonDisappearedFrame()
        assert f is not None


class TestGestureFrame:
    def test_with_gesture(self):
        f = GestureFrame(gesture="wave", x=0.5, y=0.3)
        assert f.gesture == "wave"
        assert f.x == 0.5
        assert f.y == 0.3

    def test_defaults(self):
        f = GestureFrame(gesture="wave")
        assert f.x == 0.0
        assert f.y == 0.0


class TestTelegramMessageFrame:
    def test_create(self):
        f = TelegramMessageFrame(sender="Alice", text="Hello!", chat_id=12345)
        assert f.sender == "Alice"
        assert f.text == "Hello!"
        assert f.chat_id == 12345
