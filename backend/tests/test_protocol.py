from __future__ import annotations

from virtual_assistant_be.core.protocol import (
    AnimationCmd,
    StateUpdate,
    SpeakCmd,
    ListenIndicator,
    ThinkIndicator,
    GoEvent,
    GoCommand,
    serialize,
    parse,
    OUTGOING_TYPES,
    INCOMING_TYPES,
)


class TestSerialize:
    def test_animation_cmd(self):
        result = serialize(AnimationCmd(name="greet"))
        assert result == {"type": "animation", "name": "greet"}

    def test_animation_cmd_with_params(self):
        result = serialize(AnimationCmd(name="speak", params={"viseme": 3}))
        assert result == {"type": "animation", "name": "speak", "params": {"viseme": 3}}

    def test_animation_cmd_default(self):
        result = serialize(AnimationCmd())
        assert result == {"type": "animation", "name": ""}

    def test_state_update(self):
        result = serialize(StateUpdate(connected=True))
        assert result == {"type": "state", "connected": True}

    def test_state_update_disconnected(self):
        result = serialize(StateUpdate(connected=False))
        assert result == {"type": "state", "connected": False}

    def test_speak_cmd(self):
        result = serialize(SpeakCmd(text="hello world"))
        assert result == {"type": "speak", "text": "hello world"}

    def test_speak_cmd_empty(self):
        result = serialize(SpeakCmd())
        assert result == {"type": "speak", "text": ""}

    def test_listen_indicator_active(self):
        result = serialize(ListenIndicator(active=True))
        assert result == {"type": "listen", "active": True}

    def test_listen_indicator_inactive(self):
        result = serialize(ListenIndicator(active=False))
        assert result == {"type": "listen", "active": False}

    def test_think_indicator(self):
        result = serialize(ThinkIndicator(active=True))
        assert result == {"type": "think", "active": True}


class TestParse:
    def test_go_event(self):
        msg = parse({"type": "event", "name": "animation_finished", "params": {"name": "greet"}})
        assert isinstance(msg, GoEvent)
        assert msg.name == "animation_finished"
        assert msg.params == {"name": "greet"}

    def test_go_event_no_params(self):
        msg = parse({"type": "event", "name": "ready"})
        assert isinstance(msg, GoEvent)
        assert msg.name == "ready"
        assert msg.params is None

    def test_go_command(self):
        msg = parse({"type": "command", "name": "ready"})
        assert isinstance(msg, GoCommand)
        assert msg.name == "ready"
        assert msg.params is None

    def test_go_command_with_params(self):
        msg = parse({"type": "command", "name": "start_listening", "params": {"device": "mic1"}})
        assert isinstance(msg, GoCommand)
        assert msg.name == "start_listening"
        assert msg.params == {"device": "mic1"}

    def test_unknown_type(self):
        msg = parse({"type": "unknown", "name": "test"})
        assert msg is None

    def test_empty_dict(self):
        msg = parse({})
        assert msg is None


class TestConstants:
    def test_outgoing_types(self):
        assert OUTGOING_TYPES == {"animation", "state", "listen", "think", "speak", "device"}

    def test_incoming_types(self):
        assert INCOMING_TYPES == {"event", "command"}
