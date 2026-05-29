from dataclasses import dataclass

from pipecat.frames.frames import DataFrame


@dataclass
class PersonAppearedFrame(DataFrame):
    person_name: str | None = None


@dataclass
class PersonDisappearedFrame(DataFrame):
    pass


@dataclass
class GestureFrame(DataFrame):
    gesture: str
    x: float = 0.0
    y: float = 0.0


@dataclass
class TelegramMessageFrame(DataFrame):
    sender: str
    text: str
    chat_id: int
