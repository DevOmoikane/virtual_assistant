extends CanvasLayer

@onready var bg: ColorRect = $Background
@onready var lines: VBoxContainer = $Background/Margin/VBox

var _visible_override: bool = true


func _ready() -> void:
	visible = true
	#bg.modulate = Color(0, 0, 0, 0.7)


func _input(event: InputEvent) -> void:
	if event.is_action_pressed("ToggleDebug"):
		_visible_override = not _visible_override
		visible = _visible_override


func _set_line(idx: int, label: String, value: String) -> void:
	var c = lines.get_child(idx) as HBoxContainer
	if not c:
		return
	var lbl = c.get_child(0) as Label
	var val = c.get_child(1) as Label
	if lbl:
		lbl.text = label
	if val:
		val.text = value


func set_connection(backend: bool) -> void:
	_set_line(0, "CONN", "🟢" if backend else "🔴")


func set_mode(mode: String) -> void:
	_set_line(1, "MODE", mode)


func set_heard(text: String) -> void:
	_set_line(2, "HEARD", text)


func set_spoken(text: String) -> void:
	_set_line(3, "SPOKEN", text)
