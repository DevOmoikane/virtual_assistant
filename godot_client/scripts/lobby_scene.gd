extends Node3D

@onready var websocket_node: Node = $websocket
@onready var character_node: Node = $Carlitos
@onready var debug_overlay: CanvasLayer = $DebugOverlay


func _ready() -> void:
	websocket_node.backend_connected.connect(_on_ws_connected)
	websocket_node.backend_disconnected.connect(_on_ws_disconnected)
	websocket_node.bridge_connected.connect(_on_bridge_connected)
	websocket_node.bridge_disconnected.connect(_on_bridge_disconnected)
	websocket_node.execute_action.connect(_on_execute_action)
	websocket_node.speaking.connect(_on_speaking)
	websocket_node.spoken.connect(_on_spoken)
	websocket_node.listening.connect(_on_listening)
	websocket_node.thinking.connect(_on_thinking)
	websocket_node.heard.connect(_on_heard)

	debug_overlay.set_mode("idle")
	debug_overlay.set_connection(false, false)


func _on_ws_connected() -> void:
	debug_overlay.set_connection(true, websocket_node._bridge_connected)
	if character_node.has_method("set_connected"):
		character_node.set_connected()


func _on_ws_disconnected() -> void:
	debug_overlay.set_connection(false, websocket_node._bridge_connected)
	if character_node.has_method("set_disconnected"):
		character_node.set_disconnected()


func _on_bridge_connected() -> void:
	debug_overlay.set_connection(websocket_node._was_connected, true)


func _on_bridge_disconnected() -> void:
	debug_overlay.set_connection(websocket_node._was_connected, false)


func _on_execute_action(action: String) -> void:
	if character_node.has_method("execute_action"):
		character_node.execute_action(action)


func _on_speaking() -> void:
	if character_node.has_method("execute_action"):
		character_node.execute_action("speak")


func _on_spoken(text: String) -> void:
	debug_overlay.set_mode("speaking")
	debug_overlay.set_spoken(text)


func _on_listening(active: bool) -> void:
	debug_overlay.set_mode("listening" if active else "idle")
	if character_node.has_method("execute_action"):
		character_node.execute_action("listen" if active else "idle")


func _on_thinking(active: bool) -> void:
	debug_overlay.set_mode("thinking" if active else "idle")
	if character_node.has_method("execute_action"):
		character_node.execute_action("think" if active else "idle")


func _on_heard(text: String) -> void:
	debug_overlay.set_heard(text)
