extends Node3D

@onready var websocket_node: Node = $websocket
@onready var character_node: Node = $Carlitos


func _ready() -> void:
	websocket_node.connected.connect(_on_ws_connected)
	websocket_node.disconnected.connect(_on_ws_disconnected)
	websocket_node.execute_action.connect(_on_execute_action)
	websocket_node.speaking.connect(_on_speaking)
	websocket_node.listening.connect(_on_listening)
	websocket_node.thinking.connect(_on_thinking)


func _on_ws_connected() -> void:
	if character_node.has_method("set_connected"):
		character_node.set_connected()


func _on_ws_disconnected() -> void:
	if character_node.has_method("set_disconnected"):
		character_node.set_disconnected()


func _on_execute_action(action: String) -> void:
	if character_node.has_method("execute_action"):
		character_node.execute_action(action)


func _on_speaking() -> void:
	if character_node.has_method("execute_action"):
		character_node.execute_action("speak")


func _on_listening() -> void:
	if character_node.has_method("execute_action"):
		character_node.execute_action("listen")


func _on_thinking() -> void:
	if character_node.has_method("execute_action"):
		character_node.execute_action("think")
