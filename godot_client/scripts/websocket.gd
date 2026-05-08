extends Node

@export var websocket_url: String = "ws://localhost:7700/api/ws"
@export var character: Character

var socket: WebSocketPeer = WebSocketPeer.new()
var should_reconnect: bool = true
var reconnect_timer: float = 0.0
var reconnect_delay: float = 3.0

signal connected()
signal disconnected()


func _ready():
	_connect_to_server()


func _connect_to_server():
	var err = socket.connect_to_url(websocket_url)
	if err == OK:
		print("Connecting to %s..." % websocket_url)
	else:
		push_error("WebSocket connection failed: ", err)
		_schedule_reconnect()


func _schedule_reconnect():
	if should_reconnect:
		reconnect_timer = reconnect_delay


func _process(delta):
	if reconnect_timer > 0:
		reconnect_timer -= delta
		if reconnect_timer <= 0:
			socket = WebSocketPeer.new()
			_connect_to_server()
		return

	socket.poll()
	var state = socket.get_ready_state()

	match state:
		WebSocketPeer.STATE_OPEN:
			while socket.get_available_packet_count():
				var packet = socket.get_packet()
				if socket.was_string_packet():
					var text = packet.get_string_from_utf8()
					_handle_message(text)

		WebSocketPeer.STATE_CLOSING:
			pass

		WebSocketPeer.STATE_CLOSED:
			var code = socket.get_close_code()
			print("WebSocket closed with code: %d" % code)
			emit_signal("disconnected")
			_schedule_reconnect()


func _on_connected():
	print("WebSocket connected, sending ready...")
	var msg = JSON.stringify({"type": "command", "name": "ready"})
	socket.send_text(msg)
	emit_signal("connected")


func _handle_message(raw: String):
	var json = JSON.new()
	var err = json.parse(raw)
	if err != OK:
		push_error("Invalid JSON from server: ", raw)
		return

	var data = json.data
	if typeof(data) != TYPE_DICTIONARY:
		return

	var msg_type = data.get("type", "")
	match msg_type:
		"animation":
			var anim_name = data.get("name", "idle")
			if character:
				character.execute_action(anim_name)
		"state":
			var connected_state = data.get("connected", false)
			if character:
				if connected_state:
					character.set_connected()
				else:
					character.set_disconnected()
		"speak":
			var text = data.get("text", "")
			print("Server says: ", text)
		"listen":
			var active = data.get("active", false)
			print("Listening: ", active)
		"think":
			var active = data.get("active", false)
			print("Thinking: ", active)
		_:
			print("Unknown message type: ", msg_type)


func _on_connection_succeeded(_proto = ""):
	_on_connected()


func _on_connection_error():
	push_error("WebSocket connection error")
	_schedule_reconnect()


func _on_data_received():
	pass


func disconnect_from_server():
	should_reconnect = false
	socket.close()
