extends Node

@export var websocket_url: String = "ws://localhost:7700/api/ws"

var socket: WebSocketPeer = WebSocketPeer.new()
var should_reconnect: bool = true
var reconnect_timer: float = 0.0
var reconnect_delay: float = 3.0

var _connected_sent: bool = false
var _was_connected: bool = false

signal connected()
signal disconnected()
signal execute_action(action: String)
signal speaking()
signal listening()
signal thinking()


func _ready():
	_connect_to_server()


func _connect_to_server():
	_connected_sent = false
	var err = socket.connect_to_url(websocket_url)
	if err == OK:
		print("Connecting to %s..." % websocket_url)
	else:
		push_error("WebSocket connection failed: %d" % err)
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
		WebSocketPeer.STATE_CONNECTING:
			pass

		WebSocketPeer.STATE_OPEN:
			if not _connected_sent:
				_connected_sent = true
				_was_connected = true
				print("WebSocket connected, sending ready...")
				var msg = JSON.stringify({"type": "command", "name": "ready"})
				socket.send_text(msg)
				emit_signal("connected")

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
			if _was_connected:
				_was_connected = false
				emit_signal("disconnected")
			_schedule_reconnect()


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
			emit_signal("execute_action", anim_name)
		"state":
			var s = data.get("connected", false)
			if s:
				pass
			else:
				pass
		"speak":
			var text = data.get("text", "")
			print("Server says: ", text)
			emit_signal("speaking")
		"listen":
			var active = data.get("active", false)
			print("Mic listening: ", active)
			emit_signal("listening")
		"think":
			var active = data.get("active", false)
			print("Thinking: ", active)
			emit_signal("thinking")
		_:
			print("Unknown message type: ", msg_type)


func disconnect_from_server():
	should_reconnect = false
	socket.close()
