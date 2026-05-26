extends Node

@export var websocket_url: String = "ws://10.73.19.131:7700/api/ws"
@export var bridge_url: String = "ws://localhost:7802"

var socket: WebSocketPeer = WebSocketPeer.new()
var bridge_socket: WebSocketPeer = WebSocketPeer.new()
var should_reconnect: bool = true
var bridge_should_reconnect: bool = true
var reconnect_timer: float = 0.0
var bridge_reconnect_timer: float = 0.0
var reconnect_delay: float = 3.0

var _connected_sent: bool = false
var _was_connected: bool = false
var _bridge_connected: bool = false

signal connected()
signal disconnected()
signal execute_action(action: String)
signal speaking()
signal listening()
signal thinking()


func _ready():
	_connect_to_server()
	_connect_to_bridge()


func _connect_to_server():
	_connected_sent = false
	var err = socket.connect_to_url(websocket_url)
	if err == OK:
		print("Connecting to %s..." % websocket_url)
	else:
		push_error("WebSocket connection failed: %d" % err)
		_schedule_reconnect()


func _connect_to_bridge():
	var err = bridge_socket.connect_to_url(bridge_url)
	if err == OK:
		print("Connecting to bridge at %s..." % bridge_url)
	else:
		push_error("Bridge connection failed: %d" % err)
		_schedule_bridge_reconnect()


func _schedule_reconnect():
	if should_reconnect:
		reconnect_timer = reconnect_delay


func _schedule_bridge_reconnect():
	if bridge_should_reconnect:
		bridge_reconnect_timer = reconnect_delay


func _poll_socket(ws: WebSocketPeer, was_connected: bool, connected_sent: bool, is_bridge: bool) -> Dictionary:
	var result = {
		"was_connected": was_connected,
		"connected_sent": connected_sent,
	}

	ws.poll()
	var state = ws.get_ready_state()

	match state:
		WebSocketPeer.STATE_CONNECTING:
			pass

		WebSocketPeer.STATE_OPEN:
			if not connected_sent:
				connected_sent = true
				was_connected = true
				if not is_bridge:
					print("Backend connected, sending ready...")
					var msg = JSON.stringify({"type": "command", "name": "ready"})
					ws.send_text(msg)
					emit_signal("connected")

			while ws.get_available_packet_count():
				var packet = ws.get_packet()
				if ws.was_string_packet():
					var text = packet.get_string_from_utf8()
					_handle_message(text)

		WebSocketPeer.STATE_CLOSED:
			if was_connected:
				was_connected = false
				if not is_bridge:
					emit_signal("disconnected")
				_schedule_reconnect() if not is_bridge else _schedule_bridge_reconnect()

	result["was_connected"] = was_connected
	result["connected_sent"] = connected_sent
	return result


func _process(delta):
	if reconnect_timer > 0:
		reconnect_timer -= delta
		if reconnect_timer <= 0:
			socket = WebSocketPeer.new()
			_connect_to_server()
		return

	if bridge_reconnect_timer > 0:
		bridge_reconnect_timer -= delta
		if bridge_reconnect_timer <= 0:
			bridge_socket = WebSocketPeer.new()
			_connect_to_bridge()
		# Don't return — keep polling backend even if bridge is reconnecting

	var r = _poll_socket(socket, _was_connected, _connected_sent, false)
	_was_connected = r["was_connected"]
	_connected_sent = r["connected_sent"]

	r = _poll_socket(bridge_socket, _bridge_connected, true, true)
	_bridge_connected = r["was_connected"]


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
	bridge_should_reconnect = false
	socket.close()
	bridge_socket.close()


func send_to_backend(data: Dictionary) -> void:
	if socket.get_ready_state() == WebSocketPeer.STATE_OPEN:
		socket.send_text(JSON.stringify(data))
