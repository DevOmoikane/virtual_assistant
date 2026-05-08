extends Node
class_name Character

@onready var animation_player: AnimationPlayer = $UAL2_Standard/AnimationPlayer
@onready var mesh: MeshInstance3D = $carlitos_low_poly2/root/GeneralSkeleton/carlitos

@export var disconnected_shader: ShaderMaterial

var _current_animation: String = "imported/idle_breathing"


func _ready() -> void:
	animation_player.play(_current_animation)


func execute_action(action: String) -> void:
	match action:
		"idle":
			_play("imported/idle_breathing")
		"greet":
			_play("Yes")
		"listen":
			_play("Idle_FoldArms")
		"think":
			_play("Idle_FoldArms")
		"nod":
			_play("Yes")
		"shake":
			_play("Idle_No")
		"surprised":
			_play("Chest_Open")
		"speak":
			_play("Idle_TalkingPhone")
		_:
			_play("imported/idle_breathing")


func _play(name: String) -> void:
	if animation_player.has_animation(name):
		_current_animation = name
		animation_player.play(name)
	else:
		print("Animation not found: ", name)


func set_disconnected() -> void:
	mesh.set_surface_override_material(0, disconnected_shader)


func set_connected() -> void:
	mesh.set_surface_override_material(0, null)


func toggle_connected() -> void:
	if mesh.get_surface_override_material(0) == null:
		set_disconnected()
	else:
		set_connected()
