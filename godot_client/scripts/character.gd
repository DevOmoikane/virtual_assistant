extends Node
class_name Character

@onready var animation_player: AnimationPlayer = $UAL2_Standard/AnimationPlayer
@onready var mesh: MeshInstance3D = $carlitos_low_poly2/root/GeneralSkeleton/carlitos

@export var disconnected_shader: ShaderMaterial

# Called when the node enters the scene tree for the first time.
func _ready() -> void:
	animation_player.play("imported/idle_breathing")

# Called every frame. 'delta' is the elapsed time since the previous frame.
func _process(delta: float) -> void:
	pass

func execute_action(action: String) -> void:
	pass

func set_disconnected() -> void:
	mesh.set_surface_override_material(0, disconnected_shader)
	
func set_connected() -> void:
	mesh.set_surface_override_material(0, null)

func toggle_connected() -> void:
	if mesh.get_surface_override_material(0) == null:
		set_disconnected()
	else:
		set_connected()
