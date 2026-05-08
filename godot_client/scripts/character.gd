extends Node
class_name Character

@onready var animation_player: AnimationPlayer = $UAL2_Standard/AnimationPlayer

# Called when the node enters the scene tree for the first time.
func _ready() -> void:
	animation_player.play("imported/idle_breathing")

# Called every frame. 'delta' is the elapsed time since the previous frame.
func _process(delta: float) -> void:
	pass

func execute_action(action: String) -> void:
	pass
