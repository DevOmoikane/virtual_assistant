shader_type spatial;
render_mode unshaded, depth_draw_never;

group_uniforms Highlight;
uniform vec4 shine_color : source_color = vec4(1.0);

uniform float shine_speed : hint_range(0.0, 25.0) = 1.0;
uniform float shine_width : hint_range(0.0, 5.0) = 0.5;
uniform float cycle_interval : hint_range(0.0, 10.0) = 1.0;

uniform float max_radius : hint_range(0.1, 10.0) = 2.0;

uniform bool inward = false;

uniform float angle_fade : hint_range(0.0, 1.0) = 1.0;

float sawtooth(float x) {
	return 2.0 * (
		(x + shine_speed * TIME)
		/ (shine_width + cycle_interval)
		- floor(
			0.5 + (x + shine_speed * TIME)
			/ (shine_width + cycle_interval)
		)
	);
}

void fragment() {

	// Local object space position
	vec3 local_pos = VERTEX;

	// Circular distance from center
	float dist = length(local_pos.xy);

	// Reverse direction if needed
	float projection = inward
		? (max_radius - dist)
		: dist;

	float frequency =
		ceil(
			sawtooth(projection)
			- sawtooth(projection + shine_width)
		);

	ALBEDO = shine_color.rgb;

	ALPHA = clamp(
		(1.0 - dot(NORMAL, VIEW) * angle_fade)
		* frequency
		* shine_color.a,
		0.0,
		1.0
	);
}