use crate::components::camera::CameraSettings;
use bevy::prelude::*;
use bevy_input::mouse::AccumulatedMouseMotion;

pub fn orbit(
    mut camera: Single<&mut Transform, With<Camera>>,
    camera_settings: Res<CameraSettings>,
    mouse_buttons: Res<ButtonInput<MouseButton>>,
    mouse_motion: Res<AccumulatedMouseMotion>,
    time: Res<Time>,
) {
    let delta = mouse_motion.delta;
    let mut delta_roll = 0.0;
    if mouse_buttons.pressed(MouseButton::Left) {
        let delta_pitch = delta.y * camera_settings.pitch_speed;
        let delta_yaw = delta.x * camera_settings.yaw_speed;
        delta_roll *= camera_settings.roll_speed * time.delta_secs();
        let (yaw, pitch, roll) = camera.rotation.to_euler(EulerRot::YXZ);
        let pitch = (pitch + delta_pitch).clamp(
            camera_settings.pitch_range.start,
            camera_settings.pitch_range.end,
        );
        let roll = roll + delta_roll;
        let yaw = yaw + delta_yaw;
        camera.rotation = Quat::from_euler(EulerRot::YXZ, yaw, pitch, roll);

        let target = Vec3::ZERO;
        camera.translation = target - camera.forward() * camera_settings.orbit_distance;
    }
}
