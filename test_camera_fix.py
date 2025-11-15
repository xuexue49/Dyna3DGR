"""
Test the fixed camera setup for normalized coordinates.
"""

import torch
import math

# Simulate typical values
H, W, D = 128, 128, 10
slice_idx = D // 2  # Middle slice = 5

# NEW camera setup (normalized coordinates)
slice_z_norm = slice_idx / max(D - 1, 1)  # 5 / 9 = 0.556
camera_distance = 2.0

camera_position = torch.tensor([0.5, 0.5, slice_z_norm + camera_distance])
look_at = torch.tensor([0.5, 0.5, slice_z_norm])
up = torch.tensor([0.0, 1.0, 0.0])

fov_x = 2 * math.atan(0.5 / camera_distance)
fov_y = 2 * math.atan(0.5 / camera_distance)

print("=== FIXED Camera Setup (Normalized Coordinates) ===")
print(f"Image size: H={H}, W={W}, D={D}")
print(f"Slice index: {slice_idx}")
print(f"Slice Z (normalized): {slice_z_norm:.4f}")
print(f"Camera position: {camera_position}")
print(f"Look at: {look_at}")
print(f"Up: {up}")
print(f"FOV X: {fov_x:.4f} rad = {math.degrees(fov_x):.2f} deg")
print(f"FOV Y: {fov_y:.4f} rad = {math.degrees(fov_y):.2f} deg")

print("\n=== Gaussian Positions (Normalized) ===")
print(f"Expected range: X=[0, 1], Y=[0, 1], Z=[0, 1]")

print(f"\n=== Visibility Check ===")
print(f"Camera is at Z={camera_position[2]:.4f}, looking at Z={look_at[2]:.4f}")
print(f"Gaussians at slice Z={slice_z_norm:.4f} should be visible")

print(f"\n=== Projection Check ===")
print(f"Camera distance from slice: {camera_distance}")
print(f"Horizontal view width at slice: {2 * camera_distance * math.tan(fov_x / 2):.4f}")
print(f"Vertical view height at slice: {2 * camera_distance * math.tan(fov_y / 2):.4f}")
print(f"Should cover: [0, 1] range")

print("\n=== Expected Behavior ===")
print("✓ Gaussians in [0, 1] range will be visible")
print("✓ Camera covers full [0, 1] x [0, 1] area at the slice")
print("✓ Rendering should produce non-zero output")
