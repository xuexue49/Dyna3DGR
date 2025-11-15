"""
Debug script to check CUDA renderer camera and Gaussian positions.
"""

import torch
import math

# Simulate typical values
H, W, D = 128, 128, 10
slice_idx = D // 2  # Middle slice = 5

# Camera setup (current implementation)
sx, sy, sz = 1.0, 1.0, 1.0
camera_position = torch.tensor([W * sx / 2, H * sy / 2, slice_idx + 10.0])
look_at = torch.tensor([W * sx / 2, H * sy / 2, float(slice_idx)])
up = torch.tensor([0.0, 1.0, 0.0])

fov_x = 2 * math.atan(W * sx / (2 * 10.0))
fov_y = 2 * math.atan(H * sy / (2 * 10.0))

print("=== Camera Setup ===")
print(f"Image size: H={H}, W={W}, D={D}")
print(f"Slice index: {slice_idx}")
print(f"Camera position: {camera_position}")
print(f"Look at: {look_at}")
print(f"Up: {up}")
print(f"FOV X: {fov_x:.4f} rad = {math.degrees(fov_x):.2f} deg")
print(f"FOV Y: {fov_y:.4f} rad = {math.degrees(fov_y):.2f} deg")

# Typical Gaussian positions (initialized in a grid)
# Assuming Gaussians are in range [0, W] x [0, H] x [0, D]
print("\n=== Typical Gaussian Positions ===")
print(f"Expected range: X=[0, {W}], Y=[0, {H}], Z=[0, {D}]")

# Check if Gaussians at slice_z would be visible
print(f"\n=== Visibility Check ===")
print(f"Camera is at Z={camera_position[2]:.1f}, looking at Z={look_at[2]:.1f}")
print(f"Gaussians at slice Z={slice_idx} should be visible")

# Check projection
print(f"\n=== Projection Check ===")
distance = camera_position[2] - slice_idx  # 10.0
print(f"Camera distance from slice: {distance}")
print(f"Horizontal view width at slice: {2 * distance * math.tan(fov_x / 2):.2f}")
print(f"Vertical view height at slice: {2 * distance * math.tan(fov_y / 2):.2f}")
print(f"Should cover: W={W}, H={H}")

# Problem diagnosis
print("\n=== PROBLEM DIAGNOSIS ===")
print("Issue: rendered_images all zeros")
print("\nPossible causes:")
print("1. Gaussians not initialized in correct coordinate range")
print("2. Camera looking at wrong direction")
print("3. Projection matrix incorrect")
print("4. Gaussians too small or transparent")
print("\nRecommended fixes:")
print("1. Check Gaussian initialization range matches [0, W] x [0, H] x [0, D]")
print("2. Verify deformed_xyz is in correct coordinate system")
print("3. Add debug prints for Gaussian positions and scales")
print("4. Try simpler camera setup with identity transforms")
