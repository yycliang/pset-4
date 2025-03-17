def sphere_tracing(
    ray_origins,
    ray_directions,
    model,
    t_near=0.0,
    t_far=3.0,
    max_iter=256,
    epsilon=1e-4,
):
    """
    Perform sphere tracing to find the intersection of rays with the implicit model.

    Args:
        ray_origins: [H, W, 3] origin points for rays
        ray_directions: [H, W, 3] direction vectors for rays
        model: Implicit model to compute the SDF
        t_near: Near plane distance
        t_far: Far plane distance
        max_iter: Maximum number of iterations for sphere tracing
        epsilon: Distance threshold for stopping

    Returns:
        image: [H, W, 3] rendered image
    """
    device = ray_origins.device
    H, W, _ = ray_origins.shape

    # Initialize output image
    image = torch.zeros(H, W, 3, device=device)

    # Flatten rays for batched processing
    ray_origins = ray_origins.reshape(-1, 3)
    ray_directions = ray_directions.reshape(-1, 3)

    # Initialize t values for ray marching
    t = torch.full((H * W,), t_near, device=device)

    # Active mask (tracks which rays are still marching)
    active = torch.ones_like(t, dtype=torch.bool, device=device)

    for _ in range(max_iter):
        if not active.any():
            break  # Stop if all rays have finished

        # Compute current points along active rays
        points = ray_origins[active] + t[active].unsqueeze(-1) * ray_directions[active]

        # Query SDF and color from the model
        sdf, color = model(points)

        # Find which rays have hit the surface
        hit = sdf.squeeze(-1) < epsilon
        hit_indices = torch.nonzero(active)[hit]

        # Assign color to pixels where rays hit a surface
        image.view(-1, 3)[hit_indices] = color[hit]

        # Update active rays: deactivate those that hit or exceeded t_far
        active[hit_indices] = False
        active[t >= t_far] = False

        # Advance remaining active rays
        t[active] += sdf.squeeze(-1)[~hit]

    return image.view(H, W, 3)
