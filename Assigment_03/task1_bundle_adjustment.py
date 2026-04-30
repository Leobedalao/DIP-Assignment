from __future__ import annotations
import argparse
import math
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt


IMAGE_SIZE = 1024
CX = IMAGE_SIZE / 2.0
CY = IMAGE_SIZE / 2.0


def euler_angles_to_matrix_xyz(euler_angles: torch.Tensor) -> torch.Tensor:
    """把 XYZ 顺序的欧拉角转换为旋转矩阵。
    配置pytorch3D时无法安装，使用纯 PyTorch 实现，不依赖 PyTorch3D。
    """
    x_angle, y_angle, z_angle = torch.unbind(euler_angles, dim=-1)

    cos_x, sin_x = torch.cos(x_angle), torch.sin(x_angle)
    cos_y, sin_y = torch.cos(y_angle), torch.sin(y_angle)
    cos_z, sin_z = torch.cos(z_angle), torch.sin(z_angle)

    zeros = torch.zeros_like(x_angle)
    ones = torch.ones_like(x_angle)
    rx = torch.stack(
        (
            torch.stack((ones, zeros, zeros), dim=-1),
            torch.stack((zeros, cos_x, -sin_x), dim=-1),
            torch.stack((zeros, sin_x, cos_x), dim=-1),
        ),
        dim=-2,
    )
    ry = torch.stack(
        (
            torch.stack((cos_y, zeros, sin_y), dim=-1),
            torch.stack((zeros, ones, zeros), dim=-1),
            torch.stack((-sin_y, zeros, cos_y), dim=-1),
        ),
        dim=-2,
    )
    rz = torch.stack(
        (
            torch.stack((cos_z, -sin_z, zeros), dim=-1),
            torch.stack((sin_z, cos_z, zeros), dim=-1),
            torch.stack((zeros, zeros, ones), dim=-1),
        ),
        dim=-2,
    )
    return rx @ ry @ rz


def load_observations(
    data_dir: Path,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    """
    加载 2D 观测数据，转换成扁平数组。
    """

    points2d = np.load(data_dir / "points2d.npz")
    view_keys = sorted(points2d.keys())

    first_view = points2d[view_keys[0]]
    num_points = first_view.shape[0]


    view_indices: list[np.ndarray] = []
    point_indices: list[np.ndarray] = []
    xy_observed: list[np.ndarray] = []

    for view_id, key in enumerate(view_keys):
        obs = points2d[key][:num_points]
        visible = obs[:, 2] > 0.5
        visible_point_ids = np.nonzero(visible)[0]

        view_indices.append(np.full(len(visible_point_ids), view_id, dtype=np.int64))
        point_indices.append(visible_point_ids.astype(np.int64))
        xy_observed.append(obs[visible, :2].astype(np.float32))

    view_idx = torch.from_numpy(np.concatenate(view_indices)).to(device)
    point_idx = torch.from_numpy(np.concatenate(point_indices)).to(device)
    xy = torch.from_numpy(np.concatenate(xy_observed)).to(device)

    return view_idx, point_idx, xy, len(view_keys), num_points

def initialize_camera_parameters(
    num_views: int,
    radius: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """初始化相机外参
    """
    euler_angles = torch.zeros((num_views, 3), device=device)

    translations = torch.zeros((num_views, 3), device=device)
    translations[:, 2] = -radius

    return euler_angles, translations


def initialize_points3d(num_points: int, device: torch.device) -> torch.Tensor:
    """初始化待优化的 3D 点。初始化点云位于原点附近，尺度约为 1。
    """
    points = torch.empty((num_points, 3), device=device)
    points.uniform_(-0.8, 0.8)
    points[:, 2].mul_(0.5)
    return points


def project_points(
    points3d: torch.Tensor,
    euler_angles: torch.Tensor,
    translations: torch.Tensor,
    log_focal: torch.Tensor,
    view_idx: torch.Tensor,
    point_idx: torch.Tensor,
) -> torch.Tensor:
    """把 3D 点投影到对应 view 的 2D 像素坐标。

    对每条观测，先取出对应相机外参和对应 3D 点：
        [Xc, Yc, Zc] = R @ [X, Y, Z]^T + T

    然后使用的投影公式：
        u = -f * Xc / Zc + cx
        v =  f * Yc / Zc + cy
    """
    rotations = euler_angles_to_matrix_xyz(euler_angles)
    r = rotations[view_idx]
    t = translations[view_idx]
    p = points3d[point_idx]

    camera_points = torch.bmm(r, p.unsqueeze(-1)).squeeze(-1) + t
    x_c, y_c, z_c = camera_points.unbind(dim=-1)

    focal = torch.exp(log_focal)
    u = -focal * x_c / z_c + CX
    v = focal * y_c / z_c + CY
    return torch.stack((u, v), dim=-1)


def reprojection_loss(pred_xy: torch.Tensor, observed_xy: torch.Tensor) -> torch.Tensor:
    """计算重投影误差
    """
    return torch.nn.functional.smooth_l1_loss(pred_xy, observed_xy, beta=2.0)


def export_obj(points3d: torch.Tensor, colors_path: Path, output_path: Path) -> None:
    """导出带颜色的 OBJ 点云。
    """
    points_np = points3d.detach().cpu().numpy()
    colors = load_point_colors(colors_path, len(points_np))

    with output_path.open("w", encoding="utf-8") as f:
        for point, color in zip(points_np, colors):
            f.write(
                "v "
                f"{point[0]:.8f} {point[1]:.8f} {point[2]:.8f} "
                f"{color[0]:.6f} {color[1]:.6f} {color[2]:.6f}\n"
            )


def load_point_colors(colors_path: Path, num_points: int) -> np.ndarray:
    """读取点云颜色，并统一转换到 [0, 1] 范围。"""
    colors = np.load(colors_path).astype(np.float32)[:num_points]
    if colors.max() > 1.0:
        colors = colors / 255.0
    return np.clip(colors, 0.0, 1.0)


def plot_loss_curve(loss_history: list[float], output_path: Path) -> None:
    """保存优化过程中的 loss 曲线。
    """
    steps = np.arange(1, len(loss_history) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(steps, loss_history, linewidth=1.5)
    plt.xlabel("Iteration")
    plt.ylabel("Reprojection Loss")
    plt.title("Bundle Adjustment Loss Curve")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_point_cloud(
    points3d: torch.Tensor,
    colors_path: Path,
    output_path: Path,
    max_points: int,
) -> None:
    """保存最终 3D 点云的预览图。
    """
    points_np = points3d.detach().cpu().numpy()
    colors = load_point_colors(colors_path, len(points_np))

    if len(points_np) > max_points:
        sample_ids = np.linspace(0, len(points_np) - 1, max_points, dtype=np.int64)
        points_np = points_np[sample_ids]
        colors = colors[sample_ids]

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        points_np[:, 0],
        points_np[:, 1],
        points_np[:, 2],
        c=colors,
        s=1,
        depthshade=False,
    )
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Reconstructed 3D Point Cloud")

    center = points_np.mean(axis=0)
    radius = np.max(np.ptp(points_np, axis=0)) / 2.0
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.view_init(elev=20, azim=-60)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task 1: Bundle Adjustment")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="数据目录")
    parser.add_argument("--output", type=Path, default=Path("task1_result.obj"), help="输出 OBJ 路径")
    parser.add_argument("--loss-plot", type=Path, default=Path("task1_loss_curve.png"), help="loss 曲线图保存路径")
    parser.add_argument("--point-cloud-plot", type=Path, default=Path("task1_point_cloud.png"), help="3D 点云预览图保存路径")
    parser.add_argument("--plot-max-points", type=int, default=20000, help="绘制 3D 点云预览图时最多显示的点数")
    parser.add_argument("--iters", type=int, default=2000, help="优化迭代次数")
    parser.add_argument("--batch-size", type=int, default=50000, help="每步采样的可见观测数量；设为 0 使用全部观测")
    parser.add_argument("--lr-points", type=float, default=2e-2, help="3D 点学习率")
    parser.add_argument("--lr-camera", type=float, default=5e-3, help="相机外参学习率")
    parser.add_argument("--lr-focal", type=float, default=1e-3, help="焦距学习率")
    parser.add_argument("--fov-deg", type=float, default=60.0, help="焦距初始化使用的视场角")
    parser.add_argument("--camera-radius", type=float, default=2.5, help="相机绕物体初始化的半径")
    parser.add_argument("--log-every", type=int, default=50, help="每隔多少步打印一次 loss")
    parser.add_argument("--seed", type=int, default=0, help="随机种子")
    parser.add_argument( "--device",default="cuda:2")
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device)

    view_idx, point_idx, observed_xy, num_views, num_points = load_observations(
        args.data_dir,
        device,
    )

    fov_rad = math.radians(args.fov_deg)
    focal_init = IMAGE_SIZE / (2.0 * math.tan(fov_rad / 2.0))
    log_focal = torch.nn.Parameter(torch.tensor(math.log(focal_init), device=device))

    euler_init, translation_init = initialize_camera_parameters(num_views, args.camera_radius, device)
    euler_angles = torch.nn.Parameter(euler_init)
    translations = torch.nn.Parameter(translation_init)
    points3d = torch.nn.Parameter(initialize_points3d(num_points, device))

    optimizer = torch.optim.Adam(
        [
            {"params": [points3d], "lr": args.lr_points},
            {"params": [euler_angles, translations], "lr": args.lr_camera},
            {"params": [log_focal], "lr": args.lr_focal},
        ]
    )

    total_observations = len(observed_xy)
    print(f"device={device}")
    print(f"views={num_views}, points={num_points}, visible_observations={total_observations}")
    print(f"initial_focal={focal_init:.3f}")

    loss_history: list[float] = []
    for step in range(1, args.iters + 1):
        if args.batch_size and args.batch_size > 0 and args.batch_size < total_observations:
            batch_ids = torch.randint(total_observations, (args.batch_size,), device=device)
        else:
            batch_ids = torch.arange(total_observations, device=device)

        pred_xy = project_points(
            points3d,
            euler_angles,
            translations,
            log_focal,
            view_idx[batch_ids],
            point_idx[batch_ids],
        )
        loss = reprojection_loss(pred_xy, observed_xy[batch_ids])

        loss_history.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step == 1 or step % args.log_every == 0 or step == args.iters:
            focal_value = torch.exp(log_focal).item()
            print(f"iter {step:05d} | loss={loss.item():.6f} | focal={focal_value:.3f}")

    export_obj(points3d, args.data_dir / "points3d_colors.npy", args.output)
    print(f"已导出点云: {args.output}")

    plot_loss_curve(loss_history, args.loss_plot)
    print(f"已保存 loss 曲线: {args.loss_plot}")

    plot_point_cloud(
        points3d,
        args.data_dir / "points3d_colors.npy",
        args.point_cloud_plot,
        args.plot_max_points,
    )
    print(f"已保存 3D 点云预览图: {args.point_cloud_plot}")


if __name__ == "__main__":
    main()
