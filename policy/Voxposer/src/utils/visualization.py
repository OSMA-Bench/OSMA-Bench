import numpy as np
from utils.utils import get_logger

logger = get_logger(__name__)


def save_image(array, save_path="tmp.png"):
    from PIL import Image
    Image.fromarray(array).save(save_path)
    logger.debug(f"saved {save_path}")


def save_array(array, save_name="tmp.npy"):
    np.save(save_name, array)


def visualize_voxel(voxel_maps, voxel_size=0.1):
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    for voxel_map in voxel_maps:
        ax.voxels(voxel_map.astype(bool),
                  facecolors='cyan',
                  edgecolors='gray',
                  alpha=0.7)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Voxel Visualization')
    plt.tight_layout()
    plt.show()


def visualize_voxel_from_file(path):
    """Load a .npy voxel/pixel map and visualize it."""
    load_map = np.load(path)
    if load_map.ndim == 2:
        load_map = np.stack([load_map] * load_map.shape[0], axis=2)
    visualize_voxel([load_map])


def visualize_point_cloud(ply_path, remove_outliers=True, nb_neighbors=30, std_ratio=1.0):
    """Load a .ply point cloud and visualize with optional outlier removal."""
    import open3d as o3d
    import matplotlib.pyplot as plt
    pcd = o3d.io.read_point_cloud(ply_path)
    points = np.asarray(pcd.points)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='red', s=1, label='raw')
    if remove_outliers:
        pcd_clean, _ = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
        clean_points = np.asarray(pcd_clean.points)
        ax.scatter(clean_points[:, 0], clean_points[:, 1], clean_points[:, 2], c='blue', s=1, label='filtered')
    ax.legend()
    plt.tight_layout()
    plt.show()


def save_map_to_image(array, path, save_path="tmp.png"):
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(['white', 'black', 'red', 'blue', 'cyan'])
    display = array.copy()
    for (r, c) in path:
        display[int(r), int(c)] = 2
    display[int(path[0][0]), int(path[0][1])] = 3
    display[int(path[-1][0]), int(path[-1][1])] = 4
    plt.matshow(display, cmap=cmap)
    plt.colorbar()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.debug(f"Saved map image to {save_path}")


def save_video_images(controller_infos, keyword, save_path="tmp.mp4"):
    import cv2, subprocess, tempfile, os
    images = [controller_infos[k][keyword] for k in controller_infos]
    height, width, _ = images[0].shape
    # Write with mp4v first, then re-encode to H.264 for browser/VSCode playback
    tmp_path = save_path + ".tmp.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(tmp_path, fourcc, 24, (width, height))
    for img in images:
        # [::-1]-flipped obs views have negative strides and floats slip in —
        # cv2.VideoWriter silently writes garbage (black/striped frames) for
        # both. Normalise to contiguous uint8 RGB first.
        img = np.asarray(img)
        if img.dtype != np.uint8:
            _mx = float(img.max()) if img.size else 1.0
            img = (img * (255.0 if _mx <= 1.001 else 1.0)).clip(0, 255).astype(np.uint8)
        img = np.ascontiguousarray(img[..., :3])
        # mixed frame sizes corrupt VideoWriter output (striped/garbled frames
        # for any frame whose shape differs from the writer's WxH).
        if (img.shape[1], img.shape[0]) != (width, height):
            img = cv2.resize(img, (width, height))
        video.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    video.release()
    # Re-encode to H.264
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", tmp_path, "-c:v", "libx264", "-pix_fmt", "yuv420p",
             "-movflags", "+faststart", "-loglevel", "error", save_path],
            check=True)
        os.remove(tmp_path)
    except (subprocess.CalledProcessError, FileNotFoundError):
        # ffmpeg unavailable — fall back to mp4v file
        os.replace(tmp_path, save_path)
