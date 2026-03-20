import cv2
import numpy as np
import gradio as gr

# Global variables for storing source and target control points
points_src = []
points_dst = []
image = None

# Reset control points when a new image is uploaded
def upload_image(img):
    global image, points_src, points_dst
    points_src.clear()
    points_dst.clear()
    image = img
    return img

# Record clicked points and visualize them on the image
def record_points(evt: gr.SelectData):
    global points_src, points_dst, image
    x, y = evt.index[0], evt.index[1]

    # Alternate clicks between source and target points
    if len(points_src) == len(points_dst):
        points_src.append([x, y])
    else: 
        points_dst.append([x, y])

    # Draw points (blue: source, red: target) and arrows on the image
    marked_image = image.copy()
    for pt in points_src:
        cv2.circle(marked_image, tuple(pt), 1, (255, 0, 0), -1)  # Blue for source
    for pt in points_dst:
        cv2.circle(marked_image, tuple(pt), 1, (0, 0, 255), -1)  # Red for target

    # Draw arrows from source to target points
    for i in range(min(len(points_src), len(points_dst))):
        cv2.arrowedLine(marked_image, tuple(points_src[i]), tuple(points_dst[i]), (0, 255, 0), 1)

    return marked_image


# Point-guided image deformation
def point_guided_deformation(image, source_pts, target_pts, alpha=1.0, eps=1e-8):
    """
    Return
    ------
        A deformed image.
    """ 
    ### FILL: Implement MLS or RBF based image warping
    img = np.array(image)
    source_pts = np.array(source_pts)
    target_pts = np.array(target_pts)

    h, w = img.shape[:2]
    grid_x, grid_y = np.meshgrid(
        np.arange(w, dtype=np.float32),
        np.arange(h, dtype=np.float32)
    )
    v_dst = np.stack([grid_x, grid_y], axis=-1).reshape(-1, 2) 
    #反向映射 
    p = target_pts   
    q = source_pts 
     #计算权重   
    diff = p[None, :, :] - v_dst[:, None, :]   
    dist2 = np.sum(diff * diff, axis=2) + eps 
    w_i = 1.0 / (dist2 ** alpha)               
    #计算质心
    w_sum = np.sum(w_i, axis=1, keepdims=True)            
    p_star = (w_i @ p) / w_sum                            
    q_star = (w_i @ q) / w_sum                           
    #中心化
    phat = p[None, :, :] - p_star[:, None, :]   
    qhat = q[None, :, :] - q_star[:, None, :]   
    # 求逆矩阵
    a00 = np.sum(w_i * phat[:, :, 0] * phat[:, :, 0], axis=1)
    a01 = np.sum(w_i * phat[:, :, 0] * phat[:, :, 1], axis=1)
    a11 = np.sum(w_i * phat[:, :, 1] * phat[:, :, 1], axis=1)

    b00 = np.sum(w_i * phat[:, :, 0] * qhat[:, :, 0], axis=1)
    b01 = np.sum(w_i * phat[:, :, 0] * qhat[:, :, 1], axis=1)
    b10 = np.sum(w_i * phat[:, :, 1] * qhat[:, :, 0], axis=1)
    b11 = np.sum(w_i * phat[:, :, 1] * qhat[:, :, 1], axis=1)
    det = a00 * a11 - a01 * a01
    det = np.where(np.abs(det) < eps, 1.0, det)

    inv_a00 = a11 / det
    inv_a01 = -a01 / det
    inv_a11 = a00 / det

    m00 = inv_a00 * b00 + inv_a01 * b10
    m01 = inv_a00 * b01 + inv_a01 * b11
    m10 = inv_a01 * b00 + inv_a11 * b10
    m11 = inv_a01 * b01 + inv_a11 * b11
    v_offset = v_dst - p_star
    src_x = v_offset[:, 0] * m00 + v_offset[:, 1] * m10 + q_star[:, 0]
    src_y = v_offset[:, 0] * m01 + v_offset[:, 1] * m11 + q_star[:, 1]
    map_x = np.clip(src_x, 0, w - 1).reshape(h, w).astype(np.float32)
    map_y = np.clip(src_y, 0, h - 1).reshape(h, w).astype(np.float32)
    #插值
    warped_image = cv2.remap(
        img,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE
    )

    return warped_image

def run_warping():
    global points_src, points_dst, image

    warped_image = point_guided_deformation(image, np.array(points_src), np.array(points_dst))

    return warped_image

# Clear all selected points
def clear_points():
    global points_src, points_dst
    points_src.clear()
    points_dst.clear()
    return image

# Build Gradio interface
with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label="Upload Image", interactive=True, width=800)
            point_select = gr.Image(label="Click to Select Source and Target Points", interactive=True, width=800)

        with gr.Column():
            result_image = gr.Image(label="Warped Result", width=800)

    run_button = gr.Button("Run Warping")
    clear_button = gr.Button("Clear Points")

    input_image.upload(upload_image, input_image, point_select)
    point_select.select(record_points, None, point_select)
    run_button.click(run_warping, None, result_image)
    clear_button.click(clear_points, None, point_select)

demo.launch()
