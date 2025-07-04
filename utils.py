import gc
import cv2
import math
import base64
import numpy as np
from collections import defaultdict

def read_img(img_path):
  return cv2.cvtColor(cv2.imread(img_path),cv2.COLOR_BGR2RGB)

def draw_masks_fromList(
    image, chosen_index, contours, origin_size_mask,
    labels, colors, alpha = 0.4,
    contour_color = (0,0,0), contour_line_weight = 3):
  masked_image = image.copy()
  contour_list = []
  for i, mask_index in enumerate(chosen_index):
    contour = contours[mask_index]
    contour_list.append(contour)
    mask = cv2.drawContours(np.zeros(origin_size_mask), [contour], -1, (255), -1)
    # mask[offset_masks[i][0]:offset_masks[i][1],...] = masks_generated[i]

    if mask.shape[0]!= image.shape[0] or mask.shape[1]!= image.shape[1]:
      mask = cv2.resize(mask, (image.shape[1], image.shape[0]))

    # bbox, max_contour = mask_to_bbox(mask, return_contour=True)
    # contour_list.append(max_contour)

    masked_image = np.where(np.repeat(mask[:, :, np.newaxis], 3, axis=2),
                            np.asarray(colors[int(labels[i][-1])], dtype='uint8'),
                            masked_image)

    masked_image = masked_image.astype(np.uint8)
    del mask

  gc.collect()

  image = cv2.addWeighted(image, alpha, masked_image, (1-alpha), 0)
  image = cv2.drawContours(image, contour_list, -1, contour_color, contour_line_weight)

  return image

def open_image(path: str):
    with open(path, "rb") as p:
        file = p.read()
        return f"data:image/png;base64,{base64.b64encode(file).decode()}"
    
def numpy_to_base64_cv2(image_array):
    _, buffer = cv2.imencode('.jpg', image_array)  # Use .png if needed
    base64_str = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/png;base64,{base64_str}"

def convert_bbox(x, y, w, h, w1, h1, *args):
    """
    Convert a bounding box from one image shape to another.

    Parameters:
    - x1, y1, x2, y2: Coordinates of the bounding box in the original image.
    - w1, h1: Width and height of the original image.
    - w2, h2: Width and height of the new image.

    Returns:
    - new_x1, new_y1, new_x2, new_y2: Scaled coordinates of the bounding box in the new image.
    """
    # # Calculate scaling factors
    x_center = float(x)
    y_center = float(y)
    bbox_width = float(w)
    bbox_height = float(h)

    # Convert to pixel coordinates
    new_x1 = int((x_center - bbox_width / 2) * w1)
    new_y1 = int((y_center - bbox_height / 2) * h1)
    new_x2 = int((x_center + bbox_width / 2) * w1)
    new_y2 = int((y_center + bbox_height / 2) * h1)

    return new_x1, new_y1, new_x2, new_y2

def rgb_to_hex(r, g, b):
    return '#{:02x}{:02x}{:02x}'.format(r, g, b)

def get_color_distance(color1, color2):
    """Calculate the perceptual distance between two colors in RGB space"""
    r1, g1, b1 = color1
    r2, g2, b2 = color2
    return np.sqrt((r1-r2)**2 + (g1-g2)**2 + (b1-b2)**2)

def generate_distinct_colors(n_colors, min_distance=100):
    """
    Generate n visually distinct colors with minimum distance between them
    Returns a list of BGR colors
    """
    colors = []
    attempts = 0
    max_attempts = 1000
    
    while len(colors) < n_colors and attempts < max_attempts:
        # Try different color generation strategies
        if attempts % 3 == 0:
            # Strategy 1: Use HSV with different saturation and value
            hue = np.random.randint(0, 180)
            sat = np.random.randint(150, 255)
            val = np.random.randint(150, 255)
            hsv_color = np.uint8([[[hue, sat, val]]])
            new_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0][0]
        elif attempts % 3 == 1:
            # Strategy 2: Use RGB with high values
            new_color = np.random.randint(100, 255, 3)
        else:
            # Strategy 3: Use complementary colors
            hue = np.random.randint(0, 180)
            hsv_color = np.uint8([[[hue, 255, 255]]])
            new_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0][0]
        
        # Check if the new color is distinct enough from existing colors
        if not colors or all(get_color_distance(new_color, c) > min_distance for c in colors):
            colors.append(tuple(new_color.tolist()))
        
        attempts += 1
    
    # If we couldn't generate enough distinct colors, use a predefined set
    if len(colors) < n_colors:
        predefined_colors = [
            (0, 0, 255),      # Red
            (0, 255, 0),      # Green
            (255, 0, 0),      # Blue
            (0, 255, 255),    # Yellow
            (255, 0, 255),    # Magenta
            (255, 255, 0),    # Cyan
            (0, 128, 255),    # Orange
            (128, 0, 255),    # Purple
            (0, 255, 128),    # Lime
            (255, 128, 0),    # Sky Blue
        ]
        colors.extend(predefined_colors[:n_colors - len(colors)])
    
    return colors

def get_class_color_mapping(class_names):
    """
    Generate a mapping of class names to distinct colors
    Returns a dictionary where keys are class names and values are BGR colors
    """
    # Get unique classes
    unique_classes = list(set(class_names))
    
    # Generate distinct colors for each unique class
    colors = generate_distinct_colors(len(unique_classes),min_distance=150)
    
    # Create mapping dictionary
    color_mapping = dict(zip(unique_classes, colors))
    
    return color_mapping

def xywhrn2xywhr(bbox, width, height):
    cx,cy,w,h,angle = bbox

    cx = cx*width
    w = w*width

    cy = cy*height
    h = h*height
    return cx, cy, w, h, angle

def xywhrn2xyxyxyxy(bbox, width, height, origin_width, origin_height):
    cx,cy,w,h,angle = xywhrn2xywhr(bbox, origin_width, origin_height)

    # Calculate the four corner points
    # Half dimensions
    w_half, h_half = w / 2, h / 2
    
    # Rotation matrix
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    
    # Corner points in local coordinate system (relative to center)
    corners_local = np.array([
        [-w_half, -h_half],  # bottom-left
        [w_half, -h_half],   # bottom-right
        [w_half, h_half],    # top-right
        [-w_half, h_half]    # top-left
    ])
    
    # Apply rotation
    rotation_matrix = np.array([
        [cos_a, -sin_a],
        [sin_a, cos_a]
    ])
    
    corners_rotated = corners_local @ rotation_matrix.T
    
    # Translate to center position
    corners = corners_rotated + np.array([cx, cy])
    
    # Convert to integer coordinates for cv2
    # corners = corners.astype(np.int32)
    corners[:,0] = corners[:,0]*width/origin_width
    corners[:,1] = corners[:,1]*height/origin_height
    corners = corners.astype(np.int32)
    
    return corners

def rotate_image(image, angle_rad):
    """
    Rotates an image by a given angle, expanding the image size to fit.
    This is useful for visualizing the cropped patch with its original orientation.

    Args:
        image (np.ndarray): The image to rotate (e.g., a cropped patch).
        angle_rad (float): The angle of rotation in radians.
        background_color (tuple): The color for the new background areas.

    Returns:
        np.ndarray: The rotated image.
    """
    if image is None or image.size == 0:
        return np.array([])

    angle_deg = math.degrees(angle_rad)
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # Get the rotation matrix, then grab the sine and cosine
    # We use a negative angle because cv2 rotates counter-clockwise
    M = cv2.getRotationMatrix2D((cX, cY), -angle_deg, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # Compute the new bounding dimensions of the image to fit the entire rotation
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # Adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # Perform the actual rotation and return the image
    rotated_image = cv2.warpAffine(image, M, (nW, nH))
    return rotated_image

def crop_obb(image, bbox_xywhr_abs, target_size=(420,420)):
    """
    Crops a single oriented bounding box from an image.

    Args:
        image (np.ndarray): The input image.
        bbox_xywhr_abs (tuple): A single bounding box in absolute
                                (x_center, y_center, w, h, angle_rad) format.

    Returns:
        np.ndarray: The cropped (but not resized) image patch.
    """
    x_center, y_center, w, h, angle_rad = bbox_xywhr_abs

    # Calculate the 4 corner points of the rotated bounding box
    v_w = np.array([math.cos(angle_rad), math.sin(angle_rad)]) * (w / 2)
    v_h = np.array([-math.sin(angle_rad), math.cos(angle_rad)]) * (h / 2)

    p1 = np.array([x_center, y_center]) - v_w - v_h  # top-left
    p2 = np.array([x_center, y_center]) + v_w - v_h  # top-right
    p3 = np.array([x_center, y_center]) + v_w + v_h  # bottom-right
    p4 = np.array([x_center, y_center]) - v_w + v_h  # bottom-left

    src_pts = np.array([p1, p2, p3, p4], dtype="float32")

    # Destination points for a straightened w x h image
    dst_pts = np.array([
        [0, 0],
        [w - 1, 0],
        [w - 1, h - 1],
        [0, h - 1]
    ], dtype="float32")

    # Get the perspective transform matrix and warp the image
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(image, M, (int(w), int(h)))
    if warped.size == 0:
        return warped # Skip if cropping failed (e.g., zero area)

    # Resize to the target size
    assert isinstance(target_size, tuple) or isinstance(target_size, list)
    # resized_image = cv2.resize(warped, target_size, interpolation=cv2.INTER_AREA)
    warped = rotate_image(warped, angle_rad)
    return warped #resized_image


def draw_bounding_boxes(image, bounding_boxes, class_names, orgin_shape, alpha=0.3, bbox_format="xyxy"):
    """
    Draw semi-transparent masks over objects with distinct colors for each class.
    
    Args:
        image (numpy.ndarray): Input image in BGR format
        bounding_boxes (list): List of bounding boxes in format [x1, y1, x2, y2]
        class_names (list): List of class names corresponding to each bounding box
        alpha (float): Transparency of the masks (0.0 to 1.0)
        
    Returns:
        tuple: (numpy.ndarray, dict) - Image with drawn masks and color mapping dictionary
    """
    # Create a copy of the image to avoid modifying the original
    output_image = image.copy()
    
    # Get color mapping for classes
    color_mapping = get_class_color_mapping(class_names)
    # print(f"color_mapping in draw: {color_mapping}")
    
    # Create an overlay image
    overlay = output_image.copy()
    
    # Draw each mask
    h1,w1 = orgin_shape
    h2,w2 = output_image.shape[:-1]
    for box, class_name in zip(bounding_boxes, class_names):
        
        # Create a colored mask
        mask = np.zeros_like(overlay)
        color = color_mapping[class_name]

        if bbox_format == 'xyxyn':
            x1, y1, x2, y2 = convert_bbox(*box, w2, h2)    
            mask[y1:y2, x1:x2] = color #(color[-1],color[0],color[1])
            
            # Add the mask to the overlay
            overlay = cv2.addWeighted(overlay, 1.0, mask, float(alpha), 0)
            
            # Draw class name
            # (text_width, text_height), _ = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(overlay, 
                        (x1, y1),
                        (x2, y2),
                        color, -1)
            # cv2.putText(overlay, class_name,
            #             (x1, y1 - 10),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        if bbox_format == 'xywhrn':
            corners = xywhrn2xyxyxyxy(box, w2, h2, w1, h1)
            mask = cv2.fillPoly(mask, [corners], color=color)
            overlay = cv2.addWeighted(overlay, 1.0, mask, float(alpha), 0)
            overlay = cv2.fillPoly(overlay, [corners], color=color)
    
    # Blend the overlay with the original image
    output_image = cv2.addWeighted(output_image, 1 - alpha, overlay, alpha, 0)
    
    return output_image, color_mapping