import cv2
import gc
import numpy as np
import base64
import cv2
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
    # scale_x = w2 / w1
    # scale_y = h2 / h1

    # # Scale the bounding box coordinates
    # new_x1 = int(x1 * scale_x)
    # new_y1 = int(y1 * scale_y)
    # new_x2 = int(x2 * scale_x)
    # new_y2 = int(y2 * scale_y)
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

def draw_bounding_boxes(image, bounding_boxes, class_names, orgin_shape, alpha=0.3):
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
        # x1, y1, x2, y2 = box
        x1, y1, x2, y2 = convert_bbox(*box, w2, h2)
        color = color_mapping[class_name]
        
        # Create a colored mask
        mask = np.zeros_like(overlay)
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
    
    # Blend the overlay with the original image
    output_image = cv2.addWeighted(output_image, 1 - alpha, overlay, alpha, 0)
    
    return output_image, color_mapping
