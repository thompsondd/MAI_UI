import cv2
import numpy as np
from sklearn.cluster import DBSCAN
from collections import defaultdict

# ============================== Utils ========================================
def rxywh2rxyxy(x,y,w,h):
    return x-w/2,y-h/2,x+w/2,y+h/2

def cal_centriod(x1,y1,x2,y2):
    return (x1+x2)/2, (y1+y2)/2

def convert_xyxy_bbox(x1, y1, x2, y2, w1, h1, *args):
    """
    Convert a bounding box from one image shape to another.

    Parameters:
    - x1, y1, x2, y2: Coordinates of the bounding box in the original image.
    - w1, h1: Width and height of the original image.

    Returns:
    - new_x1, new_y1, new_x2, new_y2: Scaled coordinates of the bounding box in the new image.
    """
    new_x1 = int(x1 * w1)
    new_y1 = int(y1 * h1)
    new_x2 = int(x2 * w1)
    new_y2 = int(y2 * h1)

    return new_x1, new_y1, new_x2, new_y2

def validate_string(input_string, rules):
    # Parse the input string into blocks of consecutive same characters
    blocks = []
    block_positions = {}  # Map block_id to character positions
    
    if not input_string:
        # Check if all rules are optional (min=0)
        is_valid = all(rule.get("min", 0) == 0 for rule in rules)
        return is_valid, 100 if is_valid else 0, {}, [], [], [], {}
    
    current_char = input_string[0]
    current_count = 1
    block_start = 0
    
    for i in range(1, len(input_string)):
        if input_string[i] == current_char:
            current_count += 1
        else:
            blocks.append((current_char, current_count))
            block_positions[len(blocks)-1] = list(range(block_start, block_start + current_count))
            current_char = input_string[i]
            current_count = 1
            block_start = i
    
    blocks.append((current_char, current_count))
    block_positions[len(blocks)-1] = list(range(block_start, block_start + current_count))
    
    # Check if blocks match the rules
    rule_index = 0
    block_index = 0
    valid_blocks = []
    invalid_blocks = []
    ignored_blocks = []
    violation_rule_mapping = {}  # Maps block_id to rule_id that it violates
    
    # Keep track of which rules were satisfied
    satisfied_rules = []
    
    # Process blocks against rules
    while block_index < len(blocks) and rule_index < len(rules):
        char, count = blocks[block_index]
        rule = rules[rule_index]

        # print(f"block_index: {block_index} | rule_index: {rule_index}")
        
        # If current block doesn't match current rule
        if char != rule["name"]:
            # If current rule is optional (min=0), skip it and try next rule
            if rule["min"] == 0:
                satisfied_rules.append(rule_index)
                rule_index += 1
                continue
            # Otherwise, this block violates the rule
            else:
                invalid_blocks.append(block_index)
                # Track which rule this block violated
                violation_rule_mapping[block_index] = rule_index
                block_index += 1
                rule_index +=1 
                continue
        
        # Current block matches current rule name
        if count < rule["min"]:
            invalid_blocks.append(block_index)
            # Track which rule this block violated
            violation_rule_mapping[block_index] = rule_index
        elif count > rule["max"]:
            invalid_blocks.append(block_index)
            # Track which rule this block violated
            violation_rule_mapping[block_index] = rule_index
        else:
            valid_blocks.append(block_index)
            satisfied_rules.append(rule_index)
            
        block_index += 1
        rule_index += 1
        
    # Handle any remaining blocks based on new requirement
    if rule_index >= len(rules):
        # We've finished all rules, so any remaining blocks should be ignored
        while block_index < len(blocks):
            ignored_blocks.append(block_index)
            block_index += 1
    else:
        # We still have rules but ran out of blocks
        while rule_index < len(rules):
            rule = rules[rule_index]
            # If a remaining rule is not optional, it's a violation
            if rule["min"] > 0:
                # No specific positions to mark since these rules weren't matched to any blocks
                pass
            else:
                # Optional rule (min=0) is satisfied even if no block matched it
                satisfied_rules.append(rule_index)
            rule_index += 1
    
    # Calculate validation percentage
    total_characters = len(input_string)
    # Get all invalid positions
    violated_positions = []
    for block_id in invalid_blocks:
        violated_positions.extend(block_positions[block_id])
    
    valid_character_count = total_characters - len(violated_positions)
    validation_percentage = (valid_character_count / total_characters) * 100 if total_characters > 0 else 0
    
    # String is valid if all rules were satisfied and no blocks violated rules (ignored blocks don't affect validity)
    is_valid = len(invalid_blocks) == 0 and len(satisfied_rules) == len(rules)
    
    return is_valid, validation_percentage, block_positions, valid_blocks, invalid_blocks, ignored_blocks, violation_rule_mapping, blocks

def validate_lines(rules, lines_info):
    result = defaultdict(dict)
    for (line_idx, info), rule in zip(lines_info.items(), rules) :
        is_valid, percentage, block_positions, valid_blocks, invalid_blocks, ignored_blocks, violation_rule_mapping, blocks = validate_string(
            info['label'],
            rule
        )
        result[line_idx]["is_valid"] = is_valid
        result[line_idx]["valid_percent"] = percentage
        result[line_idx]["block_positions"] = block_positions
        result[line_idx]["invalid_blocks"] = invalid_blocks
        result[line_idx]["valid_blocks"] = valid_blocks
        result[line_idx]["blocks_error"] = {}
        for block_idx, rule_idx in violation_rule_mapping.items():
            error_type = ""
            if blocks[block_idx][0] != rule[rule_idx]["name"]:
                error_type = 'Invalid Product'
            if blocks[block_idx][1] < rule[rule_idx]["min"]:
                error_type = f'Qty must be greater or equal to {rule[rule_idx]["min"]}'
            if blocks[block_idx][1] > rule[rule_idx]["max"]:
                error_type = f'Qty must be less or equal to {rule[rule_idx]["max"]}'
            
            
            result[line_idx]["blocks_error"][block_idx] = {
                "violated_rule": [rule_idx, rule[rule_idx]],
                "error_type": error_type,
                "message": f"Block {block_idx+1} violent rule [{rule[rule_idx]}] -> {error_type}"
            }
    return result

def row_identify(orig_shape, xyxy_bbox, cvt2rel=True):

    # Convert bboxes to center coordinates relative to the original shape
    scale_x = orig_shape[0] if cvt2rel else 1
    scale_y = orig_shape[1] if cvt2rel else 1
    cxcy_bbox = list(map(lambda bbox: [(bbox[0]+bbox[2])/(2*scale_x), (bbox[1]+bbox[3])/(2*scale_y)], xyxy_bbox))
    
    # Apply DBSCAN clustering based on the y-coordinate (cy) to identify rows
    db = DBSCAN(eps=0.09, min_samples=3).fit(cxcy_bbox)
    labels = db.labels_
    
    # Create a mapping of original indices to bboxes and their centers
    bbox_data = []
    for i, (bbox, center, label) in enumerate(zip(xyxy_bbox, cxcy_bbox, labels)):
        if label != -1:  # Exclude noise points (if any)
            bbox_data.append({
                'index': i,
                'bbox': bbox,
                'center': center,
                'label': label
            })
    
    # Sort lines by cy (vertically from top to bottom)
    clusters = {}
    for data in bbox_data:
        label = data['label']
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(data)
    
    # Calculate average y-coordinate for each cluster for sorting
    avg_y_by_cluster = {}
    for label, items in clusters.items():
        avg_y = sum(item['center'][1] for item in items) / len(items)
        avg_y_by_cluster[label.item()] = avg_y
    
    # Sort clusters by average y-coordinate
    sorted_clusters = sorted(avg_y_by_cluster.items(), key=lambda x: x[1])
    
    # Create the final result with reindexed lines
    result = {}
    for new_line_id, (label, _) in enumerate(sorted_clusters):
        # Sort bboxes within this line by x-coordinate (left to right)
        sorted_bboxes = sorted(clusters[label], key=lambda x: x['center'][0])
        
        # Store original indices of the sorted bboxes
        result[new_line_id] = [item['index'] for item in sorted_bboxes]
    
    return result #, avg_y_by_cluster, sorted_clusters

def get_line_info(img_class_info, line_info):
    bbox_line = defaultdict(lambda: {
        "bbox":[],
        "centroid":[],
        "id":[],
        "label":[],
    })
    for line_idx, bboxes in line_info.items():
        for bbox_idx in bboxes:
            bbox = rxywh2rxyxy(*img_class_info['bbox'][bbox_idx])
            bbox_line[line_idx]['bbox'].append(bbox)
            bbox_line[line_idx]['centroid'].append(cal_centriod(*bbox))
            bbox_line[line_idx]['id'].append(bbox_idx)
            bbox_line[line_idx]['label'].append(img_class_info['label'][bbox_idx])
    return bbox_line
def get_block_bbox_one_line(block_positions, bbox):
    block_bbox = {block_idx:[float("INF"),float("INF"),float("-INF"),float("-INF")] for block_idx in block_positions.keys()}

    for block_idx, pos in block_positions.items():
        for pos_idx in pos:
            x1,y1,x2,y2 = block_bbox[block_idx]
            a,b,c,d = bbox[pos_idx]
            block_bbox[block_idx][0] = min(x1, a)
            block_bbox[block_idx][1] = min(y1, b)
            block_bbox[block_idx][2] = max(x2, c)
            block_bbox[block_idx][3] = max(y2, d)
    return block_bbox

def draw_block_one_line(img, block_bbox, invalid_blocks, alpha= 0.7):
    overlay = img.copy()
    output_image = img.copy()
    for block_id, block_b in block_bbox.items():
        x1,y1,x2,y2 = convert_xyxy_bbox(*block_b, 662, 1033)
        if block_id in invalid_blocks:
            
            # Create a colored mask
            mask = np.zeros_like(img)
            mask[y1:y2, x1:x2] = [255,0,0]
            cv2.rectangle(overlay, 
                        (x1, y1),
                        (x2, y2),
                        [255,0,0], -1)
            
            # Add the mask to the overlay
            overlay = cv2.addWeighted(overlay, 1.0, mask, float(alpha), 0)
        else:
            overlay = cv2.rectangle(overlay, (x1,y1), (x2,y2), color=[0,200,150], thickness=5)

    return cv2.addWeighted(output_image, 1 - alpha, overlay, alpha, 0)

def validate_img(classify_result, img, rules):
    lines = row_identify(classify_result['img_shape'], [rxywh2rxyxy(x,y,w,h) for (x,y,w,h) in classify_result['bbox']], False)
    line_info = get_line_info(classify_result, lines)

    result = validate_lines(rules, line_info)

    test = img.copy()
    error_message = {}

    for ln, lni in result.items():
        block_bbox = get_block_bbox_one_line(lni["block_positions"], line_info[ln]["bbox"])
        test= draw_block_one_line(test, block_bbox, lni["invalid_blocks"], alpha= 0.7)
        if lni["blocks_error"]!= {}:
            # print(f"Line {ln+1}:")
            error_message[ln+1] = [bvals["message"] for bvals in lni["blocks_error"].values()]
            # for bix, bvals in lni["blocks_error"].items():
            #     print("\t"+bvals["message"])
    return test, error_message
    