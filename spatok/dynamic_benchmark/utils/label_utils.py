import math
import numpy as np



def cross_product(a, b):
    # a and b are 2D vectors
    return a[0]*b[1] - a[1]*b[0]


def rotation_with_cross_product(v1, v2):
    # using cross product between v1 and v2, determine which direction of rotation is needed to make v1 to v2
    c = cross_product(v1, v2)
    # get angle between the two vectors using the cross product
    angle = np.arcsin(c / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    if angle > 0:
        direction = 'counterclockwise'
    elif angle < 0:
        direction = 'clockwise'
    else:
        direction = 'straight or parallel'
    return c, angle, direction


def lane_change_direction(p1, p2, q1, q2):
    # entity moving from p1 to p2 and cross edge from q1 to q2 or from q2 to q1
    # return right if the entity should change to the right lane, left if the entity should change to the left lane, 0 if no need to change lane
    v1 = p2 - p1
    if np.dot(v1, q1 - q2) == 0:
        return 'unknown'
    if np.dot(v1, q1-q2) > 0:
        v2 = q1 - q2 
    else: #if np.dot(v1, q2-q1) > 0:
        v2 = q2 - q1
    _, angle, direction = rotation_with_cross_product(v1, v2)
    if direction == 'counterclockwise':
        return angle, 'right'
    elif direction == 'clockwise':
        return angle, 'left'
    else:
        return angle, direction
    

def side_with_cross_product(p, tp, q, tq):
    # decide if p is on the left or right side of the line from q to q+tq*c
    # decide if q is on the left or right side of the line from p to p+tq*c
    side = []
    v1 = np.array([np.cos(tp), np.sin(tp)])
    v2 = np.array([np.cos(tq), np.sin(tq)])
    ref_v = p - q 
    c, angle, direction = rotation_with_cross_product(v2, ref_v)
    if direction == 'counterclockwise':
        side.append('left')
    elif direction == 'clockwise':
        side.append('right')
    else:
        side.append('on the line')
    ref_v = q - p
    c, angle, direction = rotation_with_cross_product(v1, ref_v)
    if direction == 'counterclockwise':
        side.append('left')
    elif direction == 'clockwise':
        side.append('right')
    else:
        side.append('on the line')
    return side


def analyze_directions(points, angle_threshold_degrees=15):
    """
    Takes a list of (x, y) coordinates and returns a list of directional movements,
    using angle-based thresholds to determine primary direction.
    
    Args:
        points (list): List of tuples representing (x, y) coordinates
        angle_threshold_degrees (float): Threshold angle in degrees to determine if movement
                                      is primarily in one direction
    
    Returns:
        list: List of strings describing the direction of movement between consecutive points
    """
    if len(points) < 2:
        return []
    
    directions = []
    
    for i in range(len(points) - 1):
        x1, y1 = points[i]
        x2, y2 = points[i + 1]
        
        # Calculate deltas
        delta_x = x2 - x1
        delta_y = y2 - y1
        
        # Handle no movement case
        if delta_x == 0 and delta_y == 0:
            directions.append("no movement")
            continue
        
        # Calculate angle from horizontal (in degrees)
        if delta_x == 0:  # Vertical movement
            angle = 90
        else:
            # arctan gives angle in radians, convert to degrees
            angle = abs(math.degrees(math.atan(delta_y / delta_x)))
        
        # Determine direction based on angle and signs of deltas
        if angle < angle_threshold_degrees:
            # Primarily horizontal movement
            if delta_x > 0:
                directions.append("left to right")
            else:
                directions.append("right to left")
        elif angle > (90 - angle_threshold_degrees):
            # Primarily vertical movement
            if delta_y > 0:
                directions.append("down to up")
            else:
                directions.append("up to down")
        else:
            # Diagonal movement
            x_dir = "right" if delta_x > 0 else "left"
            y_dir = "up" if delta_y > 0 else "down"
            directions.append(f"{y_dir} to {x_dir}")
    
    return directions



def analyze_segments(segments, angle_threshold_degrees=15):
    """
    Analyzes a list of segments, where each segment is defined by (x1, y1, x2, y2).
    
    Args:
        segments (list): List of tuples representing segments as (x1, y1, x2, y2)
        angle_threshold_degrees (float): Threshold angle for direction determination
        
    Returns:
        tuple: (segment_movements, segment_directions)
            - segment_movements: How one segment moves to the next, based on normal vectors
            - segment_directions: Direction of each segment itself
    """
    if len(segments) == 0:
        return [], []
    
    # Analyze direction of each segment itself first
    segment_directions = []
    segment_normals = []
    
    for seg in segments:
        x1, y1, x2, y2 = seg
        # Create points for segment endpoints
        segment_points = [(x1, y1), (x2, y2)]
        
        # Get direction of the segment (will be a single direction)
        segment_dir = analyze_directions(segment_points, angle_threshold_degrees)
        if segment_dir:  # Should always have one item, but check to be safe
            segment_directions.append(segment_dir[0])
        else:
            segment_directions.append("no direction")
        
        # Calculate normal vector for the segment
        dx = x2 - x1
        dy = y2 - y1
        
        # Normalize the vector
        length = math.sqrt(dx**2 + dy**2)
        if length > 0:
            # Calculate two possible normal vectors (perpendicular to segment)
            nx1, ny1 = -dy/length, dx/length  # Rotate 90° clockwise
            nx2, ny2 = dy/length, -dx/length  # Rotate 90° counterclockwise
            
            # Store both possible normals
            segment_normals.append(((nx1, ny1), (nx2, ny2)))
        else:
            # If segment has zero length, use default
            segment_normals.append(((0, 0), (0, 0)))
    
    # Analyze movements between consecutive segments using normal vectors
    segment_movements = []
    
    for i in range(len(segments) - 1):
        # Get current and next segment
        current_seg = segments[i]
        next_seg = segments[i + 1]
        
        # Calculate displacement vector from current to next segment
        cur_midx = (current_seg[0] + current_seg[2]) / 2
        cur_midy = (current_seg[1] + current_seg[3]) / 2
        next_midx = (next_seg[0] + next_seg[2]) / 2
        next_midy = (next_seg[1] + next_seg[3]) / 2
        
        disp_x = next_midx - cur_midx
        disp_y = next_midy - cur_midy
        
        # Check alignment with normal vectors
        normals = segment_normals[i]
        
        # Calculate dot products with both possible normals
        dot1 = disp_x * normals[0][0] + disp_y * normals[0][1]
        dot2 = disp_x * normals[1][0] + disp_y * normals[1][1]
        
        # Find the normal with highest absolute dot product (best alignment)
        if abs(dot1) >= abs(dot2):
            best_normal = normals[0]
            alignment = dot1
        else:
            best_normal = normals[1]
            alignment = dot2
        
        # Determine direction based on the alignment
        if abs(alignment) < 0.01:  # Very small alignment, movement is parallel to segment
            # Use original direction calculation as fallback
            points = [(cur_midx, cur_midy), (next_midx, next_midy)]
            dir_result = analyze_directions(points, angle_threshold_degrees)
            # Convert string "from to to" to tuple (from, to)
            if dir_result:
                parts = dir_result[0].split(" to ")
                if len(parts) == 2:
                    segment_movements.append((parts[0], parts[1]))
                else:
                    segment_movements.append(("undetermined", "undetermined"))
            else:
                segment_movements.append(("undetermined", "undetermined"))
        else:
            # Movement is along the normal direction or opposite
            nx, ny = best_normal
            
            # Calculate movement direction based on normal vector
            if alignment > 0:
                # Movement is in the direction of the normal
                movement_angle = math.degrees(math.atan2(ny, nx))
            else:
                # Movement is opposite to the normal
                movement_angle = math.degrees(math.atan2(-ny, -nx))
            
            # Determine from-to direction based on movement angle
            if -22.5 <= movement_angle < 22.5:
                direction = ("left", "right")
            elif 22.5 <= movement_angle < 67.5:
                direction = ("down", "right")
            elif 67.5 <= movement_angle < 112.5:
                direction = ("down", "up")
            elif 112.5 <= movement_angle < 157.5:
                direction = ("down", "left")
            elif movement_angle >= 157.5 or movement_angle < -157.5:
                direction = ("right", "left")
            elif -157.5 <= movement_angle < -112.5:
                direction = ("up", "left")
            elif -112.5 <= movement_angle < -67.5:
                direction = ("up", "down")
            else:  # -67.5 <= movement_angle < -22.5
                direction = ("up", "right")
            
            segment_movements.append(direction)
    
    return segment_movements, segment_directions


##### part 2: use mask dilation to find the relative position 


from scipy.ndimage import binary_dilation, distance_transform_edt, distance_transform_edt, binary_dilation
import cv2

def minimal_distance_fast(mask1, mask2):
    dt = distance_transform_edt(~mask2)  # Distance to nearest True in mask2
    pts1 = np.argwhere(mask1)
    distances = dt[pts1[:, 0], pts1[:, 1]]
    return np.min(distances)


def maximal_distance_fast(mask1, mask2):
    dt = distance_transform_edt(~mask2)  # Distance to nearest True in mask2
    pts1 = np.argwhere(mask1)
    distances1 = dt[pts1[:, 0], pts1[:, 1]]
    dt = distance_transform_edt(~mask1)  # Distance to nearest True in mask1
    pts2 = np.argwhere(mask2)
    distances2 = dt[pts2[:, 0], pts2[:, 1]]
    return min(np.max(distances1), np.max(distances2))



def one_side_region_universal(mask_longer, mask_shorter, steps=10):
    # 1. Extract points
    pts_longer = np.argwhere(mask_longer)
    pts_shorter = np.argwhere(mask_shorter)

    # 2. Build KDTree for fast nearest neighbor search (longer <- shorter)
    tree_longer = cKDTree(pts_longer)
    dists, indices = tree_longer.query(pts_shorter)

    # 3. Compute vectors from longer to shorter
    matched_longer_pts = pts_longer[indices]
    vectors = matched_longer_pts - pts_shorter

    # 4. Compute average direction
    avg_vector = np.mean(vectors, axis=0)
    norm = np.linalg.norm(avg_vector) + 1e-8
    direction = - avg_vector / norm  # Normalize

    # 5. For each point in longer, step along this direction
    pixel_jumps = np.arange(steps)
    shifts = (direction[None, :] * pixel_jumps[:, None]).astype(int)  # shape (steps, 2)

    # Apply shifts to each longer point
    new_pts = pts_longer[:, None, :] + shifts[None, :, :]  # (N, steps, 2)
    new_pts = new_pts.reshape(-1, 2)

    # 6. Clip points to image bounds
    valid = (new_pts[:, 0] >= 0) & (new_pts[:, 0] < mask_longer.shape[0]) & \
            (new_pts[:, 1] >= 0) & (new_pts[:, 1] < mask_longer.shape[1])

    new_pts = new_pts[valid]

    # 7. Fill the mask
    new_mask = np.copy(mask_longer)
    new_mask[new_pts[:, 0], new_pts[:, 1]] = True

    new_mask = cv2.dilate(new_mask.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1)
    new_mask = new_mask.astype(bool)
    return new_mask


def extract_region_between_two_masks(mask1, mask2):
    dis = maximal_distance_fast(mask1, mask2)
    dilated1 = one_side_region_universal(mask1, mask2, dis)
    dilated2 = one_side_region_universal(mask2, mask1, dis)
    result = dilated1 | dilated2
    return result


def visualize_region_between_two_masks(mask1, mask2):
    region = extract_region_between_two_masks(mask1, mask2)
    # # Create a blank image
    # img = np.zeros((*mask1.shape, 3), dtype=np.uint8)
    
    # # Color the regions
    # img[region] = [0, 0, 255]  # red for the region between
    # img[mask1] = [255, 0, 0]  # blue for mask1
    # img[mask2] = [0, 255, 0]  # Green for mask2
    
    # cv2.imwrite('region_between_masks.png', img)
    return region



def TODO_for_Una():
    print("TODO: Implement the necessary utility function to obtain labels of relative positions between segments and objects in the image.")



if __name__ == "__main__":
    TODO_for_Una()