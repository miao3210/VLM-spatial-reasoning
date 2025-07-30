import pdb 
import warnings
import math
import numpy as np
import cv2
import networkx as nx


def distance(p1, p2):
    """Euclidean distance between p1=(x1,y1), p2=(x2,y2)."""
    return math.hypot(p1[0]-p2[0], p1[1]-p2[1])


def line_length(seg):
    """Length of a line segment seg=(x1,y1,x2,y2)."""
    x1,y1,x2,y2 = seg
    return distance((x1,y1),(x2,y2))


def parametric_point(seg, t):
    """
    Given seg=(x1,y1,x2,y2) and parameter t in [0..1],
    return the point P(t) = (x1 + t*(x2-x1), y1 + t*(y2-y1)).
    """
    x1,y1,x2,y2 = seg
    return ( x1 + t*(x2-x1), y1 + t*(y2-y1) )


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


def cross_product_array(a, b):
    if a.shape[-1] == 2 and b.shape[-1] == 2:
        if len(a.shape) == len(b.shape):
            return a[...,0]*b[...,1] - a[...,1]*b[...,0]
        elif len(a.shape) == 2 and len(b.shape) == 1:
            return a[:,0]*b[1] - a[:,1]*b[0]
        elif len(a.shape) == 1 and len(b.shape) == 2:
            return a[0]*b[:,1] - a[1]*b[:,0]
        else:
            warnings.warn('The input arrays should have the same number of dimensions or 1 or 2 dimensions.')
            pdb.set_trace()
    else:
        warnings.warn('The last dimension of the input arrays should be 2.')
        pdb.set_trace()


def rotation_with_cross_product_array(v1, v2):
    # using cross product between v1 and v2, determine which direction of rotation is needed to make v1 to v2
    c = cross_product_array(v1, v2)
    # get angle between the two vectors using the cross product
    angle = np.arcsin(c / (np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1)))
    # direction = np.where(angle > 0, 'counterclockwise', np.where(angle < 0, 'clockwise', 'straight or parallel'))
    clockwise = angle < 0
    counterclockwise = angle > 0
    straight = angle == 0
    return c, angle, clockwise, counterclockwise, straight



def find_closest_point_on_line(p, p1, p2):
    # p is the point, p1 and p2 are the two endpoints of the segment
    # find the point on the line passing the segment closest to p
    # p = p1 + u(p2-p1)
    # p = p[:2]
    # if isinstance(p1, list):
    #     p1 = np.array(p1)
    # elif isinstance(p1, dict):
    #     p1 = np.array([p1['X'], p1['Y']])
    # if isinstance(p2, list):
    #     p2 = np.array(p2)
    # elif isinstance(p2, dict):
    #     p2 = np.array([p2['X'], p2['Y']])
    u = np.dot(p-p1, p2-p1) / np.dot(p2-p1, p2-p1)
    inrange = False
    if u < 0:
        closest = p1
    elif u > 1:
        closest = p2
    else:
        inrange = True
        closest = p1 + u*(p2-p1)
    distance = np.linalg.norm(p-closest)
    return closest, distance, inrange


def find_closest_point_on_line(p, p1, p2):
    # p is the point, p1 and p2 are the two points on the line
    # find the closest point on the line to p
    # p = p1 + u(p2-p1)
    u = np.dot(p-p1, p2-p1) / np.dot(p2-p1, p2-p1)
    u_thre = u // 0.1 * 0.1
    inrange = False
    if u_thre < 0:
        closest = p1
    elif u_thre > 1:
        closest = p2
    else:
        inrange = True
        closest = p1 + u*(p2-p1)
    distance = np.linalg.norm(p-closest)
    return closest, distance, inrange


def find_closest_point_on_segment(p, roadsegment):
    '''
    road segment has the format of [[x1, y1, x2, y2], ...]
    if there are multiple segments in this road segment, find the closest point on the roadsegment
    '''
    if len(roadsegment) < 2:
        warnings.warn('roadsegment has less than 2 points. This roadsegment will be skipped')
        return None, None, None
    min_distance = 1e9
    closest_point = None
    closest_inrange = None
    closest_index = None
    for i in range(len(roadsegment)-1):
        p_, distance, inrange = find_closest_point_on_line(p, roadsegment[i], roadsegment[i+1])
        if distance < min_distance:
            min_distance = distance
            closest_point = p_
            closest_inrange = inrange
            closest_index = i
    if closest_point is None:
        warnings.warn('closest_point is None')
    return closest_point, min_distance, closest_inrange, closest_index


def split_segment_at_t(seg, t, eps=1e-9):
    """
    Split seg at parameter t in [0..1].
    Returns up to two sub-segments. If t=0 or t=1, we effectively don't split.
    """
    if t<eps:
        # entire seg is from t=0..1 => no "first" portion
        return [seg]  # no real split
    if t>1-eps:
        # no "second" portion
        return [seg]

    x1,y1,x2,y2 = seg
    # point at t
    mx,my = parametric_point(seg, t)
    segA = (x1,y1, mx,my)
    segB = (mx,my, x2,y2)
    return [segA, segB]


def angle_of_segment(seg):
    """
    Return angle in degrees [-180..180) for the direction from (x1,y1) to (x2,y2).
    """
    x1,y1,x2,y2 = seg
    return math.degrees(math.atan2(y2-y1, x2-x1))


def distance_between_segments(seg1, seg2):
    '''
    the 2 segments are almost parallel
    return the distance between the 2 segments
    '''
    v1 = np.array([seg1[2]-seg1[0], seg1[3]-seg1[1]])
    v2 = np.array([seg2[2]-seg2[0], seg2[3]-seg2[1]])
    if np.dot(v1, v2) > 0:
        v_vertical = np.array([seg1[0]-seg2[0], seg1[1]-seg2[1]])
    else:
        v_vertical = np.array([seg1[2]-seg2[0], seg1[3]-seg2[1]])
    area = cross_product(v1, v_vertical)
    if area == 0:
        return 0
    else:
        # get the distance between the 2 segments
        # d = area / np.linalg.norm(v1)
        # d = area / np.linalg.norm(v2)
        d = 2 * np.abs(area) / (np.linalg.norm(v1) + np.linalg.norm(v2))
        return d
        

def min_pool(img, window_size=5):
    '''
    use torch.nn.max_pool2d to perform max pooling on the image
    '''
    import torch
    import torch.nn.functional as F
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img = - torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0)  # convert to tensor and add batch dimension
    img = F.max_pool2d(img, kernel_size=window_size, stride=1, padding=window_size//2)  # apply max pooling
    img = - img.squeeze(0).permute(1, 2, 0).byte()  # remove batch dimension and convert back to uint8
    return img.numpy()  # convert back to numpy array


def max_pool(img, window_size=5):
    '''
    use torch.nn.max_pool2d to perform max pooling on the image
    '''
    import torch
    import torch.nn.functional as F
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0)  # convert to tensor and add batch dimension
    img = F.max_pool2d(img, kernel_size=window_size, stride=1, padding=window_size//2)  # apply max pooling
    img = img.squeeze(0).permute(1, 2, 0).byte()  # remove batch dimension and convert back to uint8
    return img.numpy()  # convert back to numpy array


def cluster_related_features(pairs):
    G = nx.Graph()
    G.add_edges_from(pairs)
    clusters = list(nx.connected_components(G))
    return clusters


def get_k_largest_components_by_l2(labels, k=5):
    component_data = []

    # Skip label 0 (background)
    for label in range(1, labels.max() + 1):
        coords = np.column_stack(np.where(labels == label))
        if coords.size == 0:
            continue

        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)

        # L2 norm of the diagonal
        l2 = np.sqrt((x_max - x_min) ** 2 + (y_max - y_min) ** 2)

        component_data.append((label, l2, (x_min, y_min, x_max, y_max)))

    # Sort by L2 norm descending
    sorted_components = sorted(component_data, key=lambda x: x[1], reverse=True)

    # Return top-k
    return sorted_components[:k]
    


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


def analyze_segments_version_midpoint(segments, angle_threshold_degrees=15):
    """
    Analyzes a list of segments, where each segment is defined by (x1, y1, x2, y2).
    
    Args:
        segments (list): List of tuples representing segments as (x1, y1, x2, y2)
        angle_threshold_degrees (float): Threshold angle for direction determination
        
    Returns:
        tuple: (segment_movements, segment_directions)
            - segment_movements: How one segment moves to the next
            - segment_directions: Direction of each segment itself
    """
    if len(segments) == 0:
        return [], []
    
    # Calculate midpoints of each segment to analyze movement between segments
    midpoints = []
    for seg in segments:
        x1, y1, x2, y2 = seg
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2
        midpoints.append((mid_x, mid_y))
    
    # Analyze movements between consecutive segments using midpoints
    segment_movements = analyze_directions(midpoints, angle_threshold_degrees)
    
    # Analyze direction of each segment itself
    segment_directions = []
    for x1, y1, x2, y2 in segments:
        # Create points for segment endpoints
        segment_points = [(x1, y1), (x2, y2)]
        # Get direction of the segment (will be a single direction)
        segment_dir = analyze_directions(segment_points, angle_threshold_degrees)
        if segment_dir:  # Should always have one item, but check to be safe
            segment_directions.append(segment_dir[0])
        else:
            segment_directions.append("no direction")
    
    return segment_movements, segment_directions



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


# Example usage
if __name__ == "__main__":
    # Example segments as (x1, y1, x2, y2)
    segments = [
        (0, 0, 5, 0),      # Horizontal segment pointing right
        (10, 5, 15, 5),    # Horizontal segment shifted up and right
        (20, 20, 20, 30),  # Vertical segment pointing up
        (15, 35, 5, 40)    # Diagonal segment
    ]
    
    # Analyze segments
    movements, directions = analyze_segments(segments)
    
    # Print results
    print("Segments:", segments)
    print("\nSegment Movements (between segments):", movements)
    print("\nSegment Directions (of each segment):", directions)