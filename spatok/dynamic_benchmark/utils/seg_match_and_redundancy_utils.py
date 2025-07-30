import warnings
import pdb
import numpy as np
import cv2
from .basic_math_utils import *



def share_endpoint(seg1, seg2, eps=10):
    """
    Return True if seg1 and seg2 have endpoints that lie within eps distance.
    seg1 and seg2 are tuples (x1,y1,x2,y2).
    """
    endpoints1 = [(seg1[0], seg1[1]), (seg1[2], seg1[3])]
    endpoints2 = [(seg2[0], seg2[1]), (seg2[2], seg2[3])]
    for p in endpoints1:
        for q in endpoints2:
            if distance(p, q) < eps:
                return True
    return False


def group_segments(segments, eps=10):
    """
    Group segments into connected components. Two segments are connected if they share
    an endpoint (within eps). Returns a list of groups, where each group is a list of indices.
    """
    n = len(segments)
    # Build connectivity graph (adjacency list)
    connectivity = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            if share_endpoint(segments[i], segments[j], eps):
                connectivity[i].append(j)
                connectivity[j].append(i)
    # Use BFS to group connected segments
    visited = [False]*n
    groups = []
    for i in range(n):
        if not visited[i]:
            group = []
            stack = [i]
            visited[i] = True
            while stack:
                idx = stack.pop()
                group.append(idx)
                for neighbor in connectivity[idx]:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        stack.append(neighbor)
            groups.append(group)
    return groups


def find_main_segments(segments, n_clusters=2):
    if len(segments.shape) == 3:
        segments = segments.reshape(-1, 4)
        
    seg_len = np.linalg.norm(segments[:, 2:4] - segments[:, 0:2], axis=1)
    # use k-means to cluster the segments based on their length
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(seg_len.reshape(-1, 1))
    labels = kmeans.labels_
    # find the cluster with the larger center value
    cluster_centers = kmeans.cluster_centers_.flatten()
    main_cluster = np.argmax(cluster_centers)
    # return the segments that belong to the main cluster
    main_segments = segments[labels == main_cluster]
    main_seg_len = np.linalg.norm(main_segments[:, 2:4] - main_segments[:, 0:2], axis=1)
    return main_segments, main_seg_len


def find_redundant_segments_kmeans(segments, redundancy=2):
    '''
    this is for the non-intersected trajectories
    for road segments, this does not work well
    '''
    # Compute midpoints for each segment.
    midpoints = np.column_stack(((segments[:, 0] + segments[:, 2]) / 2,
                                (segments[:, 1] + segments[:, 3]) / 2)).astype(np.float32)
    
    # Setup criteria for cv2.kmeans: 100 iterations or epsilon of 0.2.
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    ret, labels, centers = cv2.kmeans(midpoints, len(segments)//redundancy, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    # groupd redundant segments based on cluster labels.
    redundant_segments = []
    for i in range(len(centers)):
        cluster_indices = np.where(labels.flatten() == i)[0]
        if len(cluster_indices) <= redundancy:
            # Only keep segments that belong to the same cluster and exceed the redundancy threshold.
            redundant_segments.append(segments[cluster_indices])
    return redundant_segments, centers, labels


def summarize_overlap_segments(segments):
    '''
    this function is used to summarize the segments that are nearly parallel and have the same direction
    the returned segment is the combination of the segments, i.e. one segment that covers all the segments
    '''
    reorder_segs = []
    for seg in segments:
        if seg[0] == seg[2]:
            if seg[1] > seg[3]:
                seg = np.array([seg[2], seg[3], seg[0], seg[1]])
            else:
                seg = np.array([seg[0], seg[1], seg[2], seg[3]])
        elif seg[0] > seg[2]:
            seg = np.array([seg[2], seg[3], seg[0], seg[1]])
        else:
            seg = np.array([seg[0], seg[1], seg[2], seg[3]])
        reorder_segs.append(seg)
    x1s = np.array([seg[0] for seg in reorder_segs])
    y1s = np.array([seg[1] for seg in reorder_segs])
    x2s = np.array([seg[2] for seg in reorder_segs])
    y2s = np.array([seg[3] for seg in reorder_segs])
    x1 = np.min(x1s)
    y1 = np.min(y1s)
    x2 = np.max(x2s)
    y2 = np.max(y2s)
    return np.array([x1, y1, x2, y2])


def remove_redundant_segments(segments, angle_thresh=np.deg2rad(5), distance_thresh=3.0):
    '''
    this version allows the 2 sides of the segment have different lengths
    the 2 segments should be nearly parallel and have the same direction
    '''
    # print('remove redundant segments ...')
    if len(segments.shape) == 3:
        segments = segments.reshape(-1, 4)
    # Non-naive case: group segments based on orientation and their overlap range.
    # Compute segment orientations.
    angles = np.arctan2(segments[:, 3] - segments[:, 1], segments[:, 2] - segments[:, 0]) % np.pi
    # Compute segment lengths.
    lengths = np.sqrt((segments[:, 2] - segments[:, 0])**2 + (segments[:, 3] - segments[:, 1])**2)
    # Compute midpoints.
    midpoints = np.column_stack(((segments[:, 0] + segments[:, 2]) / 2, (segments[:, 1] + segments[:, 3]) / 2))
    # Compute normal vector for each segment (rotate angle by 90°).
    normals = np.column_stack((np.cos(angles + np.pi/2),  np.sin(angles + np.pi/2)))
    
    # Group segments by quantized orientation.
    groups = {}
    reset_key = round(np.pi / angle_thresh)
    for i, theta in enumerate(angles):
        # Use a quantized key so that segments within angle_thresh fall in the same bin.
        key = round(theta / angle_thresh)
        if key == reset_key:
            key = 0
        groups.setdefault(key, []).append(i)
    
    selected_segments = []
    # Compare segments within each orientation group.
    for key, idx_list in groups.items():
        distance = np.zeros((len(idx_list), len(idx_list)))
        close_subgroups = []
        for i, idx1 in enumerate(idx_list):
            for j in range(i+1, len(idx_list)):
                idx2 = idx_list[j]
                # Compute distance between midpoints.
                dis1 = np.dot(midpoints[idx1] - midpoints[idx2], normals[idx1])
                dis2 = np.dot(midpoints[idx1] - midpoints[idx2], normals[idx2])
                distance[i, j] = np.max([dis1, dis2])
                if np.abs(distance[i, j]) < distance_thresh:
                    # print('add to close_subgroups: ', idx1, idx2, 'distance: ', distance[i, j])
                    found = False
                    for k, subg in enumerate(close_subgroups):
                        if idx1 in subg or idx2 in subg:
                            subg.append(idx1)
                            subg.append(idx2)
                            close_subgroups[k] = list(set(subg))
                            found = True
                            break
                    if not found:
                        close_subgroups.append([idx1, idx2])

        for subg in close_subgroups:
            # if all mid points close enough, keep the shortest segment
            if len(subg) > 1:
                subsegs = segments[subg]
                # reorder_segs = []
                # for idx in subg:
                #     seg = segments[idx]
                #     if seg[0] == seg[2]:
                #         if seg[1] > seg[3]:
                #             seg = np.array([seg[2], seg[3], seg[0], seg[1]])
                #         else:
                #             seg = np.array([seg[0], seg[1], seg[2], seg[3]])
                #     elif seg[0] > seg[2]:
                #         seg = np.array([seg[2], seg[3], seg[0], seg[1]])
                #     else:
                #         seg = np.array([seg[0], seg[1], seg[2], seg[3]])
                #     reorder_segs.append(seg)
                new_seg = summarize_overlap_segments(subsegs)
                selected_segments.append(new_seg)
                # endpoint1inrange = np.zeros((len(subg), len(subg)))
                # endpoint2inrange = np.zeros((len(subg), len(subg)))
                # endpoint1closest = np.zeros((len(subg), len(subg), 2))
                # endpoint2closest = np.zeros((len(subg), len(subg), 2))
                # for i, idx1 in enumerate(subg):
                #     for j, idx2 in enumerate(subg):
                #         p = segments[idx1][:2]
                #         p1 = segments[idx2][:2]
                #         p2 = segments[idx2][2:]
                #         closest1, distance1, inrange1 = find_closest_point_on_line(p, p1, p2)
                #         endpoint1inrange[i, j] = inrange1
                #         endpoint1closest[i, j] = closest1
                #         p = segments[idx1][2:]
                #         closest2, distance2, inrange2 = find_closest_point_on_line(p, p1, p2)
                #         endpoint2inrange[i, j] = inrange2
                #         endpoint2closest[i, j] = closest2
                        
                # pre_selected = []
                # for i, idx1 in enumerate(subg):
                #     for j in range(i+1, len(subg)):
                #         idx2 = subg[j]
                #         if i==j:
                #             continue
                #         cnt_inrange = 0
                #         if endpoint1inrange[i, j]:
                #             cnt_inrange += 1
                #         if endpoint2inrange[i, j]:
                #             cnt_inrange += 1
                #         if endpoint1inrange[j, i]:
                #             cnt_inrange += 1
                #         if endpoint2inrange[j, i]:
                #             cnt_inrange += 1
                #         if cnt_inrange == 0:
                #             # no endpoints are in range, they are not redundant
                #             pre_selected.append(idx1)
                #             pre_selected.append(idx2)
                #         elif cnt_inrange == 1:
                #             warnings.warn('Found one endpoint is in range. Currently view this as one endpoint is the same.')
                #             pre_selected.append(idx1)
                #             pre_selected.append(idx2)
                #         pre_selected = list(set(pre_selected))
                #         if 24 in [idx1, idx2]:
                #             pdb.set_trace()

                # for i, idx1 in enumerate(subg):
                #     for j in range(i+1, len(subg)):
                #         keep_idx1 = True
                #         keep_idx2 = True
                #         idx2 = subg[j]
                #         if i==j:
                #             continue
                #         cnt_inrange = 0
                #         if endpoint1inrange[i, j]:
                #             cnt_inrange += 1
                #         if endpoint2inrange[i, j]:
                #             cnt_inrange += 1
                #         if endpoint1inrange[j, i]:
                #             cnt_inrange += 1
                #         if endpoint2inrange[j, i]:
                #             cnt_inrange += 1
                #         if cnt_inrange >= 2:
                #             closest1 = endpoint1closest[i, j]
                #             closest2 = endpoint2closest[i, j]
                #             coverage1 = np.linalg.norm(closest1 - closest2) / lengths[idx1]
                #             closest3 = endpoint1closest[j, i]
                #             closest4 = endpoint2closest[j, i]
                #             coverage2 = np.linalg.norm(closest3 - closest4) / lengths[idx2]
                #             if coverage1 > 0.9 and coverage2 > 0.9:
                #                 if lengths[idx1] > lengths[idx2]:
                #                     keep_idx1 = True
                #                     keep_idx2 = False
                #                 else:
                #                     keep_idx1 = False
                #                     keep_idx2 = True
                #             elif coverage1 > 0.9:
                #                 keep_idx1 = False 
                #                 keep_idx2 = True
                #             elif coverage2 > 0.9:
                #                 keep_idx1 = True
                #                 keep_idx2 = False
                            
                #             pre_selected = list(set(pre_selected))   
                #             if keep_idx1:
                #                 pre_selected.append(idx1)
                #             elif idx1 in pre_selected:
                #                 pre_selected.remove(idx1)
                #             if keep_idx2:
                #                 pre_selected.append(idx2)
                #             elif idx2 in pre_selected:
                #                 pre_selected.remove(idx2)
                #             pre_selected = list(set(pre_selected))                         
                #         else:
                #             pass
                # for p in pre_selected:
                #     selected_segments.append(segments[p])
            else:
                selected_segments.append(segments[subg[0]])
    
    return np.array(selected_segments)


def merge_redundant_segments(redundant_segments):
    '''
    this is for the xodr style road image only
    '''
    merged_segments = []
    for pair in redundant_segments:
        if not len(pair) == 2:
            warnings.warn(f'Redundant segments should be pairs, found {len(pair)} segments.')
            pdb.set_trace()
        seg1, seg2 = pair
        direc1 = np.array([seg1[2] - seg1[0], seg1[3] - seg1[1]])
        direc2 = np.array([seg2[2] - seg2[0], seg2[3] - seg2[1]])
        # Check if the segments are nearly parallel (same angle)
        if np.abs(np.dot(direc1, direc2) / (np.linalg.norm(direc1) * np.linalg.norm(direc2))) > 0.9:
            # check if the direction is the same
            if np.dot(direc1, direc2) > 0:
                # merge segments by taking the extreme endpoints
                merged_seg = np.mean([seg1, seg2], axis=0)
            else:
                # merge segments by taking the extreme endpoints
                reorder_seg2 = np.array([seg2[2], seg2[3], seg2[0], seg2[1]])  # reverse the direction of seg2
                merged_seg = np.mean([seg1, reorder_seg2], axis=0)
            merged_segments.append(merged_seg)
    return np.array(merged_segments)


def filter_useful_segments(segments, n_clusters=2, redundancy=2):
    main_segments, lengths = find_main_segments(segments, n_clusters=n_clusters)
    redundant_segments, centers, labels = find_redundant_segments_kmeans(main_segments, redundancy)
    merged_segments = merge_redundant_segments(redundant_segments)
    return merged_segments


def flip_segment(seg):
    """Flip a segment [x1, y1, x2, y2] to [x2, y2, x1, y1]."""
    return np.array([seg[2], seg[3], seg[0], seg[1]], dtype=seg.dtype)

def reorder_segments(segments):
    """
    Reorder segments based on endpoint connectivity.
    
    Steps:
      1. For each segment (of shape (N,4)), extract its two endpoints.
      2. For each endpoint, find the closest endpoint from a different segment (if within threshold).
      3. This produces a connection mapping from (seg_index, endpoint_index) to (other_seg, other_endpoint).
      4. Determine the chain start as a segment having an endpoint with no incoming connection.
      5. Reconstruct the chain by following the connections.
         If the connection is from a segment’s second endpoint, flip it so that its start connects.
    
    Parameters:
      segments: numpy array of shape (N,4), where each row is [x1, y1, x2, y2].
      threshold: maximum distance to consider endpoints as connected.
      
    Returns:
      reordered_segments: numpy array of shape (N,4) representing the connected chain.
    """
    N = segments.shape[0]
    if N == 0:
        return segments

    # Step 1. Build an array of endpoints and a mapping to (segment index, endpoint index).
    endpoints = np.zeros((2 * N, 2), dtype=np.float32)
    mapping = []  # list of tuples (segment index, endpoint index), where endpoint index 0 means first endpoint, 1 means second.
    for i in range(N):
        endpoints[2*i] = segments[i][:2]
        mapping.append((i, 0))
        endpoints[2*i+1] = segments[i][2:]
        mapping.append((i, 1))
        
    # Step 2. For each endpoint, find its nearest neighbor from a different segment.
    connections = {}  # key: (seg, ep) --> value: (other_seg, other_ep)
    for idx in range(2 * N):
        seg_i, ep_i = mapping[idx]
        pt = endpoints[idx]
        best_dist = np.inf
        best_conn = None
        for j in range(2 * N):
            if j == idx:
                continue
            seg_j, ep_j = mapping[j]
            if seg_j == seg_i:
                continue
            d = np.linalg.norm(pt - endpoints[j])
            if d < best_dist:
                best_dist = d
                best_conn = (seg_j, ep_j)
            connections[(seg_i, ep_i)] = best_conn

    # Step 3. Count incoming connections for each endpoint.
    incoming = {}  # key: (seg, ep) --> count
    for key, val in connections.items():
        incoming[val] = incoming.get(val, 0) + 1

    # Step 4. Find a starting segment.
    # We'll choose a segment that has at least one endpoint with no incoming connection.
    start_seg = None
    start_ep = None
    for i in range(N):
        for ep in [0, 1]:
            if (i, ep) not in incoming:
                start_seg = i
                start_ep = ep
                break
        if start_seg is not None:
            break
    # If every endpoint has an incoming connection (e.g. in a cycle), pick segment 0 arbitrarily.
    if start_seg is None:
        start_seg = 0
        start_ep = 0

    # Determine the proper orientation for the starting segment.
    # We want its start (the "used" endpoint) to be the one with no incoming connection.
    if start_ep == 0:
        current_seg = segments[start_seg]
    else:
        current_seg = flip_segment(segments[start_seg])
    chain = [(start_seg, current_seg)]
    used = {start_seg}
    current_endpoint = current_seg[2:4]  # current chain end

    # Step 5. Reconstruct the chain.
    # Instead of strictly following the connection mapping (which is defined for endpoints),
    # we can search among the remaining segments for the one whose one endpoint is closest to current_endpoint.
    # That way we “respect” the connection graph but allow for some flexibility.
    while len(used) < N:
        best_next = None
        best_next_ep = None
        best_dist = np.inf
        for i in range(N):
            if i in used:
                continue
            # For segment i, consider both endpoints.
            for ep in [0, 1]:
                pt = segments[i][:2] if ep == 0 else segments[i][2:]
                d = np.linalg.norm(current_endpoint - pt)
                if d < best_dist:
                    best_dist = d
                    best_next = i
                    best_next_ep = ep
        if best_next is None:
            break  # no further connection within threshold
        # If the chosen connection is at endpoint index 1, we flip the segment so that its start is near current_endpoint.
        next_seg = segments[best_next] if best_next_ep == 0 else flip_segment(segments[best_next])
        chain.append((best_next, next_seg))
        used.add(best_next)
        current_endpoint = next_seg[2:4]

    # Produce the final reordered segments array.
    reordered_segments = np.array([seg for _, seg in chain], dtype=np.float32)
    return reordered_segments


def find_matched_segment(seg_of_interest, segments, angle_thresh=np.deg2rad(10), strict=True):
    '''
    this function is used to find the matched road boundary segment from the road boundary collection
    i.e. the segment that is likely to be the other side of the road boundary of the same road branch
    args:
        seg_of_interest: the segment of interest
        segments: the collection of road boundary segments
        angle_thresh: the unit for the orientation quantization for matching
        strict: 
            if False, only requires the orientation to be close and at least one endpoint is in range
            if True, also requires both midpoints to be in range and both midpoints in range 

    '''
    segments = np.array(segments).reshape(-1, 4)
    # compute the distance and remove the seg_of_interest from the segments
    distances = np.linalg.norm(segments - seg_of_interest.reshape(-1, 4), axis=1)
    min_idx = np.argmin(distances)
    if distances[min_idx] < 1e-3:
        same_seg = True
    else:
        same_seg = False
        # print('The segment of interest is not in the segments collection.')
    # Compute segment orientations.
    angles = np.arctan2(segments[:, 3] - segments[:, 1], segments[:, 2] - segments[:, 0]) % np.pi
    # Compute segment lengths.
    lengths = np.sqrt((segments[:, 2] - segments[:, 0])**2 + (segments[:, 3] - segments[:, 1])**2)
    # Compute midpoints.
    midpoints = np.column_stack(((segments[:, 0] + segments[:, 2]) / 2, (segments[:, 1] + segments[:, 3]) / 2))
    # Compute normal vector for each segment (rotate angle by 90°).
    normals = np.column_stack((np.cos(angles + np.pi/2),  np.sin(angles + np.pi/2)))

    # update distance along the normal of interest direction
    distance = np.zeros((len(segments), 1))
    for i in range(len(segments)):
        distance[i] = np.dot(midpoints[i] - seg_of_interest[:2], normals[i])
    
    angle_of_interest = np.arctan2(seg_of_interest[3] - seg_of_interest[1], seg_of_interest[2] - seg_of_interest[0]) % np.pi
    length_of_interest = np.sqrt((seg_of_interest[2] - seg_of_interest[0])**2 + (seg_of_interest[3] - seg_of_interest[1])**2)
    midpoint_of_interest = np.array([(seg_of_interest[0] + seg_of_interest[2]) / 2, (seg_of_interest[1] + seg_of_interest[3]) / 2])
    normal_of_interest = np.array([np.cos(angle_of_interest + np.pi/2),  np.sin(angle_of_interest + np.pi/2)])

    # find the road segments that share the orientation
    angles_q = np.round(angles / angle_thresh)
    reset_q = round(np.pi / angle_thresh)
    angles_q[angles_q==reset_q] = 0
    angle_of_interest_q = np.round(angle_of_interest / angle_thresh)
    angle_of_interest_q = 0 if angle_of_interest_q == reset_q else angle_of_interest_q
    pre_matched = np.where(np.abs(angles_q - angle_of_interest_q) < 2)[0]
    # print('matched by orientation: ', pre_matched)
    if len(pre_matched) == 1:
        if same_seg and pre_matched[0] == min_idx:
            return None, None, []
        matched = pre_matched[0]
        return segments[matched].reshape(-1, 4), pre_matched, [1]
    elif len(pre_matched) == 0:
        warnings.warn('No segments have similar orientation.')
        return None, None, []

    # find the road segment that have overlap region with the segment of interest
    midpoint_inrange = np.zeros((len(pre_matched), 2))
    endpoint_inrange = np.zeros((len(pre_matched), 4))
    for i, idx in enumerate(pre_matched):
        # point from the segment of interest
        p1 = segments[idx][:2]
        p2 = segments[idx][2:]
        p = midpoint_of_interest
        # print(f'p: {p}, p1: {p1}, p2: {p2}')
        closest, distance, inrange = find_closest_point_on_line(p, p1, p2)
        midpoint_inrange[i, 0] = inrange
        p = seg_of_interest[:2]
        # print(f'p: {p}, p1: {p1}, p2: {p2}')
        closest, distance, inrange = find_closest_point_on_line(p, p1, p2)
        endpoint_inrange[i, 0] = inrange
        p = seg_of_interest[2:]
        # print(f'p: {p}, p1: {p1}, p2: {p2}')
        closest, distance, inrange = find_closest_point_on_line(p, p1, p2)
        endpoint_inrange[i, 1] = inrange
        # point from candidate segment
        p1 = seg_of_interest[:2]
        p2 = seg_of_interest[2:]
        p = midpoints[idx]
        # print(f'p: {p}, p1: {p1}, p2: {p2}')
        closest, distance, inrange = find_closest_point_on_line(p, p1, p2)
        midpoint_inrange[i, 1] = inrange
        p = segments[idx][:2]
        # print(f'p: {p}, p1: {p1}, p2: {p2}')
        closest, distance, inrange = find_closest_point_on_line(p, p1, p2)
        endpoint_inrange[i, 2] = inrange
        p = segments[idx][2:]
        # print(f'p: {p}, p1: {p1}, p2: {p2}')
        closest, distance, inrange = find_closest_point_on_line(p, p1, p2)
        endpoint_inrange[i, 3] = inrange
    
    if strict:
        # the midpoint should be inrange for both segments and the endpoint inrange should be at least 2
        midpoint_match = np.where(midpoint_inrange[:, 0] + midpoint_inrange[:, 1] > 0)[0]
        endpoint_match = np.where(np.sum(endpoint_inrange, axis=1) > 1)[0]
        matched = np.intersect1d(midpoint_match, endpoint_match)
        sec_matched = pre_matched[matched]
        # print('matched by midpoint and endpoint: ', sec_matched)
        if len(sec_matched) == 1:
            if same_seg and sec_matched[0] == min_idx:
                return None, None, []
            return segments[sec_matched].reshape(-1, 4), sec_matched, [1]
    else:
        endpoint_match = np.where(np.sum(endpoint_inrange, axis=1) > 1)[0]
        sec_matched = pre_matched[endpoint_match]
        # print('matched by endpoint: ', sec_matched)
        if len(sec_matched) == 1:
            return segments[sec_matched].reshape(-1, 4), sec_matched, [1]
        
    # return the closest segment on both sides
    sec_matched_midpoint = midpoints[sec_matched]
    vec = sec_matched_midpoint - seg_of_interest[:2]
    vec_ref = seg_of_interest[2:] - seg_of_interest[:2]
    cross_product, angle, clockwise, counterclockwise, straight = rotation_with_cross_product_array(vec, vec_ref)
    final_match = []
    num_side = []
    # print('clockwise: ', clockwise, ', counterclockwise: ', counterclockwise)
    if clockwise.sum() > 0:
        # find the closest one
        clockwise_match = sec_matched[clockwise]
        if same_seg:
            clockwise_match = [idx for idx in clockwise_match if idx != min_idx]
        clockwise_num = len(clockwise_match)
        clockwise_match = np.array(clockwise_match)
        if clockwise_num > 0:
            dis = distances[clockwise_match]
            final_match.append(clockwise_match[np.argsort(dis)])
            num_side.append(len(clockwise_match))
    if counterclockwise.sum() > 0:
        counterclockwise_match = sec_matched[counterclockwise]
        if same_seg:
            counterclockwise_match = [idx for idx in counterclockwise_match if idx != min_idx]
        counterclockwise_num = len(counterclockwise_match)
        counterclockwise_match = np.array(counterclockwise_match)
        if counterclockwise_num > 0:
            dis = distances[counterclockwise_match]
            final_match.append(counterclockwise_match[np.argsort(dis)])
            num_side.append(len(counterclockwise_match))
        
    if len(final_match) > 0:
        final_match = np.concatenate(final_match)
        # print('final match: ', final_match, ', num of segments on each side: ', num_side)
        return segments[final_match].reshape(-1, 4), final_match, num_side
    else:
        return None, None, []
    


def remove_redundant_road_boundary_segments_and_get_pair_original(segments):
    '''
    this function first find matched segments, and calculate the distance between the segments and the most closest 2 segments
    then use k-means to cluster the distance and decide a threshold
    finally, remove the segments that are too close to each other
    '''
    print('remove redundant road segments ...')
    distances = []
    # res1 = []
    # res2 = []
    # res3 = []
    grouped_segments = []
    for k in range(len(segments)):
        seg1 = segments[k]
        seg_matched, indices, side = find_matched_segment(seg1, segments)
        # res1.append(seg_matched)
        # res2.append(indices)
        # res3.append(side)
        if seg_matched is not None:
            for seg, idx in zip(seg_matched, indices):
                dis = distance_between_segments(seg1, seg)
                distances.append(dis)
                grouped_segments.append([k, idx])
    # print(distances)
    # use np.hist to find the distance distribution
    distances = np.array(distances)
    grouped_segments = np.array(grouped_segments)
    # hist, bin_edges = np.histogram(distances, bins=20)
    # if hist[1] > 0:
    #     warnings.warn('Found unsupported distance distribution.')
    #     pdb.set_trace()
    # threshold = bin_edges[1]
    threshold = distances.max() / 10
    redundant_pairs = grouped_segments[distances < threshold]
    if len(redundant_pairs) == 0:
        warnings.warn('No redundant segments found.')
        return segments
    # count how many times a segment is in the redundant pairs
    clusters = cluster_related_features(redundant_pairs)
    selected_segments = []
    for c in clusters:
        c = list(c)
        if len(c) == 1:
            # only one segment in the cluster, keep it
            selected_segments.append(segments[c[0]].reshape(-1,4))
        elif len(c) == 2:
            # merge the 2 segments
            merged_segment = merge_redundant_segments([[segments[c[0]], segments[c[1]]]])
            selected_segments.append(merged_segment.reshape(-1,4))
        elif len(c) > 2:
            G = nx.Graph()
            for g in redundant_pairs:
                if g[0] in c and g[1] in c:
                    G.add_edge(g[0], g[1])
            if nx.is_bipartite(G):
                set1, set2 = nx.bipartite.sets(G)
            if len(set1) < len(set2):
                set1, set2 = set2, set1
            # add all segments in set1 to the pruned segments
            for idx in set1:
                selected_segments.append(segments[idx].reshape(-1,4))

    selected_segments = np.concatenate(selected_segments, axis=0)
    print(selected_segments)
    ## get matched pairs of road segments
    matched_boundaries = []
    pair_distances = []
    for k in range(len(selected_segments)):
        seg1 = selected_segments[k]
        seg_matched, indices, side = find_matched_segment(seg1, selected_segments)
        if seg_matched is not None:
            for seg, idx in zip(seg_matched, indices):
                dis = distance_between_segments(seg1, seg)
                pair_distances.append(dis)
                matched_boundaries.append([k, idx])
    final_pairs = []
    # Create an undirected graph
    G = nx.Graph()
    G.add_edges_from(matched_boundaries)
    # Find connected components
    components = list(nx.connected_components(G))
    for subg in components:
        final_pairs.append([list(subg), selected_segments[list(subg)].reshape(-1,4)])
        if len(subg) != 2:
            warnings.warn('Found pair of road boundaries that has numbers of segments other than 2.')
    return selected_segments, matched_boundaries, final_pairs


def split_segment(to_split, ref1, ref2):
    # return the splited to_split segment
    # find the closest point on the line segment
    ref_points = []
    p = to_split[:2]
    ref1_close_to = None 
    ref2_close_to = None
    # find the reference point on ref1s
    p1 = ref1[:2]
    p2 = ref1[2:]
    closest1, distance1, inrange1 = find_closest_point_on_line(p, p1, p2)
    p = to_split[2:]
    closest2, distance2, inrange2 = find_closest_point_on_line(p, p1, p2)
    if inrange1 and inrange2:
        warnings.warn('Found the segment is already covered by the matched boundary.')
        pdb.set_trace()
    elif inrange1 and not inrange2:
        ref_points.append(closest2)
        ref1_close_to = 1
    elif not inrange1 and inrange2:
        ref_points.append(closest1)
        ref1_close_to = 2
    else:
        if distance1 < distance2:
            ref_points.append(closest2)
            ref1_close_to = 1
        else:
            ref_points.append(closest1)
            ref1_close_to = 2
    # find the reference point on ref2
    p1 = ref2[:2]
    p2 = ref2[2:]
    closest1, distance1, inrange1 = find_closest_point_on_line(p, p1, p2)
    p = to_split[2:]
    closest2, distance2, inrange2 = find_closest_point_on_line(p, p1, p2)
    if inrange1 and inrange2:
        warnings.warn('Found the segment is already covered by the matched boundary.')
        pdb.set_trace()
    elif inrange1 and not inrange2:
        ref_points.append(closest2)
        ref2_close_to = 1
    elif not inrange1 and inrange2:
        ref_points.append(closest1)
        ref2_close_to = 2
    else:
        if distance1 < distance2:
            ref_points.append(closest2)
            ref2_close_to = 1
        else:
            ref_points.append(closest1)
            ref2_close_to = 2
    # split reference point
    ref = np.array(ref_points).mean(axis=0)
    # split the segment
    closest, distance, inrange = find_closest_point_on_line(ref, to_split[:2], to_split[2:])
    if inrange:
        # split the segment
        seg1 = np.array([to_split[0], to_split[1], closest[0], closest[1]])
        seg2 = np.array([closest[0], closest[1], to_split[2], to_split[3]])
    else:
        warnings.warn('Found the segment is not in range.')
        pdb.set_trace()
        seg1, seg2 = None, None
    if seg1 and seg2:
        if ref1_close_to == 1 and ref2_close_to == 2:
            return seg1, seg2
        elif ref1_close_to == 2 and ref2_close_to == 1:
            return seg2, seg1
        else:
            warnings.warn('Found both segments close to the same side of the to_split segment.')
            pdb.set_trace()
            return None, None
    else:
        warnings.warn('Found the segment is not in range.')
        pdb.set_trace()
        return None, None


def remove_redundant_road_boundary_segments(segments):
    '''
    Removes redundant road segments by computing pairwise distances,
    clustering similar segments, and merging them where necessary.
    
    Returns:
        selected_segments: The pruned or merged set of segments.
    '''
    distances = []
    grouped_segments = []
    for k in range(len(segments)):
        seg1 = segments[k]
        seg_matched, indices, side = find_matched_segment(seg1, segments)
        if seg_matched is not None:
            for seg, idx in zip(seg_matched, indices):
                dis = distance_between_segments(seg1, seg)
                distances.append(dis)
                grouped_segments.append([k, idx])
    distances = np.array(distances)
    grouped_segments = np.array(grouped_segments)
    if len(distances) == 0:
        warnings.warn('No distances calculated; returning original segments.')
        return segments

    # Define threshold using a heuristic (max distance / 10)
    threshold = distances.max() / 10
    redundant_pairs = grouped_segments[distances < threshold]

    if len(redundant_pairs) == 0:
        warnings.warn('No redundant segments found.')
        return segments

    clusters = cluster_related_features(redundant_pairs)
    selected_segments_list = []
    for c in clusters:
        c = list(c)
        if len(c) == 1:
            selected_segments_list.append(segments[c[0]].reshape(-1, 4))
        elif len(c) == 2:
            merged_segment = merge_redundant_segments([[segments[c[0]], segments[c[1]]]])
            selected_segments_list.append(merged_segment.reshape(-1, 4))
        elif len(c) > 2:
            G = nx.Graph()
            for g in redundant_pairs:
                if (g[0] in c) and (g[1] in c):
                    G.add_edge(g[0], g[1])
            if nx.is_bipartite(G):
                set1, set2 = nx.bipartite.sets(G)
                if len(set1) < len(set2):
                    set1, set2 = set2, set1
                for idx in set1:
                    selected_segments_list.append(segments[idx].reshape(-1, 4))
            else:
                # Fallback: simply include all segments from the cluster.
                for idx in c:
                    selected_segments_list.append(segments[idx].reshape(-1, 4))

    selected_segments = np.concatenate(selected_segments_list, axis=0)
    return selected_segments


def form_boundary_pairs(selected_segments):
    '''
    Forms matched pairs from selected segments by re-calculating matching boundaries.
    
    Returns:
        matched_boundaries: Raw matched pairs (indices) among the selected segments.
        final_pairs: A list of pairs (or groups) where each entry includes
                     the indices of segments in the pair and the corresponding merged segments.
    '''
    matched_boundaries = []
    pair_distances = []
    num_selected = len(selected_segments)
    for k in range(num_selected):
        seg1 = selected_segments[k]
        seg_matched, indices, side = find_matched_segment(seg1, selected_segments)
        if seg_matched is not None:
            for seg, idx in zip(seg_matched, indices):
                dis = distance_between_segments(seg1, seg)
                pair_distances.append(dis)
                matched_boundaries.append([k, idx])
    
    final_pairs = []
    G = nx.Graph()
    G.add_edges_from(matched_boundaries)
    components = list(nx.connected_components(G))
    for subg in components:
        subg_list = list(subg)
        merged_view = selected_segments[subg_list].reshape(-1, 4)
        final_pairs.append([subg_list, merged_view])
        if len(subg_list) != 2:
            warnings.warn('Found road boundary pair with a number of segments other than 2.')
    
    return matched_boundaries, final_pairs


def refine_boundary_pairs(selected_segments, matched_boundaries, final_pairs):
    '''
    Refines boundary pairs by detecting and splitting cases where the pair does not have exactly 2 segments.
    
    Returns:
        paired_road_boundary: The final road boundary pairs after applying splitting logic.
        paired_road_boundary_indices: The indices associated with these final pairs.
    '''
    paired_road_boundary_indices = []
    paired_road_boundary = []
    num = len(selected_segments)
    
    for pair in final_pairs:
        indices, merged_segment = pair[0], pair[1]
        if len(indices) != 2:
            warnings.warn('Pair with non-2 segments detected. Attempting to split a shared segment.')
            subG = nx.Graph()
            for m in matched_boundaries:
                if m[0] in indices and m[1] in indices:
                    subG.add_edge(m[0], m[1])
            if nx.is_bipartite(subG):
                set1, set2 = nx.bipartite.sets(subG)
                if min(len(set1), len(set2)) == 1 and max(len(set1), len(set2)) == 2:
                    if len(set1) == 1:
                        to_split_idx = list(set1)[0]
                        adjacent_indices = list(set2)
                    else:
                        to_split_idx = list(set2)[0]
                        adjacent_indices = list(set1)
                    
                    seg_to_split = selected_segments[to_split_idx]
                    aud0 = selected_segments[adjacent_indices[0]]
                    aud1 = selected_segments[adjacent_indices[1]]
                    pdb.set_trace()
                    seg1, seg2 = split_segment(seg_to_split, aud0, aud1)
                    
                    new_id1 = num + 1
                    new_id2 = num + 2
                    paired_road_boundary_indices.append([new_id1, adjacent_indices[0]])
                    paired_road_boundary.append([seg1, selected_segments[adjacent_indices[0]]])
                    paired_road_boundary_indices.append([new_id2, adjacent_indices[1]])
                    paired_road_boundary.append([seg2, selected_segments[adjacent_indices[1]]])
                    num += 2
                else:
                    paired_road_boundary_indices.append(indices)
                    paired_road_boundary.append(merged_segment)
            elif len(indices) == 4:
                midx = merged_segment[:, 0] + merged_segment[:, 2]
                midy = merged_segment[:, 1] + merged_segment[:, 3]
                mids = np.column_stack((midx, midy))
                # find their distance
                distances = np.linalg.norm(np.reshape(mids, (-1,1,2)) - np.reshape(mids,(1,-1,2)), axis=2)
                min_dist = [1e9, 1e9]
                min_idx = [[-1, -1], [-1, -1]]
                for i in range(len(distances)):
                    for j in range(i+1, len(distances)):
                        if distances[i][j] < min_dist[0]:
                            min_dist[0] = distances[i][j]
                            min_idx[0] = [i, j]
                        elif distances[i][j] < min_dist[1]:
                            min_dist[1] = distances[i][j]
                            min_idx[1] = [i, j]
                # remove the redundant segments
                seg1 = merged_segment[min_idx[0][0]]
                seg2 = merged_segment[min_idx[1][0]]
                new_pair = [indices[min_idx[0][0]], indices[min_idx[1][0]]]
                paired_road_boundary_indices.append(new_pair)
                paired_road_boundary.append([seg1, seg2])
            else:
                # remove redundant segments
                paired_road_boundary_indices.append(indices)
                paired_road_boundary.append(merged_segment)
        else:
            paired_road_boundary_indices.append(indices)
            paired_road_boundary.append(merged_segment)
    
        # print('pair: ', pair)
        # pdb.set_trace()
    return paired_road_boundary, paired_road_boundary_indices


def find_road_boundary_pairs(selected_segments):
    matched_boundaries, final_pairs = form_boundary_pairs(selected_segments)
    paired_road_boundary, paired_road_boundary_indices = refine_boundary_pairs(selected_segments, matched_boundaries, final_pairs)

    return paired_road_boundary, [tuple(pair) for pair in paired_road_boundary_indices]


def find_inner_edge_between_boundaries(boundary_pair, candidate_edges, candidate_indices, 
                                         angle_thresh=np.deg2rad(10), tol=1.0):
    """
    Given a pair of boundary segments (each as [x1, y1, x2, y2]) and candidate inner edge segments,
    return those edges whose orientation and location place them between the boundaries.
    
    Parameters:
        boundary_pair: tuple with two boundaries, e.g. (boundary_left, boundary_right).
        candidate_edges: np.array of shape (M, 4), where each row is [x1, y1, x2, y2] for candidate edges.
        candidate_indices: list of indices corresponding to candidate_edges.
        angle_thresh: allowed angular difference (in radians) compared to the average boundary orientation.
        tol: tolerance used when testing distances.
        
    Returns:
        filtered_edges: np.array containing only the candidate edges that lie between the boundaries.
        filtered_indices: list of indices corresponding to the filtered edges.
    """
    # Unpack boundaries
    b1, b2 = boundary_pair
    
    # Compute midpoints
    mid_b1 = np.array([(b1[0] + b1[2]) / 2, (b1[1] + b1[3]) / 2])
    mid_b2 = np.array([(b2[0] + b2[2]) / 2, (b2[1] + b2[3]) / 2])
    
    # Compute orientation angles (in [0, pi) ) for each boundary.
    angle_b1 = np.arctan2(b1[3] - b1[1], b1[2] - b1[0]) % np.pi
    angle_b2 = np.arctan2(b2[3] - b2[1], b2[2] - b2[0]) % np.pi
    avg_angle = (angle_b1 + angle_b2) / 2

    # Compute inward normals (rotate each boundary’s vector by +pi/2)
    normal_b1 = np.array([np.cos(angle_b1 + np.pi/2), np.sin(angle_b1 + np.pi/2)])
    normal_b2 = np.array([np.cos(angle_b2 + np.pi/2), np.sin(angle_b2 + np.pi/2)])
    
    # Prepare lists for valid inner edge candidates.
    valid_edges = []
    valid_indices = []
    
    # Process each candidate inner edge.
    for idx, edge in zip(candidate_indices, candidate_edges):
        # Calculate the midpoint of the candidate edge.
        mid_edge = np.array([(edge[0] + edge[2]) / 2, (edge[1] + edge[3]) / 2])
        
        # Check orientation: the candidate edge should be approximately aligned with the average orientation.
        angle_edge = np.arctan2(edge[3] - edge[1], edge[2] - edge[0]) % np.pi
        if np.abs(angle_edge - avg_angle) > angle_thresh:
            continue
        
        # Compute the signed distances from the candidate edge midpoint to each boundary.
        d1 = np.dot(mid_edge - mid_b1, normal_b1)
        d2 = np.dot(mid_edge - mid_b2, normal_b2)
        
        # For an edge truly between the boundaries, the distances should be of opposite sign (or near zero).
        if d1 * d2 < 0 or (np.abs(d1) < tol and np.abs(d2) < tol):
            valid_edges.append(edge)
            valid_indices.append(idx)
    
    return np.array(valid_edges), valid_indices


def match_road_boundary_and_edge(road_boundaries, road_boundary_indices, road_edges, 
                                 angle_thresh=np.deg2rad(10), strict=True):
    """
    For each pair of road boundaries (each provided with an associated index pair),
    search for candidate road edge segments from road_edges that could lie between them.
    The candidates are first identified using find_matched_segment and then filtered
    with find_inner_edge_between_boundaries.
    
    Parameters:
        road_boundaries: list (or np.array) where each element is a pair (tuple or list) 
                         of two boundary segments ([x1, y1, x2, y2]).
        road_boundary_indices: list of tuples (or lists) corresponding to indices for each boundary pair.
        road_edges: np.array of candidate road edge segments, each defined as [x1, y1, x2, y2].
        angle_thresh: angular threshold (in radians) for candidate matching.
        strict: flag passed along to find_matched_segment determining matching criteria.
    
    Returns:
        matched_inner_edges_list: list of np.arrays, each containing the inner edge segments for a boundary pair.
        boundary_to_edge_map: dictionary mapping a boundary pair (tuple of indices) to a list of matched edge indices.
        edge_to_boundary_map: dictionary mapping an edge index to a list of boundary pair indices where it has been matched.
    """
    matched_inner_edges_list = []
    boundary_to_edge_map = {}
    edge_to_boundary_map = {}
    
    # Process each boundary pair.
    for boundary_pair, indices in zip(road_boundaries, road_boundary_indices):
        seg1, seg2 = boundary_pair
        idx1, idx2 = indices
        
        # Get candidate inner edges for each boundary separately.
        candidates1, indices1, sides = find_matched_segment(seg1, road_edges, angle_thresh=angle_thresh, strict=strict)
        candidates2, indices2, sides = find_matched_segment(seg2, road_edges, angle_thresh=angle_thresh, strict=strict)
        
        # Combine candidates from both boundaries if available.
        if (candidates1 is None or len(candidates1) == 0) and (candidates2 is None or len(candidates2) == 0):
            warnings.warn(f'No candidate inner edges found for boundary pair {(idx1, idx2)}.')
            continue
        
        if candidates1 is None:
            candidates_combined = candidates2
            candidate_indices = indices2
        elif candidates2 is None:
            candidates_combined = candidates1
            candidate_indices = indices1
        else:
            # Concatenate candidate arrays if both sets are available.
            candidates_combined = np.concatenate([candidates1, candidates2], axis=0)
            candidate_indices = indices1 + indices2
        
        # Now filter out the candidates to only the inner edges that lie between the boundaries.
        # filtered_edges, filtered_indices = find_inner_edge_between_boundaries(boundary_pair, 
        #                                                                       candidates_combined, 
        #                                                                       candidate_indices, 
        #                                                                       angle_thresh=angle_thresh)
        matched_inner_edges_list.append(candidate_indices) # filtered_edges
        
        # Build mapping from boundary pair to inner edge indices.
        boundary_to_edge_map[(idx1, idx2)] = candidate_indices # filtered_edges
        
        # Build reverse mapping for each edge.
        for edge_idx in candidate_indices: #filtered_indices:
            if edge_idx not in edge_to_boundary_map:
                edge_to_boundary_map[edge_idx] = []
            edge_to_boundary_map[edge_idx].append((idx1, idx2))
    
    return matched_inner_edges_list, boundary_to_edge_map, edge_to_boundary_map


import numpy as np

def find_inner_edge_between_boundaries_relaxed(boundary_pair, candidate_edges, candidate_indices, tol=1.0):
    """
    Given a pair of road boundaries (each defined as [x1, y1, x2, y2])
    and a set of candidate inner edge segments (which may include lines or curves),
    return those candidate edges whose midpoints lie between the two boundaries.

    The method computes the line (infinite) defined by each boundary (using its endpoints)
    and obtains a unit normal from its orientation. Then, for each candidate edge, the midpoint is
    computed and its signed distances (via the normal) to both boundaries are calculated.
    If these two distances have opposite signs or one is near zero (within tol), the edge is considered
    as an inner edge between the boundaries.

    Parameters:
        boundary_pair: tuple or list with two boundaries, e.g. (boundary_left, boundary_right),
                       where each boundary is given as [x1, y1, x2, y2].
        candidate_edges: np.array of shape (M,4) with candidate inner edge segments.
        candidate_indices: list of indices corresponding to candidate_edges.
        tol: tolerance for considering a point effectively on a boundary.

    Returns:
        filtered_edges: np.array with the candidate edges that lie between the boundaries.
        filtered_indices: list of candidate edge indices that pass the test.
    """
    # Unpack boundaries.
    b1, b2 = boundary_pair

    # Compute midpoints for each boundary.
    mid_b1 = np.array([(b1[0] + b1[2]) / 2, (b1[1] + b1[3]) / 2])
    mid_b2 = np.array([(b2[0] + b2[2]) / 2, (b2[1] + b2[3]) / 2])
    
    # Compute orientation angles of each boundary.
    angle_b1 = np.arctan2(b1[3] - b1[1], b1[2] - b1[0])
    angle_b2 = np.arctan2(b2[3] - b2[1], b2[2] - b2[0])
    
    # Compute unit normals. (These normals should ideally point toward the interior of the road.)
    # Depending on how boundaries are defined, these might need to be flipped.
    normal_b1 = np.array([np.cos(angle_b1 + np.pi/2), np.sin(angle_b1 + np.pi/2)])
    normal_b2 = np.array([np.cos(angle_b2 + np.pi/2), np.sin(angle_b2 + np.pi/2)])
    
    valid_edges = []
    valid_indices = []
    
    # Process each candidate inner edge.
    for idx, edge in zip(candidate_indices, candidate_edges):
        # Compute the midpoint of the candidate edge.
        mid_edge = np.array([(edge[0] + edge[2]) / 2, (edge[1] + edge[3]) / 2])
        
        # Calculate signed distances from mid_edge to each boundary line.
        # (Projection of the vector from boundary midpoint to edge midpoint along the boundary's normal.)
        d1 = np.dot(mid_edge - mid_b1, normal_b1)
        d2 = np.dot(mid_edge - mid_b2, normal_b2)
        
        # If the distances have opposite signs or one is near zero, then the candidate lies between.
        if d1 * d2 < 0 or (np.abs(d1) < tol or np.abs(d2) < tol):
            valid_edges.append(edge)
            valid_indices.append(idx)
    
    return np.array(valid_edges), valid_indices



def match_road_boundary_and_edge_relaxed(road_boundaries, road_boundary_indices, road_edges, tol=1.0):
    """
    For each pair of road boundaries (each specified as a pair of segments [x1,y1,x2,y2]),
    find all candidate inner edges from road_edges that lie between the boundaries.
    
    This function uses a relaxed matching criterion: For each candidate inner edge (from the entire
    collection of road_edges), it computes the midpoint and then the signed distances to both boundary lines.
    An edge is accepted if its midpoint is positioned between the two boundaries based on their signed distances.
    
    Parameters:
        road_boundaries: list (or array) of boundary pairs, where each element is (boundary1, boundary2).
        road_boundary_indices: list of index pairs corresponding to each boundary pair.
        road_edges: np.array of candidate road edge segments, each defined as [x1, y1, x2, y2].
        tol: tolerance for the distance tests (if a distance is within tol of zero, treat it as lying on the boundary).
    
    Returns:
        matched_inner_edges_list: list of np.arrays, each corresponding to the inner edge segments matched for a boundary pair.
        boundary_to_edge_map: dictionary mapping a boundary pair (tuple of indices) to a list of matched edge indices.
        edge_to_boundary_map: dictionary mapping an inner edge candidate index to a list of boundary pair indices where it was matched.
    """
    matched_inner_edges_list = []
    boundary_to_edge_map = {}
    edge_to_boundary_map = {}
    
    # For candidate edge indices we simply use range(len(road_edges)).
    candidate_indices = list(range(len(road_edges)))
    
    # Process each boundary pair.
    for boundary_pair, indices in zip(road_boundaries, road_boundary_indices):
        idx1, idx2 = indices

        # For each boundary pair, use ALL candidate inner edges.
        filtered_edges, filtered_indices = find_inner_edge_between_boundaries_relaxed(
            boundary_pair, road_edges, candidate_indices, tol=tol
        )
        
        if filtered_edges.shape[0] == 0:
            warnings.warn(f'No inner edge candidates found for boundary pair {(idx1, idx2)}.')
        
        matched_inner_edges_list.append(filtered_edges)
        boundary_to_edge_map[(idx1, idx2)] = filtered_indices
        
        for edge_idx in filtered_indices:
            if edge_idx not in edge_to_boundary_map:
                edge_to_boundary_map[edge_idx] = []
            edge_to_boundary_map[edge_idx].append((idx1, idx2))
    
    return matched_inner_edges_list, boundary_to_edge_map, edge_to_boundary_map


#################################################################################
#################################################################################
# determine whether inner road edge segments are inside the area bounded by two nearly parallel segments
#################################################################################
#################################################################################


def is_segment_in_bounded_area(boundary_seg1, boundary_seg2, test_segments):
    """
    Check if any of the test segments pass through or lie inside the area bounded by two nearly parallel segments.
    
    Parameters:
    boundary_seg1: tuple of two points ((x1, y1), (x2, y2)) defining the first boundary segment
    boundary_seg2: tuple of two points ((x1, y1), (x2, y2)) defining the second boundary segment
    test_segments: list of segments [((x1, y1), (x2, y2)), ...] to check
    
    Returns:
    List of booleans indicating whether each test segment is inside the bounded area
    """
    # Extract points from boundary segments
    p1, p2 = boundary_seg1[:2], boundary_seg1[2:]
    p3, p4 = boundary_seg2[:2], boundary_seg2[2:]
    
    # Create the quadrilateral from the boundary segments
    quad_points = [p1, p2, p4, p3]  # Order matters for determining "inside"
    
    results = []
    for k in range(len(test_segments)):
        segment = test_segments[k]
        q1 = segment[:2]
        q2 = segment[2:]
        # print(q1, q2, quad_points)
        # Check if either endpoint of the segment is inside the quadrilateral
        endpoint1_inside = is_point_in_polygon(q1, quad_points)
        endpoint2_inside = is_point_in_polygon(q2, quad_points)
        
        # Check if the segment intersects with any of the quadrilateral sides
        intersects = False
        for i in range(4):
            side = (quad_points[i], quad_points[(i+1)%4])
            if segments_intersect(segment, side):
                intersects = True
                break
        
        # The segment is inside the bounded area if either endpoint is inside
        # or if it intersects with any side of the quadrilateral
        results.append(endpoint1_inside or endpoint2_inside or intersects)
    
    return results

def is_point_in_polygon(point, polygon_points):
    """
    Check if a point is inside a polygon using the ray casting algorithm.
    
    Parameters:
    point: (x, y) coordinate of the point to check
    polygon_points: list of (x, y) coordinates forming the polygon
    
    Returns:
    Boolean indicating whether the point is inside the polygon
    """
    x, y = point
    n = len(polygon_points)
    inside = False
    
    p1x, p1y = polygon_points[0]
    for i in range(n + 1):
        p2x, p2y = polygon_points[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        x_intersect = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= x_intersect:
                        inside = not inside
        p1x, p1y = p2x, p2y
    
    return inside

def cross_product(p1, p2, p3):
    """
    Calculate the cross product (p2 - p1) × (p3 - p1)
    Used to determine if points are clockwise/counterclockwise
    """
    return (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])

def segments_intersect(seg1, seg2):
    """
    Check if two line segments intersect
    
    Parameters:
    seg1: ((x1, y1), (x2, y2)) first line segment
    seg2: ((x1, y1), (x2, y2)) second line segment
    
    Returns:
    Boolean indicating whether the segments intersect
    """
    p1, p2 = seg1[:2], seg1[2:]
    p3, p4 = seg2
    
    # print(p1, p2, p3, p4)
    # Check if the segments share an endpoint
    if ((p1-p3)**2).sum() == 0 or ((p1-p4)**2).sum() == 0 or ((p2-p3)**2).sum() == 0 or ((p2-p4)**2).sum() == 0:
        return True
    
    # Check if line segments cross each other
    d1 = cross_product(p3, p4, p1)
    d2 = cross_product(p3, p4, p2)
    d3 = cross_product(p1, p2, p3)
    d4 = cross_product(p1, p2, p4)
    
    # If the signs of d1 and d2 are different and the signs of d3 and d4 are different,
    # then the segments intersect
    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True
    
    # Check for colinearity and overlap
    if d1 == 0 and is_point_on_segment(p1, seg2):
        return True
    if d2 == 0 and is_point_on_segment(p2, seg2):
        return True
    if d3 == 0 and is_point_on_segment(p3, seg1):
        return True
    if d4 == 0 and is_point_on_segment(p4, seg1):
        return True
    
    return False

def is_point_on_segment(point, segment):
    """
    Check if a point lies on a line segment
    
    Parameters:
    point: (x, y) coordinate of the point
    segment: ((x1, y1), (x2, y2)) line segment
    
    Returns:
    Boolean indicating whether the point lies on the segment
    """
    (p1, p2) = segment
    
    # Check if the point is collinear with the segment endpoints
    if cross_product(p1, p2, point) != 0:
        return False
    
    # Check if the point lies within the bounding box of the segment
    if (min(p1[0], p2[0]) <= point[0] <= max(p1[0], p2[0]) and
        min(p1[1], p2[1]) <= point[1] <= max(p1[1], p2[1])):
        return True
    
    return False



def match_road_boundary_and_edge_relaxed(road_boundaries, road_boundary_indices, road_edges):
    matched_segments = []
    matched_indices = {}
    reversed_matched_indices = {}
    for pair in road_boundary_indices:
        seg1 = road_boundaries[pair[0]]
        seg2 = road_boundaries[pair[1]]
        flags = is_segment_in_bounded_area(seg1, seg2, road_edges)
        for k, flag in enumerate(flags):
            if flag:
                matched_segments.append(road_edges[k])
                if tuple(pair) not in matched_indices:
                    matched_indices[tuple(pair)] = []
                matched_indices[tuple(pair)].append(k)
                if k not in reversed_matched_indices:
                    reversed_matched_indices[k] = []
                reversed_matched_indices[k].append(pair)
    
    # reorder the matched inner indices of road edges such that the first segment is the closest to the first boundary
    for k in matched_indices.keys():
        road_boundary = road_boundaries[k[0]]
        matched_indices[k] = sorted(matched_indices[k], key=lambda x: distance_between_segments(road_edges[x], road_boundary))
        # print(k, matched_indices[k])
    
    return matched_segments, matched_indices, reversed_matched_indices


def match_road_boundary_and_traj(road_boundaries, road_boundary_indices, traj_collection=None, keypoint_collection=None):
    segment_matched_indices = {}
    keypoint_matched_indices = {}
    reversed_segment_matched_indices = {}
    reversed_keypoint_matched_indices = {}

    if traj_collection is not None:
        # match traj segments to the road boundaries
        for pair in road_boundary_indices:
            seg1 = road_boundaries[pair[0]]
            seg2 = road_boundaries[pair[1]]
            traj_index = 0
            for traj_index in range(len(traj_collection)):
                traj = traj_collection[traj_index]
                traj_segments = traj['segment']
                traj_coordinates = traj['coordinate']
                
                flags = is_segment_in_bounded_area(seg1, seg2, traj_segments)
                for k, flag in enumerate(flags):
                    if flag:
                        if tuple(pair) not in segment_matched_indices:
                            segment_matched_indices[tuple(pair)] = []
                        segment_matched_indices[tuple(pair)].append((traj_index, k))
                        if (traj_index, k) not in reversed_segment_matched_indices:
                            reversed_segment_matched_indices[(traj_index, k)] = []
                        reversed_segment_matched_indices[(traj_index, k)].append(pair)
        
        # reorder the matched segment such that the first segment is the closest to the first boundary
        for k in segment_matched_indices.keys():
            road_boundary = road_boundaries[k[0]]
            segment_matched_indices[k] = sorted(segment_matched_indices[k], key=lambda x: distance_between_segments(traj_collection[x[0]]['segment'][x[1]], road_boundary))
    
    if keypoint_collection is not None:
        # match traj keypoints to the road boundaries
        for pair in road_boundary_indices:
            seg1 = road_boundaries[pair[0]]
            seg2 = road_boundaries[pair[1]]
            polygon_points = [seg1[:2], seg1[2:], seg2[2:], seg2[:2]]
            traj_index = 0
            for kp_index in range(len(keypoint_collection)):
                kp = keypoint_collection[traj_index]
                kp_coordinates = kp['coordinate']
                for k, point in enumerate(kp_coordinates):
                    flag = is_point_in_polygon(point, polygon_points)
                    if flag:
                        if tuple(pair) not in keypoint_matched_indices:
                            keypoint_matched_indices[tuple(pair)] = []
                        keypoint_matched_indices[tuple(pair)].append([traj_index, k])
                        if (traj_index, k) not in reversed_keypoint_matched_indices:
                            reversed_keypoint_matched_indices[(traj_index, k)] = []
                        reversed_keypoint_matched_indices[(traj_index, k)].append(pair)

    return segment_matched_indices, keypoint_matched_indices, reversed_segment_matched_indices, reversed_keypoint_matched_indices