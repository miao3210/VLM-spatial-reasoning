import pdb 
import warnings
import xml.etree.ElementTree as ET
import numpy as np
from scipy.integrate import quad


def load_xodr(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    return root


# Function to calculate lane width at a given sOffset using the width polynomial
def lane_width(a, b, c, d, sOffset):
    return a + b * sOffset + c * sOffset**2 + d * sOffset**3


# Function to calculate the arc length of a lane based on lane offset
def calculate_lane_length(road_length, centerline_radius, lane_offset):
    # Approximate lane length as a ratio of the lane's radius to the centerline radius
    lane_length = 0
    for road_length_individual, centerline_radius_individual in zip(road_length[1:], centerline_radius):
        lane_length += road_length_individual * (1 - lane_offset * centerline_radius_individual)
    return lane_length


def get_road_segments_and_lanes(root):
    road_segments = []
    lanes = {}
    for road in root.findall('.//road'):
        road_id = road.attrib['id']
        for geometry in road.findall('.//geometry'):
            start_position = (float(geometry.attrib['x']), float(geometry.attrib['y']))
            hdg = float(geometry.attrib['hdg'])  # Heading angle in radians
            length = float(geometry.attrib['length'])
            road_segments.append((road_id, start_position, hdg, length))
        lanes[road_id] = []
        for lane in road.findall('.//lane'):
            lane_id = lane.attrib['id']
            lane_type = lane.attrib['type']
            lane_level = lane.attrib['level']
            lanes[road_id].append((lane_id, lane_type, lane_level))
    return road_segments, lanes


def find_closest_semantic_orientation_based_on_absolute_orientation(orient_value):
    ### this function should be used to find the road orientation based on the absolute orientation value pointing from the endpoint to the intersection
    # coarse description: 8 directions
    central_orientation_8 = np.arange(-np.pi*2, np.pi*2+0.1, np.pi/4)
    semantic_desc_8 = ['east', 'northeast', 'north', 'northwest', 'west', 'southwest', 'south', 'southeast', 'east', 'northeast', 'north', 'northwest', 'west', 'southwest', 'south', 'southeast', 'east']
    dif_8 = np.abs(central_orientation_8 - orient_value)
    min_dif_index_8 = np.argmin(dif_8)

    # fine description: 4 directions but with numerical values
    central_orientation_4 = np.arange(-np.pi*2, np.pi*2+0.1, np.pi/2)
    semantic_desc_4 = ['east', 'north', 'west', 'south', 'east', 'north', 'west', 'south', 'east']
    dif_4 = np.abs(central_orientation_4 - orient_value)
    min_dif_index = np.argmin(dif_4)
    if min(dif_4) == 0:
        combination = [semantic_desc_4[min_dif_index], 1, None, None]
    else:
        second_min_dif_index = np.argsort(dif_4)[1]
        percentage = dif_4[min_dif_index] / (dif_4[min_dif_index] + dif_4[second_min_dif_index])
        combination = [semantic_desc_4[min_dif_index], 1-percentage//0.01*0.01, semantic_desc_4[second_min_dif_index], percentage//0.01*0.01]
    return semantic_desc_8[min_dif_index_8], combination


def xodr_to_dict(root):
    xodr_dict = {}
    xodr_dict['roads'] = {}
    xodr_dict['linking_roads'] = {}
    xodr_dict['connections'] = {}
    xodr_dict['direct_connections'] = {}
    
    for road in root.findall('.//road'):
        road_dict = {}
        road_id = road.attrib['id']
        road_dict['geometries'] = []
        road_length = [float(road.get('length'))]
        centerline_radius = []
        for geometry in road.findall('.//geometry'):
            geometry_dict = {}
            if geometry is not None:
                geometry_dict['s'] = float(geometry.attrib['s'])
                geometry_dict['x'] = float(geometry.attrib['x'])
                geometry_dict['y'] = float(geometry.attrib['y'])
                geometry_dict['hdg'] = float(geometry.attrib['hdg'])
                geometry_dict['length'] = float(geometry.attrib['length'])
                # Extract geometry (assuming spiral for curvature) info for length calculation TODO: add support for the arc case
                road_type = geometry[0].tag  # 'spiral' or other geometry types
                if road_type == 'spiral':
                    spiral = geometry.find('spiral')
                    curvStart = float(spiral.get('curvStart'))
                    curvEnd = float(spiral.get('curvEnd'))
                    # add curvature info to the geometry dict
                    geometry_dict['curvStart'] = curvStart
                    geometry_dict['curvEnd'] = curvEnd
                    # Approximate the centerline radius as the inverse of curvature (average curvature for simplicity)
                    centerline_radius.append( (curvStart + curvEnd) / 2 )
                road_length.append( geometry_dict['length'] )
            road_dict['geometries'].append(geometry_dict)
        predecessor = road.findall('link/predecessor')
        if predecessor is not None:
            road_dict['predecessor'] = []
            for p in predecessor:
                if p.get('elementType') == 'road':
                    road_dict['predecessor'].append({'road_id': p.get('elementId'), 'contact_point': p.get('contactPoint')})
                elif p.get('elementType') == 'junction':
                    road_dict['predecessor'].append({'junction_id': p.get('elementId')})
        successor = road.findall('link/successor')
        if successor is not None:
            road_dict['successor'] = []
            for s in successor:
                if s.get('elementType') == 'road':
                    road_dict['successor'].append({'road_id': s.get('elementId'), 'contact_point': s.get('contactPoint')})
                elif s.get('elementType') == 'junction':
                    road_dict['successor'].append({'junction_id': s.get('elementId')})
        road_dict['length'] = road_length
        
        straight_link_flag = True if len(centerline_radius) == 0 else False
        
        road_dict['lanes'] = {'left': {}, 'right': {}, 'center': {}}
        for lane in road.findall('.//lane'):
            lane_id = lane.attrib['id']
            lane_dict = {}
            lane_dict['type'] = lane.attrib['type']
            lane_dict['level'] = lane.attrib['level']
            lane_dict['width'] = lane.find('.//width')
            if lane_dict['width'] is not None:
                lane_dict['width'] = float(lane_dict['width'].attrib['a'])
            #lane_dict['sOffset'] = float(lane.attrib['sOffset'])
            # calculate the length of the lane
            if straight_link_flag:
                lane_length = road_length[0]
            else:
                width_elem = lane.find('width')
                if width_elem is not None:
                    # Get polynomial coefficients for lane width
                    a = float(width_elem.get('a'))
                    b = float(width_elem.get('b'))
                    c = float(width_elem.get('c'))
                    d = float(width_elem.get('d'))
                    # Integrate the width polynomial to get the average lane offset
                    lane_offset, _ = quad(lambda s: lane_width(a, b, c, d, s), 0, road_length[0])
                    lane_offset /= road_length[0]  # Average offset over the length of the road
                    # Calculate the lane length based on the lane offset
                    lane_length = calculate_lane_length(road_length, centerline_radius, lane_offset * (int(lane_id) - 0.5 * np.sign(int(lane_id))))
                else:
                    warnings.warn(f'No width element found for lane {lane_id}. Cannot calculate lane length. Use the road length instead.')
                    lane_length = road_length[0]
            lane_dict['length'] = lane_length
            predecessor_lane = lane.find('link/predecessor')
            if predecessor_lane is not None:
                predecessor_lane_id = predecessor_lane.get('id')
                lane_dict['predecessor_lane_id'] = predecessor_lane_id
            successor_lane = lane.find('link/successor')
            if successor_lane is not None:
                successor_lane_id = successor_lane.get('id')
                lane_dict['successor_lane_id'] = successor_lane_id
            if lane_id[0] == '-':
                road_dict['lanes']['right'][lane_id] = lane_dict
            elif lane_id[0] == '0':
                road_dict['lanes']['center'][lane_id] = lane_dict
            else:
                road_dict['lanes']['left'][lane_id] = lane_dict
        if len(road_id) > 1:
            road_dict['connecting_road'] = []
        if len(road_id) == 1:
            xodr_dict['roads'][road_id] = road_dict
        elif len(road_id) == 3 or len(road_id) == 4:
            for lane_section in road.findall('lanes/laneSection'):
                sOffset = float(lane_section.get('s'))  # starting position of lane section
                for lane in lane_section.findall('left/lane') + lane_section.findall('right/lane'):
                    lane_id = lane.get('id')        
            xodr_dict['linking_roads'][road_id] = road_dict
        else:
            warnings.warn(f'Found unexpected road id {road_id}')
            pdb.set_trace()

    # add connections to the linking roads
    for junction in root.findall('junction'):
        junction_id = junction.get('id')
        #print(f"Junction {junction_id}")
        for connection in junction.findall('connection'):
            incoming_road = connection.get('incomingRoad')
            connecting_road = connection.get('connectingRoad')
            contact_point = connection.get('contactPoint')
            if connecting_road is not None and connecting_road not in xodr_dict['linking_roads']:
                warnings.warn(f'Connecting road {connecting_road} not found in linking roads.')
                pdb.set_trace()
            if connecting_road:
                if 'connecting_road' not in xodr_dict['linking_roads'][connecting_road]:
                    xodr_dict['linking_roads'][connecting_road]['connecting_road'] = []
                xodr_dict['linking_roads'][connecting_road]['connecting_road'].append( {'road_id': incoming_road, 'contact_point': contact_point} )
            else:
                linked_road = connection.get('linkedRoad')
                if not linked_road:
                    warnings.warn(f'No connecting road or direct linked road found for connection {incoming_road} -> {connecting_road} in junction {junction_id}.')
                else:
                    linked_road_contact_point = contact_point
                    incoming_road_contact_point = None
                    for pre in xodr_dict['roads'][incoming_road]['predecessor']:
                        if 'junction_id' in pre.keys():
                            if pre['junction_id'] == junction_id:
                                incoming_road_contact_point = 'start'
                                break
                    for suc in xodr_dict['roads'][incoming_road]['successor']:
                        if 'junction_id' in suc.keys():
                            if suc['junction_id'] == junction_id:
                                incoming_road_contact_point = 'end'
                                break
                    connection_id = connection.get('id')
                    predecessor_to_successor_connection_key = str(incoming_road) + ' to ' + str(linked_road)
                    xodr_dict['direct_connections'][predecessor_to_successor_connection_key] = {'lanes': {}, 'contact_point': [incoming_road_contact_point, linked_road_contact_point]}
                    successor_to_predecessor_connection_key = str(linked_road) + ' back to ' + str(incoming_road)
                    xodr_dict['direct_connections'][successor_to_predecessor_connection_key] = {'lanes': {}, 'contact_point': [linked_road_contact_point, incoming_road_contact_point]}
                    for lane_link in connection.findall('laneLink'):
                        predecessor_lane = lane_link.get('from')
                        successor_lane = lane_link.get('to')
                        xodr_dict['direct_connections'][predecessor_to_successor_connection_key]['lanes'][predecessor_lane] = successor_lane
                        xodr_dict['direct_connections'][successor_to_predecessor_connection_key]['lanes'][successor_lane] = predecessor_lane
    
    # summarize the length of the linking roads to the connection dictionary w.r.t. the predecessor and successor
    for road_id, road in xodr_dict['linking_roads'].items():
        predecessor_to_successor = {}
        successor_to_predecessor = {}
        predecessor_to_successor = {
            'road_id': road_id,
            'lane_id': {},
            'length': {'0': road['length']}
        }
        successor_to_predecessor = {
            'road_id': road_id,
            'lane_id': {},
            'length': {'0': road['length']}
        }
        # calculate the length of each lane on the link road
        for lane_id, lane in road['lanes']['left'].items():
            lane_length = lane['length']
            predecessor_lane_id = lane['predecessor_lane_id']
            successor_lane_id = lane['successor_lane_id']
            predecessor_to_successor['lane_id'][str(predecessor_lane_id) + ' to ' + str(successor_lane_id)] = lane_id
            successor_to_predecessor['lane_id'][str(successor_lane_id) + ' to ' + str(predecessor_lane_id)] = lane_id
            # add length of different lanes on the linking roads
            predecessor_to_successor['length'][str(predecessor_lane_id) + ' to ' + str(successor_lane_id)] = lane_length
            successor_to_predecessor['length'][str(successor_lane_id) + ' to ' + str(predecessor_lane_id)] = lane_length
        for lane_id, lane in road['lanes']['right'].items():
            lane_length = lane['length']
            predecessor_lane_id = lane['predecessor_lane_id']
            successor_lane_id = lane['successor_lane_id']
            predecessor_to_successor['lane_id'][str(predecessor_lane_id) + ' to ' + str(successor_lane_id)] = lane_id
            successor_to_predecessor['lane_id'][str(successor_lane_id) + ' to ' + str(predecessor_lane_id)] = lane_id
            # add length of different lanes on the linking roads
            predecessor_to_successor['length'][str(predecessor_lane_id) + ' to ' + str(successor_lane_id)] = lane_length
            successor_to_predecessor['length'][str(successor_lane_id) + ' to ' + str(predecessor_lane_id)] = lane_length
        predecessor  = road.get('predecessor', [])
        successor = road.get('successor', [])
        if len(predecessor) == 1 and len(successor) == 1:
            predecessor_road_id = predecessor[0]['road_id']
            predecessor_contact_point = predecessor[0]['contact_point']
            successor_road_id = successor[0]['road_id']
            successor_contact_point = successor[0]['contact_point']
            predecessor_to_successor['contact_point'] = [predecessor_contact_point, successor_contact_point]
            successor_to_predecessor['contact_point'] = [successor_contact_point, predecessor_contact_point]
            xodr_dict['connections'][str(predecessor_road_id) + ' to ' + str(successor_road_id)] = predecessor_to_successor
            xodr_dict['connections'][str(successor_road_id) + ' back to ' + str(predecessor_road_id)] = successor_to_predecessor
        else:
            warnings.warn(f'Linking road {road_id} does not have exactly one predecessor and one successor.')
            pdb.set_trace()
    return xodr_dict


def get_orientations_angles_and_semantic_desc(segments):
    road_ids = []
    orientations = []
    semantic_orientation = {}
    semantic_desc = {}
    angles = []
    num = 0
    for k, segment in enumerate(segments):
        if len(segment[0]) == 1:
            num += 1
            road_id = segment[0]
            road_ids.append(road_id)
            hdg = segment[2]
            if hdg == 0 and road_id == '0':
                    orien = hdg + np.pi 
            elif road_id == '0':
                warnings.warn(f'The first road segment has a heading angle of {hdg} which is not 0')
                pdb.set_trace()
            else:
                orien = hdg
            orientations.append(orien)
            semantic_orientation[road_id] = find_closest_semantic_orientation_based_on_absolute_orientation(np.pi + orien)
    # reoder the road segments based on the orientation such that the orientation is increasing
    sorted_indices = np.argsort(orientations)
    sorted_road_ids = [road_ids[i] for i in sorted_indices]
    sorted_orientations = [orientations[i] for i in sorted_indices]
    angles = [sorted_orientations[i+1] - sorted_orientations[i] for i in range(len(sorted_orientations)-1)] + [sorted_orientations[0] - sorted_orientations[-1]]
    angles = np.array(angles) % (2*np.pi) # always positive
    # in T intersection case, identify the throughway and stem, and left shoulder and right shoulder
    if num == 3:
        throughway_angle = np.max(angles)
        throughway_right_shoulder_index = np.argmax(angles)
        if throughway_right_shoulder_index < num - 1:
            throughway_left_shoulder_index = throughway_right_shoulder_index + 1
        else:
            throughway_left_shoulder_index = 0
        for k, id in enumerate(sorted_road_ids):
            if k == throughway_left_shoulder_index:
                semantic_desc[id] = 'throughway left_shoulder'
                semantic_desc['throughway left_shoulder'] = id
            elif k == throughway_right_shoulder_index:
                semantic_desc[id] = 'throughway right_shoulder'
                semantic_desc['throughway right_shoulder'] = id
            else:
                semantic_desc[id] = 'stem'
                semantic_desc['stem'] = id
        
    return sorted_road_ids, sorted_orientations, angles, semantic_orientation, semantic_desc


def get_road_info_from_xodr(xodr_file_path):
    root = load_xodr(xodr_file_path)
    road_segments, lanes = get_road_segments_and_lanes(root)
    xodr_dict = xodr_to_dict(root)
    road_info = {
        'xodr_file_path': xodr_file_path,
        'road_dict': xodr_dict,
        'lanes': {},
        'min_possible_lane_id': {},
        'max_possible_lane_id': {},
    }
    for key, value in lanes.items():
        road_info['lanes'][key] = []
        for lane in value:
            if lane[1] == 'none':
                continue
            road_info['lanes'][key].append(lane[0])
    for road_id in road_info['lanes'].keys():
        min_possible_lane_id = min([int(lane_id) for lane_id in road_info['lanes'][road_id]])
        max_possible_lane_id = max([int(lane_id) for lane_id in road_info['lanes'][road_id]])
        road_info['min_possible_lane_id'][road_id] = min_possible_lane_id
        road_info['max_possible_lane_id'][road_id] = max_possible_lane_id
    sorted_road_ids, sorted_orientations, angles, semantic_orientation, semantic_desc = get_orientations_angles_and_semantic_desc(road_segments)
    road_info['sorted_road_ids'] = [int(road_id) for road_id in sorted_road_ids]
    road_info['sorted_orientations'] = sorted_orientations
    road_info['angles'] = angles
    road_info['semantic_orientation'] = semantic_orientation
    road_info['semantic_desc'] = semantic_desc
    return road_info
