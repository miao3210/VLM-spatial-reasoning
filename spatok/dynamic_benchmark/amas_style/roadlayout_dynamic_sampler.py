import warnings
import random
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from spatok.dynamic_benchmark.utils.roadlayout_utils import *
from spatok.dynamic_benchmark.utils.roadlayout_label_utils import *

class RoadLayoutDynamicSampler:
    def __init__(self, image_size_range=[(256, 256), (512, 512), (1024, 1024)],
                 num_roads_range=[1, 2, 3, 4, 5],
                 num_lanes_range=[1, 2, 3, 4, 5],
                 road_length_range=[20, 50, 100, 200, 300],
                 rotation_range=[k for k in range(0, 360, 15)],  # Degrees
                 object_size_range=[(20, 30), (30, 40), (40, 50)],
                 path='.'): # '/data/miao/spatok_curation/road_layout/'
        self.image_size_range = image_size_range
        self.num_roads_range = num_roads_range
        self.num_lanes_range = num_lanes_range
        self.road_length_range = road_length_range
        self.rotation_range = rotation_range
        self.object_size_range = object_size_range
        self.temperature = 1.0  # Temperature for sampling angles
        self.path = path

    def sample(self, num_samples):
        samples = []
        for index in range(num_samples):
            size = self._sample_size()
            image, label = self._sample_image(size, index=index)
            samples.append((image, label))
        return samples

    def _sample_size(self):
        return random.choice(self.image_size_range)

    def _sample_num_roads(self):
        return random.choice(self.num_roads_range)

    def _sample_num_lanes(self, num=None, with_zero=False):
        if with_zero:
            if num is not None:
                return [random.choice(self.num_lanes_range + [0]) for _ in range(num)]
            else:
                return random.choice(self.num_lanes_range + [0])
        if num is not None:
            return [random.choice(self.num_lanes_range) for _ in range(num)]
        else:
            return random.choice(self.num_lanes_range)
        
    def _sample_ramp_num_lanes(self, num=None):
        if num is not None:
            return [random.choice(self.num_lanes_range[:3]) for _ in range(num)]
        else:
            return random.choice(self.num_lanes_range[:3])
    
    def _sample_road_length(self, num=None):
        if num is not None:
            return [random.choice(self.road_length_range) for _ in range(num)]
        else:
            return random.choice(self.road_length_range)

    def _sample_curvature(self):
        return random.choice([-0.01, -0.005, -0.001, 0.001, 0.005, 0.01])
    
    def _sample_road_type(self):
        road_types = ['line', 'curve']
        return random.choices(road_types, weights=[0.7, 0.3], k=1)[0]
    
    def _sample_road_angle(self, num_roads):
        if num_roads == 1:
            return random.uniform(0, 2 * np.pi)  # Single road can have any angle
        else:
            weights = [random.random() for _ in range(num_roads+1)]
            weights[0] = 0 if random.random() < 0.5 else weights[0]  # Ensure the first angle is either 0 or 1
            if not weights[0] == 0:
                weights[-1] = 0 if random.random() < 0.5 else weights[-1] # Ensure the last angle is either 0 or 1
            weights = np.array(weights)
            weights = np.exp(weights / self.temperature) / np.sum(np.exp(weights / self.temperature))
            weights = np.cumsum(weights)
            angles = [2 * np.pi * w for w in weights[:-1]]  # Exclude the last cumulative sum
            return angles

    def _sample_object_size(self):
        return random.choice(self.object_size_range)
    
    def _sample_rotation(self):
        return random.choice(self.rotation_range)
    
    def _create_road_layout(self, num_roads, num_lanes, 
                      category=None, road_type=None, road_length=None,
                      angles=None, curvature=None,
                      left_ramp_type=None, right_ramp_type=None, index=0):
        if num_roads == 1:
            num_lanes = [num_lanes, num_lanes] if isinstance(num_lanes, (int, float)) else num_lanes
            road_type = 'curve' if road_type == 'curve' else 'line'
            road_length = 50 if road_length is None else road_length
            road = create_single_road(road_type=road_type, 
                                      left_lanes=num_lanes[0], 
                                      right_lanes=num_lanes[1], 
                                      road_length=road_length,
                                      curvature=curvature)
        else:
            if category == 'intersection':
                num_lanes = num_lanes if isinstance(num_lanes, (int, float)) else num_lanes[0]
                
                radius = np.max([num_lanes]) * 3 + 1
                road_length = 50 if road_length is None else road_length
                if isinstance(road_length, list) and not len(road_length) == num_roads:
                    warnings.warn(f'Invalid road length for intersection: {road_length}. Using default or sampled values.')
                    if len(road_length) > num_roads:
                        road_length = road_length[:num_roads]
                    else:
                        num_res = num_roads - len(road_length)
                        road_length = road_length + [random.choice(self.road_length_range) for _ in range(num_res)]
                road = create_intersection(num_ways=num_roads, 
                                           nlanes=num_lanes,
                                           road_length=road_length,
                                           angles=angles,
                                           radius=radius)
            elif category == 'interchange':
                if num_lanes is None or len(num_lanes) < 3:
                    warnings.warn(f'Invalid number of lanes for interchange: {num_lanes}. Using default or sampled values.')
                    if len(num_lanes) == 1:
                        num_lanes = [num_lanes[0], num_lanes[0], random.randint(1, 3), random.randint(1, 3)]
                    elif len(num_lanes) == 2:
                        num_lanes = [num_lanes[0], num_lanes[1], random.randint(1, 3), random.randint(1, 3)]
                elif left_ramp_type is not None and right_ramp_type is not None and len(num_lanes) < 4:
                    warnings.warn(f'Invalid number of lanes for interchange: {num_lanes}. Using default or sampled values.')
                    num_lanes = [num_lanes[0], num_lanes[1], num_lanes[2], num_lanes[2]]
                road = create_interchange(road_type=road_type,
                                          left_lanes=num_lanes[0],
                                          right_lanes=num_lanes[1],
                                          left_ramp_lanes=num_lanes[2],
                                          right_ramp_lanes=num_lanes[3],
                                          left_ramp_type=left_ramp_type,
                                          right_ramp_type=right_ramp_type,
                                          road_length=road_length,
                                          curvature=curvature)
            else:
                warnings.warn(f'Unsupported road category and road numbers: {category} and {num_roads}. Skip this sample.')
        path = f'{self.path}/road_layout_{index}.xodr'
        road.write_xml(path)
        return road, num_roads, num_lanes, category, road_type, road_length, angles, curvature, left_ramp_type, right_ramp_type, path

    def _sample_road_layout(self, index: int = 0):
        num_roads = self._sample_num_roads()
        road_type = self._sample_road_type()
        road_length = self._sample_road_length()
        angles = self._sample_road_angle(num_roads)
        curvature=self._sample_curvature()
        if num_roads == 1:
            category = 'single'
            num_lanes = self._sample_num_lanes(num=2, with_zero=True)
            while num_lanes == [0, 0]:
                num_lanes = self._sample_num_lanes(num=2, with_zero=True)
        elif num_roads == 2:
            category = 'interchange'
            flag = random.choices([1, 2, 3, 4, 5, 6], weights=[0.2, 0.2, 0.2, 0.2, 0.1, 0.1])[0]  # 1: left on ramp, 2: right on ramp, 3: left off ramp, 4: right off ramp
            if flag == 1:
                left_ramp_type = 'on-ramp'
                right_ramp_type = None
                ramp_lanes = self._sample_ramp_num_lanes(num=1) + [0]
            elif flag == 2:
                left_ramp_type = None
                right_ramp_type = 'on-ramp'
                ramp_lanes = [0] + self._sample_ramp_num_lanes(num=1)
            elif flag == 3:
                left_ramp_type = 'off-ramp'
                right_ramp_type = None
                ramp_lanes = self._sample_ramp_num_lanes(num=1) + [0]
            elif flag == 4:
                left_ramp_type = None
                right_ramp_type = 'off-ramp'
                ramp_lanes = [0] + self._sample_ramp_num_lanes(num=1)
            elif flag == 5:
                left_ramp_type = 'on-ramp'
                right_ramp_type = 'on-ramp'
                ramp_lanes = self._sample_ramp_num_lanes(num=2)
            elif flag == 6:
                left_ramp_type = 'off-ramp'
                right_ramp_type = 'off-ramp'
                ramp_lanes = self._sample_ramp_num_lanes(num=2)
            else:
                raise ValueError(f'Invalid flag {flag} for road layout sampling.')
            left_lanes = self._sample_num_lanes() if ramp_lanes[0] > 0 else self._sample_num_lanes(with_zero=True)
            right_lanes = self._sample_num_lanes() if ramp_lanes[1] > 0 else self._sample_num_lanes(with_zero=True)
            num_lanes = [left_lanes, right_lanes, ramp_lanes[0], ramp_lanes[1]]
        elif num_roads > 2:
            category = 'intersection'
            num_lanes = self._sample_num_lanes(num=1)
        else:
            raise ValueError(f'Invalid number of roads {num_roads} for road layout sampling.')
        return self._create_road_layout(num_roads=num_roads, 
                                        num_lanes=num_lanes,
                                        category=category,
                                        road_type=road_type, 
                                        road_length=road_length,
                                        angles=angles,
                                        curvature=curvature,
                                        left_ramp_type=left_ramp_type if 'left_ramp_type' in locals() else None,
                                        right_ramp_type=right_ramp_type if 'right_ramp_type' in locals() else None,
                                        index=index)    



    def _sample_image(self, size, index=0):
        '''
        Don't delete this function desc.
        This function generate a random image and its label.
        image:
        This image has the given number of roads and lanes and the type of roads.
        
        label:
        The label is a dictionary with the following keys:
        Una, please help me check if the labels I listed below in the function are reasonable and complete. If not, please modify it as you like.
        Once you finish the _get_label, please fill in the arg description here
        '''

        height, width = size
        road, num_roads, num_lanes, category, road_type, road_length, angles, curvature, left_ramp_type, right_ramp_type, path = self._sample_road_layout(index=index)
        img_path = visualize_road(path)
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {img_path}")

        # Get image center
        (height, width) = image.shape[:2]
        center = (width // 2, height // 2)
        rotation_angle = self._sample_rotation()

        # Compute rotation matrix and rotate
        M = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
        rotated = cv2.warpAffine(image, M, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))

        # Resize to (w, h)
        resized = cv2.resize(rotated, (width, height), interpolation=cv2.INTER_AREA)

        # Convert to RGB and save
        image = resized # image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        cv2.imwrite(f'{self.path}/road_layout_{index}_0.png', image)

        #todolabel = self._get_label(index, label_group='base')
        todolabel = self._get_label(path, label_group='base')

        # base_label = self._get_label(TODO, 'base')
        base_label = {
            'category': None,
            'road_type': None,
            'num_roads': None,
            'num_lanes': None, 
            'road_length': None, 
            'angles': None, 
            'curvature': None, 
            'left_ramp_type': None, 
            'right_ramp_type': None, 
        }

        # advanced_label = self._get_label(TODO, 'advanced')
        advanced_label = {
            'roads': [
                {
                    'id': None,
                    'lanes_with_side': None, # e.g. {'up': 2, 'down': 2} {'left': 1, 'right': 1}, {'upper_left': 1, 'upper_right': 3}, {'lower_left': 1, 'lower_right': 1}
                    'length': None,
                    'orientation': None,
                    'curvature': None,
                    'length': None,
                    'position': None,
                    'area': None,
                    'vertices': None,  # List of tuples representing the vertices of the road
                    'edges': [
                            {
                                'id': None,  # The id of the lanes (if inside the road) or lane (if road edge)
                                'type': None,  # e.g. 'solid yellow', 'dash yellow', 'solid white', 'dash white', 'road edge', 'curb', 'parking lot'
                                'geometry': 'line', # e.g. 'line', 'curve', 'arc'
                                'width': None,  # The width of the edge, e.g. 1
                                'vertices': None,  # List of tuples representing the vertices of the edge
                                'color': None,  # The color of the edge, e.g. (255, 255, 255) for white
                                'relative_position': None,  # the relative position in this road, e.g. 'left', 'right', 'up', 'down', 'upper_left', 'upper_right', 'lower_left', 'lower_right'
                            },
                    ],
                    'connections_with_side': [
                        {
                            'side': None,  # e.g. 'up', 'down', 'left', 'right', 'upper_left', 'upper_right', 'lower_left', 'lower_right'
                            'road_id': None,  # The id of the connected road
                            'lane_id_pair': {},  # The id of the connected lane, e.g. {1:-1, 2:-2} means the lane with id 1 on the current road is connected to the lane with id -1 on the connected road, and the lane with id 2 on the current road is connected to the lane with id -2 on the connected road
                            'ramp_type': None,  # e.g. 'on-ramp', 'off-ramp'
                            'relationship': None,  # e.g. 'left', 'right', 'straight'
                        },
                    ]
                },
            ],
            'junctions': [
                {
                    'id': None,
                    'connected_roads': [],  # List of road ids connected to this junction
                    'position': None,
                    'vertices': None,  # List of tuples representing the vertices of the junction
                    'radius': None, # the width or length of the junction, e.g. 10, or both, e.g. (20,20)
                },
            ]
        }
        label = {**base_label, **advanced_label}

        '''
        after finish the label above, please let me know and we will discuss how to augment the image and label
        '''
        # augmented_sample = #TODO 
        # augmented_base = #TODO

        return image, label


    def _get_label(self, xodr_path, label_group='base'):
        """
        Generate label by parsing the existing .xodr layout file.

        Args:
            xodr_path (str): path to the .xodr layout file.
            label_group (str): 'base' or 'advanced'.

        Returns:
            dict: structured label data
        """
        road_info = get_road_info_from_xodr(xodr_path)

        base_label = {
            'category': road_info.get('category'),
            'road_type': road_info.get('road_type'),
            'num_roads': road_info.get('num_roads'),
            'num_lanes': road_info.get('num_lanes'),
            'road_length': road_info.get('road_length'),
            'angles': road_info.get('angles'),
            'curvature': road_info.get('curvature'),
            'left_ramp_type': road_info.get('left_ramp_type'),
            'right_ramp_type': road_info.get('right_ramp_type'),
        }
        if label_group == 'base':

            return base_label

        roads_data = road_info.get('roads', [])
        junctions_data = road_info.get('junctions', [])

        return {**base_label, 'roads': roads_data, 'junctions': junctions_data}



    