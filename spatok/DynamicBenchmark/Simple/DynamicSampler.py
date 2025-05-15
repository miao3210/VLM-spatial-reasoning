import pdb 
import warnings
import numpy as np
import cv2
import random

class DynamicSampler:
    def __init__(self, image_size_range=[(256, 256), (512, 512), (1024, 1024)],
                 num_edges_range=[1, 2, 3, 4, 5],
                 num_objects_range=[1, 2, 3, 4, 5],
                 object_size_range=[(20, 30), (30, 40), (40, 50)]):
        self.image_size_range = image_size_range
        self.num_edges_range = num_edges_range
        self.num_objects_range = num_objects_range
        self.object_size_range = object_size_range

    def sample(self, num_samples):
        samples = []
        for _ in range(num_samples):
            size = self._sample_size()
            num_edges = self._sample_num_edges()
            num_objects = self._sample_num_objects()
            image, label = self._sample_image(size, num_edges, num_objects)
            samples.append((image, label))
        return samples

    def _sample_size(self):
        return random.choice(self.image_size_range)

    def _sample_num_edges(self):
        return random.choice(self.num_edges_range)

    def _sample_num_objects(self):
        return random.choice(self.num_objects_range)

    def _sample_object_size(self):
        return random.choice(self.object_size_range)
    
    def _sample_object_size(self):
        if len(self.object_size_range) == 0:
            # TODO
            pass
        return random.choice(self.object_size_range)

    def _sample_image(self, size, num_edges, num_objects):
        #TODO -- done: Una please help with this function
        '''
        This function generate a random image and its label.
        image:
        The image has the given size, number of edges and number of objects.
        cv2 use BGR color space as default, when save the image, it will be converted to RGB color space.
        label:
        The label is a dictionary with the following keys:
        - 'background': a dictionary of the background color, either {'RGB': (r, g, b)} or {'BGR': (b, g, r)}
            Note that the edge color and object color should be (255, 255, 255) - background color
        - 'edges': a list of edges, each edge is a list of segments, each segment is a tuple of (x1, y1, x2, y2)
        - 'edge width': the width of the edge, if using the default value, it will be 1
        - 'objects': a list of objects, each object is a dictionary with the following keys:
            - 'id': the id of the object
            - 'type': the type of the object, 'rectangle' in this case
            - 'color': the color of the object, (0,0,0) or (255,255,255) in this case
            - 'position': the position of the object, (center_x, center_y)
            - 'vertex': the vertices of the object, list of tupes, each tuple is (x, y)
            - 'radius': the radius of the object, None in this case
        '''
        height, width = size
        image = np.zeros((height, width, 3), dtype=np.uint8)
        label = {
            'background': {'RGB': (0, 0, 0)},
            'edges': [],
            'edge width': 1,
            'objects': []
        }

        for _ in range(num_edges):
            x1, y1 = random.randint(0, width), random.randint(0, height)
            x2, y2 = random.randint(0, width), random.randint(0, height)
            cv2.line(image, (x1, y1), (x2, y2), (255, 255, 255), 1)
            label['edges'].append([(x1, y1, x2, y2)])

        for obj_id in range(num_objects):
            obj_w, obj_h = self._sample_object_size()
            cx, cy = random.randint(0 + obj_w, width - obj_w), random.randint(0 + obj_h, height - obj_h)
            x1, y1 = cx - obj_w // 2, cy - obj_h // 2
            x2, y2 = cx + obj_w // 2, cy + obj_h // 2
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 255), -1)

            label['objects'].append({
                'id': f'obj_{obj_id}',
                'type': 'rectangle',
                'color': (255, 255, 255),
                'position': (cx, cy),
                'vertex': [(x1, y1), (x2, y1), (x2, y2), (x1, y2)],
                'radius': None
            })

        return image, label
    

if __name__ == "__main__":
    sampler = DynamicSampler()
    image, label = sampler.sample(1)
    print(image.shape)
    print(label)