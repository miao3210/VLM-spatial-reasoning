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

    def _sample_image(self, size, num_edges, num_objects):
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
