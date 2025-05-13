class DynamicSampler:
    def __init__(self, image_size_range=[(256,256), (512,512), (1024,1024)], 
                 num_edges_range=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 
                 num_objects_range=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                 object_size_range=[]):
        self.image_size_range = image_size_range
        self.num_edges_range = num_edges_range
        self.num_objects_range = num_objects_range

    def sample(self, num_samples):
        samples = []
        for _ in range(num_samples):
            size = self._sample_size()
            num_edges = self._sample_num_edges()
            num_objects = self._sample_num_objects()
            sample = self._sample_image(size, num_edges, num_objects)
            samples.append()
        return samples
    
    def _sample_size(self):
        return self._random_choice(self.image_size_range)
    
    def _sample_num_edges(self):
        return self._random_choice(self.num_edges_range)
    
    def _sample_num_objects(self):
        return self._random_choice(self.num_objects_range)
    
    def _sample_object_size(self):
        if len(self.object_size_range) == 0:
            # TODO: Una please help with this function
            pass
        return self._random_choice(self.object_size_range)

    def _sample_image(self, size, num_edges, num_objects):
        #TODO: Una please help with this function
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
        image = np.zeros((size[0], size[1], 3), dtype=np.uint8)
        label = {
            'background': {'RGB': (0, 0, 0)},
            'edges': [],
            'edge width': 1,
            'objects': []
        }
        return image, label
    

if __name__ == "__main__":
    sampler = DynamicSampler()
    image, label = sampler.sample(1)
    print(image.shape)
    print(label)