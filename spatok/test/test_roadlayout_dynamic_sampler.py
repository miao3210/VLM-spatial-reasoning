
from spatok.dynamic_benchmark.amas_style.roadlayout_dynamic_sampler import RoadLayoutDynamicSampler

import matplotlib.pyplot as plt
import cv2
import json

def run_test():
    sampler = RoadLayoutDynamicSampler()
    samples = sampler.sample(1)
    image, label = samples[0]

    # Show image
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    # plt.imshow(image)
    plt.title("Sampled Layout")
    plt.axis("off")
    plt.show()
    plt.savefig("sampled_layout.png")

    # Print label as JSON for clarity
    print("Sampled Label:\n")
    print(json.dumps(label, indent=2))


########################### New #################################


def test_get_label_structure():
    sampler = RoadLayoutDynamicSampler()  


    # Generate one sample to get a valid .xodr path
    _ = sampler.sample(1)  # Generates image and label
    xodr_path = "road_layout_0.xodr"    # use the local path
    label = sampler._get_label(xodr_path, label_group='base')

    #label = sampler._get_label(path, label_group='base') 


    # Check top-level keys
    assert 'category' in label
    assert 'roads' in label
    assert 'junctions' in label
    assert isinstance(label['roads'], list)
    assert isinstance(label['junctions'], list)

#######################################################################

if __name__ == "__main__":
    run_test()