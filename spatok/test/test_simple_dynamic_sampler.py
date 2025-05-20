from spatok.dynamic_benchmark import DynamicSampler


import matplotlib.pyplot as plt
import cv2
import json

def run_test():
    sampler = DynamicSampler()
    samples = sampler.sample(1)
    image, label = samples[0]

    # Show image
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Sampled Layout")
    plt.axis("off")
    plt.show()

    # Print label as JSON for clarity
    print("Sampled Label:\n")
    print(json.dumps(label, indent=2))

if __name__ == "__main__":
    run_test()