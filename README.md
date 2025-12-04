# VLM Grounding & Spatial Reasoning Project

## Abstract

This project focuses on the spatial reasoning problem for both abstract traffic images and real-world road scenes. An abstract traffic image is shown as follows. Despite the significant progress in the vision language model field, even the SOTA VLMs struggle with accurately describing the shape, attribute, and spatial relationships of such road network, and precisely identifying the location of the vehicles in the scenario. These limitations set inevitable obstacles for deploying VLMs onto intelligent autonomy in order to conduct scene understanding. This problem is especially challenging for autonomous vehicles.

This project aims to address this problem by first establishing a benchmark for VLM grounding and particularly focusing on the spatial relationships between components, and then creating a tokenizer. This tokenizer extracts the spatial information from the image and then interprets the location, shape, area, and attributes of all components into a structured hierarchical representation. We claim the hierarchical structure is essential because it provides correspondence between regions, roads, lanes, and the surrounding environments.

By adopting the proposed tokenizer, we translate the image into a manageable data stream where the VLM grounding process is simplified and the spatial reasoning process has a cleaner information source.

The contribution lies in three folds. First, we propose a benchmark that targets the grounding and spatial reasoning ability of VLM and particularly the relative spatial relationship. Second, we propose a tokenizer that extracts detailed spatial information and translates it into a structured format. Third, we propose a hierarchical spatial representation that improve the spatial reasoning ability of Multimodal Large Language Models (MLLMs).
