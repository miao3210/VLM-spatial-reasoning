
from typing import List, Optional
import csv
import math
import os

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

from scenariogeneration import xodr


def create_single_road(
        road_type: str,
        left_lanes: int,
        right_lanes: int,
        road_length: float = 100,
        curvature: float = 0.01
    ) -> xodr.OpenDrive:
    """
    Description:
        Create a highway representation using the OpenDRIVE format.

    Parameters:
        road_type: The type of road geometry ('line' or 'curve').
        left_lanes: The number of lanes on the left side of the road.
        right_lanes: The number of lanes on the right side of the road.
        length: The length of the road (default is 100).
        curvature: The curvature of the road, applicable only for 'curve' road type (default is 0.01).

    Returns:
        xodr.OpenDrive: An OpenDRIVE object representing the intersection.
    """
    if left_lanes < 0 or right_lanes < 0 or road_length <= 0:
        raise ValueError("Negative values for lanes or road_length are not allowed.")
    road_type = road_type.lower()
    if road_type == "line":
        road_geometry = xodr.Line(length=road_length)
    elif road_type == "curve":
        road_geometry = xodr.Arc(length=road_length, curvature=curvature)
    else:
        raise NotImplementedError(f"Road type {road_type} is not supported. Consider using 'line' or 'curve'.")
    
    roads = []
    roads.append(xodr.create_road([road_geometry], id=0, left_lanes=left_lanes, right_lanes=right_lanes))
    odr = xodr.OpenDrive("myroad")
    for r in roads:
        odr.add_road(r)
    odr.adjust_roads_and_lanes()
    return odr


def create_intersection(
        num_ways: int,
        nlanes: int = 2,
        road_length: int | list[int] = 50,
        angles: Optional[List[float]] = None,
        radius: int = 20
    ) -> xodr.OpenDrive:
    """
    Description:
        Create an intersection in an OpenDRIVE format.

    Parameters:
        num_ways (int): Number of ways (roads) in the intersection.
        nlanes (int): Number of lanes in each road. Default is 2.
        road_length (int): Length of each road. Default is 50.
        angles (List[float], optional): List of angles for the roads. If None, angles are evenly distributed. Default is None.
        radius (int): Radius of the connecting roads in the junction. Default is 20.

    Returns:
        xodr.OpenDrive: An OpenDRIVE object representing the intersection.
    """
    if nlanes < 0 or road_length <= 0:
        raise ValueError("Negative values for lanes or road_length are not allowed.")
    if num_ways < 3:
        raise ValueError("Invalid way number for an intersection.")
    if angles is None:
        angles = [i * 2 * np.pi / num_ways for i in range(num_ways)]
    else:
        assert len(angles) == num_ways, f"Invalid angles for {num_ways} ways intersection"

    # create the roads
    roads = []

    if isinstance(road_length, int):
        for i in range(num_ways):
            roads.append(xodr.create_road(
                [xodr.Line(road_length)], 
                id=i, 
                center_road_mark=xodr.STD_ROADMARK_BROKEN, 
                left_lanes=nlanes, 
                right_lanes=nlanes
            ))
    else:
        assert len(road_length) == num_ways, f"Invalid road length for {num_ways} ways intersection"
        for i in range(num_ways):
            roads.append(xodr.create_road(
                [xodr.Line(road_length[i])], 
                id=i, 
                center_road_mark=xodr.STD_ROADMARK_BROKEN, 
                left_lanes=nlanes, 
                right_lanes=nlanes
            ))

    # create the junction
    junction_roads = xodr.create_junction_roads(roads, angles, [radius])
    junction = xodr.create_junction(junction_roads, id=1, roads=roads)

    # create the opendrive and add all roads and the junction
    odr = xodr.OpenDrive("myroad")
    odr.add_junction(junction)
    for r in roads:
        odr.add_road(r)
    for j in junction_roads:
        odr.add_road(j)
    odr.adjust_roads_and_lanes()
    return odr


def create_interchange(
        road_type: str,
        left_lanes: int,
        right_lanes: int,
        left_ramp_lanes: int,
        right_ramp_lanes: int,
        left_ramp_type: str = None,
        right_ramp_type: str = None,
        road_length: float = 50,
        curvature: float = 0.01
    ) -> xodr.OpenDrive:
    """
    Description:
        Creates an interchange with ramps.

    Parameters:
        road_type: The type of the main road (line/curve).
        left_lanes: Number of lanes on the left that are not connected with ramps.
        right_lanes: Number of lanes on the right that are not connected with ramps.
        left_ramp_type: The type of left ramp (on-ramp/off-ramp).
        right_ramp_type: The type of right ramp (on-ramp/off-ramp).
        road_length: The length of each road segment.
        curvature: The curvature of the road.

    Returns:
        An OpenDrive object representing the created interchange.
    """
    junction_id = 100
    assert left_lanes > 0 and right_lanes > 0, "Number of lanes must be positive."
    assert road_length > 0, "Road length must be positive."

    # road lane number
    left_ramp_type = left_ramp_type.lower() if left_ramp_type is not None else None
    right_ramp_type = right_ramp_type.lower() if right_ramp_type is not None else None
    start_road_lanes = {'left_lanes':left_lanes, 'right_lanes':right_lanes}
    end_road_lanes = {'left_lanes':left_lanes, 'right_lanes':right_lanes}
    if left_ramp_type == "on-ramp":
        start_road_lanes['left_lanes'] += left_ramp_lanes
    elif left_ramp_type == "off-ramp":
        end_road_lanes['left_lanes'] += left_ramp_lanes
    else:
        assert left_ramp_type is None, f"Invalid ramp type {left_ramp_type}. Use 'on-ramp' or 'off-ramp'."
    if right_ramp_type == "on-ramp":
        end_road_lanes['right_lanes'] += right_ramp_lanes
    elif right_ramp_type == "off-ramp":
        start_road_lanes['right_lanes'] += right_ramp_lanes
    else:
        assert right_ramp_type is None, f"Invalid ramp type {right_ramp_type}. Use 'on-ramp' or 'off-ramp'."
  
    # create the roads
    roads = []
    start_road = xodr.create_road([xodr.Line(road_length)], id=0, center_road_mark=xodr.STD_ROADMARK_BROKEN, **start_road_lanes)
    end_road = xodr.create_road([xodr.Line(road_length)], id=1, center_road_mark=xodr.STD_ROADMARK_BROKEN, **end_road_lanes)
    start_road.add_successor(xodr.ElementType.junction, junction_id)
    end_road.add_predecessor(xodr.ElementType.junction, junction_id)
    roads.extend([start_road, end_road])

    # connect ramp and road
    junction = xodr.DirectJunctionCreator(id=junction_id, name="my direct junction")
    junction.add_connection(roads[0], roads[1])
    if left_ramp_type == "on-ramp":
        left_ramp = xodr.create_road(xodr.Spiral(0.00001, 0.01, 50), id=2, left_lanes=1, right_lanes=0)
        left_ramp.add_predecessor(xodr.ElementType.junction, junction_id)
        roads.append(left_ramp)
        junction.add_connection(
            incoming_road=start_road,
            linked_road=left_ramp,
            incoming_lane_ids=start_road_lanes["left_lanes"],
            linked_lane_ids=1
        )
    elif left_ramp_type == "off-ramp":
        left_ramp = xodr.create_road(xodr.Spiral(0.01, 0.00001, 50), id=2, left_lanes=1, right_lanes=0)
        left_ramp.add_successor(xodr.ElementType.junction, junction_id)
        roads.append(left_ramp)
        junction.add_connection(
            incoming_road=left_ramp,
            linked_road=end_road,
            incoming_lane_ids=1,
            linked_lane_ids=end_road_lanes['left_lanes']
        )
    if right_ramp_type == "on-ramp":
        right_ramp = xodr.create_road(xodr.Spiral(-0.01, -0.00001, 50), id=3, left_lanes=0, right_lanes=1)
        right_ramp.add_successor(xodr.ElementType.junction, junction_id)
        roads.append(right_ramp)
        junction.add_connection(
            incoming_road=right_ramp,
            linked_road=end_road,
            incoming_lane_ids=-1,
            linked_lane_ids=-end_road_lanes['right_lanes']
        )
    elif right_ramp_type == "off-ramp":
        right_ramp = xodr.create_road(xodr.Spiral(-0.00001, -0.01, 50), id=3, left_lanes=0, right_lanes=1)
        right_ramp.add_predecessor(xodr.ElementType.junction, junction_id)
        roads.append(right_ramp)
        junction.add_connection(
            incoming_road=start_road,
            linked_road=right_ramp,
            incoming_lane_ids=-start_road_lanes['right_lanes'],
            linked_lane_ids=-1
        )

    # create the opendrive and add all roads and the junction
    odr = xodr.OpenDrive("myroad")
    odr.add_junction_creator(junction)
    for r in roads:
        odr.add_road(r)
    odr.adjust_roads_and_lanes()
    return odr


def create_2_adjecent_interchange(
        road_type: str,
        left_lanes: int,
        right_lanes: int,
        left_ramp_lanes: int,
        right_ramp_lanes: int,
        left_ramp_type: str = None,
        right_ramp_type: str = None,
        road_length: float = 50,
        curvature: float = 0.01
    ) -> xodr.OpenDrive:
    """
    TODO:
    una, can you help me to implement this function?
    The goal is to create 3 roads, where at least two roads have ramps 
    """
    pass

def plot_road(csv_file_path, image_file_path=None):
    H_SCALE = 10
    text_x_offset = 0
    text_y_offset = 0.7
    text_size = 7

    with open(csv_file_path) as f:
        reader = csv.reader(f, skipinitialspace=True)
        positions = list(reader)

    ref_x = []
    ref_y = []
    ref_z = []
    ref_h = []

    lane_x = []
    lane_y = []
    lane_z = []
    lane_h = []

    border_x = []
    border_y = []
    border_z = []
    border_h = []

    road_id = []
    road_id_x = []
    road_id_y = []

    road_start_dots_x = []
    road_start_dots_y = []

    lane_start_dots_x = []
    lane_start_dots_y = []
    lane_start_dots_x.append([])
    lane_start_dots_y.append([])

    road_end_dots_x = []
    road_end_dots_y = []

    lane_section_dots_x = []
    lane_section_dots_y = []

    arrow_dx = []
    arrow_dy = []

    lane_arrow_dx = []
    lane_arrow_dy = []
    lane_arrow_dx.append([])
    lane_arrow_dy.append([])

    lane_ids = []
    lane_ids.append([])

    current_road_id = None
    current_lane_id = None
    current_lane_section = None
    new_lane_section = False

    for i in range(len(positions) + 1):
        if i < len(positions):
            pos = positions[i]
        
        # plot road id before going to next road
        if i == len(positions) or (pos[0] == 'lane' and i > 0):
            if current_lane_id == '0':
                if current_lane_section == '0':
                    road_id.append(int(current_road_id))
                    index = int(len(ref_x[-1])/3.0)
                    h = ref_h[-1][index]
                    road_id_x.append(ref_x[-1][index] + (text_x_offset * math.cos(h) - text_y_offset * math.sin(h)))
                    road_id_y.append(ref_y[-1][index] + (text_x_offset * math.sin(h) + text_y_offset * math.cos(h)))
                    road_start_dots_x.append(ref_x[-1][0])
                    road_start_dots_y.append(ref_y[-1][0])
                    if len(ref_x) > 0:
                        arrow_dx.append(ref_x[-1][1]-ref_x[-1][0])
                        arrow_dy.append(ref_y[-1][1]-ref_y[-1][0])
                    else:
                        arrow_dx.append(0)
                        arrow_dy.append(0)

                lane_section_dots_x.append(ref_x[-1][-1])
                lane_section_dots_y.append(ref_y[-1][-1])

            if current_lane_section == '0':
                if current_lane_id != '0':
                    lane_start_dots_x[-1].append(lane_x[-1][0])
                    lane_start_dots_y[-1].append(lane_y[-1][0])
                    lane_arrow_dx[-1].append(lane_x[-1][1]-lane_x[-1][0])
                    lane_arrow_dy[-1].append(lane_y[-1][1]-lane_y[-1][0])
                else:
                    lane_start_dots_x[-1].append(ref_x[-1][0])
                    lane_start_dots_y[-1].append(ref_y[-1][0])
                    lane_arrow_dx[-1].append(ref_x[-1][1]-ref_x[-1][0])
                    lane_arrow_dy[-1].append(ref_y[-1][1]-ref_y[-1][0])

                lane_ids[-1].append(current_lane_id)

            if current_road_id != pos[1] and i < len(positions):
                lane_start_dots_x.append([])
                lane_start_dots_y.append([])
                lane_arrow_dx.append([])
                lane_arrow_dy.append([])
                lane_ids.append([])
        
        if i == len(positions):
            break

        if pos[0] == 'lane':
            current_road_id = pos[1]
            current_lane_section = pos[2]
            current_lane_id = pos[3]
            if pos[3] == '0':
                ltype = 'ref'
                ref_x.append([])
                ref_y.append([])
                ref_z.append([])
                ref_h.append([])
            elif pos[4] == 'no-driving':
                ltype = 'border'
                border_x.append([])
                border_y.append([])
                border_z.append([])
                border_h.append([])
            else:
                ltype = 'lane'
                lane_x.append([])
                lane_y.append([])
                lane_z.append([])
                lane_h.append([])
        else:
            if ltype == 'ref':
                ref_x[-1].append(float(pos[0]))
                ref_y[-1].append(float(pos[1]))
                ref_z[-1].append(float(pos[2]))
                ref_h[-1].append(float(pos[3]))
            elif ltype == 'border':
                border_x[-1].append(float(pos[0]))
                border_y[-1].append(float(pos[1]))
                border_z[-1].append(float(pos[2]))
                border_h[-1].append(float(pos[3]))
            else:
                lane_x[-1].append(float(pos[0]))
                lane_y[-1].append(float(pos[1]))
                lane_z[-1].append(float(pos[2]))
                lane_h[-1].append(float(pos[3]))

    # Defining line styles ( - | -- ) according to lane type in the plot
    lane_type_flags = [list(flags) for flags in lane_ids]
    for i, flags in enumerate(lane_type_flags):
        flags = [flag for flag in flags if flag != '0'] # remove all the reference lanes (handled separately)
        flags[0] = flags[-1] = '-' # set the border lanes as solid lines
        for j, flag in enumerate(flags):
            if flag != '-': flags[j] = '--' # set the non-border and non-reference lanes as dashed lines
        lane_type_flags[i] = flags # update the lane type flags
    lane_type_flags = [flag for flags in lane_type_flags for flag in flags] # flatten the list

    plt.clf()
    plt.close('all')
    plt.figure(1, figsize=(10, 10))
    plt.axis('off')
    plt.gcf().set_facecolor('gray')

    # plot road ref line segments
    for i in range(len(ref_x)):
        plt.plot(ref_x[i], ref_y[i], linewidth=2.0, color='white')

    # plot driving lanes in white
    for i in range(len(lane_x)):
        width = 2.0 if lane_type_flags[i] == '-' else 1.0
        if lane_type_flags[i] == '-':
            dash_controller = {}
            line_color = 'white'
        else:
            dash_controller = {'dashes': [20, 40]}
            line_color = 'orange'
        plt.plot(lane_x[i], lane_y[i], linewidth=width, color=line_color, linestyle=lane_type_flags[i], **dash_controller)

    # plot border lanes in gray
    for i in range(len(border_x)):
        plt.plot(border_x[i], border_y[i], linewidth=3.0, color='white')

    # plot red dots indicating lane dections
    # for i in range(len(lane_section_dots_x)):
    #     plt.plot(lane_section_dots_x[i], lane_section_dots_y[i], 'o', ms=4.0, color='#BB5555')

    # for i in range(len(road_start_dots_x)):
        # plot a yellow dot at start of each road
        # plt.plot(road_start_dots_x[i], road_start_dots_y[i], 'o', ms=5.0, color='#BBBB33')
        # and an arrow indicating road direction
        # plt.arrow(road_start_dots_x[i], road_start_dots_y[i], arrow_dx[i], arrow_dy[i], width=0.1, head_width=1.0, color='#BB5555')

    # # plot road id numbers
    # for i in range(len(road_id)):
    #     plt.text(road_id_x[i], road_id_y[i], road_id[i], size=text_size, ha='center', va='center', color='green')

    # plot lane direction
    # for i in range(len(lane_start_dots_x)):
    #     for j in range(len(lane_start_dots_x[i])-1):
    #         direction = 1 if (int(lane_ids[i][j])+int(lane_ids[i][j+1])) < 0 else -1
    #         # plt.plot(lane_start_dots_x[i][j], lane_start_dots_y[i][j], 'o', ms=1.0, color='pink')
    #         plt.arrow((lane_start_dots_x[i][j]+lane_start_dots_x[i][j+1])/2,
    #                   (lane_start_dots_y[i][j]+lane_start_dots_y[i][j+1])/2,
    #                   (lane_arrow_dx[i][j]+lane_arrow_dx[i][j+1])*direction/2,
    #                   (lane_arrow_dy[i][j]+lane_arrow_dy[i][j+1])*direction/2,
    #                    width=0.1, head_width=0.3, color='blue')

    plt.gca().set_aspect('equal', 'datalim')

    #p2 = plt.figure(2)
    #for i in range(len(z)):
    #    ivec = [j for j in range(len(z[i]))]
    #    plt.plot(z[i])

    if image_file_path:
        plt.savefig(image_file_path, dpi=300)
        image = Image.open(image_file_path)
        rotated_image = image.rotate(90, expand=True)
        rotated_image.save(image_file_path)
    else:
        plt.show()


def stack_images_with_text(image1_path, image2_path, image3_path, output_path, text, summary, font_size=40):
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    # Load the images
    if isinstance(image1_path, str):
        image1 = Image.open(image1_path)
    else:
        image1 = Image.fromarray(image1_path)
    if isinstance(image2_path, str):
        image2 = Image.open(image2_path)
    else:
        image2 = Image.fromarray(image2_path)
    if isinstance(image3_path, str):
        image3 = Image.open(image3_path)
    else:
        image3 = Image.fromarray(image3_path)

    # Stack the images horizontally and handle different heights
    max_height = max(image1.height, image2.height, image3.height)
    total_width = image1.width + image2.width + image3.width

    # Create a new blank image with the appropriate dimensions
    stacked_image = Image.new('RGB', (total_width, max_height))

    # Paste both images onto the new blank image
    stacked_image.paste(image1, (0, 0))
    stacked_image.paste(image2, (image1.width, 0))
    stacked_image.paste(image3, (image1.width + image2.width, 0))

    # Use a larger font size for the text
    font = ImageFont.truetype(font_path, font_size)

    # Create an image for the text with black background
    text_image = Image.new('RGB', (total_width, 1000), color="black")  # Increase the height to accommodate larger font
    draw = ImageDraw.Draw(text_image)

    # Calculate text position
    text_x = 10
    text_y = 10

    draw.text((text_x, text_y), text.split('[Explanation]:')[0], fill="white", font=font)

    # Create final stacked image with text
    final_height = max_height + text_image.height
    final_image = Image.new('RGB', (total_width, final_height))
    final_image.paste(stacked_image, (0, 0))
    final_image.paste(text_image, (0, max_height))

    # Save the final image
    final_image.save(output_path)
    with open(os.path.splitext(output_path)[0]+".txt", "w") as f:
        f.write(text+
                "\n\n"+
                "[Summary from NHTSA]:\n\t"+
                summary
                )
        


def visualize_road(current_file_path):
    import sys
    import os
    import matplotlib.pyplot as plt

    if sys.platform.startswith("linux"):
        executable = "esmini_linux"
    else:
        executable = "esmini_mac"
    os.system(f"{os.path.join(os.path.dirname(__file__), f'../../../{executable}/bin/odrplot')} \
                {os.path.splitext(current_file_path)[0]+'.xodr'} \
                {os.path.splitext(current_file_path)[0]+'.csv'}")
    
    # plot figure method 1 
    # print(os.path.splitext(current_file_path)[0]+'.png')
    # os.system(f"python {os.path.join(os.path.dirname(__file__), f'../../{executable}/EnvironmentSimulator/Applications/odrplot/xodr.py')} \
    #            {os.path.splitext(current_file_path)[0]+'.csv'} \
    #            {os.path.splitext(current_file_path)[0]+'.png'}")
    
    # plot figure method 2
    plot_road(os.path.splitext(current_file_path)[0]+'.csv', os.path.splitext(current_file_path)[0]+'.png')
    return os.path.splitext(current_file_path)[0]+'.png'



if __name__ == "__main__":
    curvature = 0.1
    left_lanes = 2 #np.random.choice([1, 2, 3])
    right_lanes = 3 #np.random.choice([1, 2, 3])
    dir_path = os.path.join('./', f'left_{left_lanes}_right_{right_lanes}_curvature_{curvature}/')
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    path = os.path.join(dir_path, 'road.xodr')
    road_length = 100
    road_type = 'line' if curvature == 0 else 'curve'
    # Create a single road with the specified parameters
    road = create_single_road(
        road_type=road_type,
        left_lanes=left_lanes,
        right_lanes=right_lanes,
        road_length=road_length,
        curvature=curvature
    )
    # Visualize the generated road
    road.write_xml(path, prettyprint=True)
    visualize_road(path)


