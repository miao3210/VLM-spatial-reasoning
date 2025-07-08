import sys
import csv
import math
import matplotlib.pyplot as plt

H_SCALE = 10
text_x_offset = 0
text_y_offset = 0.7
text_size = 7

with open(sys.argv[1]) as f:
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

p1 = plt.figure(1)
plt.axis('off')

# plot road ref line segments
for i in range(len(ref_x)):
    plt.plot(ref_x[i], ref_y[i], linewidth=1.0, color='black')

# plot driving lanes in blue
for i in range(len(lane_x)):
    width = 1.0 if lane_type_flags[i] == '-' else 0.5
    dash_controller = {} if lane_type_flags[i] == '-' else {'dashes': [20, 40]}
    plt.plot(lane_x[i], lane_y[i], linewidth=width, color='grey', linestyle=lane_type_flags[i], **dash_controller)
    
# plot border lanes in gray
for i in range(len(border_x)):
    plt.plot(border_x[i], border_y[i], linewidth=1.0, color='#AAAAAA')

# plot red dots indicating lane dections
# for i in range(len(lane_section_dots_x)):
#     plt.plot(lane_section_dots_x[i], lane_section_dots_y[i], 'o', ms=4.0, color='#BB5555')

# for i in range(len(road_start_dots_x)):
    # plot a yellow dot at start of each road
    # plt.plot(road_start_dots_x[i], road_start_dots_y[i], 'o', ms=5.0, color='#BBBB33')
    # and an arrow indicating road direction
    # plt.arrow(road_start_dots_x[i], road_start_dots_y[i], arrow_dx[i], arrow_dy[i], width=0.1, head_width=1.0, color='#BB5555')

# plot road id numbers
for i in range(len(road_id)):
    plt.text(road_id_x[i], road_id_y[i], road_id[i], size=text_size, ha='center', va='center', color='green')

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

if sys.argv[2]:
    plt.savefig(sys.argv[2], dpi=300)
else:
    plt.show()