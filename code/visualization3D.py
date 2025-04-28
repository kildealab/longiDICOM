
# from three import BufferGeometry, Vector3, Mesh, MeshPhongMaterial, Float32BufferAttribute, Uint8Array

def distance(x1, y1, x2, y2):
    return np.sqrt((x2-x1)**2 + (y2-y1)**2)

def closest_index(x, y, array):
    dist = distance(x, y, array[0], array[1])
    index = 0
    for j in range(2, len(array), 2):
        newdist = distance(x, y, array[j], array[j+1])
        if newdist < dist:
            dist = newdist
            index = j
    return index

def render_body(slices, keys):
    pts = []  # The 3D points making up the entire body
    indices = []  # The indices of those points ordered in such a way that it forms a triangular mesh
    curr = 0  # The index of the current point (in the overall pts array, not of the current slice)
    colours = []  # The array of colours assigned to each point

    # Sort slice height (z) to be floats and in order
    keys.sort()

    # Loop through each slice to form triangle faces between current and subsequent slice
    for i in range(len(keys)):
        z = float(keys[i])  # the z value (height) of the slice
        slice = []  # stores the x and y values of each point in the slice (they all have the same z value)
        slice2 = []

        # Parse the slice data
        slice = slices[keys[i]]
#         print(len(slice))

        # Do not do for last slice since it will have already been connected to the previous slice
#         print(len(keys))
        if i < len(keys) - 1:
            slice2 = slices[keys[i+1]]
#             print(len(slice2))

        # Loop through the points in the current slice
        index = 0
        for j in range(0, len(slice), 2):
            pts.extend([slice[j], slice[j+1], z])

            # If first iteration, find starting point on second slice (this ensure both slices align)
            if j == 0 and i != len(keys)-1:
                index = closest_index(slice[j], slice[j+1], slice2)

#             # Check if point is inside one of the beams (stored in meshes array) and sets colour accordingly
#             point = Vector3(slice[j], slice[j+1], z)
#             is_inside = False

#             for mesh in meshes:
#                 if ConvexHull(mesh).containsPoint(point):
#                     is_inside = True
#                     break
#             if is_inside:
#                 colours.extend([153, 153, 255])  # blue
#             else:
#                 colours.extend([127, 127, 127])  # grey

            # Algorithm to create two triangular faces (between points [c,a,d] and [a,b,d] per point
            if j != len(slice)-2 and i != len(keys)-1:
                a = curr
                b = curr - j//2 + len(slice)//2 + ((index//2)+j//2) % (len(slice2)//2)
                c = curr + 1
                d = curr - j//2 + len(slice)//2 + ((index//2)+j//2 +1) % (len(slice2)//2)

                indices.extend([c, a, d])
                indices.extend([a, b, d])

            # For the final point in the slice, algorithm is such that it links back to first points of each slice
            if j == len(slice)-2 and i != len(keys)-1:
                a = curr
                b = curr + (((index//2) + len(slice2)//2) ) % (len(slice2)//2)
                if index == 0:
                    b += len(slice2)//2
                c = curr - j//2
                d = curr + 1 + ((index//2)) % (len(slice2)//2)

                indices.extend([c, a, d])
                indices.extend([a, b, d])

            # Algorithm to cap top and bottom slices (creates triangles across the slice instead of to next slice)
            if (i == len(keys)-1 or i == 0) and j == 0:
                for k in range(1, len(slice)//4):
                    a = curr + k - 1
                    b = curr + (len(slice)//2 - k)
                    c = curr + k
                    d = curr + (len(slice)//2 - k - 1)

                    indices.extend([c, a, d])
                    indices.extend([a, b, d])

            curr += 1
    return pts, indices

    # Create buffer geometry from points and set index and position attributes
#     geo = BufferGeometry()
#     geo.set_index(indices)
#     geo.set_attribute('position', Float32BufferAttribute(pts, 3))

#     geo.compute_bounding_sphere()
#     geo.compute_vertex_normals()

#     # Set colour attribute
#     colours = Uint8Array(colours)
#     geo.set_attribute('color', BufferAttribute(colours, 3, True))
#     geo.attributes.color.normalized = True

#     # Create final mesh
#     mesh = Mesh(geo, MeshPhongMaterial(vertex_colors=VertexColors, side=DoubleSide, shininess=0))
#     group.add(mesh)


def generate_mesh_from_RS(RS, structure, z_min=None, z_max=None):#z_cutoff=None):
    dict_contours, z_lists = get_all_ROI_contours([structure], RS)
    full_stack, keys = get_contour_stack(structure, dict_contours)
    
#     if z_cutoff is not None:
#         for key in keys.copy():
#             print(key)
#             if key < z_cutoff:
# #                 print(key)
#                 full_stack.pop(key)
#                 keys.remove(key)
        
    if z_min is not None or z_max is not None:
        if z_min is None:
            z_min = min(keys)
        if z_max is None:
            z_max = max(keys)
            
        for key in keys.copy():
            print(key)
            if key < z_min or key > z_max:
#                 print(key)
                if key in full_stack:
                    full_stack.pop(key)
                if key in keys:
                    keys.remove(key)
    print(full_stack.keys())
    
    
    return triangulate_structure(full_stack, keys)

def triangulate_structure(full_stack, keys):
    pts, indices = render_body(full_stack,keys)
    triangles = [indices[i:i+3] for i in range(0, len(indices), 3)]
    
    return pts, triangles
    
def generate_mesh_from_array(slice_stack, z_cutoff=None):
    full_stack_int = {}
    keys = []


    for slice in slice_stack:
        xi, yi, zi = slice[::3], slice[1::3], slice[2::3]
        if z_cutoff is not None and zi[0] < z_cutoff:
            continue

        keys.append(zi[0])
        # Fit spline with s=0 (passing through all points)
        tck, u = interpolate.splprep([xi, yi], s=0, per=True)

        # Evaluate spline for 1000 evenly spaced points
        xj, yj = interpolate.splev(np.linspace(0, 1, 1000), tck)
        zj = [zi[0]]*1000
        full_stack_int[zi[0]] = []
        
        for i in range(len(zj)):
            full_stack_int[zi[0]].append(xj[i]) 
            full_stack_int[zi[0]].append(yj[i])
            
    
    
    
    return triangulate_structure(full_stack_int, keys)


def get_contour_from_ROI_name(ROI_name, RS):
    '''
    Gets the contour for the requested ROI_name in the RS file.
    '''
    for i, seq in enumerate(RS.StructureSetROISequence):
        if seq.ROIName == ROI_name:
            index = i
            break

    contour_coords = [] 
    
    if 'ContourSequence' in RS.ROIContourSequence[index]:
        i=0
        for ROI_contour_seq in RS.ROIContourSequence[index].ContourSequence:
            i+=1
#             print(i)
            contour_coords.append(ROI_contour_seq.ContourData) 
    else:
        print("Warning:",ROI_name,"has no contour sequence.")
            
    return contour_coords
def get_all_ROI_contours(list_ROIs,RS):
    '''
    Get dictionary of contours for each of the ROIs in list_ROIs.
    '''
    dict_contours = {}
    z_lists = []
#     print(list_ROIs)

    for roi_name in list_ROIs:

        all_contours_ROI = get_contour_from_ROI_name(roi_name, RS)
        all_contours_ROI = sorted(all_contours_ROI, key=lambda x: x[2])
        dict_contours[roi_name] = all_contours_ROI
        z_lists.append(sorted([float(roi[2]) for roi in all_contours_ROI]))
    return dict_contours, z_lists


def get_contour_stack(contour_name, dict_contours,cutoff=-1000):
    XN = []
    YN = []
    ZN = []
    fullN = []
    full_stack_N = []
    y_prev = 0
    x_prev = 0
    for c in dict_contours[contour_name]:
    #     print(c)
        test = c
    #     full = full + list(c)
        slice = []

        if c[2] < cutoff:
            continue

        for i in range(0,len(c),3):

            x = test[i]
            y = test[i+1]
            z = test[i+2]

            if not (y== y_prev and x==x_prev): # ensure interpolation functin oworks
                XN.append(x)
                YN.append(y)
                ZN.append(z)
                fullN.append([x,y,z])
                slice = slice + [x,y,z]
            y_prev = y
            x_prev = x

        full_stack_N.append(slice)
        
            # full_stack = full_stack_N
    
    full_stack_int = {}
    keys = []


    for slice in full_stack_N:
        
    #     print(slice)
    #     break
        xi, yi, zi = slice[::3], slice[1::3], slice[2::3]
#         print(zi[0])
        # Append starting coordinates to make it closed
    #     xi = np.r_[xi, xi]
    #     yi = np.r_[yi, yi]

        keys.append(zi[0])
        # Fit spline with s=0 (passing through all points)
        tck, u = interpolate.splprep([xi, yi], s=0, per=True)

        # Evaluate spline for 1000 evenly spaced points
        xj, yj = interpolate.splev(np.linspace(0, 1, 1000), tck)
        zj = [zi[0]]*1000
        full_stack_int[zi[0]] = []
        print
        for i in range(len(zj)):
            full_stack_int[zi[0]].append(xj[i]) 
            full_stack_int[zi[0]].append(yj[i])

    return full_stack_int, keys



    