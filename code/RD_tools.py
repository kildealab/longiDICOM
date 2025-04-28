import os, sys
import pydicom as dcm
import numpy as np

from rs_tools import get_contour_from_ROI_name

# def get_RD(patient_path, CT_file):
#     # simply looks for 
#     image_path = os.path.join(patient_path,CT_file)
#     # CT = dcm.read_file(image_path)
#     RD_file = [f for f in os.listdir(image_path) if f[0:2] == 'RD'][0]
#     RD = dcm.read_file(os.path.join(image_path,RD_file))

#     #TO DO:load dose and RP fiels
#     return RD,RD_file


debug=False




# NOTE this also will look for the smallest resolution dose fil ein the case that thaere are many.
def find_dose_file(CT_path):
    # note assumes that thefile is named 'RD...', would be slower but more comprehensive to open each file
    dose_files = [f for f in os.listdir(CT_path) if 'RD' in f]
    num_dose_files = len(dose_files)
    
    if num_dose_files == 0:
        raise FileNotFoundError("ERROR: NO DOSE FILES FOUND")
    
    RD = dcm.read_file(CT_path+'/'+dose_files[0]) 
    
    if num_dose_files == 1:
        return RD
    
    elif num_dose_files > 1:
        found = True
        smallest_spacing = RD[0x0028, 0x0030]
        most_frames = RD.NumberOfFrames
        
        for dose_file in dose_files[1:]:
            rd = dcm.read_file(CT_path+'/'+dose_file)
            spacing = rd[0x0028, 0x0030]
            frames = rd.NumberOfFrames
            
            if spacing[0] < smallest_spacing[0] and spacing[1] < smallest_spacing[1]:
                smallest_spacing = spacing
                RD = rd 
            elif frames > most_frames:
                most_frames = frames
                RD = rd
    return RD


def get_dose_in_gy(RD):
    dose_grid = RD.pixel_array
    dose_grid_gy = dose_grid * RD.DoseGridScaling
    return dose_grid_gy

def get_dose_spacing(RD):
    return [RD.PixelSpacing[0],RD.PixelSpacing[1],RD.GridFrameOffsetVector[1]-RD.GridFrameOffsetVector[0]]


def get_struct_dose_values(RS,RD,structure_name):
    xpos,ypos,zpos = get_dose_xyz(RD)
    mask_struct = create_binary_mask(RS,RD,structure_name,xpos,ypos,zpos)
    dose_gy = get_dose_in_gy(RD)
    dose_values = extract_dose_values(dose_gy,mask_struct)
    return dose_values


# The following three functions were adapted from Haley's RTDSM code
def get_dose_xyz(rtdose):
    xpos = np.arange(rtdose.Columns) * rtdose.PixelSpacing[0] + rtdose.ImagePositionPatient[0]
    ypos = np.arange(rtdose.Rows) * rtdose.PixelSpacing[1] + rtdose.ImagePositionPatient[1]
    zpos = np.asarray(rtdose.GridFrameOffsetVector) + rtdose.ImagePositionPatient[2]
    return xpos, ypos, zpos


def create_binary_mask(rtstruct, rtdose, structure_name, xpos, ypos, zpos):
    # Extract the contours for the specified structure
    contours =  get_contour_from_ROI_name(structure_name,rtstruct)
    
    # Create an empty image with the same dimensions as the dose grid
    dose_shape = rtdose.pixel_array.shape
    mask_image = np.zeros(dose_shape, dtype=np.uint8)
    
    # Fill the mask image with the contours
    for contour in contours:
        points = np.array(contour).reshape(-1, 3)
        for point in points:
            # Find the closest indices in the dose grid
            x_idx = np.argmin(np.abs(xpos - point[0]))
            y_idx = np.argmin(np.abs(ypos - point[1]))
            z_idx = np.argmin(np.abs(zpos - point[2]))
            mask_image[z_idx, y_idx, x_idx] = 1
    
    return mask_image

def extract_dose_values(dose_grid, mask_image):
    # Extract the dose values where the mask is 1
    dose_values = dose_grid[mask_image == 1]
    
    return dose_values
'''

def get_struct_dose_values(rtstruct, struct_name):
    organ_contours = get_contour_from_ROI_name(struct_name,rtstruct)
    flat_org = [i for sublist in organ_contours for i in sublist]
    flat_arr = np.array(flat_org)
    resh_arr = flat_arr.reshape(-1,3)
    # Map dose grid to organ contours and interpolate dose values
    dose_values = interpolate_dose(dose_grid, np.array(resh_arr), rtdose)
    return dose_values
'''


def resample_dose_map_2D(dose_map, new_spacing, old_spacing):

    scaling_factors = [old/new for old, new in zip(old_spacing,new_spacing)]
    original_size = np.array([len(dose_map), len(dose_map[0])]) # change to 3D
    
    new_size = np.round(original_size*scaling_factors).astype(int)
    resampled_dose_map = np.zeros(new_size)

    # to do make 3d
    for i in range(new_size[0]):
        for j in range(new_size[1]):

            original_index = np.array([i,j])/scaling_factors
            resampled_dose_map[i][j] = dose_map[int(original_index[0]), int(original_index[1])]            
        
    return resampled_dose_map

def resample_dose_map_3D(dose_map, new_spacing, old_spacing):

    scaling_factors = [old/new for old, new in zip(old_spacing,new_spacing)]
    original_size = np.array([len(dose_map[0]), len(dose_map[0][0]),len(dose_map)]) # change to 3D
    
    print("OG SIZE",original_size)
    new_size_xyz = np.round(original_size*scaling_factors).astype(int)
    new_size = [new_size_xyz[2],new_size_xyz[0],new_size_xyz[1]]
    print("NEW SIZE",new_size)
    print("scaling factors",scaling_factors)
    print(original_size*scaling_factors)
    resampled_dose_map = np.zeros(new_size)
#     return resampled_dose_map
    print(resampled_dose_map.shape)

    # to do make 3d
    for k in range(new_size[0]):
        for i in range(new_size[1]):
            for j in range(new_size[2]):
           
#                 print(k)
                original_index = (np.array([i,j,k])/scaling_factors).astype(int)
#                 print(original_index)
#                 print(int(original_index[2]), int(original_index[0]),int(original_index[1]))
                resampled_dose_map[k][i][j] = dose_map[ int(original_index[2]), int(original_index[0]),int(original_index[1])]            

    return resampled_dose_map
# TO DO: check why this is diff
'''

def resample_dose_map_3D(dose_map, new_spacing, old_spacing,new_size):

    scaling_factors = [old/new for old, new in zip(old_spacing,new_spacing)]
    original_size = np.array([len(dose_map[0]), len(dose_map[0][0]),len(dose_map)]) # change to 3D
    if debug:
        print("OG SIZE",original_size)
        # new_size_xyz = np.round(original_size*scaling_factors).astype(int)
        # new_size = [new_size_xyz[2],new_size_xyz[0],new_size_xyz[1]]
        print("NEW SIZE",new_size)
        print("scaling factors",scaling_factors)
        print(original_size*scaling_factors)
    resampled_dose_map = np.zeros(new_size)
#     return resampled_dose_map
    if debug:
        print(resampled_dose_map.shape)
    # resampled_dose_map[:new_size[2], :new_size[0], :new_size[1]] = dose_map[:new_size[2], :new_size[0], :new_size[1]]

#     # to do make 3d # to do -- this is exctremely slow.
#     for k in range(new_size[0]):
#         for i in range(new_size[1]):
#             for j in range(new_size[2]):

#                 original_index = (np.array([i,j,k])/scaling_factors).astype(int)
# #                 print(original_index)
# #                 print(int(original_index[2]), int(original_index[0]),int(original_index[1]))
#                 resampled_dose_map[k][i][j] = dose_map[ int(original_index[2]), int(original_index[0]),int(original_index[1])]            
    grid_k, grid_i, grid_j = np.meshgrid(np.arange(new_size[0]), np.arange(new_size[1]), np.arange(new_size[2]), indexing='ij') 
    original_indices = (np.array([grid_i, grid_j, grid_k]).transpose(1, 2, 3, 0) / scaling_factors).astype(int)
    # print(original_indices)
      # Debugging: Print the maximum values of original indices
    if debug:
        print("Max original indices:", np.max(original_indices, axis=(0, 1, 2)))

    # Ensure indices are within bounds
    original_indices = np.clip(original_indices, 0, [len(dose_map[0]) - 1, len(dose_map[0][0]) - 1, len(dose_map) - 1])
    if debug:
        print("clip to:", [len(dose_map) - 1, len(dose_map[0][0]) - 1, len(dose_map[0]) - 1])
    # Debugging: Print the modified original indices
    # print("Clipped original indices:", original_indices)
    resampled_dose_map = dose_map[original_indices[..., 2], original_indices[..., 0], original_indices[..., 1]]
    return resampled_dose_map
'''

def resize_dose_map(dose_map,new_size, spacing, new_origin, old_origin,default=0):

    resized_dose_map = np.zeros(new_size)

    x_img = int((new_origin[0]-old_origin[0])/spacing[0])
    y_img = int((new_origin[1]-old_origin[1])/spacing[1])
    if debug:
        print(x_img,y_img)
        print((new_origin[0]-old_origin[0])/spacing[0],(new_origin[1]-old_origin[1])/spacing[1])
        print("X",x_img, len(dose_map[0]))
        print("Y",y_img, len(dose_map))
        print(x_img+len(dose_map[0]))
    
    y_end = y_img+len(dose_map)
    x_end = x_img+len(dose_map[0])
    

    if debug:
        print('xy ends',x_end,y_end)
    if y_end > 512:
        dose_map = dose_map[:(new_size[1]-y_end)]
    if x_end > 512:
        dose_map = dose_map[:,:(new_size[0]-x_end)]
          
    resized_dose_map[y_img:y_end,x_img:x_end] = dose_map
    return resized_dose_map

def resize_dose_map_3D(dose_map,new_size, spacing, new_origin, old_origin,default=0):
    
    new_size_zxy = new_size[2],new_size[0],new_size[1]
    # print(new_size_zxy)
    resized_dose_map = np.zeros(new_size_zxy)
    z_image = int((new_origin[2] - old_origin[2])/spacing[2])
    if debug:
        print("z_img",z_image, (new_origin[2] - old_origin[2])/spacing[2])
    
    len_dose_map = len(dose_map) 
    if debug:
        print("len dose map",len_dose_map)
    # Crop dose map if starting index is negative
    if z_image < 0:       
        z_image = 0
        len_dose_map = len_dose_map + z_image
    # print("len dose map after < 0",len_dose_map)     
    
    for i,resized_index in enumerate(range(z_image,z_image+len_dose_map)):  # added zimage + lendose map. idk anymore
        if debug:
            print(i,resized_index)
            print("new or (DOSE)", new_origin[2])
            print("old or (IMG)", old_origin[2])
            print("spacing", spacing)
        resized_dose_map[resized_index] = resize_dose_map(dose_map[i],[new_size[0],new_size[1]],spacing,new_origin,old_origin,default=0)
    
    return resized_dose_map



def reverse_resize_dose_map_3D(dose_map,new_size, spacing, new_origin, old_origin,default=0):
    new_size_zxy = new_size[0],new_size[1],new_size[2]
    if debug:
        print(new_size_zxy)
    resized_dose_map = np.zeros(new_size_zxy)
    z_image = int((new_origin[2] - old_origin[2])/spacing[2])
    if debug:
        print("z_img",z_image, (new_origin[2] - old_origin[2])/spacing[2])
    # 
    len_dose_map = len(dose_map)
    if debug:
        print("len dose map",len_dose_map)
    # Crop dose map if starting index is negative
    if z_image < 0:       
        z_image = 0
        len_dose_map = len_dose_map + z_image
    if debug:
        print("len dose map after < 0",len_dose_map)    
    
    # z_start = int((new_origin[2]-old_origin[2])/spacing[2])

    
    for i,resized_index in enumerate(range(z_image,z_image+new_size[0]-1)):  # added zimage + lendose map. idk anymore
        # print(i,resized_index)
        # if i > lennew_size[0]
        if debug:
        
            print(i,resized_index)
            print("new or (DOSE)", new_origin[2])
            print("old or (IMG)", old_origin[2])
            print("spacing", spacing)
            print("len dose map resized:",len(resized_dose_map))
            print("len dose map or?:", len(dose_map))
            print()
        #if dose map is smaller than resized dose map, leave with zeros?
        if i < len(dose_map):
            try:
                resized_dose_map[resized_index] = reverse_resize_dose_map(dose_map[i],[new_size[1],new_size[2]],spacing,new_origin,old_origin)
            except Exception as e:

                print("Leaving zero in dose map at index", resized_index,"because :",e)
    return resized_dose_map

def reverse_resize_dose_map(dose_map,new_size, spacing,  old_origin,new_origin):
    resized_dose_map = np.zeros(new_size)

    x_start = int((new_origin[0]-old_origin[0])/spacing[0])
    y_start = int((new_origin[1]-old_origin[1])/spacing[1])
  
    
    y_end = y_start+new_size[0]#len(dose_map)
    x_end = x_start+new_size[1]#len(dose_map[0])
        
    resized_dose_map = dose_map[y_start:y_end,x_start:x_end]
    


    return resized_dose_map


