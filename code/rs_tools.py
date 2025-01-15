# rcParams['figure.figsize'] = 11.7,8.27
# rcParams['font.size'] = 22


import os
import numpy as np
import matplotlib.pyplot as plt
def find_RS_file(path):
    for f in os.listdir(path):
        if 'RS' in f:
            return f



def find_ROI_names(RS, keyword=''):
    '''
    find_ROI_names  finds all contour names in RT Structure Set File containing keyword, 
                    while ignoring those containing 'nos' and 'z_'.
    
    :param RS: the RS file opened by pydicom
    :param keyword: The keyword to search the ROIs for. If blank returns all ROIs.
    
    :returns: list of ROI names containing keyword.
    '''
    ROI_names = []

    for seq in RS.StructureSetROISequence:
        roi_name = seq.ROIName
        if keyword.lower() in roi_name.lower() and 'nos' not in roi_name.lower() and 'z_' not in roi_name.lower():
            ROI_names.append(seq.ROIName)
    return ROI_names



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
        
        for ROI_contour_seq in RS.ROIContourSequence[index].ContourSequence:
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


def get_avg_ROI_z_and_slice(z_lists):
    '''
    Gets the average (ie middle) z-value of the set of contour slices.
    
    TO DO: divide by zero error encountered when submandibular gland contour doesn't exist.
    '''
    z_avg = 0
    # print("Z:",z_lists)

    for z_list in z_lists:
        z_avg += (np.sum(z_list)/len(z_list))

    z_avg = z_avg/len(z_lists)
    roi_slice = np.argmin(abs(z_lists[0] - z_avg)) #TO DO: what if not in this list? but it should ebf ine
    z_smg = z_lists[0][roi_slice]

    return roi_slice, z_smg

def get_lowest_ROI_z_and_slice(z_lists):
    '''
    Gets the lowest z-value of the set of contour slices.
    
    '''
    # print(z_lists)
    z_min = min([min(z) for z in z_lists])
    # print(z_lists[0])
    # print(z_min)
    roi_slice = np.argmin([abs(z-z_min) for z in z_lists[0]]) 
    # roi_slice = 0
    # print("slice",roi_slice)
    # print("zmin",z_min)

    # z_avg = 0

    # for z_list in z_lists:
    #   z_avg += (np.sum(z_list)/len(z_list))

    # z_avg = z_avg/len(z_lists)
    # roi_slice = np.argmin(abs(z_lists[0] - z_avg)) #TO DO: what if not in this list? but it should ebf ine
    # z_smg = z_lists[0][roi_slice]

    return roi_slice, z_min



def get_ROI_pixel_array(roi_array,start_x,start_y,pixel_spacing):
    '''
    Get the pixel positions (rather than the x,y coords) of the contour array so it can be plotted.
    '''
#     roi_array = dict_contours[subgland_ROI_names[0]][roi_slice]
    x = []
    y = []

    for i in range(0,len(roi_array),3):
        x.append((roi_array[i] - start_x)/pixel_spacing[0])
        y.append((roi_array[i+1] - start_y)/pixel_spacing[1]) 
        
    return x, y




def plot_ROI(image_array, x, y):
    plt.imshow(image_array)
    if len(x)==1 and len(y)==1:
        plt.plot(x,y,'ro')
    else:
        plt.plot(x, y, 'r-')
#     plt.plot(256,256,'mo')
    plt.show()


def plot_all_contours(RS,image,slice_num,origin,spacing,ignore_terms=[],legend=False):
    contours_not_on_slice = []
    plt.subplot(2, 2, 3)
    all_ROIs = [r for r in find_ROI_names(RS)]
    for term in ignore_terms:
        all_ROIs = [r for r in all_ROIs if term.lower() not in r.lower()]
    
#     dict_contours, z_lists_b = get_all_ROI_contours(all_ROIs, RS)
    colours= get_ROI_colour_dict(RS)
    z_slice = image_to_xyz_coords_single(slice_num, spacing[2],origin[2])[0]
#     xyz_to_image_coords_single(X,spacing,origin):
    # print(z_slice)
    
    
    plt.imshow(image[slice_num],cmap='gray')
    for roi in all_ROIs:
        # print(roi)
        dict_contours, z_lists = get_all_ROI_contours([roi], RS)
        # print("X")
#         if len(dict_contours) >1:
        for i,r in enumerate(dict_contours):
            if r == roi:
                break
        try:       
            roi_slice = get_ROI_slice(z_slice,z_lists[i])
            # print(len(roi_slice))
            
            c =colours[roi]
            for s in roi_slice:
                # print(s)
                # print("**")
                roi_x, roi_y = get_ROI_pixel_array(dict_contours[roi][s],origin[0],origin[1],spacing)
                plt.plot(roi_x,roi_y,'-',color=  (c[0]/255,c[1]/255,c[2]/255),label=roi)
        except Exception as e:
            contours_not_on_slice.append(roi)
#             print(roi,"not on slice.")
        
    if legend:
        plt.legend(prop={'size': 8})
    # print("Other contours not on slice:",contours_not_on_slice)
    
    

def get_ROI_colour_dict(RS):
    index_colour_dict = {}
    for ROI_contour_seq in RS.ROIContourSequence:
        index_colour_dict[ROI_contour_seq.ReferencedROINumber] = ROI_contour_seq.ROIDisplayColor
   
    name_colour_dict = {}
    for seq in RS.StructureSetROISequence:
        name_colour_dict[seq.ROIName] = index_colour_dict[seq.ROINumber]

    return name_colour_dict


def image_to_xyz_coords_single(X,spacing,origin):
    if type(X) is not list:
        X = [X]
    X_new = []
    for x in X:
        X_new.append(x*spacing+origin)
        
    return X_new