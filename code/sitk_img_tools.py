
def get_start_position_sitk(sitk_image):

    origin = sitk_image.GetOrigin()
    spacing = sitk_image.GetSpacing()
    
    return origin[0], origin[1], origin[2], spacing


