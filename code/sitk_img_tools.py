import os
import pydicom as dcm
import SimpleITK as sitk
def get_start_position_sitk(sitk_image):

    origin = sitk_image.GetOrigin()
    spacing = sitk_image.GetSpacing()
    
    return origin[0], origin[1], origin[2], spacing

       
def generate_sitk_image(DCM_path):
        """
        generate_sitk_image     Reads DICOM file at DCM_path as a SITK (SimpleITK) image.

        :param DCM_path: Path to dicom series directory. 

        :returns: The DICOM image in SITK format
        """

        series_id = ''

        for file in os.listdir(DCM_path):
                if 'CT' in file:
                        series_id = dcm.read_file(DCM_path+file).SeriesInstanceUID
                        continue
           
        fixed_reader = sitk.ImageSeriesReader()
        dicom_names = fixed_reader.GetGDCMSeriesFileNames(DCM_path, seriesID=series_id)
        fixed_reader.SetFileNames(dicom_names)
        fixed_image = fixed_reader.Execute()

        return fixed_image
        

