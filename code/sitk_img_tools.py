import os
import pydicom as dcm
import SimpleITK as sitk
import time
from pydicom.uid import generate_uid

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
        

def save_dicoms(data_directory, sitk_image, output_directory):

    # Copy relevant tags from the original meta-data dictionary (private tags are also accessible).
    # The following tags should be all those that exist in a CT or CBCT dicom file. 
    tags_to_copy = ["0010|0010", # Patient Name
                    "0010|0020", # Patient ID
                    "0010|0030", # Patient Birth Date
                    "0020|000D", # Study Instance UID, for machine consumption
                        "0020|0010", # Study ID, for human consumption
                    "0008|0020", # Study Date
                    "0008|0030", # Study Time
                    "0008|0050", # Accession Number
                    "0008|0060"  # Modality
    ]

    #The following tags were extracted from a random CBCT scan from the MUHC.
    tags_to_copy_CBCT = ["0002|0000",  #File Meta Information Group Length 
                     "0002|0001", # File Meta Information Version      
                     "0002|0002", # Media Storage SOP Class UID         
                     "0002|0003", # Media Storage SOP Instance UID      
                     "0002|0010", # Transfer Syntax UID                 
                     "0002|0012", # Implementation Class UID           
                     "0002|0013", # Implementation Version Name        
    #-------------------------------------------------
                     "0008|0005", # Specific Character Set              
                     "0008|0008", # Image Type                       
                     "0008|0012", # Instance Creation Date            
                     "0008|0013", # Instance Creation Time           
                     "0008|0016", # SOP Class UID                    
                     "0008|0018", # SOP Instance UID                 
                     "0008|0020", # Study Date                       
                     "0008|0021", # Series Date                      
                     "0008|0022", # Acquisition Date               
                     "0008|0023", # Content Date                        
                     "0008|0030", # Study Time                         
                     "0008|0031", # Series Time                      
                     "0008|0032", # Acquisition Time                  
                     "0008|0033", # Content Time                    
                     "0008|0050", # Accession Number                 
                     "0008|0060", # Modality                            
                     "0008|0070", # Manufacturer                       
                     "0008|0090", # Referring Physician's Name       
                     "0008|1090", # Manufacturer's Model Name           
                     "0010|0010", # Patient's Name                   
                     "0010|0020", # Patient ID                        
                     "0010|0030", # Patient's Birth Date             
                     "0010|0040", # Patient's Sex                    
                     "0012|0062", # Patient Identity Removed         
                     "0012|0063", # De-identification Method           
                     "0012|0064", #  De-identification Method Code Sequence  2 item(s) ---- 
                        "0008|0100", # Code Value                        
                        "0008|0102", # Coding Scheme Designator        
                        "0008|0104", # Code Meaning                   
                   #     ---------
                        "0008|0100", # Code Value                        
                        "0008|0102", # Coding Scheme Designator         
                        "0008|0104", # Code Meaning                      
                  #     ---------
                     "0018|0022", # Scan Options                      
                     "0018|0050", # Slice Thickness                    
                     "0018|0060", # KVP                                
                     "0018|0090", # Data Collection Diameter           
                     "0018|1020", # Software Versions               
                     "0018|1100", # Reconstruction Diameter          
                     "0018|1110", # Distance Source to Detector        
                     "0018|1111", # Distance Source to Patient          
                     "0018|1120", # Gantry/Detector Tilt             
                     "0018|1130", # Table Height                    
                     "0018|1140", # Rotation Direction                
                     "0018|1150", # Exposure Time                     
                     "0018|1151", # X-Ray Tube Current                
                     "0018|1152", # Exposure                           
                     "0018|1160", # Filter Type                    
                     "0018|1190", # Focal Spot(s)                  
                     "0018|1210", # Convolution Kernel            
                     "0018|5100", # Patient Position              
                     "0020|000d", # Study Instance UID                
                     "0020|000e", # Series Instance UID              
                     "0020|0010", # Study ID                         
                     "0020|0011", # Series Number                    
                     "0020|0012", # Acquisition Number              
                     "0020|0013", # Instance Number                   
                     "0020|0032", # Image Position (Patient)           
                     "0020|0037", # Image Orientation (Patient)       
                     "0020|0052", # Frame of Reference UID             
                     "0020|1040", # Position Reference Indicator        
                     "0028|0002", # Samples per Pixel               
                     "0028|0004", # Photometric Interpretation        
                     "0028|0010", # Rows                               
                     "0028|0011", # Columns                          
                     "0028|0030", # Pixel Spacing                     
                     "0028|0100", # Bits Allocated                     
                     "0028|0101", # Bits Stored                         
                     "0028|0102", # High Bit                           
                     "0028|0103", # Pixel Representation               
                     "0028|1050", # Window Center                     
                     "0028|1051", # Window Width                       
                     "0028|1052", # Rescale Intercept                   
                     "0028|1053", # Rescale Slope                      
                     "0028|1054", # Rescale Type                        
                     "300a|0122", # Patient Support Angle          
                     "300a|0129", # Table Top Longitudinal Position    
                     "300a|012a", # Table Top Lateral Position        
                     "7fe0|0010" # Pixel Data                         
                     ]

    modification_time = time.strftime("%H%M%S")
    modification_date = time.strftime("%Y%m%d")


    tags_to_add = ["0008|103e", #series description
                  ]

    
    series_IDs = sitk.ImageSeriesReader.GetGDCMSeriesIDs(data_directory)
    if not series_IDs:
        print("ERROR: given directory \""+data_directory+"\" does not contain a DICOM series.")
        sys.exit(1)
    series_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(data_directory, series_IDs[0])

    series_reader = sitk.ImageSeriesReader()
    series_reader.SetFileNames(series_file_names)

    # Configure the reader to load all of the DICOM tags (public+private):
    # By default tags are not loaded (saves time).
    # By default if tags are loaded, the private tags are not loaded.
    # Explicitly configure the reader to load tags, including the private ones.
    series_reader.MetaDataDictionaryArrayUpdateOn()
    series_reader.LoadPrivateTagsOn()
    image3D = series_reader.Execute() #image3D is the unregistered CBCT

    filtered_image = sitk_image

    # Write the 3D image as a series
    # IMPORTANT: There are many DICOM tags that need to be updated when you modify an
    #            original image. This is a delicate opration and requires knowlege of
    #            the DICOM standard. This example only modifies some. For a more complete
    #            list of tags that need to be modified see:
    #                 http://gdcm.sourceforge.net/wiki/index.php/Writing_DICOM
    writer = sitk.ImageFileWriter()
    
    # Use the study/series/frame of reference information given in the meta-data
    # dictionary and not the automatically generated information from the file IO
    writer.KeepOriginalImageUIDOn() 

    CT_dcm = dcm.read_file(data_directory +'/'+ [f for f in os.listdir(data_directory) if 'CT' in f][0])
    
    #Print all dicom tags if troubleshooting is needed:
    #print(CT_dcm)
    
    direction = filtered_image.GetDirection()

    slice_thickness = filtered_image.GetSpacing()[2]

    # Copy some of the tags and add the relevant tags indicating the change.
    # For the series instance UID (0020|000e), each of the components is a number, cannot start
    # with zero, and separated by a '.' We create a unique series ID using the date and time.
    # tags of interest:
    series_tag_values = [(k, series_reader.GetMetaData(0,k)) for k in tags_to_copy_CBCT if series_reader.HasMetaDataKey(0,k)] + \
                     [("0008|0031",modification_time), # Series Time
                      ("0008|0021",modification_date), # Series Date
                      # ("0018|0050", '3.0') #hard-coding the slice thickness because CTs always have thickness = 3.0 mm # REMOVE HARDCODING
                      ("0018|0050",str(slice_thickness))
                     ]
    
    # print(series_tag_values)


    #TO DO for generalizability: read in output directory from .env and append patient number + CT or CBCT name from local variables. 
    if not os.path.exists(output_directory):
        os.mkdir(output_directory)


    for i in range(filtered_image.GetDepth()):
        image_slice = filtered_image[:,:,i]
        # Tags shared by the series.
        for tag, value in series_tag_values:
            image_slice.SetMetaData(tag, value)
        # Slice specific tags.
        image_slice.SetMetaData("0008|0012", time.strftime("%Y%m%d")) # Instance Creation Date
        image_slice.SetMetaData("0008|0013", time.strftime("%H%M%S")) # Instance Creation Time
        image_slice.SetMetaData("0020|0032", '\\'.join(map(str,filtered_image.TransformIndexToPhysicalPoint((0,0,i))))) # Image Position (Patient)
        image_slice.SetMetaData("0020|0013", str(i)) # Instance Number
        image_slice.SetMetaData("0018|0050",str(slice_thickness))
        # image_slice.SetMetaData("0018|0050", '3.0') #setting thickness to 3.0mm just in case it wasn't previously copied
        
        # Write to the output directory and add the extension dcm, to force writing in DICOM format.
        new_UID = generate_uid()
        image_slice.SetMetaData("0008|0018", new_UID)
        writer.SetFileName(os.path.join(output_directory,'CT.'+new_UID+'.dcm'))
        writer.Execute(image_slice) 
    return()