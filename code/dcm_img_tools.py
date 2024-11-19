
from pydicom.uid import generate_uid

def get_start_position_dcm(CT_path):
    positions = []
    for f in [file for file in os.listdir(CT_path) if 'CT' in file]:
        d = dcm.dcmread(CT_path+f)
        positions.append(d.ImagePositionPatient)

    # dcm.dcmread(patient_path + '20190712_CT_05_JUL_2019/CT.1.2.246.352.221.461737655324817980512720114456327223461.dcm')

    positions = sorted(positions, key=lambda x: x[-1])
    start_z = positions[0][2]
    start_x = positions[0][0]
    start_y = positions[0][1]
    pixel_spacing = d.PixelSpacing
    
    return start_x, start_y, start_z, spacing



def get_image_slice(start_z, z_smg, spacing):
    img_slice = int((abs(start_z - z_smg)/spacing[2]))
    return img_slice


def save_dicom(slices,save_path,patient, CT_file,new_UID=False,new_id = ''):
    if new_UID:
        new_series_instance_uid = generate_uid()
        new_study_instance_uid = generate_uid()
    
    for slice in slices:
        if not os.path.exists(save_path+patient):
            os.makedirs(save_path+patient)
        if not os.path.exists(save_path+patient+'/'+CT_file):
            os.makedirs(save_path+patient+'/'+CT_file)
        
        if not new_UID:
            slice.save_as(save_path+patient+'/'+CT_file+'/'+'CT.'+slice.SOPInstanceUID+'.dcm')
        else:
            new_sop_instance_uid = generate_uid()
            
            # Replace UIDs
            slice.SOPInstanceUID = new_sop_instance_uid
            slice.SeriesInstanceUID = new_series_instance_uid
            slice.StudyInstanceUID = new_study_instance_uid
            if new_id !='':
                slice.PatientID = new_id
                # slice.PatientName = CT_path.split('/')[-2] # can add paitent name if u want
            
            # Save the updated DICOM file
            slice.save_as(output_file_path+'CT.'+new_sop_instance_uid+'.dcm')

        


def change_uids(CT_path, output_file_path):
    new_series_instance_uid = generate_uid()
    new_study_instance_uid = generate_uid()
    
    for f in [f for f in os.listdir(CT_path) if 'CT' in f]:
        # Load the DICOM file
        dicom_data = dcm.dcmread(os.path.join(CT_path,f))

        
        
        # Generate new UIDs
        new_sop_instance_uid = generate_uid()
       
        # Replace UIDs
        dicom_data.SOPInstanceUID = new_sop_instance_uid
        dicom_data.SeriesInstanceUID = new_series_instance_uid
        dicom_data.StudyInstanceUID = new_study_instance_uid

        dicom_data.PatientID = 'non_anon_test_id'
        dicom_data.PatientName = CT_path.split('/')[-2]
        
        # Save the updated DICOM file
        dicom_data.save_as(output_file_path+'CT.'+new_sop_instance_uid+'.dcm')