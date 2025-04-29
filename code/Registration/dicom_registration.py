

# from RegisterImages.WithDicomReg import register_images_with_dicom_reg, sitk
import SimpleITK as sitk
import os, sys
import pydicom as dcm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import json
from datetime import datetime
import csv

sys.path.append("../")
# from dicom_registration import *
from Registration.registration_callbacks import *
from Registration.registration_utilities import *
# from Slice_Selection.slice_selection import *
from rs_tools import *
from sitk_img_tools import *
from Registration.registration_core import *

rcParams['figure.figsize'] = 11.7,8.27
rcParams['font.size'] = 22

debug_print = False     

dict_class_UID = {'1.2.840.10008.5.1.4.1.1.2': 'CT', '1.2.840.10008.5.1.4.1.1.481.1': 'RI', '1.2.840.10008.5.1.4.1.1.4': 'MR', '1.2.840.10008.5.1.4.1.1.128':'PE'}
replan = False
second_replan = False
patient_path = ''
has_dicom_reg = True
image_dict = {}
CT_list = []

transform_save_path = 'insert-path-here'#'/data/kayla/HNC_images/transforms/'
CT_CBCT_save_path = 'insert-path-here'#'/data/kayla/HNC_images/FINAL_withreg/CT_CBCT/'
CBCT_CBCT_save_path = 'insert-path-here'#'/data/kayla/HNC_images/FINAL_withreg/CBCT_CBCT/'


#TO DO: make nicer, especialy replan part
def get_file_lists(patient_path = patient_path,return_dict=False):
        """
        get_file_lists  gets lists of CTs and CBCTs in patient folder whose dir names conform to the following formats:
                                        CT: "CT" in 9-10th index position and 23 char length.
                                        CBCT: "kV" in 9-10th index position

        """

        global CT_list
        global replan
        global image_dict

        if return_dict:
                image_dict = {}
                # image_dict.clear()
                
        replan = False
        # Get list of CT directories
        CT_list = [d for d in os.listdir(patient_path) if d[9:11] == 'CT' and len(d) == 23]
        CT_list.sort()


        CT_list.sort(key=lambda x: datetime.strptime(x[12:], "%d_%b_%Y"))
        if debug_print:
                print(CT_list)
        # Add CT UID to image dict
        CT_UIDs = []
        for CT in CT_list:
                files = [f for f in os.listdir(patient_path+CT) if 'CT' in f]
                CT_UIDs.append(dcm.read_file(patient_path+CT+"/"+files[0]).FrameOfReferenceUID)
                image_dict[CT] = {}
                image_dict[CT]['UID'] = dcm.read_file(patient_path+CT+"/"+files[0]).FrameOfReferenceUID

        # Get list of CBCT directories
        CBCT_list_replan = []
        CBCT_list = [d for d in os.listdir(patient_path) if d[9:11] == 'kV']
        if debug_print:
                print(CBCT_list)
        CBCT_list.sort()

        # Raise errors if no CTs found, or if > 2 CTs found
        if len(CT_list) == 0:
                raise NotADirectoryError('No CT directories were found. Please ensure the following naming convention was used: "YYYYMMDD_CT_DD_MMM_YYYY".')
        
        if len(CT_list) > 2: # Set replan to true if > 1 CT
                # global replan
                second_replan = True
        if len(CT_list) > 1: # Set replan to true if > 1 CT
                # global replan
                replan = True

                # if len(CT_list) != 2:
                #       raise Warning('More than 2 CT directories found. This code may not perform as expected, as it was made for exactly one replan (2 CTs)')

        if debug_print:
                print(replan)
        #TO DO: check for whne dates don't work
        #TO DO: REDO THIS PART --> GRAB DATES FROM ACTUAL TABLE, THIS ISN'T NECESSARILLY TRUE
        
        if replan:
                #date_replan = CT_list[1][0:8]
                # alternate dates since dated CTs often wrong
                # date_fx_1 = CBCT_list[0][0:8]
                fx_1s = sorted([i for i in CBCT_list if int(i.split("_")[-1][:-1]) == 1])
                if debug_print:
                        print(fx_1s)
                if len(fx_1s)==1:
                        CBCT_list_replan = []
                else:

                        date_replan = fx_1s[1].split("_")[0]
                        # if second_replan:
                        #         second_date_replan = fx_1s[2].split("_")[0]
                        #         CBCT_list_replan = [CBCT for CBCT in CBCT_list if int(CBCT[0:8]) >= int(date_replan) and int(CBCT[0:8]) < int(second_date_replan)]
                        #         CBCT_list_second_replan = [CBCT for CBCT in CBCT_list if int(CBCT[0:8]) >= int(second_date_replan)]
                        # else:
                        CBCT_list_replan = [CBCT for CBCT in CBCT_list if int(CBCT[0:8]) >= int(date_replan)]
                        
                        CBCT_list =  [CBCT for CBCT in CBCT_list if int(CBCT[0:8]) < int(date_replan)]

                if debug_print:
                        print(CBCT_list)
                        print("REPLNA", CBCT_list_replan)
                        # if second_replan:
                        #         print("2nd REPLNA", CBCT_list_second_replan)

                

                '''
                # Divide CBCT list if before or after replan
                CBCT_list_replan = [CBCT for CBCT in CBCT_list if int(CBCT[0:8]) > int(date_replan)]
                CBCT_list_same_date = [CBCT for CBCT in CBCT_list if int(CBCT[0:8]) == int(date_replan)]
                CBCT_list =  [CBCT for CBCT in CBCT_list if int(CBCT[0:8]) < int(date_replan)]

                

                # Organizing CBCTs with same date as replan CT into pre-post replan             
                for CBCT in CBCT_list_same_date:
                        # print("same date:", CBCT)
                        # print(CBCT.split('_')[-1][:-1])
                        fx = CBCT.split('_')[-1][:-1]
                        # print(fx)
                        # print(CBCT_list[-1].split('_')[-1][:-1])
                        if int(fx) > int(CBCT_list[-1].split('_')[-1][:-1]):
                                CBCT_list.append(CBCT)
                        else:
                                CBCT_list_replan.insert(0,CBCT)
                '''

                #to do - make better
                if len(CT_list) == 3:

                        # lCBCTist_replan2 = []
                        # date_fx_1 = CBCT_list[1][0:8]
                        # fx_1s = sorted([i for i in CBCT_list if int(i.split("_")[-1][:-1]) == 1])
                        
                        date_replan = fx_1s[2].split("_")[0]

                        CBCT_list_replan2 = [CBCT for CBCT in CBCT_list_replan if int(CBCT[0:8]) >= int(date_replan)]
                        CBCT_list_replan =  [CBCT for CBCT in CBCT_list_replan if int(CBCT[0:8]) < int(date_replan)]

                        image_dict[CT_list[2]]['CBCTs'] = CBCT_list_replan2
                        if debug_print:
                                print(fx_1s)
                                print("CT_list2:",CBCT_list_replan2)
                                



                image_dict[CT_list[1]]['CBCTs'] = CBCT_list_replan
                if debug_print:
                        # print(fx_1s)
                        print("CT_list1:",CBCT_list_replan)


        
        image_dict[CT_list[0]]['CBCTs'] = CBCT_list
        if debug_print:
                print("CTlist:",CBCT_list)
        if return_dict:
                return image_dict


        
 

# Should be [0,0,0]
# TO DO: Deal with isocenter
def get_acq_isocenter(CBCT_path):

        RS_files = [f for f in os.listdir(CBCT_path) if "RS" in f]
        
        if len(RS_files) == 0:
                # return [0,0,0]
                #TODO: fix
                raise FileNotFoundError('No Structure Set File Found')
        if len(RS_files) > 1:
                # raise Warning('More than one Structure Set File found')
                print("Warning: More than one Structure Set File found.")
        RS = dcm.dcmread(CBCT_path+RS_files[0])
        names = find_ROI_names(RS,'Acq')
        isocenter = get_contour_from_ROI_name(names[0], RS)
        return isocenter

# Should be [0,0,0]
# TO DO: Deal with isocenter
def get_acq_isocenter(CBCT_path):
        RS_files = [f for f in os.listdir(CBCT_path) if "RS" in f]
        
        if len(RS_files) == 0:
                raise FileNotFoundError('No Structure Set File Found')
        if len(RS_files) > 1:
                print('Warning: More than one Structure Set File found')
        RS = dcm.dcmread(CBCT_path+RS_files[0])
        names = find_ROI_names(RS,'Acq')
        isocenter = get_contour_from_ROI_name(names[0], RS)
        return isocenter


def get_unregistered_dict(patient_path = patient_path):
        global image_dict
        image_dict = {}
        get_file_lists(patient_path)
        if debug_print:
                
                print("Replan:", replan)

        for CT in image_dict:
                cbct_list = []
                for cbct in image_dict[CT]['CBCTs']:
                        
                        cbct_path = patient_path+cbct+'/'
                        CBCT_sitk = generate_sitk_image(cbct_path)
                        cbct_list.append(CBCT_sitk)
                image_dict[CT]['cbct_list'] = cbct_list
        return image_dict

def register_CT_CBCT(CT_CBCT_save_path,patient_path = patient_path):
        if debug_print:
                print(patient_path)
        global image_dict
        image_dict.clear()
        replan = False

        min_zs = []
        max_zs = []

        slices_below = []
        slices_above = []

        count = 0

        
        f = open(CT_CBCT_save_path+'reg_stats.csv', 'a',newline='')
        g = open(CT_CBCT_save_path+'mandible_positions.csv', 'a',newline='')

        # create the csv writer
        writer = csv.writer(f)
        writer_g = csv.writer(g)

        # global image_dict
        # global patient_path
        image_dict = get_unregistered_dict(patient_path)
        
        if debug_print:
                print(patient_path)
        # print(len(image_dict)


        for CT in image_dict:
          
                CT_path = patient_path + CT + "/"
        #     CT_path = patient_path+ '20190122_CT_14_JAN_2019/'
                CT_sitk = generate_sitk_image(CT_path)
                if debug_print:
                        print(CT)

                RS_file = find_RS_file(CT_path)
                RS = dcm.dcmread(CT_path+RS_file)
                subgland_ROI_names = find_ROI_names(RS,keyword='mandible')
                if len(subgland_ROI_names) ==0:
                        print("WARNING: Patient",patient,CT,"has no mandible.")
                if debug_print:
                        print(subgland_ROI_names)
                dict_contours, z_lists = get_all_ROI_contours(subgland_ROI_names, RS)
                roi_slice, z_smg = get_lowest_ROI_z_and_slice(z_lists)
                



                for i,CBCT_sitk in enumerate(image_dict[CT]['cbct_list']):
                        if debug_print:
                                print(patient_path)
                        CBCT = image_dict[CT]['CBCTs'][i]
                        if debug_print:
                                print(CBCT)
                        moving_resampled, transform, metric, condition = register_images_without_dicom_reg(CBCT_sitk, CT_sitk)
           
                        if debug_print:
                                print(metric, condition)
                
                        writer.writerow([patient_path.split("/")[-2],CBCT,metric,condition])

                        save_transformation(transform,CBCT,CT_CBCT_save_path+patient_path.split("/")[-2]+'/')

                        
                        z_trans = transform.GetParameters()[-1]
                        z_cbct = z_smg - z_trans
                        if debug_print:
                                print(z_cbct)
                        start_x, start_y, start_z, spacing = get_start_position_sitk(CBCT_sitk)
                        slice_ind = get_image_slice(start_z, z_cbct, spacing)
                        if debug_print:
                                print(slice_ind)
                        slices_below.append(slice_ind)
                        total_slices = CBCT_sitk.GetSize()[-1]
                        slice_above = total_slices - 1 - slice_ind

                        slices_above.append(slice_above)

                        writer_g.writerow([patient_path.split("/")[-2],CBCT,slice_ind,slice_above])
        
        f.close()
        g.close()
        return slices_below, slices_above, image_dict


        
def register_CBCT_CBCT_crop(slices_below, slices_above, image_dict, patient_path = patient_path):
        f = open(CBCT_CBCT_save_path+'reg_stats.csv', 'a',newline='')


        # create the csv writer
        writer = csv.writer(f)
        counter = 0

        min_start = min(slices_below)
        min_end = min(slices_above)

        for CT in image_dict:
            ref_CBCT = image_dict[CT]['CBCTs'][0]
            break

        for CT in image_dict:   
                resampled_list = []
                for i, CBCT_sitk in enumerate(image_dict[CT]['cbct_list']):
                        
                        CBCT = image_dict[CT]['CBCTs'][i]

                        mand_slice = slices_below[counter]
                        start = mand_slice - min_start
                        end = mand_slice + min_end
                        
                        # crop_cbct = CBCT_sitk[:,:,start:end]
                        crop_cbct = CBCT_sitk
                        counter+=1
                        
                        if image_dict[CT]['CBCTs'][i]==ref_CBCT:
                                ref_sitk = CBCT_sitk[:,:,start:end]#crop_cbct
                                resampled_list.append(ref_sitk)
                                continue
                        
                        if debug_print:
                                print(image_dict[CT]['CBCTs'][i])
                        moving_resampled, transform, metric, condition = register_images_without_dicom_reg(ref_sitk, crop_cbct)
        #         moving_resampled = crop_cbct
                        if debug_print:
                                print(metric, condition)
                        writer.writerow([patient_path.split("/")[-2],CBCT,metric,condition])

                        save_transformation(transform,CBCT,CBCT_CBCT_save_path+patient_path.split("/")[-2]+'/')

                        resampled_list.append(moving_resampled)
                image_dict[CT]['resampled_cbcts'] = resampled_list

        f.close()
        produce_plots(False)
        return image_dict


def register_CBCT_CT(CT, CBCT_list,use_reg_file = True,patient_path=patient_path):
        """
        register_CBCT_CT        Registers list of CBCTs to CT.

        :param CT: Name of CT directory.
        :param CBCT_list: List of names of CBCT directries to be resampled.

        :returns: List of registered sitk CBCT images and isocenters.
        """
        # use_reg_file=False
        resampled_cbct_list = []
        isocenter_list = []
        matrices = []
        start_positions = []
        registration_file = ''

        CT_sitk = generate_sitk_image(patient_path+CT+'/')

        
        # Loop through al CBCTs in list
        for cbct in CBCT_list:
                if debug_print:
                        print(cbct)
                cbct_path = patient_path+cbct+'/'
                CBCT_sitk = generate_sitk_image(cbct_path)
                isocenter = get_acq_isocenter(cbct_path)[0]
                start_position = get_start_position_sitk(CBCT_sitk)
                start_positions.append(start_position)
                # print(isocenter)

                # Find registration file for CBCT directory
                registration_file=''
                for f in os.listdir(cbct_path):
                        if f[0:2] == 'RE':
                                registration_file = cbct_path + f
                                continue
                
                if debug_print:
                        print("RE - ", registration_file)
                # If no registration file, register images with optimizer, otherwise use dicom reg file
                if registration_file =='' or use_reg_file==False:# or True: #use_reg_file == False:
                        # raise Exception("NO REGISTRATION FILES FOUND")
                    #TO DO -- REMOVE THIS, JUST THROWING EXCEPTION FOR NOW
                        print("OPTIMIZER")
                        with open("/data/kayla/reg_report2.txt", "a") as file:
                            # Write the new data to the file
                            print(patient_path.split("/")[-2], cbct, "--> no reg file\n")
                            file.write(patient_path.split("/")[-2]+ cbct+ "--> no reg file\n")
                
                        print(patient_path,cbct)
                        has_dicom_reg = False
                        resampled_cbct = None
                        # raise SystemExit()
                        resampled_cbct, transform,_,_ = register_images_without_dicom_reg(fixed_image=CT_sitk, moving_image=CBCT_sitk)
                        # registered_isocenter = register_point_without_dicom_reg(isocenter,transform)

                        # save_transformation(transform,cbct)
                 
                else:
                        _, registration_matrix = get_transformation_matrix(registration_file)
                        matrices.append(registration_matrix)
                        # print(registration_matrix)
                        transform = matrix_to_transform(registration_matrix)
                        # print(transform)
                        resampled_cbct = register_images_with_dicom_reg(fixed_image=CT_sitk, moving_image=CBCT_sitk, registration_matrix=registration_matrix)
                        registered_isocenter = register_point(isocenter, registration_matrix)
                
                resampled_cbct_list.append(resampled_cbct)
                # print("reg", registered_isocenter)
                # TO DO: FIX ISOCENTER REGISTRATION
                # if not legacy and registration_file !='':
                #       registered_isocenter = register_point(isocenter, registration_matrix)
                # else:
                # registered_isocenter = list(isocenter )# TO DO: Register point
                # isocenter_list.append(registered_isocenter)
                
        return resampled_cbct_list, isocenter_list, matrices, start_positions

# TO DO REGISTER TO CBCT 1
def register_CBCT_CBCT():
        """
        register_CBCT_CT        Registers list of CBCTs to CT.

        :param CT: Name of CT directory.
        :param CBCT_list: List of names of CBCT directries to be resampled.

        :returns: List of registered sitk CBCT images and isocenters.
        """

        ref_CBCT = image_dict[CT_list[0]]['CBCTs'][0]
        ref_sitk = generate_sitk_image(patient_path+ref_CBCT+'/')
        if debug_print:
                
                print(ref_CBCT)

        f = open('/data/kayla/HNC_images/reg_stats.csv', 'a',newline='')

        # create the csv writer
        writer = csv.writer(f)

        
        for CT in image_dict:
                resampled_cbct_list = []


                for CBCT in image_dict[CT]['CBCTs']:

                        cbct_path = patient_path+CBCT+'/'
                        CBCT_sitk = generate_sitk_image(cbct_path)
                        if CBCT == ref_CBCT:
                                resampled_cbct_list.append(CBCT_sitk)
                                continue
                        if debug_print:
                                print(CBCT)
                        resampled_cbct, transform, metric,stop  = register_images_without_dicom_reg(fixed_image=ref_sitk, moving_image=CBCT_sitk)
                        # registered_isocenter = register_point_without_dicom_reg(isocenter,transform)

                        save_transformation(transform,CBCT)
                        # write a row to the csv file
                        writer.writerow([patient_path.split("/")[-1]+"-"+CBCT,metric,stop])
                        resampled_cbct_list.append(resampled_cbct)
                image_dict[CT]['resampled_CBCTs'] = resampled_cbct_list



        # close the file
        f.close()
                




def register_replan_CBCTs(second_replan=False):
        """
        register_replan_CBCTs: Register CBCTs after replan a second time.
        """
        
        resampled_cbct_list_2 = []
        isocenter_list = []
        reg_exists = False

        
        reg_dir, registration_file = find_CT1_CT2_registration_file_v2(patient_path, CT_list, image_dict[CT_list[0]]['UID'],image_dict[CT_list[1]]['UID'])

        if second_replan:
                print("WHY SECOND REOPLANNNN")
                reg_dir, registration_file = find_CT1_CT2_registration_file_v2(patient_path, CT_list, image_dict[CT_list[1]]['UID'],image_dict[CT_list[2]]['UID'])
                if reg_dir == '':
                        reg_dir, registration_file = find_CT1_CT2_registration_file_v2(patient_path, CT_list, image_dict[CT_list[0]]['UID'],image_dict[CT_list[2]]['UID'])





        if registration_file !='':# and False:
                reg_exists = True
                moving_reference_UID, registration_matrix = get_transformation_matrix(patient_path+reg_dir+'/'+registration_file)
                transform = matrix_to_transform(registration_matrix)

                for CT in image_dict:
                        if image_dict[CT]['UID'] == moving_reference_UID:
                                moving_CT = CT
                        else:
                                reference_CT = CT
                
                try: moving_CT
                except:
                        print("ERROR: Could not find CT with reference UID", moving_reference_UID)
                        return
        

        if not reg_exists:
                # raise SystemExit()
                moving_CT = list(image_dict.keys())[1]
                reference_CT = list(image_dict.keys())[0]

        reference_CT_sitk = generate_sitk_image(patient_path+reference_CT+"/")

        image_dict[moving_CT]['isReference'] = False
        
        for CBCT_sitk in image_dict[moving_CT]['resampled_CBCTs']:
                if debug_print:
                        print("replanreg")
                if reg_exists:
                        resampled_cbct = register_images_with_dicom_reg(fixed_image=reference_CT_sitk, moving_image=CBCT_sitk, registration_matrix=registration_matrix)
                else:
                        print("replanned no reg")
                        # raise SystemExit()
                        resampled_cbct, transform,_,_ = register_images_without_dicom_reg(fixed_image=reference_CT_sitk,moving_image=CBCT_sitk)
                resampled_cbct_list_2.append(resampled_cbct)

        # for isocenter in image_dict[moving_CT]['isocenters']:
        #       registered_isocenter = register_point(isocenter, registration_matrix)
        #       isocenter_list.append(registered_isocenter)

        image_dict[moving_CT]['resampled_CBCTs'] = resampled_cbct_list_2
        # image_dict[moving_CT]['isocenters'] = isocenter_list



def apply_transforms_CBCT():

        for CT in image_dict:
                
                resampled_cbct_list = []
                
                ref_CBCT = image_dict[CT_list[0]]['CBCTs'][0]
                ref_sitk = generate_sitk_image(patient_path+ref_CBCT+'/')

                for CBCT in image_dict[CT]['CBCTs']:
                        print("Registering ", CBCT)
                        cbct_path = patient_path+CBCT+'/'
                        CBCT_sitk = generate_sitk_image(cbct_path)
                        if CBCT == ref_CBCT:
                                resampled_cbct_list.append(CBCT_sitk)
                                continue
                        
                        transform = sitk.ReadTransform(transform_save_path+patient_path.split('/')[-2]+'-'+CBCT+'.tfm')
                        resampled_cbct = sitk.Resample(CBCT_sitk,ref_sitk,transform,sitk.sitkLinear,-1000,ref_sitk.GetPixelID())
                        
                        resampled_cbct_list.append(resampled_cbct)
                image_dict[CT]['resampled_CBCTs'] = resampled_cbct_list


def find_multi_CT_regs(CT_list):
    dict_regs = {}
    # patient_path = path
    list_REs = []
    UID_list = []
    for CT in CT_list:
        files = [f for f in os.listdir(patient_path+CT+'/') if 'CT' in f]
        UID_list.append(dcm.read_file(patient_path+CT+'/'+files[0]).FrameOfReferenceUID)
    for i, CT in enumerate(CT_list):
        dict_regs[CT] = {}
        list_REs
        re_dirs = [f for f in os.listdir(patient_path+CT) if f[0:2] == 'RE']
    
        list_REs = [f for f in os.listdir(patient_path+CT) if f[0:2] == 'RE']
        registration_file = ''
        ref_CT = ''
       
        
        for j in range(0,len(list_REs)):
                for reg in list_REs:
                    print("trying", reg)
                    r1 = dcm.read_file(patient_path+CT_list[i]+'/'+reg)
                    CT1_ref_exist = False
                    CT2_ref_exist = False
                    good = True
                    for seq in r1.RegistrationSequence:
                        uid = seq.FrameOfReferenceUID
                        if uid not in UID_list:
                            good = False
                            break
                            
                    
                    if good:

                        for seq in r1.RegistrationSequence:
                            class_UID = seq.ReferencedImageSequence[0].ReferencedSOPClassUID
                            registration_matrix = np.asarray(seq.MatrixRegistrationSequence[-1].
                                                                                 MatrixSequence[-1].FrameOfReferenceTransformationMatrix)#.reshape((4, 4))
                            identity = [1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.]
        
                            
                            if list(registration_matrix) == identity:
                                ref = seq.FrameOfReferenceUID
                            else:
                                move = seq.FrameOfReferenceUID
        
        
                        dict_regs[CT][reg] = {
                            "moving": move,
                            "reference": ref
                        }
    return dict_regs
    

def get_num_refs(dict_regs):
    num_refs = {}
    for CT in dict_regs:
        for reg in dict_regs[CT]:
            reg_UID = dict_regs[CT][reg]['reference']
            if reg_UID not in num_refs:
                num_refs[reg_UID] = 0
            num_refs[reg_UID]+=1
    return num_refs

def register_double_replan(dict_regs):
    num_refs = get_num_refs(dict_regs)


    for uid in num_refs:
        if num_refs[uid] ==2: #Only works for 2 refs rn (ie two of the 3 CTs were registered to the same CT)
            ref_uid = uid
            break
    uid_dict = {}
    for CT in image_dict:
        uid_dict[image_dict[CT]['UID']] = CT    
    
    for reg_dir in dict_regs:
        for reg_file in dict_regs[reg_dir]:
            resampled_cbct_list_2 = []
            if dict_regs[reg_dir][reg_file]['reference'] == ref_uid:
                moving_CT = uid_dict[dict_regs[reg_dir][reg_file]['moving']]
                reference_CT =  uid_dict[dict_regs[reg_dir][reg_file]['reference']]
                print("Registering", moving_CT,"to",reference_CT)
                reference_CT_sitk = generate_sitk_image(patient_path+reference_CT+"/")
                image_dict[moving_CT]['isReference'] = False
                moving_reference_UID, registration_matrix = get_transformation_matrix(patient_path+reg_dir+'/'+reg_file)
                
                for CBCT_sitk in image_dict[moving_CT]['resampled_CBCTs']:
                     resampled_cbct = register_images_with_dicom_reg(fixed_image=reference_CT_sitk, moving_image=CBCT_sitk, registration_matrix=registration_matrix)
                     resampled_cbct_list_2.append(resampled_cbct)
                image_dict[moving_CT]['resampled_CBCTs'] = resampled_cbct_list_2

def produce_plots(zoom=True,second_replan=False):
        """
        produce_plots   Plot a sample of registered slices.
        """


        if second_replan:
            fig = plt.figure(figsize=(20, 20))
        else:
            fig = plt.figure(figsize=(20, 10))
            

        columns = 3
        rows = 1
        rows_replan = 2 if replan else rows
        rows_replan_second = 3 if second_replan else rows_replan

        middle_slice = int(len(sitk.GetArrayViewFromImage(image_dict[CT_list[0]]['resampled_CBCTs'][0]))/2)

        print("Replan status: ", replan)#, second_replan)
        # print("rows: ", rows_replan_second)

        # Plotting the first set of images
        for i in range(1, columns * rows + 1):
            try:
                img = sitk.GetArrayViewFromImage(image_dict[CT_list[0]]['resampled_CBCTs'][i-1])[middle_slice]
                fig.add_subplot(rows_replan_second, columns, i)
                plt.title(image_dict[CT_list[0]]['CBCTs'][i-1], fontsize=12)
                if zoom:
                    plt.imshow(img[125:375, 150:360])
                else:
                    plt.imshow(img)
            except Exception as e:
                print("Error plotting all:", e)

        # Plotting the replan set of images
        if replan:
            for i in range(columns * rows + 1, columns * rows_replan + 1):
                try:
                    img = sitk.GetArrayViewFromImage(image_dict[CT_list[1]]['resampled_CBCTs'][i - (columns * rows) - 1])[middle_slice]
                    fig.add_subplot(rows_replan_second, columns, i)
                    plt.title(image_dict[CT_list[1]]['CBCTs'][i - (columns * rows) - 1], fontsize=12)
                    if zoom:
                        plt.imshow(img[125:375, 150:360])
                    else:
                        plt.imshow(img)
                except Exception as e:
                    print("Error plotting all:", e)

        # Plotting the second replan set of images
        if second_replan:
            for i in range(columns * rows_replan + 1, columns * rows_replan_second + 1):
                try:
                    img = sitk.GetArrayViewFromImage(image_dict[CT_list[2]]['resampled_CBCTs'][i - (columns * rows_replan) - 1])[middle_slice]
                    fig.add_subplot(rows_replan_second, columns, i)
                    plt.title(image_dict[CT_list[2]]['CBCTs'][i - (columns * rows_replan) - 1], fontsize=12)
                    if zoom:
                        plt.imshow(img[125:375, 150:360])
                    else:
                        plt.imshow(img)
                except Exception as e:
                    print("Error plotting all:", e)

        plt.show()



def register_patient(path, use_reg_file=True, plot=False,ignore_CT=False,use_transforms=False,save_CBCT=False):
        """
        register_patient        Call all functions to register all images of te given patient.

        :param path: Path to the patient directory.
        :param plot: Flag to plot sample images or not.
        :param ignore_CT: Registers images to first CBCT using optimizer
        :param use_transforms: Uses already made SITK transform files if available #TO DO: add check that they are avaiallbe
        """

        global patient_path
        global replan
        global second_replan
        
        patient_path = path
        if debug_print:
                
                print(patient_path)

        image_dict.clear()
        replan = False
        has_dicom_reg = True
        zoom = False

        if debug_print:
                print("- Loading files from",patient_path," -")
        
        get_file_lists(patient_path)

        if ignore_CT:
                print("IGNORINGCTs")
                if use_transforms:
                        apply_transforms_CBCT()

                        # transform = sitk.ReadTransform('/data/kayla/HNC_images/transforms/624-20211116_kV_CBCT_1a.tfm')
                else:
                        register_CBCT_CBCT()
                zoom = False

        else:
                if len(CT_list) > 2:
                        print("SECOND REPLAN")
                        second_replan = True
                        # raise Warning('More than 2 CT directories found. This code may not perform as expected, as it was made for exactly one replan (2 CTs)')

                if debug_print:
                
                        # Register each set of CBCTs to respective CT
                        print("--------------------------------------------------------")
                        print("-                   Registering CBCTs                  -")
                        print("--------------------------------------------------------")
                if use_reg_file:
                        for CT in image_dict:
                                image_dict[CT]['resampled_CBCTs'], image_dict[CT]['isocenters'],image_dict[CT]['matrices'],image_dict[CT]['starts'] = register_CBCT_CT(CT, image_dict[CT]['CBCTs'],True,patient_path)
                                image_dict[CT]['isReference'] = True

                else:

                        print("not using reg file")
                        CT_0 = ''
                        for CT in image_dict:
                                if CT_0 == '':
                                        CT_0 = CT
                                image_dict[CT]['resampled_CBCTs']= register_CBCT_CT(CT_0, image_dict[CT]['CBCTs'],use_reg_file,patient_path)
                                image_dict[CT]['isReference'] = False
                        image_dict[CT_0]['isReference'] = True

                plt_CT = False
                if plt_CT:
                        CT_sitk = generate_sitk_image(patient_path+CT+'/')
                        img = sitk.GetArrayViewFromImage(CT_sitk)[76]
                        plt.imshow(img[125:375,150:360])
                        plt.show()


                
                if second_replan:
                        
                        if debug_print:
                
                                print("--------------------------------------------------------")
                                print("-           TWO REPLANS: Registering CBCTs           -")
                                print("--------------------------------------------------------")
                                # if use_reg_file or True:
                                print("Doing replans")

                        dict_regs = find_multi_CT_regs(sorted(list(image_dict.keys())))
                        # num_refs = get_num_refs(dict_regs)
                        register_double_replan(dict_regs)
                        

                elif replan:
                        if debug_print:
                
                                print("--------------------------------------------------------")
                                print("-              REPLAN: Registering CBCTs               -")
                                print("--------------------------------------------------------")
                                # if use_reg_file or True:
                                print("Doing replans")
                        register_replan_CBCTs()
                

        if plot: produce_plots(zoom,second_replan)


        return image_dict



if __name__ == "__main__":
        
        PATH = '/mnt/iDriveShare/Kayla/CBCT_images/kayla_extracted/' # Path to patient directories

        # Check if command line arguments correspond to existing patient directories
        for patient in sys.argv[1:]:
                patient_path = PATH+patient+"/"
                if not os.path.exists(patient_path):
                        print("Patient directory "+ patient_path + " does not exist.")
        
                register_patient(patient_path, plot = True)

        print("Done")
