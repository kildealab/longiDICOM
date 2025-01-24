def get_CT_list(patient_path):
    CT_list = [d for d in os.listdir(patient_path) if d[9:11] == 'CT' and (len(d)==23 or len(d)==24) and d[-1].isdigit()]
    
    if len(CT_list) == 0:
        print("WARNING: Patient",patient_path.split("/")[-1],"- No CT directories found.")
#         print("Removing from list.")
#         list_patients_full.remove(patient_path.split("/")[-1])
    elif len(CT_list) > 2:
        print("WARNING: Patient",patient_path.split("/")[-1],"- Found ",len(CT_list)," CT directories.")#" Remove incorrect directory(ies) and try again.")
#         print("CT Directories:", CT_list)

#     replan = True if len(CT_list) > 1 else False
    return sorted(CT_list)



def get_CBCT_list(patient_path):
    CBCT_list = [d for d in os.listdir(patient_path) if d[9:16] == 'kV_CBCT']
    return sorted(CBCT_list)

def get_CT_dict(patient_path,CT_list):
    CTs = []
    
    for CT in CT_list:
        i_CT_dict = {}
        CT_path = os.path.join(patient_path,CT)

        CT_file = sorted([f for f in os.listdir(CT_path) if f[0:2]=='CT'])[0]
        if CT_file[0:2] != 'CT':
            print("WARNING: No CT files in directory",CT_path)
            
        else:    
            d = dcm.read_file(os.path.join(CT_path,CT_file)) # Sorted ensures we grab CT file and not RT
            date = d.SeriesDate
            CTs.append({'dir_name': CT, 'date':date})
            
    return CTs

def get_CBCT_dict(patient_path,CBCT_list):
    CBCTs = []
    
    for CBCT in CBCT_list:
        i_CBCT_dict = {}
        CBCT_path = os.path.join(patient_path,CBCT)
        CBCT_file = sorted(os.listdir(CBCT_path))[0]
        if CBCT_file[0:2] != 'CT':
            print("WARNING: No CT files in directory",CBCT_path)

        else:   
            fraction = CBCT.split("_")[-1][0:-1]
            d = dcm.read_file(os.path.join(CBCT_path,sorted(os.listdir(CBCT_path))[0]))
            date = d.SeriesDate
            CBCTs.append({'dir_name': CBCT, 'date':date, 'fraction':fraction})

    return CBCTs
    

def generate_imaging_data_dict(PATH):
    imaging_data = []
    list_patients_full = [x for x in os.listdir(PATH) if 'b' not in x and 'old' not in x]
    list_patients_full.sort(key=int)
    
    for patient in list_patients_full:
        print(patient)
        try:
            patient_path = os.path.join(PATH,patient)
    
            CT_list = get_CT_list(patient_path)
            if len(CT_list)==0:
                continue
    
            CBCT_list = get_CBCT_list(patient_path)
            
            patient_dict = {}
            patient_dict['id'] = patient
            patient_dict['replan'] = 'Y' if len(CT_list) > 1 else 'N' # not necessarily true always
            patient_dict['numCTs'] = len(CT_list)
            patient_dict['numCBCTs'] = len(CBCT_list)
    
            patient_dict['CT'] = get_CT_dict(patient_path, CT_list)
            patient_dict['CBCT'] = get_CBCT_dict(patient_path, CBCT_list)
        
            imaging_data.append(patient_dict)
        except Exception as e:
            print(e)
        
        # OLD WAY
#         imaging_data[patient] = {}
#         imaging_data[patient]['replan'] = 'Y' if len(CT_list) > 1 else 'N'
#         imaging_data[patient]['numCTs'] = len(CT_list)
#         imaging_data[patient]['numCBCTs'] = len(CBCT_list)

#         imaging_data[patient]['CT'] = get_CT_dict(patient_path, CT_list)
#         imaging_data[patient]['CBCT'] = get_CBCT_dict(patient_path, CBCT_list)
    
    return imaging_data


def check_num_CT_CBCTs(imaging_data):
    for entry in imaging_data:
        if len(entry['CT']) != entry['numCTs']:
             print("Number of CT directories doesn't match actual CTs for patient",entry['id'])

    for i,entry in enumerate(imaging_data):
        if len(entry['CBCT']) != entry['numCBCTs']:
            print("Number of CBCT directories doesn't match actual CBCTs for patient",entry['id'],"--> updating...")
            entry['numCBCTs'] = len(entry['CBCT'])