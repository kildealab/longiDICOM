import os
import pydicom as dcm

def get_sorted_dicoms(patient_path):
    RT_path = os.path.join(patient_path, 'RT/')
    print(RT_path)
    RT_files = os.listdir(RT_path)
    RT_files_sorted = RT_files.sort(key=lambda x: dcm.read_file(RT_path+x).TreatmentDate)
    print(RT_files_sorted)
    
'''
Format 1: Replans restart at fx 1, organized by CT date
'''


def get_fraction_dates(patient_path):
    RT_path = os.path.join(patient_path, 'RT/')
    RT_files = os.listdir(RT_path)
    
    patient_dict = {}
    
    for file in RT_files:
        d = dcm.read_file(RT_path+file)
        fraction = d.TreatmentSessionBeamSequence[0].CurrentFractionNumber
        date = d.TreatmentDate    
        ct_date = d.StudyDate
        if ct_date not in patient_dict:
            patient_dict[ct_date] = {}
        patient_dict[ct_date][int(fraction)] = date # change f to int

    for date in patient_dict:
        patient_dict[date] = dict(sorted(patient_dict[date].items()))
    return dict(sorted(patient_dict.items()))



def generate_fraction_date_dict(PATH):
    patient_date_dict = {}
    
    for patient in sorted([f for f in os.listdir(PATH) if 'old' not in f and 'b' not in f and 'lung' not in f.lower() and '-' not in f],key=int):
        try:
            patient_date_dict[patient] = get_fraction_dates(os.path.join(PATH,patient))
        except Exception as e:
            print("WARNING! Patient",patient, "has error:", e)
        
    return patient_date_dict

'''
### Format 2: Replans continue fx number from before, no CT dates
'''
def format_fraction_numbers(fraction_dict):
    new_dict = {}
    last_fx = 0
    for ct_date in fraction_dict:
        for fx in fraction_dict[ct_date]:
            new_fx = int(fx) + last_fx
            new_dict[new_fx] = fraction_dict[ct_date][fx]
        last_fx = last_fx+int(fx)
    return new_dict

def format_all_patient_fx_numbers(patient_dict):
    new_patient_dict = {}
    for patient in patient_dict:
#         print(patient)
        new_patient_dict[patient] = format_fraction_numbers(patient_dict[patient])
    return new_patient_dict


'''
### Format 3: Replans continue fx number from before + CT dates
'''

def format_fraction_numbers(fraction_dict):
    new_dict = {}
    last_fx = 0
    for i,ct_date in enumerate(fraction_dict):
        new_dict['CT'+str(i+1)] = ct_date
        for fx in fraction_dict[ct_date]:
            new_fx = int(fx) + last_fx
            new_dict[new_fx] = fraction_dict[ct_date][fx]
        last_fx = last_fx+int(fx)
    return new_dict


def format_all_patient_fx_numbers(patient_dict):
    new_patient_dict = {}
    for patient in patient_dict:
#         print(patient)
        new_patient_dict[patient] = format_fraction_numbers(patient_dict[patient])
    return new_patient_dict