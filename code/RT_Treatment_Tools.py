import os
import pydicom as dcm

def get_sorted_RT_files(patient_path):
    RT_path = os.path.join(patient_path, 'RT/')
    RT_files = os.listdir(RT_path)
    RT_files_sorted = sorted(RT_files,key=lambda x: dcm.read_file(RT_path+x).TreatmentDate)

    return RT_files_sorted
    
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



##########################################
# Get fraction dates by PLAN ID
##########################################

def get_plan_dict(patient_path):
    RT_path = os.path.join(patient_path, 'RT/')
    RT_files = os.listdir(RT_path)

    date_fx_dict = {}

    for file in RT_files:
        d = dcm.read_file(RT_path+file)
        ref_plan = d.ReferencedRTPlanSequence[0].ReferencedSOPInstanceUID
        fraction = d.TreatmentSessionBeamSequence[0].CurrentFractionNumber
        # if fraction == '34':
        #     print(d)
    #     print(fraction)
        date = d.TreatmentDate 
        if ref_plan not in date_fx_dict:
            date_fx_dict[ref_plan] = {}

        date_fx_dict[ref_plan][fraction] = date

    
    return date_fx_dict

def arrange_fx_date(plan_dict):
    plans = list(plan_dict.keys())
    for plan in plans:
        try:
            plan_dict[plan][1]
        except:
            plan_dict.pop(plan,None)
            print("Warning: Error with plan",plan)
    sorted_plans = sorted(plan_dict, key=lambda plan: plan_dict[plan][1])

    fx_date_new = {}
    plan_num = 0
    current_last_fx = 0
    current_fx = 1

    for plan in sorted_plans:
        for fx in sorted(plan_dict[plan]):
            fx_date_new[current_fx] = plan_dict[plan][fx]
            current_fx +=1

    return fx_date_new

def get_plan_start_dates(PATH):
    patient_dict = {}
    
    for patient in sorted(os.listdir(PATH)):
        if 'b' in patient:
            continue
        patient_path = PATH + patient
        plan_dict = get_plan_dict(patient_path)
        # print(plan_dict.keys())
        plans = list(plan_dict.keys())
        # Error checking to ensure no issues before sorting, 
        # Inefficient, could be redone
        for plan in plans:
            try:
                plan_dict[plan][1]
            except Exception as e:
                print("Warning: patient",patient,'has an error with plan',plan,":",e)
                plan_dict.pop(plan,None)
        sorted_plans = sorted(plan_dict.keys(), key=lambda plan: plan_dict[plan][1])
        
        plan_start_dates = {}
        for i,plan in enumerate(sorted_plans):
            plan_start_dates[i] = plan_dict[plan][1]
        
        patient_dict[patient] = plan_start_dates

    return patient_dict


def generate_patient_fx_date_dict(PATH):
    patient_dict = {}
    
    for patient in sorted(os.listdir(PATH)):
        if 'b' in patient:
            continue
        patient_path = PATH + patient
        plan_dict = get_plan_dict(patient_path)
        fx_date = arrange_fx_date(plan_dict)
        patient_dict[patient] = fx_date
        
    return patient_dict

def get_plan_num_fractions(PATH):
    patient_dict = {}
    
    for patient in sorted(os.listdir(PATH)):
        if 'b' in patient:
            continue
        patient_path = PATH + patient
        plan_dict = get_plan_dict(patient_path)
        plans = list(plan_dict.keys())
        # Error checking to ensure no issues before sorting, 
        # Inefficient, could be redone
        for plan in plans:
            try:
                plan_dict[plan][1]
            except Exception as e:
                print("Warning: patient",patient,'has an error with plan',plan,":",e)
                plan_dict.pop(plan,None)
        sorted_plans = sorted(plan_dict, key=lambda plan: plan_dict[plan][1])
        
        plan_num_fx = {}
        for i,plan in enumerate(sorted_plans):
            plan_num_fx[i] = len(plan_dict[plan])
        
        patient_dict[patient] = plan_num_fx

    return patient_dict



def get_patients_exact_fx_num(patient_dict_fx_dates, num):
    list_num = []

    for patient in sorted(patient_dict_fx_dates.keys()):
        num_fx = len(patient_dict_fx_dates[patient])
        if num_fx == num:
            list_num.append(patient)
    return list_num


def get_patients_exceeding_fx_num(patient_dict_fx_dates, num):
    list_over = []

    for patient in sorted(patient_dict_fx_dates.keys()):
        num_fx = len(patient_dict_fx_dates[patient])
        if num_fx > num:
            list_over.append(patient)

    return list_over