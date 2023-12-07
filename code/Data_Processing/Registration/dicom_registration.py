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

rcParams['figure.figsize'] = 11.7,8.27
rcParams['font.size'] = 22



dict_class_UID = {'1.2.840.10008.5.1.4.1.1.2': 'CT', '1.2.840.10008.5.1.4.1.1.481.1': 'RI', '1.2.840.10008.5.1.4.1.1.4': 'MR', '1.2.840.10008.5.1.4.1.1.128':'PE'}
replan = False
patient_path = ''
has_dicom_reg = True
image_dict = {}
CT_list = []

transform_save_path = '/data/kayla/HNC_images/transforms/'

# CT1
# CT2
def find_ROI_names(RS, keyword=''):
	ROI_names = []

	for seq in RS.StructureSetROISequence:
		roi_name = seq.ROIName
		if keyword.lower() in roi_name.lower() and 'nos' not in roi_name.lower() and 'z_' not in roi_name.lower():
			ROI_names.append(seq.ROIName)
	return ROI_names


def get_contour_from_ROI_name(ROI_name, RS):
	for i, seq in enumerate(RS.StructureSetROISequence):
		if seq.ROIName == ROI_name:
			index = i
			break

	contour_coords = []

	for ROI_contour_seq in RS.ROIContourSequence[index].ContourSequence:
		contour_coords.append(ROI_contour_seq.ContourData)

	return contour_coords

#TO DO: make nicer, especialy replan part
def get_file_lists():
	"""
	get_file_lists 	gets lists of CTs and CBCTs in patient folder whose dir names conform to the following formats:
					CT: "CT" in 9-10th index position and 23 char length.
					CBCT: "kV" in 9-10th index position

	"""

	global CT_list

	# Get list of CT directories
	CT_list = [d for d in os.listdir(patient_path) if d[9:11] == 'CT' and len(d) == 23]
	CT_list.sort()


	CT_list.sort(key=lambda x: datetime.strptime(x[12:], "%d_%b_%Y"))

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
	print(CBCT_list)
	CBCT_list.sort()

	# Raise errors if no CTs found, or if > 2 CTs found
	if len(CT_list) == 0:
		raise NotADirectoryError('No CT directories were found. Please ensure the following naming convention was used: "YYYYMMDD_CT_DD_MMM_YYYY".')
	
	elif len(CT_list) > 1: # Set replan to true if > 1 CT
		global replan
		replan = True

		# if len(CT_list) != 2:
		# 	raise Warning('More than 2 CT directories found. This code may not perform as expected, as it was made for exactly one replan (2 CTs)')

 
	#TO DO: check for whne dates don't work
	#TO DO: REDO THIS PART --> GRAB DATES FROM ACTUAL TABLE, THIS ISN'T NECESSARILLY TRUE
	
	if replan:
		#date_replan = CT_list[1][0:8]
		# alternate dates since dated CTs often wrong
		# date_fx_1 = CBCT_list[0][0:8]
		fx_1s = sorted([i for i in CBCT_list if int(i.split("_")[-1][:-1]) == 1])
		print(fx_1s)
		date_replan = fx_1s[1].split("_")[0]

		CBCT_list_replan = [CBCT for CBCT in CBCT_list if int(CBCT[0:8]) >= int(date_replan)]
		CBCT_list =  [CBCT for CBCT in CBCT_list if int(CBCT[0:8]) < int(date_replan)]

		print(CBCT_list)
		print("REPLNA", CBCT_list_replan)

		

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
			print(fx_1s)
			date_replan = fx_1s[2].split("_")[0]

			CBCT_list_replan2 = [CBCT for CBCT in CBCT_list_replan if int(CBCT[0:8]) >= int(date_replan)]
			CBCT_list_replan =  [CBCT for CBCT in CBCT_list_replan if int(CBCT[0:8]) < int(date_replan)]

			image_dict[CT_list[2]]['CBCTs'] = CBCT_list_replan2


		image_dict[CT_list[1]]['CBCTs'] = CBCT_list_replan


	
	image_dict[CT_list[0]]['CBCTs'] = CBCT_list


	
	
def generate_sitk_image(DCM_path):
	"""
	generate_sitk_image	Reads DICOM file at DCM_path as a SITK (SimpleITK) image.

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


def get_transformation_matrix(registration_file):
	"""
	get_transformation_matrix	Gets the transformation matrix from the DICOM Registration file.

	:param registration_file: Path to registration file. 

	:returns: The Frame of Reference UID of the moving image (to be registered) and the registration matrix
	"""

	reg = dcm.read_file(registration_file)

	# Find the trasnformation matrix that is not the identity matrix
	for seq in reg.RegistrationSequence:

		registration_matrix = np.asarray(seq.MatrixRegistrationSequence[-1].
										 MatrixSequence[-1].FrameOfReferenceTransformationMatrix)#.reshape((4, 4))
		identity = [1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.]

		if list(registration_matrix) != identity:
			return seq.FrameOfReferenceUID, registration_matrix.reshape(4,4)

def matrix_to_transform(registration_matrix):
	affine_transform = sitk.AffineTransform(3)
	registration_matrix = np.linalg.inv(registration_matrix)
	affine_transform.SetMatrix(registration_matrix[:3, :3].ravel())
	affine_transform.SetTranslation(registration_matrix[:3, -1])
	return affine_transform

def register_images_with_dicom_reg2(fixed_image: sitk.Image, moving_image: sitk.Image, registration_matrix,
									min_value=-1000, method=sitk.sitkLinear):
	"""
	register_images_with_dicom_reg2	Register a moving image to fixed image using known registratio_matrix from dicom registration file.
									Function modified from https://github.com/brianmanderson/RegisteringImages.

	:param fixed_image: The fixed image
	:param moving_image: The moving image
	:param registration_matrix: the DICOM registration matrix
	:param min_value: Value to put as background in resampled image (-1000 for HU)
	:param method: interpolating method, recommend sitk.sitkLinear for images and sitk.sitkNearestNeighbor for masks

	:return: The registered moving SITK image.
	"""

	affine_transform = sitk.AffineTransform(3)
	registration_matrix = np.linalg.inv(registration_matrix)
	affine_transform.SetMatrix(registration_matrix[:3, :3].ravel())
	affine_transform.SetTranslation(registration_matrix[:3, -1])
	moving_resampled = sitk.Resample(moving_image, fixed_image, affine_transform, method, min_value,
									 moving_image.GetPixelID())
	return moving_resampled


def register_point(point, registration_matrix):
	"""
	register_point	Register a point according to a registration_matrix.

	:param point: The point to register.
	:param registration_matrix: the DICOM registration matrix.


	:return: The registered point.
	"""

	affine_transform = sitk.AffineTransform(3)

	registration_matrix = np.linalg.inv(registration_matrix)
	affine_transform.SetMatrix(registration_matrix[:3, :3].ravel())
	affine_transform.SetTranslation(registration_matrix[:3, -1])
	inverse_transform = affine_transform.GetInverse() #https://discourse.itk.org/t/simpleitk-registration-working-but-applying-the-transform-to-a-single-point-does-not/5728/4
	point_resampled = inverse_transform.TransformPoint(point) 

	return point_resampled

def register_point_without_dicom_reg(point, transform):
	point_resampled = transform.TransformPoint(point)
	return point_resampled


def register_images_without_dicom_reg(fixed_image,moving_image):
	"""
	NOTE: Code modified from taken https://simpleitk.org/SPIE2019_COURSE/04_basic_registration.html

	register_images_without_dicom_reg	Registers 2 images using optimizer when DICOM Reg file isn't available.

	:param fixed_image: The fixed sitk image.
	:param moving_image: The moving sitk image to registered.

	:returns: The registered moving image.
	"""

	#initial alignment of the two volumes
	'''
	print("JELLOOOO???")
	initial_transform = sitk.CenteredTransformInitializer(sitk.Cast(fixed_image,moving_image.GetPixelID()), 
											  moving_image, 
											  sitk.Euler3DTransform(), 
											  sitk.CenteredTransformInitializerFilter.GEOMETRY)
	
	registration_method = sitk.ImageRegistrationMethod()

	# registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
	registration_method.SetMetricAsMeanSquares()
	registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
	registration_method.SetMetricSamplingPercentage(0.01)

	registration_method.SetInterpolator(sitk.sitkLinear)

	registration_method.SetOptimizerAsGradientDescent(learningRate=1, numberOfIterations=200)
	
	# Scale the step size differently for each parameter, this is critical!!!
	registration_method.SetOptimizerScalesFromPhysicalShift() 

	registration_method.SetInitialTransform(initial_transform, inPlace=False)

	## Plot optimization
	registration_method.AddCommand(sitk.sitkStartEvent, registration_callbacks.metric_start_plot)
	registration_method.AddCommand(sitk.sitkEndEvent, registration_callbacks.metric_end_plot)
	registration_method.AddCommand(sitk.sitkIterationEvent, 
	                               lambda: registration_callbacks.metric_plot_values(registration_method))

	final_transform_v1 = registration_method.Execute(sitk.Cast(fixed_image, sitk.sitkFloat32), 
													 sitk.Cast(moving_image, sitk.sitkFloat32))

	print('Final metric value:',(registration_method.GetMetricValue())	)
    # print('Optimizer\'s stopping condition,', (registration_method.GetOptimizerStopConditionDescription()))
	
	

	# Resample the moving image onto the fixed image's grid
	resampler = sitk.ResampleImageFilter()
	resampler.SetReferenceImage(fixed_image)
	resampler.SetInterpolator(sitk.sitkLinear)
	resampler.SetDefaultPixelValue(100)
	resampler.SetTransform(final_transform)

	moving_resampled = sitk.Resample(moving_image,fixed_image,final_transform_v1,sitk.sitkLinear,-1000,moving_image.GetPixelID())
	# print(final_transform_v1)
	'''
	initial_transform = sitk.CenteredTransformInitializer(sitk.Cast(fixed_image,moving_image.GetPixelID()), moving_image, 
                                                      sitk.AffineTransform(3), sitk.CenteredTransformInitializerFilter.GEOMETRY)

	registration_method = sitk.ImageRegistrationMethod()

	registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
	# registration_method.SetMetricAsMeanSquares()
	# registration_method.SetMetricAsCorrelation()
	registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
	registration_method.SetMetricSamplingPercentage(0.01)

	registration_method.SetInterpolator(sitk.sitkLinear)

	# registration_method.SetOptimizerAsGradientDescent(learningRate=.05, numberOfIterations=500,convergenceMinimumValue=1e-6)#, convergenceWindowSize=10)
	registration_method.SetOptimizerAsRegularStepGradientDescent(
	        learningRate=.5,
	        minStep=1e-4,
	        numberOfIterations=2000,
	        gradientMagnitudeTolerance=1e-8,
	    )
	# Scale the step size differently for each parameter, this is critical!!!
	# registration_method.SetOptimizerScalesFromPhysicalShift()
	registration_method.SetOptimizerScalesFromIndexShift()



	registration_method.SetInitialTransform(initial_transform, inPlace=False)

	## Plot optimization
	registration_method.AddCommand(sitk.sitkStartEvent, metric_start_plot)
	registration_method.AddCommand(sitk.sitkEndEvent, metric_end_plot)
	registration_method.AddCommand(sitk.sitkIterationEvent, 
	                               lambda: metric_plot_values(registration_method))

	final_transform_v1 = registration_method.Execute(sitk.Cast(fixed_image, sitk.sitkFloat32), 
	                                                 sitk.Cast(moving_image, sitk.sitkFloat32))

	print('Final metric value:',(registration_method.GetMetricValue()))
	print('Optimizer\'s stopping condition,', (registration_method.GetOptimizerStopConditionDescription()))


	moving_resampled = sitk.Resample(moving_image,fixed_image,final_transform_v1,sitk.sitkLinear,-1000,moving_image.GetPixelID())
	# print(final_transform_v1)


	return moving_resampled, final_transform_v1, registration_method.GetMetricValue(), registration_method.GetOptimizerStopConditionDescription()


def register_CBCT_CT(CT, CBCT_list,use_reg_file = True):
	"""
	register_CBCT_CT	Registers list of CBCTs to CT.

	:param CT: Name of CT directory.
	:param CBCT_list: List of names of CBCT directries to be resampled.

	:returns: List of registered sitk CBCT images and isocenters.
	"""
	
	resampled_cbct_list = []
	isocenter_list = []
	registration_file = ''

	CT_sitk = generate_sitk_image(patient_path+CT+'/')

	# Loop through al CBCTs in list
	for cbct in CBCT_list:
		print(cbct)
		cbct_path = patient_path+cbct+'/'
		CBCT_sitk = generate_sitk_image(cbct_path)
		isocenter = get_acq_isocenter(cbct_path)[0]
		# print(isocenter)

		# Find registration file for CBCT directory
		registration_file=''
		for f in os.listdir(cbct_path):
			if f[0:2] == 'RE':
				registration_file = cbct_path + f
				continue
		
		print("RE - ", registration_file)
		# If no registration file, register images with optimizer, otherwise use dicom reg file
		if registration_file =='':# or True: #use_reg_file == False:
			print("OPTIMIZER")
		
			print(CT)
			has_dicom_reg = False
			resampled_cbct = None
			raise SystemExit()
			# resampled_cbct, transform = register_images_without_dicom_reg(fixed_image=CT_sitk, moving_image=CBCT_sitk)
			# registered_isocenter = register_point_without_dicom_reg(isocenter,transform)

			# save_transformation(transform,cbct)
		 
		else:
			_, registration_matrix = get_transformation_matrix(registration_file)
			transform = matrix_to_transform(registration_matrix)
			# print(transform)
			resampled_cbct = register_images_with_dicom_reg2(fixed_image=CT_sitk, moving_image=CBCT_sitk, registration_matrix=registration_matrix)
			registered_isocenter = register_point(isocenter, registration_matrix)
			
		resampled_cbct_list.append(resampled_cbct)
		# print("reg", registered_isocenter)
		# TO DO: FIX ISOCENTER REGISTRATION
		# if not legacy and registration_file !='':
		# 	registered_isocenter = register_point(isocenter, registration_matrix)
		# else:
		# registered_isocenter = list(isocenter )# TO DO: Register point
		isocenter_list.append(registered_isocenter)
		
	return resampled_cbct_list, isocenter_list

# TO DO REGISTER TO CBCT 1
def register_CBCT_CBCT():
	"""
	register_CBCT_CT	Registers list of CBCTs to CT.

	:param CT: Name of CT directory.
	:param CBCT_list: List of names of CBCT directries to be resampled.

	:returns: List of registered sitk CBCT images and isocenters.
	"""

	ref_CBCT = image_dict[CT_list[0]]['CBCTs'][0]
	ref_sitk = generate_sitk_image(patient_path+ref_CBCT+'/')
	print(ref_CBCT)

	f = open('/data/kayla/HNC_images/reg_stats.csv', 'w')

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
		



	
def find_CT1_CT2_registration_file_v2(patient_path, CT_list, CT1_ref, CT2_ref):
	"""
	find_CT1_CT2_registration_file_v2	Finds the registration file for 2 CTs within the patient directory.

	:param patient_path: Path to patient directory.
	:param CT_list: List of CT directories.
	:param CT1_ref: One of the CT reference UIDs.
	:param CT2_ref: Another one fo th CT reference UIDs.

	:returns: The directory name of the fixed (reference) CT and the registration_file.
	"""
	
		
	list_REs= []
	
	REs_outside = False
	RE_found = False

	# Find REG files in CT direcories
	for CT in CT_list:
		print("Searching for RE...")
		print(CT)
		re_dirs = [f for f in os.listdir(patient_path+CT) if f[0:2] == 'RE']
		if len(re_dirs)!=0:
			RE_found = True
		list_REs.append([f for f in os.listdir(patient_path+CT) if f[0:2] == 'RE'])
		print(list_REs)

	# If no RE files found, check for RE files outside of CT diretcories
	if (not RE_found):
		REs_outside = True
		list_REs.append([f for f in os.listdir(patient_path) if f[0:2] == 'RE'])

	# Check each registration file to see if they reference both of the CT UIDs
	registration_file = ''
	ref_CT = ''
	for i in range(0,len(list_REs)):
		for reg in list_REs[i]:
			if REs_outside:
				r1 = dcm.read_file(patient_path+'/'+reg)
			else:
				r1 = dcm.read_file(patient_path+CT_list[i]+'/'+reg)
			CT1_ref_exist = False
			CT2_ref_exist = False
			print("-------",reg,"-------")
			for seq in r1.RegistrationSequence:
				class_UID = seq.ReferencedImageSequence[0].ReferencedSOPClassUID
		#         if (dict_class_UID[class_UID] =='MR'):
		#             list_REs[reg_index].remove(reg)

				# print(dict_class_UID[class_UID],seq.FrameOfReferenceUID)

				if (seq.FrameOfReferenceUID == CT1_ref):
					CT1_ref_exist = True
				elif (seq.FrameOfReferenceUID == CT2_ref):
					CT2_ref_exist = True

			if (CT1_ref_exist and CT2_ref_exist):
				ref_CT = CT_list[i]
				registration_file = reg
				reg_index = i
				print("************************************************************************************************")
				print("*               ",registration_file,"               *")
				print("************************************************************************************************")
				continue

	return ref_CT, registration_file

def register_replan_CBCTs():
	"""
	register_replan_CBCTs: Register CBCTs after replan a second time.
	"""
	
	resampled_cbct_list_2 = []
	isocenter_list = []
	reg_exists = False

	

	reg_dir, registration_file = find_CT1_CT2_registration_file_v2(patient_path, CT_list, image_dict[CT_list[0]]['UID'],image_dict[CT_list[1]]['UID'])

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
		raise SystemExit()
		moving_CT = list(image_dict.keys())[1]
		reference_CT = list(image_dict.keys())[0]

	reference_CT_sitk = generate_sitk_image(patient_path+reference_CT+"/")

	image_dict[moving_CT]['isReference'] = False
	
	for CBCT_sitk in image_dict[moving_CT]['resampled_CBCTs']:
		print("replanreg")
		if reg_exists:
			resampled_cbct = register_images_with_dicom_reg2(fixed_image=reference_CT_sitk, moving_image=CBCT_sitk, registration_matrix=registration_matrix)
		else:
			print("replanned no reg")
			raise SystemExit()
			# resampled_cbct, transform = register_images_without_dicom_reg(fixed_image=reference_CT_sitk,moving_image=CBCT_sitk)
		resampled_cbct_list_2.append(resampled_cbct)

	# for isocenter in image_dict[moving_CT]['isocenters']:
	# 	registered_isocenter = register_point(isocenter, registration_matrix)
	# 	isocenter_list.append(registered_isocenter)

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




def produce_plots(zoom=True):
	"""
	produce_plots	Plot a sample of registered slices.
	"""

	fig = plt.figure(figsize=(20, 10))
	
	columns = 3
	rows = 1
	rows_replan = 2 if replan else rows
	print("Replan status: ",replan)

	for i in range(1, columns*rows +1):
		try:
			img = sitk.GetArrayViewFromImage(image_dict[CT_list[0]]['resampled_CBCTs'][i-1])[76]
		#     img = sitk.GetArrayViewFromImage(resampled_cbct_list_2[i-1])[75]
			fig.add_subplot(rows_replan, columns, i)
			plt.title(image_dict[CT_list[0]]['CBCTs'][i-1], fontsize=12)
			if zoom:
				plt.imshow(img[125:375,150:360])
			else:
				plt.imshow(img)
		except:
			print("Error plotting all")

	if replan:

		for i in range(columns*rows +1, columns*rows_replan +1):
			try:
				img = sitk.GetArrayViewFromImage(image_dict[CT_list[1]]['resampled_CBCTs'][i-(columns*rows+1)])[76]
			#     img = sitk.GetArrayViewFromImage(resampled_cbct_list_2[i-1])[75]
				fig.add_subplot(rows_replan, columns, i)
				plt.title(image_dict[CT_list[1]]['CBCTs'][i-(columns*rows+1)], fontsize=12)
				if zoom:
					plt.imshow(img[125:375,150:360])
				else:
					plt.imshow(img)
			except:
				print("error plotting all")
	
	plt.show()

def save_transformation(transform,cbct_name):
	# save_path = '/data/kayla/HNC_images/transforms/'
	save_name = patient_path.split('/')[-2]+'-'+cbct_name+'.tfm'
	sitk.WriteTransform(transform, transform_save_path+save_name)

# def save_transformation(transform,cbct_name):
	
# 	save_name = patient_path.split('/')[-2]+'-'+cbct_name+'.tfm'
# 	sitk.WriteTransform(transform, save_path+save_name)



def register_patient(path, use_reg_file=False, plot=False,ignore_CT=False,use_transforms=False):
	"""
	register_patient	Call all functions to register all images of te given patient.

	:param path: Path to the patient directory.
	:param plot: Flag to plot sample images or not.
	:param ignore_CT: Registers images to first CBCT using optimizer
	:param use_transforms: Uses already made SITK transform files if available #TO DO: add check that they are avaiallbe
	"""

	global patient_path
	global replan
	
	patient_path = path
	print(patient_path)

	image_dict.clear()
	replan = False
	has_dicom_reg = True
	zoom = True

	
	print("- Loading files from",patient_path," -")
	
	get_file_lists()

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
			raise Warning('More than 2 CT directories found. This code may not perform as expected, as it was made for exactly one replan (2 CTs)')


		# Register each set of CBCTs to respective CT
		print("--------------------------------------------------------")
		print("-                   Registering CBCTs                  -")
		print("--------------------------------------------------------")
		# if use_reg_file:
		for CT in image_dict:
			image_dict[CT]['resampled_CBCTs'], image_dict[CT]['isocenters'] = register_CBCT_CT(CT, image_dict[CT]['CBCTs'])
			image_dict[CT]['isReference'] = True
		# else:

		# 	print("not using reg file")
		# 	CT_0 = ''
		# 	for CT in image_dict:
		# 		if CT_0 == '':
		# 			CT_0 = CT
		# 		image_dict[CT]['resampled_CBCTs'], image_dict[CT]['isocenters'] = register_CBCT_CT(CT_0, image_dict[CT]['CBCTs'],use_reg_file)
		# 		image_dict[CT]['isReference'] = False
		# 	image_dict[CT_0]['isReference'] = True

		plt_CT = False
		if plt_CT:
			CT_sitk = generate_sitk_image(patient_path+CT+'/')
			img = sitk.GetArrayViewFromImage(CT_sitk)[76]
			plt.imshow(img[125:375,150:360])
			plt.show()


		
		if replan:
			print("--------------------------------------------------------")
			print("-              REPLAN: Registering CBCTs               -")
			print("--------------------------------------------------------")
			# if use_reg_file or True:
			print("Doing replans")
			register_replan_CBCTs()

	if plot: produce_plots(zoom)
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
