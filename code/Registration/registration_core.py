import pydicom as dcm
import SimpleITK as sitk
import os, sys
import numpy as np
import matplotlib.pyplot as plt

from Registration.registration_callbacks import *
from Registration.registration_utilities import *

sys.path.append('..')
from sitk_img_tools import generate_sitk_image

debug_print = False

def get_image_UID(image_path):
	UID = ''
	for f in os.listdir(image_path):
		try:
			UID = dcm.read_file(os.path.join(image_path,f)).FrameOfReferenceUID
		except:
			continue
		if UID != '':
			return UID
	return UID


def find_registration_file_two_images(image_path1, image_path2, opt_parent_search_path=''):
	"""
	find_registration_file_two_images       Finds the registration file for 2 images, assuming the reg file  is in one of those paths.
											For a deeper search and two images in the same parent path, see find_CT1_CT2_registration_file_v2.

	:param image_path1: Path to first image.
	:param image_path2: Path to second image.
	:param opt_parent_search_path: Optional parent directory path to search for registration files in, if not expected in image paths.

	:returns: The directory name of the fixed (reference) CT and the registration_file.
	"""
	debug_print=True

	# Get the UIDs of the images			
	ref1 = get_image_UID(image_path1)
	ref2 = get_image_UID(image_path2)

	list_REs= []
		
	REs_outside = False
	RE_found = False

	path_search = [image_path1, image_path2]

	if opt_parent_search_path != '':
		path_search.append(opt_parent_search_path)

	for image_path in path_search:

		# Find REG files in the image directories
		# Note assumes file name starts with 'RE', for a more robust (but less efficient) search, will need to open up files.
		list_REs.append([f for f in os.listdir(image_path) if f[0:2] == 'RE'])		

	# Check if no reg files found
	if len(list_REs) == 0:
		print("No registration files found.")
		return '', ''



	# Check each registration file to see if they reference both of the CT UIDs
	registration_file = ''
	ref_CT = ''

	identity = [1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.]

	for i in range(0,len(list_REs)):
		for reg in list_REs[i]:
				
			r1 = dcm.read_file(os.path.join(path_search[i],reg))
			img1_ref_exist = False
			img2_ref_exist = False

			
			for seq in r1.RegistrationSequence:
				class_UID = seq.ReferencedImageSequence[0].ReferencedSOPClassUID


				# if (seq.FrameOfReferenceUID == ref1):
				# 	img1_ref_exist = True
				# elif (seq.FrameOfReferenceUID == ref2):
				# 	img2_ref_exist = True

				registration_matrix = np.asarray(seq.MatrixRegistrationSequence[-1].MatrixSequence[-1].FrameOfReferenceTransformationMatrix)#.reshape((4, 4))
			   
				
				if list(registration_matrix) == identity:
					ref = seq.FrameOfReferenceUID

				else:
					move = seq.FrameOfReferenceUID
			
		
			if move == ref1 and ref == ref2:
				registration_file = os.path.join(path_search[i],reg)
				return image_path2, registration_file
			if move == ref2 and ref == ref1:
				registration_file = os.path.join(path_search[i],reg)
				return image_path1, registration_file 

	print("Warning: no registration file found")

	return ref_CT, registration_file
						
def find_CT1_CT2_registration_file_v2(patient_path, CT_list, CT1_ref, CT2_ref):
		"""
		find_CT1_CT2_registration_file_v2       Finds the registration file for 2 CTs within the patient directory.

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
				if debug_print:
				
						print("Searching for RE...")
						print(CT)
				re_dirs = [f for f in os.listdir(patient_path+CT) if f[0:2] == 'RE']
				if len(re_dirs)!=0:
						RE_found = True
				list_REs.append([f for f in os.listdir(patient_path+CT) if f[0:2] == 'RE'])
				if debug_print:
				
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
						if debug_print:
				
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
								if debug_print:
						
										print("************************************************************************************************")
										print("*               ",registration_file,"               *")
										print("************************************************************************************************")
								continue

		return ref_CT, registration_file

def get_transformation_matrix(registration_file):
		"""
		get_transformation_matrix       Gets the transformation matrix from the DICOM Registration file.

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

def plot_reg_results_two_images(ref_sitk, moving_sitk, resampled_moving_sitk):
	middle_slice = int(len(sitk.GetArrayFromImage(ref_sitk))/2)

	plt.subplot(1,3,2)
	plt.imshow(sitk.GetArrayFromImage(ref_sitk)[middle_slice])
	plt.title("Reference Image")

	plt.subplot(1,3,1)
	plt.imshow(sitk.GetArrayFromImage(moving_sitk)[middle_slice])
	plt.title("Moving Image")


	plt.subplot(1,3,3)
	plt.imshow(sitk.GetArrayFromImage(resampled_moving_sitk)[middle_slice])
	plt.title("Registered Moving Image")
	
	

# def get_reference_moving

def register_two_images(image_path1, image_path2, allow_optimizer = True, force_optimizer = False, plot_result = False, opt_RE_search_path=''):
		"""
		register_two_images        Registers list of CBCTs to CT.

		:param CT: Name of CT directory.
		:param CBCT_list: List of names of CBCT directries to be resampled.

		:returns: List of registered sitk CBCT images and isocenters.
		"""

		registration_file = ''

		# isocenter = get_acq_isocenter(cbct_path)[0]

		if not force_optimizer:
			reference_image_path, registration_file = find_registration_file_two_images(image_path1, image_path2)

		
		if not allow_optimizer and registration_file == '':
			print("No registration file found. Please set allow_optimizer to True if you would like to use register the images without the DICOM file.")
			return None 

		if registration_file != '' and not force_optimizer:
			print("Using existing registration file...")

			reference_image_path, registration_file = find_registration_file_two_images(image_path1, image_path2, opt_RE_search_path)

		
			moving_image_path = image_path1 if reference_image_path == image_path2 else image_path2

			moving_sitk = generate_sitk_image(moving_image_path)
			reference_sitk = generate_sitk_image(reference_image_path)

			_, registration_matrix = get_transformation_matrix(registration_file)

			transform = matrix_to_transform(registration_matrix)
		  
			resampled_moving = register_images_with_dicom_reg(fixed_image=reference_sitk, moving_image=moving_sitk, registration_matrix=registration_matrix)
			# registered_isocenter = register_point(isocenter, registration_matrix)
			if plot_result:
				plot_reg_results_two_images(reference_sitk, moving_sitk, resampled_moving)
			return resampled_moving, moving_image_path

		if registration_file == '' or force_optimizer:
			print("Using optimizer...")

			reference_sitk = generate_sitk_image(image_path1)
			moving_sitk = generate_sitk_image(image_path2)

			
			resampled_moving, transform,_,_ = register_images_without_dicom_reg(fixed_image=reference_sitk, moving_image=moving_sitk)
			if plot_result:
				plot_reg_results_two_images(reference_sitk, moving_sitk, resampled_moving)
			return resampled_moving, image_path2
	

# Adapted from Brian M Anderson's RegisteringImages repository # Source: https://github.com/brianmanderson/RegisteringImages/blob/main/src/RegisterImages/WithDicomReg.py
def register_images_with_dicom_reg(fixed_image: sitk.Image, moving_image: sitk.Image, registration_matrix,
																		min_value=-1000, method=sitk.sitkLinear):
		"""
		register_images_with_dicom_reg Register a moving image to fixed image using known registratio_matrix from dicom registration file.
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
		register_point  Register a point according to a registration_matrix.

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

		register_images_without_dicom_reg       Registers 2 images using optimizer when DICOM Reg file isn't available.

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

		print('Final metric value:',(registration_method.GetMetricValue())      )
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
						numberOfIterations=500,
						gradientMagnitudeTolerance=1e-8,
						relaxationFactor = 0.8

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



def save_transformation(transform,cbct_name,save_path):
		# save_path = '/data/kayla/HNC_images/transforms/'
		save_name = cbct_name+'.tfm'

		if not os.path.isdir(save_path):
				os.mkdir(save_path)
		sitk.WriteTransform(transform, save_path+save_name)

# def save_transformation(transform,cbct_name):
		
#       save_name = patient_path.split('/')[-2]+'-'+cbct_name+'.tfm'
#       sitk.WriteTransform(transform, save_path+save_name)

def save_registered_CBCTs(input_path,save_path, patient, image_dict):
	patient_save_path = os.path.join(save_path,patient)
	if not os.path.isdir(patient_save_path):
		os.mkdir(patient_save_path)
	for CT in image_dict:
		for i,CBCT_name in enumerate(image_dict[CT]['CBCTs']):
			CBCT_path = os.path.join(input_path,CBCT_name)
			# print(CBCT_name)
			CBCT_sitk = image_dict[CT]['resampled_CBCTs'][i]
			CBCT_save_path = os.path.join(patient_save_path, CBCT_name)
			if not os.path.isdir(CBCT_save_path):
				os.mkdir(CBCT_save_path)

			save_dicoms(CBCT_path, CBCT_sitk, CBCT_save_path)    

