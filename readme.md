# longiDICOM

A Python toolkit for longitudinal DICOM image registration, analysis, and visualization. It was made for registering and preprocessing our longitudinal radiotherapy dataset consisting of CBCT images, planning CTs, and DICOM-RT data (Structure Set, Dose, Treatment Record, etc.).
If you're looking for registering two DICOM imagess, with or without a DICOM registration file, see `Examples/Sample Notebook - General Registration.ipynb` and the accompanying code in `code/Registration/dicom_registration.py`.

## Features
- DICOM image tools for reading, processing, and visualizing medical images
- Registration tools for aligning images across timepoints (using DICOM REG files and from scratch)
- Longitudinal data management
- Slice selection utilities for selecting anatomy across images based on Structure Set data and Registration
- RT treatment, Dose, and Structure Set analysis

## Directory Structure

```
code/
    dcm_img_tools.py           # DICOM image processing functions
    json_utils.py              # JSON data utilities
    longitudinal_data.py       # Longitudinal data management
    RD_tools.py                # Radiotherapy dose tools
    rs_tools.py                # Radiotherapy Structure Set tools
    RT_Treatment_Tools.py      # DICOM RT treatment analysis
    sitk_img_tools.py          # SimpleITK image tools
    visualization3D.py         # 3D visualization utilities
    Registration/
        dicom_registration.py  #  DICOM image registration
        ...
    Slice_Selection/
Examples/
    Sample Notebook - General Registration.ipynb                # Registering two images
    Sample Notebook - Registration CBCT Dataset.ipynb           # Registration longitudinal CBCTs (pretty specific to our dataset)
    Sample Notebook - Get Treatment Fraction Dates.ipynb        # Extract dates for each treatment fraction from DICOM RT Treatment Records
    Sample Notebook - RT Dose.ipynb                             # Extract dose information from RT Dose files, including for specific structures
    Sample Notebook - Using Registration & Slice Selection Code.ipynb # Very specific to our dataset
```

## Getting Started
1. Clone the repository.
```
git clone https://github.com/kildealab/longiDICOM.git
cd longiDICOM
```
2. Install required Python packages (requirements.txt coming soon).
3. Explore the example notebooks in the `Examples/` folder for usage demonstrations.

Alternatively instead of cloning, if you want this embedded into your project and edit it, I would consider using Git Submodules (instructions coming soon)

## Contributions
We welcome contributions! If you are interested in contributing, please fork the repository and create a pull request with your changes.

1. Fork the repository.
2. Create a new branch: `git checkout -b feature-name`
3. ake your changes and commit them.
4. Push your branch: `git push origin feature-name`
5. Create a pull request.

## Contact

For support or questions, please email Kayla O'Sullivan-Steben at kayla.osullivan-steben@mail.mcgill.ca.

## Disclaimer
This is not the most beautifully packaged or polished code, but I hope it still proves useful. I have also only tested this code on our in-house data. That being said, I happily welcome suggestions, improvements, and contributions!
