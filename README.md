# Face-Symmetry-dlib
The current project deploys a Python dlib library to detect a face from the uploaded picture and allocate 68 facial landmarks. Enhanced code retrieves (x, y) coordinates from each point to calculate the Euclidean distance between facial feature points pairwise. The facial symmetry index is therefore calculated on the basis of the sum of pairwise deviations between main facial attributes from their midpoint on the face (e.g. the difference between the left-side jaw-center distance and right-side jaw-center distance + the difference between the left-corner nose-center distance and right-corner nose-center distance + etc.).

# Important!
Running the code requires the installation of heavy dependencies on your computer locally. Once you put all the files downloaded from the current repository on your Python environment you should be getting warning errors from the output that will require you to install packages. 
Please, install them all. Also make sure to have both "dlib-face-recognition_resnet_model_v1.dat" and "shape_predictor_68_face_landmarks.dat" in the same directory as your main.py file. Since the secon .dat file is too heavy, please download it from https://drive.google.com/file/d/1v0nGz_rvGeWp3eiJg_50-9IyLPTIoWwj/view?usp=share_link.

main.py contains the face detection and symmetry index calculation code, while detectface.py returns the picture with allocated landmarks. To return a pucture with allocated 

# How to run this code on pycharm?
1. Create a new Project (File->New Project)
2. Find VCS on the top panel -> Get from Version Control -> paste current repo's URL -> Clone
3. Download "shape_predictor_68_face_landmarks.dat" file from https://drive.google.com/file/d/1v0nGz_rvGeWp3eiJg_50-9IyLPTIoWwj/view?usp=share_link
4. Copy it to the same directory where all the other files are
5. Create a configuration for the current projet (select the same directory)
6. Install CMake Library (Python Packages on lower panel -> type CMake)
7. Install dlib library (this step takes a while)
8.  -m pip install --upgrade pip
9.  Install opencv-python
10.  Install cv2
