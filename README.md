# Face-Symmetry-dlib
The current project deploys a Python dlib library to detect a face from the uploaded picture and allocate 68 facial landmarks. Enhanced code retrieves (x, y) coordinates from each point to calculate the Euclidean distance between facial feature points pairwise. The facial symmetry index is therefore calculated on the basis of the sum of pairwise deviations between main facial attributes from their midpoint on the face (e.g. the difference between the left-side jaw-center distance and right-side jaw-center distance + the difference between the left-corner nose-center distance and right-corner nose-center distance + etc.).