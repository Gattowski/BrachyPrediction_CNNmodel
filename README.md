This machine learning project was part of my March-August 2025 Internship in the DKFZ where it consists of two different models.

The first one is a forest regressor model that inputs the coordinates of the template (XYZ) of the prostate of a single patient and outputs the predicted coordinates of the seed positions (XYZ) that are placed correctly, using the lowest objective function value as the weight loss impact for each sample.
It correctly predicts the seed coordinates for one patient.

The second one is a 3DUNet model that inputs the prostate architecture in a XYZ cube and outputs the predicted XYZ cube of seed Coordinates that are inside the prostate cube dimensions. It predicts the coordinates for patients with a range of 0.81 - 1.20 times the size of the given prostate from the matRad mat file.
