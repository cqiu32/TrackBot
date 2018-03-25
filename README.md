# TrackBot
A final project for 8803-AI for Robotics at Georgia Tech: a program that is used to predict the motion of a robot in a control envirnonment 


Instructions:


Our team originally had decided to go with KNN as the algorithm of choice but we've been able to modify particle filter logic with a bit of knn-like logic. Hence, we have 2 versions of algorithms we'd like to submit. Our preferred algorithm is named finalproject.py. But the KNN version is named finalprojectKNN.py

Option 1. (Our preferred option)
Run python grading.py <CUR_DIR>

Option 2. 
To use the KNN version, please run

mv finalproject.py finalprojectPF.py
mv finalprojectKnn.py finalproject.py

Run python grading.py <CUR_DIR>

To switch back, 
mv finalproject.py finalprojectKnn.py
mv finalprojectPF.py finalproject.py


File description:

utilities.py - Some helper code used in final project.py
model - holds the generated k-nn models specifically for ubuntu VM environment; which will be used in finalprojectKnn.py
