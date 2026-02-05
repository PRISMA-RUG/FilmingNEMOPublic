# Filming NEMO (Video-Based Machine Learning for Essential Tremor and Cortical Myoclonus)
This is the associated repository with the publication *We Can Explain: A Non-Invasive and Interpretable Video Approach to Classifying Essential Tremor and Cortical Myoclonus*. To run it, you will need the following:
- A [MediaPipe model](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker). For this publication, we use the lite full body model.
- pip install -r requirements.txt
- A .json file with the saved MediaPipe coordinates.

You can configure running parameters in 'evaluate_with_model.py', then run it to train the model! After each run, the code will generate a task set based on your landmarks, which is loaded dynamically depending on your generation settings.
