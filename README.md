# HandsOn: Indian Sign Language Detection Model
![framework](https://img.shields.io/badge/framework-flask-red)
![libraries](https://img.shields.io/badge/libraries-tensorflow,opencv,mediapipe-green)
![models](https://img.shields.io/badge/models-lstm,mediapipe_holstic-yellow)

LSTM and Mediapipe integrated model deployed on Flask that detects real-time dynamic Indian sign language (ISL) on web browser.

## Web Application Home Page
<img width="1344" alt="Homepage" src="https://github.com/user-attachments/assets/1e700503-6b29-4337-bbf4-33e6f9599fec">



## Video Demo of Web App


https://github.com/user-attachments/assets/58b74e18-b7d1-4cf4-9812-cc6871171b22




### Indian Sign Language Gestures

1. Hello
   
![](/outputs/Hi.gif)

2. Home

![](/outputs/Home.gif)

3. Good

   

![](/outputs/Good.gif)

4. ImSorry

![](/outputs/ImSorry.gif)

## LSTM Model Architecture
```
# LSTM Sequential Model using Keras
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,258))) #each video has input shape of 30 frames of 258 keypoints: X.shape (Handpose + Bodypose)
model.add(Dropout(0.2))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(Dropout(0.2))
model.add(LSTM(64, return_sequences=False, activation='relu')) #next layer is a dense layer so we do not return sequences here
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))

# Compile defines the loss function, the optimizer and the metrics. 
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
```

## System Architecture
![Methodology](https://github.com/user-attachments/assets/c1356c99-f5dc-4a27-bc4d-662d72bb2352)

## Model Evaluation

### Model Training Graphs

1. Train & Test Categorical Accuracy over epochs
   
![output1](https://github.com/user-attachments/assets/969372ec-565c-4cff-9671-03369fa5f4d4)

2. Train & Test Loss over epochs
   
![output2](https://github.com/user-attachments/assets/478e7dea-142b-4c61-bcb3-eff31b4c1267)

### Confusion Matrix

![Train Test Confusin Matrix](https://github.com/user-attachments/assets/405eb0c1-9a8a-42cc-9e08-e06e51d5a676)

### F1 Score
![f1score](https://github.com/user-attachments/assets/dae30802-629e-4a8b-9bf7-1613a85b50bc)


## Discussion of Findings
1. The findings of this project illuminate significant advancements in the application of artificial intelligence to sign language generation via LSTM networks and MediaPipe.
2. The high accuracy and low latency achieved by the proposed model demonstrate its potential utility as a reliable tool for real-time sign language communication. However, the issues identified with fluidity and the handling of complex gestures underscore the intrinsic challenges in fully automating sign language translation.
3. The diversity of sign languages, coupled with regional dialects and individual styles, adds layers of complexity to the task of sign language recognition and generation. Addressing these nuances is critical for the advancement of universal, accessible communication tools for the deaf and hard-of-hearing community.
4. While the proposed model marks a forward step in AI-powered accessibility tools, the exploration of enhanced neural network models, more extensive datasets, and hybrid approaches is recommended to overcome the current limitations.


