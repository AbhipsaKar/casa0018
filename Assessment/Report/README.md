# Forest Fire detector- Image classification model

#### Youtube link: https://youtu.be/vTjy-qHdXjo
#### Edge impulse project: https://studio.edgeimpulse.com/studio/86904/
#### Project Page: https://github.com/AbhipsaKar/casa0018/tree/main/Assessment/Projects/Final%20Project



## Introduction
Over 18 million hectares were destroyed in the infamous Australia bushfires in 2019 to 2020 alone (Sorri, 2021). Owing to the presence of kindling in the forests, a small fire quickly spreads causing extensive damage to wildlife and environment. Research comparing various methods of Forest fire detection (Alkhatib, 2014) revealed that optical detection systems have an edge over human and satellite detection systems in terms of area of coverage and low detection turnaround time, but are undermined due to a high rate of false alarms and the need for manual review of video footage. These factors could be overcome with an additional deep learning unit in the existing video surveillance networks that could send an alert to the nearest Fire station as soon as it detects a fire. 
The research landscape in intelligent optical fire detection systems include experimenting various metrics of measurement. While models like Zhang et al. (2016) have used CNN based models to detect fire from UAV mounted cameras, Celik & Ma( 2009) tried to identify fire pixel detection from color video footages measuring the colour and illuminance.  Liu & Ahuja(2004) measure the spectral pattern of scene to detect forest fires. 

## Research Question
Can Image classification deep learning model be used to identify forest fires using low resolution color images taken from video camera?

## Application Overview

 ![image](https://user-images.githubusercontent.com/91799774/163422014-e40df6ae-ba7e-48d1-9b08-5d3da150ebe0.png)
  <figcaption><i><h6>Figure 1: Application diagram of Fire detector ML project</h6></i></figcaption>


The application (Figure 1) consists of the microcontroller which hosts the ML model and operation code. Every 2 seconds, an image is captured from the camera, resized to the input size of model and sent for prediction. The output of the model is used to light the red device LED in case of fire or blue device LED in case of no fire. 2 external LEDs are used to display the fire alert outside the enclosure.

As the microcontroller, I chose the Arduino Nano 33 owing to its low power consumption which is a critical parameter for outdoor deployment. The tiny ML kit(Figure 2) comes with this microcontroller as well as a camera that can be used with the Arduino library created on Edge impulse.

![image](https://user-images.githubusercontent.com/91799774/163422523-d789a40f-e887-4532-8312-d344a84c52ee.png)
<figcaption><i><h6>Figure 2: TinyML kit and its components</h6></i></figcaption>

The final application complete with a 3-D printed enclosure(Figure 3 and Figure 4) keeps it protected from the elements.
![image](https://user-images.githubusercontent.com/91799774/163422732-2b6edcfc-8046-4287-8b71-31acd0b18fd9.png)
<figcaption><i><h6>Figure 3: Fire detector physical components</h6></i></figcaption>

![image](https://user-images.githubusercontent.com/91799774/163422955-b1fc34ed-2ee2-4c2c-9584-950f5bf190ef.png)

<figcaption><i><h6>Figure 4: Fire detector enclosure</h6></i></figcaption>

## Data

The initial set of data was downloaded from Kaggle website (Saied, 2020) which contained labelled data for 2 categories: Fire and Not Fire. There were 968 images in total (724 Fire images and 244 Not Fire images). An additional set of images was downloaded from images.cv(Image datasets for Computer Vision and machine learning) website to supplement the 'Not fire' dataset as it was found that the Non-fire dataset was too small. The final number of data items collected was 1793(1119 'Not fire' images and 724 'Fire' images)
 Observations on the dataset:
1.	Dataset covered multiple forest types(Figure: 5)
2.	Dataset contains images of varied exposure levels(Figure: 6)
3.	Not fire dataset contained images that could be easily confused with Fire(Figure: 7). For Ex: Sunrise/Sunset, Autumn scene, sunny day
4.	Fire dataset contained images from forest as well as buildings or cities(Figure: 8)
5.	The color of Fire in fire dataset was mostly red with pictures of black smoke(Figure: 9)

![image](https://user-images.githubusercontent.com/91799774/163423075-f1d6471e-ce02-4fc0-8f41-c00e918e0574.png)
<figcaption><i><h6>Figure 5: Examples of multiple forest types</h6></i></figcaption>

![image](https://user-images.githubusercontent.com/91799774/163423180-36559508-b274-4523-963e-60f82b0e5e11.png)
<figcaption><i><h6>Figure 6: Examples of different exposures</h6></i></figcaption>

![image](https://user-images.githubusercontent.com/91799774/163423281-a168b7d3-e365-4ef8-a36f-0d39421b8e6d.png)
<figcaption><i><h6>Figure 7:confusing pictures in Non fire category</h6></i></figcaption>

![image](https://user-images.githubusercontent.com/91799774/163423547-e8e08df7-daa9-45a7-af95-f03095257cf1.png)
<figcaption><i><h6>Figure 8: Fire dataset: indoor and outdoor</h6></i></figcaption>

![image](https://user-images.githubusercontent.com/91799774/163423595-2627d867-1eed-4a1c-add1-c71c4f1e206c.png)
<figcaption><i><h6>Figure 9: Pictures of smoke</h6></i></figcaption>

## Model
After failing to deploy my own CNN model which weighed 500 kb and other transfer learning models weighing over 100 kb, it was observed that models over 100 kb failed to allocate tensor space on the Arduino Nano 33 BLE. The base model chosen thereafter is the lightweight MobileNet v1.0 architecture(figure 10) which takes a 96*96 colour image as input i.e having input tensor shape [96,96,3]. 

![image](https://user-images.githubusercontent.com/91799774/163423860-bd97e1fa-721c-4a8b-8bba-ba55624e7371.png)
<figcaption><i><h6>Figure 10: Base Model architecture</h6></i></figcaption>

![image](https://user-images.githubusercontent.com/91799774/163423882-cb0e7223-ab27-4f7d-b6ef-9a813719d98a.png)
<figcaption><i><h6>Figure 11: Transfer learning model architecture</h6></i></figcaption>

## Experiments
Table 1 shows the experiments executed to change the hyperparameters and fine tune the model once the model was finalized. As the initial accuracy achieved with default parameters was already good, the confusion matrix on Edge impulse did not change at all with the multiple experiments and made it difficult to visualize the effect of these changes. I used the ‘Edit as python notebook’ option(Figure:12) on the Transfer learning(Impulse design ) tab to import the code to Google colab and used the matplot library to differentiate the changes and check for signs of overfitting/underfitting.

<figcaption><i><h6>Table 1: Hyperparameters finetuning</h6></i></figcaption>

![image](https://user-images.githubusercontent.com/91799774/163424276-8d701a30-734b-43d5-b128-d50ffad85f9e.png)

![image](https://user-images.githubusercontent.com/91799774/163424315-14af830a-ba74-46a7-9584-604617dea3fb.png)
<figcaption><i><h6>Figure 12: Python notebook import option</h6></i></figcaption>

The graph of the metrics with the set of hyperparameters highlighted in bold is given below (Figure 13) followed by its confusion matrix from Edge impulse(Figure 14). The graph shows no signs of overfitting/underfitting and maintains a low rate of false negatives which means that less fires go undetected. Even with the same set of hyperparameters, the metrics vary slightly from one run to the next.
![image](https://user-images.githubusercontent.com/91799774/163424453-22af67f1-0f69-49a1-a930-e777e6013925.png)

<figcaption><i><h6>Figure 13: Model metrics graph</h6></i></figcaption>

![image](https://user-images.githubusercontent.com/91799774/163424494-524dbf8e-84d8-4277-bd36-941c66749d5c.png)

<figcaption><i><h6>Figure 14: Confusion matrix</h6></i></figcaption>

The edge impulse adds data augmentation using methods such as resizing, cropping, flipping, and changing brightness. This helps to create variation in the dataset. The model training on Edge impulse(Figure 15) also adds a fine-tuning process of 10 epochs before fixing the weights of base model.

![image](https://user-images.githubusercontent.com/91799774/163424601-e5237eeb-68aa-430e-a930-bde08d6b3f19.png)
<figcaption><i><h6>Figure 15: Transfer learning page (Edge impulse)</h6></i></figcaption>

## Deployment
The model created in Edge impulse weighs 62.1Kb as shown in the Deployment page on Edge impulse(Figure 16).

![image](https://user-images.githubusercontent.com/91799774/163425084-35bfa218-a5ce-4153-83e4-de0d9348eaf4.png)
<figcaption><i><h6>Figure 16: Confusion matrix-quantized</h6></i></figcaption>
A function, RespondToDetection(), was created to send the classification output to the LEDs( and print on the serial output) which included an averaging algorithm to smooth out any faulty readings(Figure 17).

![image](https://user-images.githubusercontent.com/91799774/163425236-6341ce91-2ad8-4cad-a3d2-3d551ae13701.png)
<figcaption><i><h6>Figure 17: Serial output during test</h6></i></figcaption>

## Results and Observations
A test run on Edge impulse (Figure 18) from the supplied dataset revealed that the model performed with an impressive accuracy of 97.07% and a false negative rate of 3.4%.
![image](https://user-images.githubusercontent.com/91799774/163425608-3aef975c-9e5a-45d3-b237-8b8499f5dd52.png)
<figcaption><i><h6>Figure 18: Test accuracy performance</h6></i></figcaption>

An analysis of the misclassified images suggests that pictures with a reddish tint confuse the ML model. On the other hand, some smaller fires are often misidentified (Figure 19).
![image](https://user-images.githubusercontent.com/91799774/163425734-0b9e01a5-2c40-4913-af30-8813a54005e5.png)
<figcaption><i><h6>Figure 19: Misclassified pictures from test run</h6></i></figcaption>

After deploying the model on Tiny ML kit, 2 testing methods were adopted. 
In the first method (Figure 20), unseen pictures were displayed in front of the camera and the readings were taken from serial window as well as LED output

![image](https://user-images.githubusercontent.com/91799774/163425994-8a5b99f8-536d-4208-885b-eb0f5ca37276.png)
<figcaption><i><h6>Figure 20: Test method 1(from laptop)</h6></i></figcaption>

From the below snapshot(Figure 21), it is evident that some pictures which have a smokey texture are misclassified as fire.

![image](https://user-images.githubusercontent.com/91799774/163426133-4c96a3b5-ca1c-4415-b065-dea9e9864093.png)
<figcaption><i><h6>Figure 21: Classification results displayed in textbox(RED: incorrect , GREEN: correct)</h6></i></figcaption
  
In the second test method (Figure 22), the Tiny ML kit was pointed at the scene to get a real time classification. 
 
![image](https://user-images.githubusercontent.com/91799774/163426338-5de2903b-7ba4-4903-aa5a-66b351c0809b.png)
<figcaption><i><h6>Figure 22: Test method 2(through real time scene capture)</h6></i></figcaption
  
As color in the scene is critical, a classification test done for colors showed that red color in the scene mostly results in a fire classification (Figure 23).
 
![image](https://user-images.githubusercontent.com/91799774/163426573-edbd5041-8355-4622-8c24-9060f2f50bcb.png)
<figcaption><i><h6>Figure 23: Color classification test(RED: incorrect , GREEN: correct)</h6></i></figcaption>

The classification from indoor scenes(Figure 24) resulted in unreliable readings depending on light source and exposure because the ‘Not fire’ dataset consisted of outdoor images.

  ![image](https://user-images.githubusercontent.com/91799774/163426775-8a7849e0-035b-4385-b050-264177a697b8.png)
  <figcaption><i><h6>Figure 24: Indoor test classification (RED: incorrect , GREEN: correct)</h6></i></figcaption>

Finally, a test was conducted in the outdoor scene where the scene could only be test for ‘Not fire’ conditions. The setup was tested for various exposure conditions as well as varied color scenes(Figure 25). The output was almost always ‘Not-fire’ with a rare one off faulty ‘Fire’ readings. The averaging algorithm added in ‘RespondToDetection()’ corrected these readings.

![image](https://user-images.githubusercontent.com/91799774/163427048-f3b1b0d2-403d-4c07-97a5-def2943dae98.png)
<figcaption><i><h6>Figure 25: Outdoor classification test(Not fire)</h6></i></figcaption>

The next stage of texting could include testing in the real forest with a PTZ camera to test the actual efficacy. The testing process is incomplete due to lack of real outdoor fire scenes.

## Conclusion
The system, although performs well would need substantial investments and are best suited to improve pre-existing optical fire detectors. The idea of the system is to ultimately have a GPS enabled Pan-Tilt-Zoom camera at each connected sensor node to send a fire alert to MQTT. The fire stations subscribed to the MQTT topic would get an alert from a camera node as soon as it detects fire, along with the node’s exact location. The research on hybrid sensor networks suggest that multiple sensors (camera and temperature sensor) are more successful by overcoming individual sensor limitations. Hence, additional sensors could be used to supplement this system to improve accuracy.


## Bibliography
<ul>
  <li>
Sorri, A., (2021). Fighting fire with cameras – an innovative and logical step forward. Secure Insights. Available at: https://www.axis.com/blog/secure-insights/camera-fire-fighting/ [Accessed March 21, 2022].
  </li>
<li>
Alkhatib, A. A. A. (2014) ‘A Review on Forest Fire Detection Techniques’, International Journal of Distributed Sensor Networks. doi: 10.1155/2014/597368.
Zhang, Q., Xu, J., Xu, L. and Guo, H., (2016), January. Deep convolutional neural networks for forest fire detection. In Proceedings of the 2016 international forum on management, education and information technology application. Atlantis Press.
</li>
  <li>
Celik, T. & Ma, K.-K. (2009) ‘Computer Vision based fire detection in color images’. IEEE Xplore. Available at: https://ieeexplore.ieee.org/abstract/document/5045970?casa_token=QHzJmTDfdFQAAAAA%3AfFsQGva8oEixyiA6wMLmLFzNlBpt3gPHxiICX8Gvll3PtcRRQQ0kjfFFgkMA0o1wyqOQhig [Accessed April 9, 2022]. 
</li>
  <li>
Liu, C.-B. & Ahuja, N., (2004). Vision based fire detection. IEEE Xplore. Available at: https://ieeexplore.ieee.org/document/1333722/references#references [Accessed April 9, 2022]. 
    </li>
<li>
 Saied, A., (2020). Fire dataset. Kaggle. Available at: https://www.kaggle.com/datasets/phylake1337/fire-dataset/ [Accessed March 21, 2022].
 </li>
 <li>
  Image datasets for Computer Vision and machine learning. Download Forest image classifcation dataset for your computer vision project. Available at: https://images.cv/dataset/forest-image-classification-dataset [Accessed March 21, 2022].
 </li>
</ul>


## Declaration of Authorship

I, Abhipsa kar, confirm that the work presented in this assessment is my own. Where information has been derived from other sources, I confirm that this has been indicated in the work.


Abhipsa kar

14th Apr, 2022
