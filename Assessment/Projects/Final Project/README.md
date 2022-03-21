# Forest Fire detector using Transfer learning image classification method.
![image](https://user-images.githubusercontent.com/91799774/159302414-b98c1d13-df3e-4b44-ab7f-17b26907f7da.png)


Youtube link: https://youtu.be/GpoTbInmQ1w

Arduino library for ML model: https://github.com/AbhipsaKar/casa0018/blob/main/Assessment/Projects/Final%20Project/ei-firedetect_transfer-arduino-1.0.14.zip

Arduino sketch for Arduino Nano 33 BLE: https://github.com/AbhipsaKar/casa0018/blob/main/Assessment/Projects/Final%20Project/fire_detector.ino

## Project overview
Over 18 million hectares were destroyed in the infamous Australia bushfires in 2019 to 2020 alone( Sorri, 2021). Owing to the presence of kindling in the forests, a small fire quickly spreads destroying all flora and fauna in its path. Countries have utilised various setups to detect these fires ranging from drones, satellites, cameras to manned fire towers(Knaus,2020).
Croatia, a country constantly affected by wildfires is now setting up video surveillance networks to detect these forest fires before they become uncontrollable. 
However, a major issue in the existing systems is the fact that the video footage from these systems must be reviewed manually which could result in a delay in the fire detection( Sorri, 2021). 
Additionally, as forest fires usually occur in uninhabited areas, it takes more time for the fire to be detected and handled. Temperature or satellite based fire alerts are often inadequate because they can only detect the fire once it becomes big.

It would , therefore, be beneficial to add an additional deep learning unit to the exisiting video surveillance networks that could send an alert to the nearest Fire station as soon as it detects a fire. The idea of the system is to ultimately have a GPS enabled PTZ camera at each node which is Wifi enabled to send a fire alert to MQTT. The fire stations, which would have subscribed to the MQTT topic would get an alert from a camera node as soon as it detects fire, along with its gps coordinates.

## Research question
Can Image classification deep learning model be used to detect fires with reliable accuracy and use the traditional video surveillance setups to automate the process.

## Project Design
### Data collection
The initial set of data was downloaded from Kaggle website(Saied, 2020) which contained labelled data for 2 categories: Fire and Not Fire.
There were 968 images in total(724 Fire images and 244 Not Fire images)
Observations on the dataset:
1. Dataset covered multiple forest types.
2. Dataset contains images of varied exposure levels.
3. Not fire dataset contained images that could be easily confused with Fire. For Ex: Sunrise/Sunset, Autumn scene, sunny day
4. Fire dataset contained images from forest as well as buildings or cities.
5. The colour of Fire in fire dataset was mostly red with black smoke.

On using this dataset, although the resulting accuracy was good, in practical testing, the system classified most items as Fire. 
This was because the number of Not fire images was too low compared to the number of Fire images.
An additional set of images was downloaded from images.cv(Image datasets for Computer Vision and machine learning) website to supplement the 'Not fire' dataset.
The final number of data items collected was 1793(1119 'Not fire' images and 724 'Fire' images)

### Model Design
After the data collection step on Edge impulse, the next step generates features for the model design for the training set of data(1416 data items):
![Picture3](https://user-images.githubusercontent.com/91799774/159271822-8fc9dc7b-0c2c-47e6-afd4-5dbd806ca19c.png)

Each image of the training set is redrawn based on these features and can be visualised in the Feature Explorer:
![image](https://user-images.githubusercontent.com/91799774/159272771-0fc8fc15-a1a1-42b1-a451-7ff22d56be78.png)

In the next step, the model can be trained for transfer learning though choosing various parameters like base model, model design, data augmentation, learning step etc.

#### Design iteration 1:
MobileNet v1 0.1 
No data balancing or data augmentation
All other settings at default:
![image](https://user-images.githubusercontent.com/91799774/159273901-72ebb668-c785-4e45-8b82-07ac60c2e432.png)

Although the model accuracy was good, it was also important for the model not to miss detecting 'Fire'.
In order to decrease the rate of false negatives from 15.7%, multiple iterations were carried out. 
1. Add data augmentation and balancing: false negatives at 12.5%
2. Add data augmentation and balancing and changed loss function to 'Binary_crossentropy': false negatives at 12.3%

#### Design iteration 2:
MobileNet v1 0.2
No data balancing or data augmentation:
All other settings at default:
![image](https://user-images.githubusercontent.com/91799774/159275778-9016a85e-225b-4179-b458-51a1ca84d78d.png)

Although both model accuracy and rate of false negatives was excellent, the model weighed 131 Kb and could not run on the Arduino Nano with error "Could not allocate memory for TensorFlow Arena".

#### Design iteration 3:
MobileNet v1 0.1
Data balancing
Data augmentation
Additional dense layer of 8 neurons:
![image](https://user-images.githubusercontent.com/91799774/159276440-0fad5706-fe02-423c-b1d7-fc713784e50f.png)

With a good accuracy of 97% and low false negatives rate of 4.9%, this model has produced the best results by far.
The model resizes each input image to 96*96 RGB array i.e tensor with input shape [96,96,3]
The model uses the following data augmentation methods:
1. Flips the image randomly
2. Increase the image size, then randomly crop it down to the original dimensions
3. Vary the brightness of the image

The model uses Adam optimiser and 'Categorical_crossentropy' loss function to converge.

After training the base model with the updated parameters, the model was retrained for new set of data.
![image](https://user-images.githubusercontent.com/91799774/159277243-0493f5a8-7b90-4f4b-9a75-64c43c8e75c8.png)

In the next step, Arduino library was chosen as the deployment option to generate a quantised int8 model weighing 62 kb
![image](https://user-images.githubusercontent.com/91799774/159278482-17b09e71-68af-4896-ad61-eb94d0de6822.png)

Comparing the confusion matrix, the the rate of false negatives increases to 10.7%(6.0 + 4.7). However, this model still performs better than the initial model in Design iteration 1.
![image](https://user-images.githubusercontent.com/91799774/159278765-6e0d450a-4290-4dd9-848f-272068a5a254.png)

### Model build
The build includes:
1. Tiny ML kit(Arduino Nano 33 BLE, OV7675 camera, Tiny ML shield)
2. 2 red LEDs
3. 3d printed Tower enclosure
4. 3d printed dragon(The dragon for 3D-printable Modular Castle playset,2017)

![image](https://user-images.githubusercontent.com/91799774/159295183-1a9debfe-43e9-4b08-804e-b4e3eb66d1ab.png)

The Arduino nano 33 BLE contains onboard red and blue LEDs which are used to denote the classification of final output: Red when 'Fire', Blue when 'Not Fire'.
2 Additional Red Leds have been added externally to be able to signal the Fire alert from outside the enclosure.

### Experiments and results 
To test the performance of the ML model, a test run was performed on Edge impulse to classify unseen pictures from the collected dataset.
![image](https://user-images.githubusercontent.com/91799774/159280186-fcc1715d-ae58-41b9-b3c9-c33707617b7d.png)

The rate of accuracy was good at 96.54%. After deploying the model on the Arduino Nano, I used images infront on the camera because it is not possible
to simulate forest fire scenario.
![image](https://user-images.githubusercontent.com/91799774/159302384-6c1db2c1-54ec-4bd1-8e1d-feb578ac0a10.png)


The test images were tested and the results compared in indoor vs outdoor environment and light vs dark environment.
The time of day did not seem to affect the reading even though some of the classification was done on the laptop which had its own brightness for display.
There was some difference in reading between indoor and outdoor scenes. Very bright scenes were sometimes misclassified as Fire although the exact correlation between the level of brightness and classification could not be determined.
<img width="483" alt="Test" src="https://user-images.githubusercontent.com/91799774/159281216-4bc8bf9a-3d12-4ee6-afca-6ba775d4ee4f.PNG">
<img width="524" alt="test01" src="https://user-images.githubusercontent.com/91799774/159281288-472b75bf-73d3-4ae7-ad20-2c1736ec0711.PNG">

It was observed that there were sudden one-off fire alert classifications from the model even though there was no discernible reason for the classification.
To avoid this, an averaging algorithm was added in the output handler. 
In this method, the last 10 classification readings of the model were saved in a global array. In each iteration of the loop(), a count of the number of 'Fire' readings and 'Not fire' readings was extracted from the array. The final output was the reading whose count was greater. In short, fire would be alerted only if atleast 6 of the last 10 readings were 'Fire'. As each reading was taken 2 seconds apart, this algorithm would only delay the alert by a further 10 seconds but would bolster the reliability the readings.
![image](https://user-images.githubusercontent.com/91799774/159286060-a3482aa1-f4eb-4ffc-be09-3609a599c890.png)


### Critical reflection 

 There were some observations during the testing process.
1. It was required to Point the camera directly to the fire scene. In a real world scenario, we would need a PTZ camera to capture pictures from every angle.
2. It was observed that every time the setup starts, the first few readings misclassify the environment as 'Fire'. Also, When the exposure level in the environment suddenly changes, the model misclassifies the env as 'Fire'.
3. As the model is trained on forest images, the indoor lights are often misclassified as 'Fire'. So are objects with reddish color. This might give faulty readings in Autumn when leaves change color even though the 'Not fire' dataset contains forests in Autumn which are correctly classified during the test phase.
4. Small lights like lighter fire(fires of white colour) are not recognised by the model.

Further experimentation:
The rate of false negatives could possibly be decreased further by finding a model design which was bigger than this design but still small enough to run on the Arduino Nano 33.
To be able to send a fire alert though MQTT, it would be required to use a different board that supports Wifi. Also, to ensure that the fire alert was indeed identified correctly, the setup could send a picture of the scene for manual review. This would prevent Fire station personnel to prepare and reach the scene just to find out that it was a false alert.

### References:
1. Anon, 2017. The dragon for 3D-printable Modular Castle playset. Download free STL file The Dragon for 3D-printable Modular Castle Playset • 3D print template ・ Cults. Available at: https://cults3d.com/en/3d-model/game/the-dragon-for-3d-printable-modular-castle-playset [Accessed March 21, 2022]. 
2. Saied, A., 2020. Fire dataset. Kaggle. Available at: https://www.kaggle.com/datasets/phylake1337/fire-dataset/ [Accessed March 21, 2022]. 
3. Image datasets for Computer Vision and machine learning. Download Forest image classifcation dataset for your computer vision project. Available at: https://images.cv/dataset/forest-image-classification-dataset [Accessed March 21, 2022]. 
4. Knaus, C., 2020. Early warning: human detectors, drones and the race to control Australia’s extreme bushfires. The guardian. Available at: https://www.theguardian.com/australia-news/2020/oct/25/early-warning-human-detectors-drones-and-the-race-to-control-australias-extreme-bushfires.[Accessed March 21, 2022]. 
5.  Sorri, A., 2021. Fighting fire with cameras – an innovative and logical step forward. Secure Insights. Available at: https://www.axis.com/blog/secure-insights/camera-fire-fighting/ [Accessed March 21, 2022]. 
