# Udacity SDC Engineer - Capstone Project

### Team Fast and Furious: Carla Drift

#### Team members
- Gleb Podkolzin (lead)
- Ajith Rai
- Allen Hsu
- David Altimira
- Ming Wong

# General approach

We started by implementing a waypoint updater based on walkthrough lectures. The waypoint updater publishes at a
 frequency of 50hz. At every cycle, we send a maximum set of waypoints ahead of the car to the `final_waypoints` publisher.
 
Each waypoint includes a target velocity value. When there is no stopline nearby, we simply publish these waypoints
 as given. However, if there is a stopline the car must obey, what happens is we gradually bring the target velocity
  of the car down to 0 to make sure the car stops before the stopline waypoint.
  
A `camera_info` publisher publishes the current camera image to the `/image_color` topic at 10hz. We subscribe to
 `/image_color` inside the `tl_detector` node and cache the current image received. The goal of `tl_detector` is to
  perform the necessary image processing and publish the next stopline. We do this at a frequency of 10hz as well.
  Additionally, we run the most recent available camera image through a neural network (described below) to detect
   any traffic lights with their status.
   
Since the neural network takes longer than 100msec (speed needed to maintain a constant 10hz) to return results, at
 every cycle we set a
 boolean to keep track of when we are currently processing an image. As a result, each detection cycle uses the latest
  image and there is less chance of lag due to triggering a new cycle of traffic light detection before the previous
   one completes, causing stale results - one of the problems we ran into initially.
   
For each detected traffic light state, we use a couple techniques to filter out possible noise:

- Using detections with a confidence level higher than `0.5`
-  Waiting until we identify the same state at
 least 2 times before updating the prediction.
 
 Both of these techniques guard against unpredictable behavior due to
  errors in detection. For the second method, 2 was a number chosen to balance confidence in the prediction with
   reacting quickly enough to changing conditions.
   
Another byproduct of slow detection speeds is the possibility of the car seeing a yellow light and
 interpreting it too slowly, allowing the light to turn red and causing the car to run the red light. With 2 cycles to
  update predictions, the worst case is ~3 detections, because the light can change in the next frame after the
   previous cycle of detection kicked off.
   
To solve this, we assume the next stopline applies unless either:
- we see a `green` light or
- we identify a
 `yellow` light and we are close enough to it to exclude the possibility of the light turning red before
    we cross the line.

This ensures that in the worst case scenario, the car will always stop at the next stopline (assuming it is not mis
-detecting red lights as green).

# Traffic light detection

For an autonomous car it is important to detect the traffic lights so that the car can adjust its speed based on the
 light's state: red, green or yellow. For example, when the car navigates across different waypoints, the car might want to travel at its maximum speed. However, as soon as it detects a red light, it should reduce the speed (and stop). This is done by adjusting the waypoints' speed.

To detect the traffic lights,  the system (and in particular the Perception module) should have a model that can detects correctly traffic lights and its state (red, yellow or green). Furthermore, apart from a high degree accuracy (as a detection error could cause a car accident), the traffic light classifier might need to run fast, i.e. be efficient.

## Models and architectures

There are different models and architectures we considered for the traffic light detection.

- Faster-rcnn: this is a type of architecture/model that have two training phase. First it has a region proposal network to obtain the region of interest and finally there is a classification network.
- SSD (Single Shot Detection) architecture: Instead of having a two training phase, this type of architecture have just one training phase. Therefore, this architecture can be trained end-to-end.

Also, there are implementation of neural networks that are optimized to run efficiently. These are named MobileNets, which are are neural networks that can run efficiently and can therefore meet the efficiency requirement.

Our approach was to reuse existing models from the TensorFlow detection model zoo [2]: collection of detection models pre-trained on different datasets on the task of object detection. These pre-trained models would be our starting point and we will re-train these models with our dataset generated for our training.

The different models we aimed to re-train and evaluate their accuracy and performance were:

- ssd_inception_v2_coco.
- ssd_mobilenet_v2_coco.
- faster_rcnn_inception_v2_coco

### Methodology approach taken

- Train the different 3 models in a simulation data set (set the same parameters for the three models).
- Test the accuracy and performance of these three models and select the one that is more suitable for our purposes.
- From the model selected, adjust the parameters (batch size, number of step) in order to improve the accuracy levels.
- Train a model for the simulated dataset, and freeze and export the graph.
- Train a model for the real world dataset, and freeze and export another graph.

### Settings of the models

Each model has a particular settings (.config file) that we needed to tune. The different parameters that we adjusted:

- Model:
  - Adjust the number of classes to 4 classes.
  - max_detection_per_class (100) and max_total_detections (300) to 10 and 10 respectively.
- Train_config:
  - find_tune_checkpoints: directory where the pre-trained model is placed.
  - Number steps: 1000.
- Eval_config:
  - num_examples: number of images in the evaluation data.
- Eval_input_reader and train_input_reader:
  - input_path and label_map_path to the .record files and label_map_pbtxt respectively.


## Datasets

The dataset to be used is important in order to get a good model [1]. Different characteristics that we might need to take into account in order to get the more general model as possible: (1) data should be balanced (similar amount of samples for the different classes to classify (red, green and yellow traffic light images), and (2) it might not be necessary to obtain real world images, but good quality synthetic images can achieve a high degree of generalization [1].

### Methodology to generate datasets

####  Data augmentation

# References

[1] https://anyverse.ai/2019/06/19/synthetic-vs-real-world-data-for-traffic-light-classification/
[2] https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md
