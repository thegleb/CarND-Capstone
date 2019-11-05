# Udacity SDC Engineer - Capstone Project

### Team Fast and Furious: Carla Drift

#### Team members
- Gleb Podkolzin (lead)
- Ajith Rai
- Allen Hsu
- David Altimira
- Ming Wong

# General approach

We started by implementing a waypoint updater based on walkthrough lectures. The waypoint updater publishes at a frequency of 50hz. At every cycle, we send a maximum set of waypoints ahead of the car to the `final_waypoints` publisher.

Each waypoint includes a target velocity value. When there is no stopline nearby, we simply publish these waypoints as given. However, if there is a stopline the car must obey, what happens is we gradually bring the target velocity of the car down to 0 to make sure the car stops before the stopline waypoint.

A `camera_info` publisher publishes the current camera image to the `/image_color` topic at 10hz. We subscribe to `/image_color` inside the `tl_detector` node and cache the current image received. The goal of `tl_detector` is to perform the necessary image processing and publish the next stopline. We do this at a frequency of 10hz as well. Additionally, we run the most recent available camera image through a neural network (described below) to detect any traffic lights with their status.

Since the neural network takes longer than 100msec (speed needed to maintain a constant 10hz) to return results, at every cycle we set a boolean to keep track of when we are currently processing an image. As a result, each detection cycle uses the latest image and there is less chance of lag due to triggering a new cycle of traffic light detection before the previous one completes, causing stale results - one of the problems we ran into initially.

For each detected traffic light state, we use a couple techniques to filter out possible noise:

- Using detections with a confidence level higher than `0.5`
- Waiting until we identify the same state at least 2 times before updating the prediction.

Both of these techniques guard against unpredictable behavior due to errors in detection. For the second method, 2 was a number chosen to balance confidence in the prediction with reacting quickly enough to changing conditions.

Another byproduct of slow detection speeds is the possibility of the car seeing a yellow light and interpreting it too slowly, allowing the light to turn red and causing the car to run the red light. With 2 cycles to update predictions, the worst case is ~3 detections, because the light can change in the next frame after the previous cycle of detection kicked off.

To solve this, we assume the next stopline applies unless either:
- we see a `green` light or
- we identify a `yellow` light and we are close enough to it to exclude the possibility of the light turning red before we cross the line.

This ensures that in the worst case scenario, the car will always stop at the next stopline (assuming it is not mis -detecting red lights as green).

# Traffic light detection

For an autonomous car it is important to detect the traffic lights so that the car can adjust its speed based on the light's state: red, green or yellow. For example, when the car navigates across different waypoints, the car might want to travel at its maximum speed. However, as soon as it detects a red light, it should reduce the speed (and stop). This is done by adjusting the waypoints' speed.

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

- Train the different models in a simulation data set.
- Evaluate accuracy and speed for each of these models.
- Select a model.
- Check how we could improve the model selected.
- Train the same model for the site data.
- Freeze and export the graphs of both models (one for the simulation and the other for the real site). The exported graphs will be the ones to be used in the simulated and real vehicle.

### Settings of the models

Each model has a particular settings (.config file) that we needed to tune. The different parameters that we adjusted:

- Model:
  - Adjust the number of classes to 4 classes.
  - max_detection_per_class (100) and max_total_detections (300) to 10 and 10 respectively.
- Train_config:
  - find_tune_checkpoints: directory where the pre-trained model is placed.
  - Number steps: 10000/20000.
- Eval_config:
  - num_examples: number of images in the evaluation data.
- Eval_input_reader and train_input_reader:
  - input_path and label_map_path to the .record files and label_map_pbtxt respectively.


## Datasets

The dataset to be used is important in order to get a good model [1]. Different characteristics that we might need to take into account in order to get the more general model as possible: (1) data should be balanced (similar amount of samples for the different classes to classify (red, green and yellow traffic light images), and (2) it might not be necessary to obtain real world images, but good quality synthetic images can achieve a high degree of generalization [1].

### Simulation data

We used two different datasets. For training the models we used an external one [3]. For evaluating the model, we created another dataset that was generated with a script. This script consisted in taking random background images, and placing random number of traffic lights (red, green or yellow) at random positions and with some transformations such as rotation or scaling.

### Site data


## Results of the model analysis

### Models trained with simulator data

We used different models with different steps and we evaluated the performance based on the four categories (see below picture):
- Correct: If the traffic light is detected and labelled correctly.
- Incorrect: If the traffic light is detected and labelled incorrectly.
- No Detected: If the traffic light is not detected.
- Background: If we detect a traffic light on a background.

As we can see in the picture the fast_rcnn_inception_v2 with 10000 steps performed better than the other models. The rest of the models performed similarly. We can also observer that there were very few traffic lights that were detected incorrectly but there are quite a few traffic lights not detected.

![](./img/simulator_model_performance.png "Model performance using simulator data")

We also measured the efficienty (i.e. the amount of time the model to process data).

![](./img/simulator_model_timings.png "Model performance using simulator data")

The above figure we can see that the fast_rcnn_inception_v2 is quite slow and the fasted model is the ssd_mobilenet. Although the fast_rcnn is the more accurate we decided to discard this model owing to it is too slow. Since all the other models performed similary, we chose the ssd_mobilenet_v1_coco_20000 as our model.

To further analyse the selected model, we also decided to evaluate its performance for different traffic light sizes. This could indicate if there are some traffic sizes that the model performed better than others and whether we need to feed more data (and which traffic light sizes would be better the new data to have).

We decided to have 10 categories of bounding boxes sizes. To obtain the range of traffic light sizes for each category, we calculated the maximum size and the minimum size of the traffic lights. Then, the size was obtained by $step=\frac{(max_size-min_size)}{10}$. So the first category in the graph (category 0) had the following range of traffic light size: [min_size, min_size+step]. The last category had [max_size-step, max_size].

 ![](./img/simulator_ssd_mobilenet_bbox_performance.png "Traffic light sizes analsysi for the ssd mobilened model")

 We can see that the model performed very well in detecting correctly the traffic light. However, we can see that for very small traffic lights, the model could not detect them. This could also explain why we have seen during all this analysis lots of traffic lights not detected. One explanation for this is that our evaluation script produced too small traffic lights that the model was not trained for. In our case, we did not consider this as a big concern as if we miss a traffic light that is too far away, this traffic light will probably be detected as we are approaching it and safely stop the car if we need to stop.

### Site model



### Methodology to generate datasets

####  Data augmentation

# Future work
- Train the model with smaller traffic lights so that we do not miss these traffic lights in our detection.

# References

[1] https://anyverse.ai/2019/06/19/synthetic-vs-real-world-data-for-traffic-light-classification/
[2] https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md
[3] https://github.com/alex-lechner/Traffic-Light-Classification#linux
