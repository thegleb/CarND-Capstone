# Traffic light detection

For an autonomous car it is important to detect the traffic lights so that the car can adjust its speed based on the light's state: red, green or yellow. For example, when the car navigates across different waypoints, the car might want to travel at its maxium speed. However, as soon as it detects a red light, it should reduce the speed (and stop). This is done by adjusting the waypoints' speed.

To detect the traffic lights,  the system (and in particular the Perception module) should have a model that can detects correctly traffic lights and its state (red, yellow or green). Furthermore, apart from a high degree accuracy (as a detection error could cause a car accident), the traffic light classifier might need to run fast, i.e. be efficient.

## Models and arquitectures

There are different models and arquitectures we considered for the traffic light detection.

- Faster-rcnn: this is a type of arquitecture/model that have two training phase. First it has a region proposal network to obtain the region of interest and finally there is a classification network.
- SSD (Single Shot Detection) arquitecture: Instead of having a two training phase, this type of arquitecture have just one training phase. Therefore, this arquitecture can be trained end-to-end.

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

Each model has a particular settings (.config file) that we needed to tune. The different paramters that we adjusted:

- Model:
  - Adjust the number of classes to 4 classses.
  - max_detection_per_class (100) and max_total_detections (300) to 10 and 10 respectively.
- Train_config:
  - find_tune_checkpoints: directoy where the pre-trained model is placed.
  - Number steps: 1000.
- Eval_config:
  - num_examples: number of images in the evaluation data.
- Eval_input_reader and train_input_reader:
  - input_path and label_map_path to the .recrod files and label_map_pbtxt respectively.


## Datasets

The dataset to be used is important in order to get a good model [1]. Different characteristics that we might need to take into account in order to get the more general model as possible: (1) data should be balanced (similar amount of samples for the different classes to classify (red, green and yellow traffic light images), and (2) it might not be necessary to obtain real world images, but good quality synthetic images can achieve a high degree of generalization [1].

### Methodology to generate datasets

####  Data augmentation

# References

[1] https://anyverse.ai/2019/06/19/synthetic-vs-real-world-data-for-traffic-light-classification/
[2] https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md
