library("reshape2")
library("ggplot2")

dataModel <- read.csv(file="evaluation_model_accuracy_site_13nov_site_real_adj.csv", header=FALSE, sep=",")
names(dataModel) <- c("ModelName", "NumSteps", "TrainDataName", "TestEvaluation", "Synthetic", "TrafficLight", "TotalSamples", "CorrectDetected", "IncorrectDetected", "NotDetected", "DetectedBackgroundAsTrafficLight")

dataModel$TrafficLight <- factor(dataModel$TrafficLight,
                    levels = c(1,2,3),
                    labels = c("Green", "Red", "Yellow"))

dataModel.reshape <- melt(dataModel, id.vars = c("ModelName", "TrafficLight"),  measure.vars = c("CorrectDetected", "IncorrectDetected", "NotDetected", "DetectedBackgroundAsTrafficLight"))
names(dataModel.reshape) <- c("ModelName", "TrafficLight", "Measurement", "Count")

dataModel.reshape$Measurement <- factor(dataModel.reshape$Measurement,
                                        levels = c("CorrectDetected", "IncorrectDetected", "NotDetected", "DetectedBackgroundAsTrafficLight"),
                                        labels = c("Correct", "Incorrect", "No Detected", "Background"))

dataModel.reshape$ModelName <- factor(dataModel.reshape$ModelName,
                                      levels = c("ssd_inception_v2_coco_10000_synthetic_ajith_test", "ssd_mobilenet_v1_coco_20000_gamma_test", "ssd_mobilenet_v1_coco_40000_external_site_test"),
                                      labels = c("ssd_inception_v2_coco_10000", "ssd_mobilenet_v1_coco_20000_gamma", "ssd_mobilenet_v1_coco_40000"))

ggplot(data=dataModel.reshape, aes(x=Measurement, y=Count, fill = ModelName)) +
  geom_bar(stat="identity",  position = "dodge") + 
  scale_fill_discrete(name = "Model") +
  ggtitle("ssd mobilenet v1 coco 20000 evaluation")#+ 
  #facet_grid(TrafficLight ~ .)

ggplot(data=dataModel.reshape, aes(x=Measurement, y=Count)) +
  geom_bar(stat="identity",  position = "dodge") + 
  scale_fill_discrete(name = "Model") +
  ggtitle("ssd mobilenet v1 coco 20000 evaluation") + 
  facet_grid(TrafficLight ~ .)

# now to evaluate/analyse the accuracy based on the bounding box
processData <- function(data, numBBoxCat) {
  minBBoxSize <- min(data$BoxSize)
  maxBBoxSize <- max(data$BoxSize)
  step <- (maxBBoxSize-minBBoxSize)/numBBoxCat
  itBBoxSize <- minBBoxSize
  iBBox <- 0
  
  df <- data.frame()
  modelName <- as.character(data$ModelName[1])
  
  while (itBBoxSize < maxBBoxSize) {
    rangeMin <- itBBoxSize
    rangeMax <- itBBoxSize + step
    
    #print(rangeMin)
    #print(rangeMax)
    
    data_tmp <- data[data$BoxSize >= rangeMin,]
    data_tmp <- data_tmp[data_tmp$BoxSize <= rangeMax,]
    
    #print("num rows")
    #print(nrow(data_tmp))
    
    correctDetected <- sum(data_tmp$CorrectDetected)
    incorrectDetected <- sum(data_tmp$IncorrectDetected)
    notDetected <- sum(data_tmp$NotDetected)
    totalSamples <-sum(data_tmp$TotalSamples)
    
    #print(correctDetected)
    #print(incorrectDetected)
    #print(notDetected)
    #print(totalSamples)
    
    df2 <- rbind(df, c(iBBox, correctDetected, incorrectDetected, notDetected, totalSamples))
    #df2 <- rbind(df, c(as.character(data$ModelName[1]), iBBox, correctDetected, incorrectDetected, notDetected, totalSamples))
    df <- df2
    #print(paste(paste(rangeMin, " , "), rangeMax))
    itBBoxSize = rangeMax
    iBBox <- iBBox + 1
  }
  names(df) <- c("BBoxSizeLevel", "Correct", "Incorrect", "NotDetected", "TotalSamples")
  return (df)
}

dataModelBBox <- read.csv(file="evaluation_model_accuracy_bbox_site_12nov.csv", header=FALSE, sep=",")
names(dataModelBBox) <- c("ModelName", "NumSteps", "TrainDataName", "TestEvaluation", "TrafficLight", "BoxSize", "CorrectDetected", "IncorrectDetected", "NotDetected", "TotalSamples")
dataModelBBox$TrafficLight <- factor(dataModelBBox$TrafficLight,
                                 levels = c(1,2,3),
                                 labels = c("Green", "Red", "Yellow"))

dataModelBBox <- dataModelBBox[dataModelBBox$ModelName == "ssd_mobilenet_v1_coco_20000_gamma_test",]

dataModelBBox_Process <- processData(dataModelBBox, 10)

dataModelBBox_Process.reshape <- melt(dataModelBBox_Process, id.vars = c("BBoxSizeLevel"),  measure.vars = c("Correct", "Incorrect", "NotDetected"))
names(dataModelBBox_Process.reshape) <- c("BBoxSizeLevel", "Measurement", "Count")
dataModelBBox_Process.reshape$BBoxSizeLevel <- as.factor(dataModelBBox_Process.reshape$BBoxSizeLeve)

ggplot(dataModelBBox_Process.reshape, aes(x=BBoxSizeLevel, y=Count, group=Measurement)) +
  geom_line(aes(color=Measurement))+
  geom_point(aes(color=Measurement))

# Analysis of the existing database

dataDB <- read.csv(file="evaluation_simulator_training_dataset.csv", header=FALSE, sep=",")
names(dataDB) <- c("Light", "xmin", "xmax", "ymin", "ymax")
table(dataDB$Light)

# Analyse model with different adjustments of gamma in the test set

dataGamma <- read.csv(file="evaluation_model_accuracy_gamma_12nov.csv", header=FALSE, sep=",")
names(dataGamma) <- c("ModelName", "NumSteps", "TrainDataName", "TestEvaluation", "Synthetic", "TrafficLight", "TotalSamples", "CorrectDetected", "IncorrectDetected", "NotDetected", "DetectedBackgroundAsTrafficLight", "Gamma")

dataGamma$TrafficLight <- factor(dataGamma$TrafficLight,
                                 levels = c(1,2,3),
                                 labels = c("Green", "Red", "Yellow"))

dataGamma.reshape <- melt(dataGamma, id.vars = c("TrafficLight", "Gamma"),  measure.vars = c("CorrectDetected"))
names(dataGamma.reshape) <- c("TrafficLight", "Gamma","Measurement", "Count")

ggplot(data=dataGamma.reshape, aes(x=Gamma, y=Count)) +
  geom_point() +
  geom_line() +
  ggtitle("Detected correctly") +
  facet_grid(TrafficLight ~ .)


