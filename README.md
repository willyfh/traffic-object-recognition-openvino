# Trafic Object Recognition - OpenVINO

Implementation of  object detection and semantic segmentation of traffic objects in the front facing car camera  using OpenVINO's pretrained models.
- **Object Detection:** Identify vehicles by drawing the bounding boxes on the detected objects. [vehicle-detection-adas-0002
](https://docs.openvinotoolkit.org/latest/_models_intel_vehicle_detection_adas_0002_description_vehicle_detection_adas_0002.html)
- **Semantic Segmentation:**
Classify objects as roads, sidewalks, buildings, walls, fences, poles, traffic lights, traffic signs, vegetation, terrain, sky, people, passengers, cars, trucks, buses, trains, motorcycles, bicycles, or electric vehicles. [semantic-segmentation-adas-0001
](https://docs.openvinotoolkit.org/latest/_models_intel_semantic_segmentation_adas_0001_description_semantic_segmentation_adas_0001.html)
- **Road Segmentation:** Classify objects as roads, curbs, painted lines, or backgrounds. [road-segmentation-adas-0001
](https://docs.openvinotoolkit.org/latest/_models_intel_road_segmentation_adas_0001_description_road_segmentation_adas_0001.html)




## Demo


![4K Mask RCNN COCO Object detection and segmentation #2](https://gitlab.com/willyfitrahendria/trafic-object-recognition-openvino/-/raw/master/demo/driving_jakarta.gif)

## Prerequisite
*  Setup OpenVINO. [OpenVINO™ toolkit Documentation
](https://docs.openvinotoolkit.org/latest/index.html)
*  Download [the models](https://software.intel.com/en-us/openvino-toolkit/documentation/pretrained-models)

## How to start the program
Required args:
> python app.py -m {model_name} -t {model_type}

For more details:
> python app.py -h 