
# Installation process of Tfod1

This file will help to install the tfod setup in your system.

  * create dir 'tfod' this dir will be root dir all the files & dirs for further 
    operations will be stored in this dir only .




#### step 1 : create fresh conda environment and activate it 


```bash
    $ conda create -n env_name python=3.6 -y 

    $  conda activate env_name
```

#### step2 : Now we need to install/Download some repos
https://github.com/tensorflow/models/tree/v1.13.0

   - open your git bash  & write 
```bash 
   $  git clone https://github.com/tensorflow/models.git
```

#### step3 : Download the model.tar.gz file (model can be anything , it's your choice)
http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz 


#### step4 : Download the Utils folder 
https://drive.google.com/file/d/12F5oGAuQg7qBM_267TCMt_rlorV-M7gf/view?usp=sharing 


#### step5 : Install the Basic requirements 

```bash 
    $  pip install pillow lxml Cython contextlib2 jupyter matplotlib pandas opencv-python tensorflow==1.14.0

```



#### step 6 : Install protobuf package 

```bash
   $  conda install -c anaconda protobuf
   ```
#### step7 : convert protobuf file to python files


         substeps : 
         - all the protobuf files are present in "tfod/models/research/protos" dir 
         - to store the converted protobuf-> .py files into current dir use --python_out=.
         - to store at any path use --python_out='provide_path_here_where_you_want_to_store'

 * set the path to "tfod/models/research" given as below 
```bash
      cd tfod/models/research

      $  protoc object_detection/protos/*.proto --python_out=.

   ```

#### step8 : For the installtion of object detection library install setup.py
   * change path to "tfod/models/research" 
   * for linux & windows command is same 

```bash 
      cd tfod/models/research
      python setup.py install 

```

#### step 9 : let's check the installtion is properly done or not 

* go to research folder & open jupyter notebook

```bash 
      cd tfod/models/research 

      ( for windows )
            $ jupyter notebook

      ( for linux  )
            $ jupyter-notebook

   Note : inside the object_detection there will be .pynb file and open this file 
      - try to run this file & see whether it's working properly or not 
      - if it run then our installtion is successed.
```

# <  model training  > 
## file Handling & movements 
* #### move faster-rcnn dir to research dir 
* #### move images dir , training dir , generate_tfrecords.py , xml_to_csv.py files from Utils dir to research dir 
* #### tfod/research/object_detection/legacy/"train.py" move this train.py file to research folder
* #### copy  nets , deployment dirs from research/slim to research dir 
* #### copy export_inference_graph.py file from research/object_detection to the research dir 
# step1 : 
* ## Annotation of Images 

for annotation we need to install labelimg
https://tzutalin.github.io/labelImg/

   - this library help to annotated images 

```bash 
      annotated files are available in .xml format
```

for each image there will be seperate .xml file 

# step 2 : 
we need to convert this .xml files into single .csv file 
using 'xml_to_csv.py'

```bash 
   file conversions in future -> 
                     xml ==> csv ==> tfrecords 
```

- conversions of xml to csv 
```bash 
         $ cd tfod/models/research
         $ python xml_to_csv.py

```

# step 3 :

- conversion of csv file to tfrecord format 
for that we'll use "generate_tfrecords.py" file 
```bash 
- do the changes in generate_tfrecord.py file according to your dataset.
```

```bash 
      $ cd tfod/models/research
      $ python generate_tfrecord.py --csv_input=images/train_labels.csv --image_dir=images/train --output_path=train.record
      $ python generate_tfrecord.py --csv_input=images/test_labels.csv --image_dir=images/test --output_path=test.record
```

# Step4 : 
    * for training model we required .config file 
    * this config fill will be present in tfod/models/research/object_detection/sample/configs dir
    * select the appropriate corresponding  .config file from this dir & copy this file into research/training dir
    *  make sure that modelname and config file name should be same.

``` bash 
   1. make changes in training/labelmap.pbtxt file according to your dataset
        put the info in dict format like =>

         item {
                id = 1 
                name = 'cat' 
                }
         item {
                id = 1 
                name = 'dog' 
                }
         item {
                id = 1 
                name = 'horse' 
                }

    2. open training/model_name.config file & make changes in this file 
            changes : 
             num_classes : < how many classes in your dataset >
             fine_tune_checkpoint : 'pretrained_model-path/model.ckpt'
            input_path : 'train.record'
            labelmap_path : 'training/labelmap.pbtxt'
            input_path : 'test.record'
            labelmap_path : 'training/labelmap_path.pbtxt'

            num_steps : 'For how many times you want to train model'
```
# Step 5 : now we'll start training

```bash 
        $  cd tfod/models/research
        $  python train.py --logtostderr --train_dir=training/ --pipeline_config_path=training/"Your_model_name".config
          
          # in my case i am using "faster_rcnn_inception_v2_coco" model hence 

        $  python train.py --logtostderr --train_dir=training/ --pipeline_config_path=training/faster_rcnn_inception_v2_coco.config

```
- it'll create modelckpt (model checkpoint) in  "research/training" directory
- any time you can interrupt / stop training & resume from the same point 

# step 6 : Resume training process from last checkpoint  

```bash 
    same command 

    $  python train.py --logtostderr --train_dir=training/ --pipeline_config_path=training/faster_rcnn_inception_v2_coco.config

```

- model will start training from last fine_tune_checkpoint

# step 7 : convert ckpt file to frozen-inference-graph

```bash 
    need to make some changes in following command
        1. mention your last ckpt point (for eg : 1000)
        2. You can chage output dir also (means where you want to store this output file ) (here : inference_graph)
        3. You can change config path also (depends on usecases ) 

    $  python export_inference_graph.py --input_type image_tensor --pipeline_config_path training/faster_rcnn_inception_v2_coco.config --trained_checkpoint_prefix training/model.ckpt-1000 --output_directory inference_graph

```

  for prediction purpose we'll use this file 


# step 8 : 
```bash

  import cv2
 
cap = cv2.VideoCapture(0)
with detection_graph.as_default():
  with tf.Session(graph=detection_graph) as sess:
    while True:
      ret, image_np = cap.read()
      # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
      image_np_expanded = np.expand_dims(image_np, axis=0)
      image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
      # Each box represents a part of the image where a particular object was detected.
      boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
      # Each score represent how level of confidence for each of the objects.
      # Score is shown on the result image, together with the class label.
      scores = detection_graph.get_tensor_by_name('detection_scores:0')
      classes = detection_graph.get_tensor_by_name('detection_classes:0')
      num_detections = detection_graph.get_tensor_by_name('num_detections:0')
      # Actual detection.
      (boxes, scores, classes, num_detections) = sess.run([boxes, scores, classes, num_detections],feed_dict={image_tensor: image_np_expanded})
      # Visualization of the results of a detection.
      vis_util.visualize_boxes_and_labels_on_image_array(
          image_np,
          np.squeeze(boxes),
          np.squeeze(classes).astype(np.int32),
          np.squeeze(scores),
          category_index,
          use_normalized_coordinates=True,
          line_thickness=8)
 
      cv2.imshow('object detection', cv2.resize(image_np, (800,600)))
      if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
 
cap.release()

```














  
## Authors

- [@shubhamchau78](https://www.linkedin.com/in/shubham-chaudhari-3a7270176)

  
## Tech Stack

**python , tensorflow 1.14.0**


  