
# tfod Mask-Rcnn setup (Image Segmentation)

This file will help to install the tfod-Mask-Rcnn setup in your system.

  * create dir 'project' .
  * this dir will be root dir all the files & dirs for further 
    operations will be stored in this dir only .


## setup


#### step 1 : create fresh conda environment and activate it 


```bash
    $ conda create -n env_name python=3.6 -y 

    $  conda activate env_name
```
#### step 2 : Install requieremets.txt
* [requieremets.txt](https://github.com/shubhamchau222/tensorflow-Basic-Installations/blob/main/mask_rcnn_requirements.txt)


```bash 
        $ pip install -r  requieremets.txt     

        
```

#### step3 : Now we need to install/Download some repos
https://github.com/tensorflow/models/tree/v1.13.0

- open your git bash  & write 
```bash 
   $  git clone https://github.com/tensorflow/models.git
```

#### step 4 : Download pretrained (model can be anything , it's your choice)
- https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md
- http://download.tensorflow.org/models/object_detection/mask_rcnn_inception_v2_coco_2018_01_28.tar.gz
- rename it as 'mask'

#### step 5 : Install protobuf package 

```bash
            $  conda install -c anaconda protobuf
   ```

#### step 6  : convert protobuf file to python files


         substeps : 
         - all the protobuf files are present in "tfod/models/research/protos" dir 
         - to store the converted protobuf-> .py files into current dir use --python_out=.
         - to store at any path use --python_out='provide_path_here_where_you_want_to_store'

```bash
            cd project/models/research

            $  protoc object_detection/protos/*.proto --python_out=.

   ```

#### step 7 : Arrangment of training data 


        - create 'train_data' dir inside ./project/models/research/object_detection dir 

         # dir structure will as below 

         ./project/models/research/object_detection =>
             # create the following dirs 
            - train_data 
                    - testimg       (dir to store test images )
                    - testjson      (dir to store test json annotation  )
                    - trainimg      (dir to store train images )
                    - trainjson     (dir to store train json annotation  )  

```bash 
    # create the dirs trgh cmd line (optional - you can create by GUI )

    $  cd ./project/models/research/object_detection
    $  mkdir train_data
    $  mkdir ./train_data/testimg
    $  mkdir ./train_data/testjson
    $  mkdir ./train_data/trainimg
    $  mkdir ./train_data/trainjson

```


#### step 8 : adding Extra files into the ./research dir 

- **add following files in project/models/research dir ....**

    - [**create_tf_record.py**](https://github.com/shubhamchau222/tensorflow-Basic-Installations/blob/master/mask_rcnn_imp_conversion_files/create_tf_records.py)
    - [**read_pbtxt.py**](https://github.com/shubhamchau222/tensorflow-Basic-Installations/blob/master/mask_rcnn_imp_conversion_files/read_pbtxt.py)
    - [**read_pbtxt_file.py**](https://github.com/shubhamchau222/tensorflow-Basic-Installations/blob/master/mask_rcnn_imp_conversion_files/read_pbtxt_file.py)
    - [**string_int_label_map_pb2.py**](https://github.com/shubhamchau222/tensorflow-Basic-Installations/blob/master/mask_rcnn_imp_conversion_files/string_int_label_map_pb2.py)

#### **step 9 : creating the labelmap.pbtxt** 

    1. create the labelmap.pbtxt in dir (./models/research/object_detection/data)
    put the info in dict format like =>

```bash
       item {
                id = 1 
                name = 'person' 
                }
         item {
                id = 1 
                name = 'dog' 
                }
```

#### **step 10 : creating the tfrecords** 

- data type path is like 
```
    images => [labelim]=> .json => [create_tf_record.py] => tfrecords

    - mask_rcnn accept data in .tfrecords format 
    - convert .json to .record using [create_tf_record.py]
```

- need to do **some changes in [create_tf_record.py] file**

  changes : set the Following paths as per your dataset & dir structure
  **note : path will be different in your case**

**creating custom_train_dog.record**
```bash 
    trainImagePath = "../research/object_detection/traindata/trainimg"
    trainImageJsonPath = "../research/object_detection/traindata/trainjson"
    labelMapPath = "../research/object_detection/data/labelmapdog.pbtxt"
    outputFolderPath = "../research/object_detection/data/custom_train_dog.record"
``` 
**run file [create_tf_record.py]**

-similarely for testing data also (need to changes path )

**creating custom_train_dog.record**
```bash 
    trainImagePath = "../research/object_detection/traindata/testimg"
    trainImageJsonPath = "../research/object_detection/traindata/testjson"
    labelMapPath = "../research/object_detection/data/labelmapdog.pbtxt"
    outputFolderPath = "../research/object_detection/data/custom_test_dog.record"
``` 
**run file[create_tf_record.py]**

- **now we have custom_train_dog.record and custom_test_dog.record file in dir (./research/object_detection/data)**

-------------------------------------------------------------------------------

# **<  model training  >** 
## file Handling & movements 
* #### move mask(downloaded pretrained model ) dir to research dir 
* #### tfod/research/object_detection/legacy/"train.py" copy this **train.py** file to **research folder**
* #### copy  **nets , deployment dirs** from **research/slim** to **research dir** 
* #### copy ****export_inference_graph.py**** file from **./research/object_detection** to the **research dir**
* #### create dir custom_training inside research dir 

``` 
# comments
  - copy labelmap.pbtxt from ./objectdetction/data to ./research/customMask_training
  - copy model configuration file  from ./object_detection/samples/configs/mask_rcnn_inception_v2_coco.config to the ./customMask_training

```

```bash 
   # commands 

    $ cd ./research
    $ mkdir custom_training    
    $ cp ./object_detection/samples/configs/mask_rcnn_inception_v2_coco.config ./custom_training
    $ cp ./object_detection/data/labelmapdog.pbtxt ./custom_training/

```


## **step 11 : make changes in corresponding config file** 

```bash 
# make changes in mask_rcnn_inception_v2_coco.config
 $ cd ./research./customMask_training
    
            -line no : 127 ( set fine tune checkpointpath)
            - line no :10 ( set the number of classes )
            - line no :142  ( set the train.record file path )
            - line no :145  ( set the labelmap.pbtxt file path )
            - line no :158  ( set the test.record file path )
            - line no :160  ( set the labelmap.pbtxt file path  )
```


## **Step 12 : now we'll start training**

```bash 
        $  cd project/models/research
        $  python train.py --logtostderr --train_dir=training/ --pipeline_config_path=training/"Your_model_name".config
          
          # in my case i am using "mask_rcnn_inception_v2_coco" model hence 

        $  python train.py --logtostderr --train_dir=custom_training/ --pipeline_config_path=custom_training/mask_rcnn_inception_v2_coco.config

```

## **step 13 : Resume training process from last checkpoint**  

```bash 
    same command 

    $  $  python train.py --logtostderr --train_dir=custom_training/ --pipeline_config_path=custom_training/mask_rcnn_inception_v2_coco.config
```

- model will start training from last fine_tune_checkpoint

## **step14 : convert ckpt file to frozen-inference-graph**

```bash 
    need to make some changes in following command
        1. mention your last ckpt point (for eg : 1000)
        2. You can chage output dir also (means where you want to store this output file ) (here : inference_graph)
        3. You can change config path also (depends on usecases ) 

    $  python export_inference_graph.py --input_type image_tensor --pipeline_config_path custom_training/mask_rcnn_inception_v2_coco.config --trained_checkpoint_prefix custom_training/model.ckpt-1000 --output_directory ready_to_go

```

- Converted Model will be stored at loc "./research/ready_to_go" dir 
- this model can be use for prediction 


  
## Created by 

- [@shubhamchau78](https://www.linkedin.com/in/shubham-chaudhari-3a7270176)

  
## Tech Stack

**python3.6 , tensorflow 1.14.0**














        















  