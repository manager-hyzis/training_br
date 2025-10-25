# -*- coding: utf-8 -*-
"""Roboflow Detection Plate - TFLite Object Detection.ipynb

Adapted from the official Roboflow TFLite training notebook.
This notebook trains a MobileNetSSDv2 model for license plate detection optimized for mobile devices.

# Introduction

In this notebook, we use TensorFlow Lite to prepare a custom model on the 
DetectionPlate dataset for low-end (e.g. mobile) devices. It takes three steps:

1. Import our dataset from Roboflow using the roboflow library
2. Train a TensorFlow2 Object Detection Model (MobileNetSSDv2 - optimized for mobile)
3. Convert the model to TensorFlow Lite

## Workflow

This model detects and classifies license plates in images:
1. **Detection**: Locates the license plate in the image (returns bounding box coordinates)
2. **Classification**: Identifies the plate type:
   - 1: placa carro (old car plate)
   - 2: placa carro mercosul (Mercosul car plate)
   - 3: placa moto (old motorcycle plate)
   - 4: placa moto mercosul (Mercosul motorcycle plate)
3. **Output**: Returns coordinates (x, y, width, height) for cropping
4. **Next Step**: Crop the detected region and send to a classification model for further processing

Dataset: DetectionPlate (olhodeaguia/detectionplate-soevy v11)
Format: COCO JSON (_annotations.coco.json)
Resolution: 192x192 pixels
Model: MobileNetSSDv2 (lightweight, optimized for mobile/edge devices)
"""

# ============================================================================
# STEP 1: IMPORT AND PREPARE DATA
# ============================================================================

import os
import sys
import shutil
import glob
import urllib.request
import tarfile
import re
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt

print("TensorFlow version:", tf.__version__)

# ============================================================================
# Download dataset from Roboflow (COCO Format)
# ============================================================================

print("\n[1/3] Downloading dataset from Roboflow...")

# Install roboflow library
import subprocess
subprocess.check_call([sys.executable, "-m", "pip", "install", "roboflow", "-q"])

from roboflow import Roboflow

# Initialize Roboflow
rf = Roboflow(api_key="SDfnuMydLG5k2Nq7dlny")
project = rf.workspace("olhodeaguia").project("detectionplate-soevy")
version = project.version(11)  # Version 11 with COCO format (192x192)

# Download dataset in COCO format
dataset = version.download("coco")

print(f"Dataset downloaded to: {dataset.location}")

# ============================================================================
# Prepare directory structure (following standard pattern)
# ============================================================================

# Create /content/train and /content/valid directories (Colab standard)
os.makedirs("/content/train", exist_ok=True)
os.makedirs("/content/valid", exist_ok=True)
os.makedirs("/content/test", exist_ok=True)

# Copy dataset to standard locations
train_src = os.path.join(dataset.location, "train")
valid_src = os.path.join(dataset.location, "valid")
test_src = os.path.join(dataset.location, "test")

# Copy train data
if os.path.exists(train_src):
    for item in os.listdir(train_src):
        src = os.path.join(train_src, item)
        dst = os.path.join("/content/train", item)
        if os.path.isdir(src):
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)
    print(f"✓ Train data copied to /content/train")

# Copy valid data
if os.path.exists(valid_src):
    for item in os.listdir(valid_src):
        src = os.path.join(valid_src, item)
        dst = os.path.join("/content/valid", item)
        if os.path.isdir(src):
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)
    print(f"✓ Valid data copied to /content/valid")

# Copy test data
if os.path.exists(test_src):
    for item in os.listdir(test_src):
        src = os.path.join(test_src, item)
        dst = os.path.join("/content/test", item)
        if os.path.isdir(src):
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)
    print(f"✓ Test data copied to /content/test")

# Verify directories exist
train_dir = "/content/train"
valid_dir = "/content/valid"
test_dir = "/content/test"

assert os.path.exists(train_dir), f"Train directory not found: {train_dir}"
assert os.path.exists(valid_dir), f"Valid directory not found: {valid_dir}"
assert os.path.exists(test_dir), f"Test directory not found: {test_dir}"

# Count images
train_images = len(os.listdir(os.path.join(train_dir, 'images'))) if os.path.exists(os.path.join(train_dir, 'images')) else 0
valid_images = len(os.listdir(os.path.join(valid_dir, 'images'))) if os.path.exists(os.path.join(valid_dir, 'images')) else 0
test_images = len(os.listdir(os.path.join(test_dir, 'images'))) if os.path.exists(os.path.join(test_dir, 'images')) else 0

print(f"\nDataset Summary:")
print(f"  Train images: {train_images}")
print(f"  Valid images: {valid_images}")
print(f"  Test images: {test_images}")

# ============================================================================
# STEP 2: SETUP TENSORFLOW OBJECT DETECTION API
# ============================================================================

print("\n[2/3] Setting up TensorFlow Object Detection API...")

# Clone tensorflow models repository
if not os.path.exists("models"):
    print("Cloning TensorFlow models repository...")
    os.system("git clone --quiet https://github.com/tensorflow/models.git")
else:
    print("Models repository already exists")

# Install required packages
print("Installing required packages...")
os.system("pip install -q tf_slim")
os.system("apt-get install -qq protobuf-compiler python-pil python-lxml python-tk")
os.system("pip install -q Cython contextlib2 pillow lxml matplotlib")
os.system("pip install -q pycocotools")
os.system("pip install -q lvis==0.5.3")

# Compile protobuf files
print("Compiling protobuf files...")
os.chdir("models/research")
os.system("protoc object_detection/protos/*.proto --python_out=.")

# Set Python path
os.environ['PYTHONPATH'] += ':/content/models/research/:/content/models/research/slim/'

# Test the installation
print("Testing TensorFlow Object Detection API installation...")
os.system("python object_detection/builders/model_builder_test.py")

os.chdir("../..")

# ============================================================================
# STEP 3: PREPARE DATA DIRECTORY FOR TRAINING
# ============================================================================

print("\n[3/3] Preparing data directory for training...")

# Create data directory structure (following standard pattern)
data_dir = "/content/tensorflow-object-detection-faster-rcnn/data"
os.makedirs(data_dir, exist_ok=True)

# Copy train and valid data to data directory
train_data_dir = os.path.join(data_dir, "train")
valid_data_dir = os.path.join(data_dir, "valid")

os.makedirs(train_data_dir, exist_ok=True)
os.makedirs(valid_data_dir, exist_ok=True)

# Copy from /content/train to data directory
if os.path.exists(os.path.join("/content/train", "images")):
    shutil.copytree(os.path.join("/content/train", "images"), 
                    os.path.join(train_data_dir, "images"), 
                    dirs_exist_ok=True)
    print(f"✓ Train images copied")

if os.path.exists(os.path.join("/content/train", "labels")):
    shutil.copytree(os.path.join("/content/train", "labels"), 
                    os.path.join(train_data_dir, "labels"), 
                    dirs_exist_ok=True)
    print(f"✓ Train labels copied")

if os.path.exists(os.path.join("/content/train", "_annotations.coco.json")):
    shutil.copy(os.path.join("/content/train", "_annotations.coco.json"),
               os.path.join(train_data_dir, "_annotations.coco.json"))
    print(f"✓ Train annotations copied")

# Copy from /content/valid to data directory
if os.path.exists(os.path.join("/content/valid", "images")):
    shutil.copytree(os.path.join("/content/valid", "images"), 
                    os.path.join(valid_data_dir, "images"), 
                    dirs_exist_ok=True)
    print(f"✓ Valid images copied")

if os.path.exists(os.path.join("/content/valid", "labels")):
    shutil.copytree(os.path.join("/content/valid", "labels"), 
                    os.path.join(valid_data_dir, "labels"), 
                    dirs_exist_ok=True)
    print(f"✓ Valid labels copied")

if os.path.exists(os.path.join("/content/valid", "_annotations.coco.json")):
    shutil.copy(os.path.join("/content/valid", "_annotations.coco.json"),
               os.path.join(valid_data_dir, "_annotations.coco.json"))
    print(f"✓ Valid annotations copied")

# ============================================================================
# STEP 4: CONFIGURE MODEL AND TRAINING
# ============================================================================

print("\nConfiguring model for training...")

# Model configuration
MODELS_CONFIG = {
    'ssd_mobilenet_v2': {
        'model_name': 'ssd_mobilenet_v2_coco_2018_03_29',
        'pipeline_file': 'ssd_mobilenet_v2_coco.config',
        'batch_size': 12
    },
}

selected_model = 'ssd_mobilenet_v2'
MODEL = MODELS_CONFIG[selected_model]['model_name']
pipeline_file = MODELS_CONFIG[selected_model]['pipeline_file']
batch_size = MODELS_CONFIG[selected_model]['batch_size']

# Training parameters
num_steps = 100000  # Increase for better accuracy
num_eval_steps = 50

print(f"Model: {MODEL}")
print(f"Batch size: {batch_size}")
print(f"Training steps: {num_steps}")

# ============================================================================
# STEP 5: DOWNLOAD AND PREPARE BASE MODEL
# ============================================================================

print("\nDownloading pre-trained base model...")

MODEL_FILE = MODEL + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'
DEST_DIR = 'pretrained_model'

if not os.path.exists(MODEL_FILE):
    print(f"Downloading {MODEL_FILE}...")
    urllib.request.urlretrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)

tar = tarfile.open(MODEL_FILE)
tar.extractall()
tar.close()

os.remove(MODEL_FILE)
if os.path.exists(DEST_DIR):
    shutil.rmtree(DEST_DIR)
os.rename(MODEL, DEST_DIR)

fine_tune_checkpoint = os.path.join(DEST_DIR, "model.ckpt")
print(f"Fine-tune checkpoint: {fine_tune_checkpoint}")

# ============================================================================
# STEP 6: CREATE LABEL MAP (COCO Format - Named Classes)
# ============================================================================

print("\nCreating label map file (COCO format with named classes)...")

# Classes from COCO format - IDs 1-4 (ID 0 is reserved for "plates" supercategory)
classes = {
    1: 'placa carro',
    2: 'placa carro mercosul',
    3: 'placa moto',
    4: 'placa moto mercosul'
}

# Create label map in TensorFlow format (matching COCO IDs)
label_map_content = ""
for class_id, class_name in classes.items():
    label_map_content += f"""item {{
  id: {class_id}
  name: '{class_name}'
}}\n"""

label_map_path = os.path.join(data_dir, "label_map.pbtxt")
with open(label_map_path, 'w') as f:
    f.write(label_map_content)

# Create COCO format names file (one class per line, indexed by ID)
coco_names_path = os.path.join(data_dir, "classes.names")
with open(coco_names_path, 'w') as f:
    for class_id in sorted(classes.keys()):
        f.write(f"{class_id}: {classes[class_id]}\n")

print(f"✓ Label map created: {label_map_path}")
print(f"✓ COCO classes file created: {coco_names_path}")
print(f"✓ Classes ({len(classes)} total):")
for class_id, class_name in classes.items():
    print(f"    {class_id}: {class_name}")

# ============================================================================
# STEP 7: CONFIGURE TRAINING PIPELINE
# ============================================================================

print("\nConfiguring training pipeline...")

pipeline_fname = os.path.join('models/research/object_detection/samples/configs/', pipeline_file)

assert os.path.isfile(pipeline_fname), f'Pipeline file not found: {pipeline_fname}'

# Get number of classes
num_classes = len(classes)

# Read and modify pipeline configuration
with open(pipeline_fname) as f:
    s = f.read()

with open(pipeline_fname, 'w') as f:
    # Set fine_tune_checkpoint
    s = re.sub('fine_tune_checkpoint: ".*?"',
               f'fine_tune_checkpoint: "{fine_tune_checkpoint}"', s)
    
    # Set training record file
    train_record = os.path.join(os.path.abspath(data_dir), "train", "train.record")
    s = re.sub(
        '(input_path: ".*?)(train.record)(.*?")', 
        f'input_path: "{train_record}"', s)
    
    # Set validation record file
    val_record = os.path.join(os.path.abspath(data_dir), "valid", "val.record")
    s = re.sub(
        '(input_path: ".*?)(val.record)(.*?")', 
        f'input_path: "{val_record}"', s)
    
    # Set label_map_path
    s = re.sub(
        'label_map_path: ".*?"', 
        f'label_map_path: "{label_map_path}"', s)
    
    # Set training batch_size
    s = re.sub('batch_size: [0-9]+',
               f'batch_size: {batch_size}', s)
    
    # Set training steps
    s = re.sub('num_steps: [0-9]+',
               f'num_steps: {num_steps}', s)
    
    # Set number of classes
    s = re.sub('num_classes: [0-9]+',
               f'num_classes: {num_classes}', s)
    
    f.write(s)

print(f"Pipeline configured with {num_classes} classes")

# ============================================================================
# STEP 8: TRAIN THE MODEL
# ============================================================================

print("\nStarting model training...")
print("This may take a while depending on the number of steps and GPU availability")

# Use absolute path for model directory
model_dir = os.path.abspath('models/research/training/')
os.makedirs(model_dir, exist_ok=True)

os.chdir('models/research')

train_command = (
    f"python object_detection/model_main.py "
    f"--pipeline_config_path={pipeline_fname} "
    f"--model_dir={model_dir} "
    f"--alsologtostderr "
    f"--num_train_steps={num_steps} "
    f"--num_eval_steps={num_eval_steps}"
)

print(f"Running: {train_command}")
print(f"Model directory: {model_dir}")
os.system(train_command)

os.chdir("../..")

# ============================================================================
# STEP 9: EXPORT TRAINED MODEL FOR TFLITE
# ============================================================================

print("\nExporting trained model for TFLite conversion...")

os.chdir('models/research')

output_directory = './fine_tuned_model'
tflite_directory = './fine_tuned_model/tflite'

# Find the latest checkpoint
if not os.path.exists(model_dir):
    print(f"❌ Model directory not found: {model_dir}")
    print("Training may not have completed successfully.")
    sys.exit(1)

lst = os.listdir(model_dir)
lst = [l for l in lst if 'model.ckpt-' in l and '.meta' in l]

if lst:
    print(f"✓ Found {len(lst)} checkpoints")
    steps = np.array([int(re.findall('\d+', l)[0]) for l in lst])
    last_model = lst[steps.argmax()].replace('.meta', '')
    last_model_path = os.path.join(model_dir, last_model)
    
    print(f"Using checkpoint: {last_model_path}")
    
    # Export inference graph
    export_command = (
        f"python object_detection/export_inference_graph.py "
        f"--input_type=image_tensor "
        f"--pipeline_config_path={pipeline_fname} "
        f"--output_directory={output_directory} "
        f"--trained_checkpoint_prefix={last_model_path}"
    )
    print(f"Exporting inference graph...")
    os.system(export_command)
    
    # Export TFLite graph
    export_tflite_command = (
        f"python object_detection/export_tflite_ssd_graph.py "
        f"--input_type=image_tensor "
        f"--pipeline_config_path={pipeline_fname} "
        f"--output_directory={tflite_directory} "
        f"--trained_checkpoint_prefix={last_model_path}"
    )
    print(f"Exporting TFLite graph...")
    os.system(export_tflite_command)
else:
    print("No checkpoints found. Training may not have completed successfully.")

os.chdir("../..")

# ============================================================================
# STEP 10: CONVERT TO TFLITE WITH UINT8 QUANTIZATION AND RGB FORMAT
# ============================================================================

print("\nConverting model to TensorFlow Lite format with UINT8 quantization...")
print("Format: RGB (3 channels)")
print("Quantization: UINT8 (8-bit unsigned integer)")

tflite_graph_path = os.path.join('models/research/fine_tuned_model/tflite/tflite_graph.pb')

if os.path.exists(tflite_graph_path):
    # Convert using TFLite converter with UINT8 quantization
    convert_command = (
        "tflite_convert "
        "--input_shape=1,192,192,3 "
        "--input_arrays=normalized_input_image_tensor "
        "--output_arrays=TFLite_Detection_PostProcess,TFLite_Detection_PostProcess:1,"
        "TFLite_Detection_PostProcess:2,TFLite_Detection_PostProcess:3 "
        "--allow_custom_ops "
        "--inference_type=QUANTIZED_UINT8 "
        "--inference_input_type=QUANTIZED_UINT8 "
        "--mean_values=128 "
        "--std_dev_values=128 "
        f"--graph_def_file={tflite_graph_path} "
        "--output_file=detection_plate_model.tflite"
    )
    print(f"Running TFLite conversion with UINT8 quantization...")
    os.system(convert_command)
    
    print("\n✓ TFLite model successfully created: detection_plate_model.tflite")
    print("  - Quantization: UINT8 (8-bit)")
    print("  - Format: RGB (3 channels)")
    print("  - Input size: 192x192 (COCO format)")
    print("  - Optimized for mobile/edge devices")
else:
    print(f"TFLite graph not found at {tflite_graph_path}")

# ============================================================================
# STEP 11: TEST INFERENCE (OPTIONAL)
# ============================================================================

print("\nTesting inference on sample images...")

test_images_dir = os.path.join(dataset.location, "test", "images")
test_images = glob.glob(os.path.join(test_images_dir, "*.*"))[:5]

if test_images:
    print(f"Found {len(test_images)} test images")
    for img_path in test_images:
        print(f"  - {os.path.basename(img_path)}")
else:
    print("No test images found")

print("\n" + "="*70)
print("TRAINING COMPLETE!")
print("="*70)
print(f"TFLite model saved as: detection_plate_model.tflite")
print(f"Model directory: {os.path.abspath('models/research/fine_tuned_model')}")
print("\nClasses (YOLO format - Named):")
for idx, class_name in enumerate(classes):
    print(f"  {idx}: {class_name}")
print("="*70)

# ============================================================================
# STEP 12: INFERENCE EXAMPLE WITH CROP
# ============================================================================

print("\n" + "="*70)
print("INFERENCE EXAMPLE - How to use the model")
print("="*70)

inference_code = """
# Example: Using the trained TFLite model for inference with crop
# Model: UINT8 quantized, RGB format

import tensorflow as tf
import cv2
import numpy as np
from PIL import Image

# Load TFLite model (UINT8 quantized)
interpreter = tf.lite.Interpreter(model_path='detection_plate_model.tflite')
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load image
image = cv2.imread('test_image.jpg')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB format
height, width = image.shape[:2]

# Resize to model input size (192x192 for DetectionPlate COCO format)
input_size = 192
resized_image = cv2.resize(image_rgb, (input_size, input_size))

# Prepare input for UINT8 quantized model
# Input range: 0-255 (UINT8)
input_data = np.expand_dims(resized_image, axis=0).astype(np.uint8)

# Run inference
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()

# Get detections
detections = interpreter.get_tensor(output_details[0]['index'])
detection_classes = interpreter.get_tensor(output_details[1]['index'])
detection_scores = interpreter.get_tensor(output_details[2]['index'])

# Process detections (COCO format - IDs 1-4)
class_names = {
    1: 'placa carro',
    2: 'placa carro mercosul',
    3: 'placa moto',
    4: 'placa moto mercosul'
}
confidence_threshold = 0.5

for i in range(len(detection_scores[0])):
    score = detection_scores[0][i]
    if score > confidence_threshold:
        # Get bounding box coordinates (normalized 0-1)
        box = detections[0][i]
        y_min, x_min, y_max, x_max = box
        
        # Convert to pixel coordinates
        x_min_px = int(x_min * width)
        y_min_px = int(y_min * height)
        x_max_px = int(x_max * width)
        y_max_px = int(y_max * height)
        
        # Get class (COCO format - IDs 1-4)
        class_id = int(detection_classes[0][i])
        class_name = class_names.get(class_id, f'Unknown ({class_id})')
        
        # Draw bounding box
        cv2.rectangle(image, (x_min_px, y_min_px), (x_max_px, y_max_px), (0, 255, 0), 2)
        cv2.putText(image, f'{class_name} ({score:.2f})', 
                   (x_min_px, y_min_px - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # CROP the detected plate region
        cropped_plate = image[y_min_px:y_max_px, x_min_px:x_max_px]
        
        # Save or process the cropped plate
        # cv2.imwrite(f'plate_{class_name}_{i}.jpg', cropped_plate)
        # Send cropped_plate to classification model for further processing

# Display result
cv2.imshow('Detections', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""

print(inference_code)
print("="*70)
