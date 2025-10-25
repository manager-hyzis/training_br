# -*- coding: utf-8 -*-
"""Roboflow TFLite Object Detection - DetectionPlate"""

import os
from pathlib import Path

os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

try:
    import google.colab  # type: ignore
    IS_COLAB = True
except ImportError:  # pragma: no cover
    IS_COLAB = False

if IS_COLAB:
    BASE_DIR = Path('/content')
else:
    BASE_DIR = Path('/home/manager/Desktop/training_br')

# GUARD: Se estamos dentro de models/research, volta para BASE_DIR
current_dir = Path.cwd()
if 'models/research' in str(current_dir):
    os.chdir(str(BASE_DIR))
    print(f"🔄 Voltando para BASE_DIR: {BASE_DIR}")

RESEARCH_DIR = BASE_DIR / 'models' / 'research'
DATA_ROOT = BASE_DIR / 'data'
TEST_IMAGES_SRC = BASE_DIR / 'test' / 'test'

print("🔍 Verificando dependências...")
!pip install tensorflow==2.19 protobuf==4.25.3 -q

num_steps = 100000
num_eval_steps = 50

print(f"📁 BASE_DIR: {BASE_DIR}")
print(f"📁 RESEARCH_DIR: {RESEARCH_DIR}")

MODELS_CONFIG = {
    'ssd_mobilenet_v2': {
        'model_name': 'ssd_mobilenet_v2_coco_2018_03_29',
        'pipeline_file': 'ssd_mobilenet_v2_coco.config',
        'batch_size': 12
    },
    'faster_rcnn_inception_v2': {
        'model_name': 'faster_rcnn_inception_v2_coco_2018_01_28',
        'pipeline_file': 'faster_rcnn_inception_v2_pets.config',
        'batch_size': 12
    },
    'rfcn_resnet101': {
        'model_name': 'rfcn_resnet101_coco_2018_01_28',
        'pipeline_file': 'rfcn_resnet101_pets.config',
        'batch_size': 8
    },
}

selected_model = 'ssd_mobilenet_v2'
MODEL = MODELS_CONFIG[selected_model]['model_name']
pipeline_file = MODELS_CONFIG[selected_model]['pipeline_file']
batch_size = MODELS_CONFIG[selected_model]['batch_size']

import shutil
import glob
import urllib.request
import tarfile
import re
import numpy as np
import six.moves.urllib as urllib

os.chdir(str(BASE_DIR))

if not (BASE_DIR / 'models').exists():
    print("📥 Clonando TensorFlow models...")
    !git clone --quiet https://github.com/tensorflow/models.git
else:
    print("✅ TensorFlow models já existe")

print("📦 Instalando dependências do sistema...")
!pip install tf_slim -q
!apt-get install -qq protobuf-compiler python-pil python-lxml python-tk 2>/dev/null || true
!pip install -q Cython contextlib2 pillow lxml matplotlib
!pip install -q pycocotools
!pip install lvis==0.5.3 -q

if (RESEARCH_DIR / 'object_detection').exists():
    print(f"✅ TensorFlow Object Detection já existe em {RESEARCH_DIR}")
    if not (RESEARCH_DIR / 'object_detection' / 'protos' / 'string_int_label_map_pb2.py').exists():
        print("🔧 Compilando protos...")
        os.chdir(str(RESEARCH_DIR))
        !protoc object_detection/protos/*.proto --python_out=.
        os.chdir(str(BASE_DIR))
    else:
        print("✅ Protos já compilados")
else:
    print("⚠️  TensorFlow Object Detection não encontrado!")

pythonpath_extra = f"{str(RESEARCH_DIR)}:{str(RESEARCH_DIR / 'slim')}"
existing_pythonpath = os.environ.get('PYTHONPATH')
os.environ['PYTHONPATH'] = (
    f"{existing_pythonpath}:{pythonpath_extra}" if existing_pythonpath else pythonpath_extra
)

print("📦 Instalando Roboflow...")
!pip install roboflow -q

from roboflow import Roboflow

if (DATA_ROOT / 'train' / 'plates.tfrecord').exists() and (DATA_ROOT / 'test' / 'plates.tfrecord').exists():
    print("✅ Dataset TFRecord já existe, pulando download...")
    dataset_root = DATA_ROOT
else:
    print("📥 Baixando dataset Roboflow...")
    rf = Roboflow(api_key="SDfnuMydLG5k2Nq7dlny")
    project = rf.workspace("olhodeaguia").project("detectionplate-soevy")
    version = project.version(11)
    dataset = version.download("tfrecord", location=str(BASE_DIR / 'train'))
    print(f"✅ Dataset baixado em: {dataset.location}")
    dataset_root = Path(dataset.location)

train_dest_dir = DATA_ROOT / 'train'
test_dest_dir = DATA_ROOT / 'test'
train_dest_dir.mkdir(parents=True, exist_ok=True)
test_dest_dir.mkdir(parents=True, exist_ok=True)

def _find_tfrecord(root: Path, subset: str) -> Path:
    subset_aliases = {
        'train': {'train', 'training'},
        'test': {'test', 'testing'},
        'val': {'val', 'valid', 'validation'}
    }
    aliases = subset_aliases.get(subset, {subset})
    candidates = []
    for path in root.rglob('*.tfrecord'):
        parts_lower = {part.lower() for part in path.parts}
        stem_lower = path.stem.lower()
        if parts_lower & aliases or any(alias in stem_lower for alias in aliases):
            candidates.append(path)
    if not candidates:
        candidates = list(root.rglob('*.tfrecord'))
    if not candidates:
        raise FileNotFoundError(f"No TFRecord files found in {root}")
    return candidates[0]

def _find_label_map(root: Path) -> Path:
    candidates = list(root.rglob('*label_map.pbtxt'))
    if not candidates:
        raise FileNotFoundError(f"No label map (.pbtxt) found in {root}")
    # Prefer file inside train subset if available
    for candidate in candidates:
        if 'train' in {part.lower() for part in candidate.parts}:
            return candidate
    return candidates[0]

train_record_src = _find_tfrecord(dataset_root, 'train')
test_record_src = _find_tfrecord(dataset_root, 'test')
if not test_record_src:
    test_record_src = _find_tfrecord(dataset_root, 'val')
train_label_src = _find_label_map(dataset_root)

if str(train_record_src) != str(train_dest_dir / 'plates.tfrecord'):
    shutil.copy(train_record_src, train_dest_dir / 'plates.tfrecord')
    print(f"✅ Copiado: {train_record_src.name}")
else:
    print(f"✅ Train TFRecord já está no lugar correto")

if str(train_label_src) != str(train_dest_dir / 'plates_label_map.pbtxt'):
    shutil.copy(train_label_src, train_dest_dir / 'plates_label_map.pbtxt')
    print(f"✅ Copiado: {train_label_src.name}")
else:
    print(f"✅ Label map já está no lugar correto")

if str(test_record_src) != str(test_dest_dir / 'plates.tfrecord'):
    shutil.copy(test_record_src, test_dest_dir / 'plates.tfrecord')
    print(f"✅ Copiado: {test_record_src.name}")
else:
    print(f"✅ Test TFRecord já está no lugar correto")

test_record_fname = str(test_dest_dir / 'plates.tfrecord')
train_record_fname = str(train_dest_dir / 'plates.tfrecord')
label_map_pbtxt_fname = str(train_dest_dir / 'plates_label_map.pbtxt')

assert os.path.isfile(train_record_fname), f'Train TFRecord not found: {train_record_fname}'
assert os.path.isfile(test_record_fname), f'Test TFRecord not found: {test_record_fname}'
assert os.path.isfile(label_map_pbtxt_fname), f'Label map not found: {label_map_pbtxt_fname}'
print(f'✅ Train TFRecord: {train_record_fname}')
print(f'✅ Test TFRecord: {test_record_fname}')
print(f'✅ Label map: {label_map_pbtxt_fname}')

MODEL_FILE = MODEL + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'
DEST_DIR = str(RESEARCH_DIR / 'pretrained_model')

if (RESEARCH_DIR / 'pretrained_model' / 'model.ckpt.meta').exists():
    print(f"✅ Modelo pré-treinado já existe em {DEST_DIR}")
else:
    print(f"📥 Baixando modelo pré-treinado {MODEL}...")
    max_retries = 3
    for attempt in range(max_retries):
        try:
            if not (os.path.exists(MODEL_FILE)) or os.path.getsize(MODEL_FILE) < 1000000:
                print(f"  Tentativa {attempt + 1}/{max_retries}...")
                urllib.request.urlretrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
            
            tar = tarfile.open(MODEL_FILE)
            tar.extractall()
            tar.close()
            break
        except (EOFError, tarfile.ReadError, urllib.error.URLError) as e:
            print(f"  ❌ Erro ao baixar/extrair: {e}")
            if os.path.exists(MODEL_FILE):
                os.remove(MODEL_FILE)
            if attempt == max_retries - 1:
                raise ValueError(f"Falha ao baixar modelo após {max_retries} tentativas")
            continue
    
    os.remove(MODEL_FILE)
    if (os.path.exists(DEST_DIR)):
        shutil.rmtree(DEST_DIR)
    
    if os.path.exists(MODEL):
        os.rename(MODEL, DEST_DIR)
        print(f"✅ Modelo extraído para {DEST_DIR}")
    else:
        print(f"❌ Error: {MODEL} not found after extraction")
        print(f"Available files: {os.listdir('.')}")

!echo {DEST_DIR}
!ls -alh {DEST_DIR}

fine_tune_checkpoint = os.path.join(DEST_DIR, "model.ckpt")
fine_tune_checkpoint


pipeline_fname = str(RESEARCH_DIR / 'object_detection' / 'samples' / 'configs' / pipeline_file)

assert os.path.isfile(pipeline_fname), '`{}` not exist'.format(pipeline_fname)

def get_num_classes(pbtxt_fname):
    from object_detection.utils import label_map_util
    label_map = label_map_util.load_labelmap(pbtxt_fname)
    categories = label_map_util.convert_label_map_to_categories(
        label_map, max_num_classes=90, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    return len(category_index.keys())

num_classes = get_num_classes(label_map_pbtxt_fname)
with open(pipeline_fname) as f:
    s = f.read()
with open(pipeline_fname, 'w') as f:

    # fine_tune_checkpoint
    s = re.sub('fine_tune_checkpoint: ".*?"',
               'fine_tune_checkpoint: "{}"'.format(fine_tune_checkpoint), s)

    # tfrecord files train and test.
    s = re.sub(
        '(input_path: ".*?)(train.record)(.*?")', 'input_path: "{}"'.format(train_record_fname), s)
    s = re.sub(
        '(input_path: ".*?)(val.record)(.*?")', 'input_path: "{}"'.format(test_record_fname), s)

    # label_map_path
    s = re.sub(
        'label_map_path: ".*?"', 'label_map_path: "{}"'.format(label_map_pbtxt_fname), s)

    # Set training batch_size.
    s = re.sub('batch_size: [0-9]+',
               'batch_size: {}'.format(batch_size), s)

    # Set training steps, num_steps
    s = re.sub('num_steps: [0-9]+',
               'num_steps: {}'.format(num_steps), s)

    # Set number of classes num_classes.
    s = re.sub('num_classes: [0-9]+',
               'num_classes: {}'.format(num_classes), s)
    f.write(s)

!cat {pipeline_fname}

model_dir = str(RESEARCH_DIR / 'training')
os.makedirs(model_dir, exist_ok=True)

!wget https://bin.equinox.io/c/4VmDzA7iaHb/ngrok-stable-linux-amd64.zip
!unzip -o ngrok-stable-linux-amd64.zip

LOG_DIR = model_dir
get_ipython().system_raw(
    'tensorboard --logdir {} --host 0.0.0.0 --port 6006 &'
    .format(LOG_DIR)
)

get_ipython().system_raw('./ngrok http 6006 &')


!python {RESEARCH_DIR}/object_detection/model_main_tf2.py \
    --pipeline_config_path={pipeline_fname} \
    --model_dir={model_dir} \
    --alsologtostderr \
    --num_train_steps={num_steps} \
    --sample_1_of_n_eval_examples=1

!ls {model_dir}


output_directory = str(RESEARCH_DIR / 'fine_tuned_model')
tflite_directory = str(RESEARCH_DIR / 'fine_tuned_model' / 'tflite')

import tensorflow as tf

latest_checkpoint = tf.train.latest_checkpoint(model_dir)
if latest_checkpoint is None:
    raise ValueError(f'No checkpoints found in {model_dir}. Run the training cell first to generate model checkpoints.')
print(f'Using checkpoint: {latest_checkpoint}')

!python {str(RESEARCH_DIR)}/object_detection/exporter_main_v2.py \
    --input_type=image_tensor \
    --pipeline_config_path={pipeline_fname} \
    --trained_checkpoint_dir={model_dir} \
    --output_directory={output_directory}

!python {str(RESEARCH_DIR)}/object_detection/export_tflite_graph_tf2.py \
    --pipeline_config_path={pipeline_fname} \
    --trained_checkpoint_dir={model_dir} \
    --output_directory={tflite_directory}

!ls {output_directory}


pb_fname = os.path.join(os.path.abspath(output_directory), "frozen_inference_graph.pb")
print(pb_fname)
assert os.path.isfile(pb_fname), '`{}` not exist'.format(pb_fname)

if TEST_IMAGES_SRC.exists():
    shutil.copytree(TEST_IMAGES_SRC, DATA_ROOT.parent / 'test', dirs_exist_ok=True)

PATH_TO_CKPT = pb_fname
PATH_TO_LABELS = label_map_pbtxt_fname
PATH_TO_TEST_IMAGES_DIR =  os.path.join(repo_dir_path, "test")

assert os.path.isfile(pb_fname)
assert os.path.isfile(PATH_TO_LABELS)
TEST_IMAGE_PATHS = glob.glob(os.path.join(PATH_TO_TEST_IMAGES_DIR, "*.*"))
assert len(TEST_IMAGE_PATHS) > 0, 'No image found in `{}`.'.format(PATH_TO_TEST_IMAGES_DIR)
print(TEST_IMAGE_PATHS)

!ls {str(BASE_DIR / 'tensorflow-object-detection-faster-rcnn')}


import six.moves.urllib as urllib
import sys
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

sys.path.append("..")
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')


label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(
    label_map, max_num_classes=num_classes, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)

IMAGE_SIZE = (12, 8)


def run_inference_for_single_image(image, graph):
    with graph.as_default():
        with tf.Session() as sess:
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {
                output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in [
                'num_detections', 'detection_boxes', 'detection_scores',
                'detection_classes', 'detection_masks'
            ]:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                        tensor_name)
            if 'detection_masks' in tensor_dict:
                detection_boxes = tf.squeeze(
                    tensor_dict['detection_boxes'], [0])
                detection_masks = tf.squeeze(
                    tensor_dict['detection_masks'], [0])
                real_num_detection = tf.cast(
                    tensor_dict['num_detections'][0], tf.int32)
                detection_boxes = tf.slice(detection_boxes, [0, 0], [
                                           real_num_detection, -1])
                detection_masks = tf.slice(detection_masks, [0, 0, 0], [
                                           real_num_detection, -1, -1])
                detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                    detection_masks, detection_boxes, image.shape[0], image.shape[1])
                detection_masks_reframed = tf.cast(
                    tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                tensor_dict['detection_masks'] = tf.expand_dims(
                    detection_masks_reframed, 0)
            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

            output_dict = sess.run(tensor_dict,
                                   feed_dict={image_tensor: np.expand_dims(image, 0)})

            output_dict['num_detections'] = int(
                output_dict['num_detections'][0])
            output_dict['detection_classes'] = output_dict[
                'detection_classes'][0].astype(np.uint8)
            output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
            output_dict['detection_scores'] = output_dict['detection_scores'][0]
            if 'detection_masks' in output_dict:
                output_dict['detection_masks'] = output_dict['detection_masks'][0]
    return output_dict

count = 5
for i, image_path in enumerate(TEST_IMAGE_PATHS):
    if i > count:
        break
    image = Image.open(image_path)
    image_np = load_image_into_numpy_array(image)
    image_np_expanded = np.expand_dims(image_np, axis=0)
    output_dict = run_inference_for_single_image(image_np, detection_graph)
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        output_dict['detection_boxes'],
        output_dict['detection_classes'],
        output_dict['detection_scores'],
        category_index,
        instance_masks=output_dict.get('detection_masks'),
        use_normalized_coordinates=True,
        line_thickness=8)
    plt.figure(figsize=IMAGE_SIZE)
    plt.imshow(image_np)

!tflite_convert \
  --input_shape=1,192,192,3 \
  --input_arrays=normalized_input_image_tensor \
  --output_arrays=TFLite_Detection_PostProcess,TFLite_Detection_PostProcess:1,TFLite_Detection_PostProcess:2,TFLite_Detection_PostProcess:3 \
  --allow_custom_ops \
  --graph_def_file={tflite_directory}/tflite_graph.pb \
  --output_file="{str(RESEARCH_DIR)}/fine_tuned_model/final_model.tflite"

if IS_COLAB:
    from google.colab import drive
    drive.mount('/content/drive')
    # Change the final TFLite destination here
    !cp {str(RESEARCH_DIR)}/fine_tuned_model/final_model.tflite "/content/drive/My Drive/"
else:
    print(f"✅ TFLite model saved to: {str(RESEARCH_DIR)}/fine_tuned_model/final_model.tflite")
    print(f"📁 Copy it manually from {str(RESEARCH_DIR)}/fine_tuned_model/ to your desired location.")

import zipfile
from datetime import datetime

print("\n" + "="*60)
print("📦 COMPACTANDO ARTEFATOS DE TREINAMENTO...")
print("="*60)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
zip_filename = f"training_artifacts_{timestamp}.zip"
zip_path = BASE_DIR / zip_filename

artifacts_to_backup = [
    RESEARCH_DIR / 'training',
    RESEARCH_DIR / 'fine_tuned_model',
    DATA_ROOT,
]

with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for artifact_dir in artifacts_to_backup:
        if artifact_dir.exists():
            print(f"  ➕ Adicionando {artifact_dir.name}...")
            for root, dirs, files in os.walk(artifact_dir):
                for file in files:
                    file_path = Path(root) / file
                    arcname = file_path.relative_to(BASE_DIR)
                    zipf.write(file_path, arcname)
        else:
            print(f"  ⚠️  {artifact_dir.name} não encontrado, pulando...")

zip_size_mb = zip_path.stat().st_size / (1024 * 1024)
print(f"\n✅ Arquivo compactado: {zip_filename}")
print(f"📊 Tamanho: {zip_size_mb:.2f} MB")

if IS_COLAB:
    print("\n📥 Iniciando download do Colab...")
    from google.colab import files
    files.download(str(zip_path))
    print("✅ Download concluído!")
else:
    print(f"\n📁 Arquivo salvo em: {zip_path}")
    print("💾 Copie manualmente para seu computador ou nuvem.")

"""Your TFLite file is now in your Drive as "final_model.tflite", ready to use with your project on-device! For specific device tutorials, check out the official TensorFlow Lite [Android Demo](https://github.com/tensorflow/examples/tree/master/lite/examples/object_detection/android), [iOS Demo](https://github.com/tensorflow/examples/tree/master/lite/examples/object_detection/ios), or [Raspberry Pi Demo](https://github.com/tensorflow/examples/tree/master/lite/examples/object_detection/raspberry_pi). [link text](https://)"""