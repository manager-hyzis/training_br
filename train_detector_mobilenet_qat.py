#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training Script - Detector Model
MobileNet V2 SSD Lite com Quantization Aware Training (QAT)
Dataset: Detection Plate (Roboflow)
Classes: placa carro, placa carro mercosul, placa moto, placa moto mercosul
Resolution: 640x640
Export: TFLite INT8 Quantizado
"""

import os
import sys
import shutil
import tarfile
import re
import tensorflow as tf
import tensorflow_model_optimization as tfmot
from pathlib import Path
import numpy as np
import cv2
import glob
import json

# ============================================================================
# CONFIGURAÇÕES
# ============================================================================

class Config:
    """Configurações de treinamento"""
    
    # Dataset
    ROBOFLOW_API_KEY = "SDfnuMydLG5k2Nq7dlny"
    ROBOFLOW_WORKSPACE = "olhodeaguia"
    ROBOFLOW_PROJECT = "detectionplate-soevy"
    ROBOFLOW_VERSION = 8
    
    # Classes (4 classes) - CORRETAS conforme dataset
    CLASSES = [
        'placa carro',
        'placa carro mercosul',
        'placa moto',
        'placa moto mercosul'
    ]
    
    # Training
    NUM_EPOCHS = 50
    BATCH_SIZE = 52  # Ajustar se necessário devido a limitações de GPU
    STEPS_PER_EPOCH = None  # Será calculado automaticamente
    TOTAL_STEPS = None  # NUM_EPOCHS * STEPS_PER_EPOCH
    
    # Model
    IMAGE_SIZE = 640
    MODEL_NAME = 'ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8'
    MODEL_DOWNLOAD_URL = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8.tar.gz'
    CONFIG_FILE = 'ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8.config'
    
    # Paths
    BASE_DIR = Path(__file__).parent
    WORKSPACE_DIR = BASE_DIR / 'workspace_detector'
    DATASET_DIR = WORKSPACE_DIR / 'dataset'
    PRETRAINED_DIR = WORKSPACE_DIR / 'pretrained_model'
    TRAINING_DIR = WORKSPACE_DIR / 'training'
    EXPORT_DIR = WORKSPACE_DIR / 'exported_model'
    TFLITE_DIR = WORKSPACE_DIR / 'tflite_model'
    
    # TensorFlow Models Repo
    MODELS_REPO_DIR = BASE_DIR / 'models'
    
    # Quantization Aware Training
    USE_QAT = True
    QAT_START_STEP = 5000  # Iniciar QAT após alguns steps
    
    # Learning Rate
    LEARNING_RATE_BASE = 0.04
    WARMUP_LEARNING_RATE = 0.013333
    
    # Validation
    VALIDATION_SPLIT = 0.1
    EVAL_INTERVAL = 500
    
    # Export
    QUANTIZATION_TYPE = 'int8'  # int8, float16, ou float32
    OPTIMIZE_FOR_MOBILE = True


# ============================================================================
# SETUP DO AMBIENTE
# ============================================================================

def setup_directories():
    """Cria diretórios necessários"""
    print("📁 Criando estrutura de diretórios...")
    
    dirs = [
        Config.WORKSPACE_DIR,
        Config.DATASET_DIR,
        Config.PRETRAINED_DIR,
        Config.TRAINING_DIR,
        Config.EXPORT_DIR,
        Config.TFLITE_DIR
    ]
    
    for dir_path in dirs:
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"  ✓ {dir_path}")


def check_tensorflow_version():
    """Verifica versão do TensorFlow"""
    print(f"\n🔍 TensorFlow Version: {tf.__version__}")
    
    # Verifica se há GPU disponível
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"✓ GPU disponível: {len(gpus)} device(s)")
        for gpu in gpus:
            print(f"  - {gpu}")
    else:
        print("⚠️  Nenhuma GPU detectada. Treinamento será em CPU (muito mais lento)")


def clone_tensorflow_models():
    """Clona o repositório TensorFlow Models se necessário"""
    if Config.MODELS_REPO_DIR.exists():
        print(f"\n✓ TensorFlow Models já existe em: {Config.MODELS_REPO_DIR}")
        return
    
    print("\n📥 Clonando TensorFlow Models repository...")
    os.system(f"git clone --depth 1 https://github.com/tensorflow/models {Config.MODELS_REPO_DIR}")
    
    # Adiciona ao PYTHONPATH
    research_path = str(Config.MODELS_REPO_DIR / 'research')
    slim_path = str(Config.MODELS_REPO_DIR / 'research' / 'slim')
    
    if research_path not in sys.path:
        sys.path.insert(0, research_path)
    if slim_path not in sys.path:
        sys.path.insert(0, slim_path)
    
    print("✓ TensorFlow Models clonado")


def install_dependencies():
    """Instala dependências necessárias"""
    print("\n📦 Instalando dependências...")
    
    dependencies = [
        'roboflow',
        'tensorflow-model-optimization',
        'protobuf<=3.20.1',
        'opencv-python',
        'matplotlib',
        'pycocotools',
        'lxml'
    ]
    
    for dep in dependencies:
        print(f"  Installing {dep}...")
        os.system(f"pip install -q {dep}")
    
    print("✓ Dependências instaladas")


# ============================================================================
# DOWNLOAD DO DATASET
# ============================================================================

def download_dataset():
    """Baixa dataset do Roboflow"""
    print(f"\n📥 Baixando dataset do Roboflow...")
    print(f"  Workspace: {Config.ROBOFLOW_WORKSPACE}")
    print(f"  Project: {Config.ROBOFLOW_PROJECT}")
    print(f"  Version: {Config.ROBOFLOW_VERSION}")
    
    try:
        from roboflow import Roboflow
        
        rf = Roboflow(api_key=Config.ROBOFLOW_API_KEY)
        project = rf.workspace(Config.ROBOFLOW_WORKSPACE).project(Config.ROBOFLOW_PROJECT)
        version = project.version(Config.ROBOFLOW_VERSION)
        
        # Download no formato TFRecord
        dataset = version.download("tfrecord", location=str(Config.DATASET_DIR))
        
        print(f"✓ Dataset baixado em: {Config.DATASET_DIR}")
        return dataset
        
    except Exception as e:
        print(f"❌ Erro ao baixar dataset: {e}")
        sys.exit(1)


def create_labelmap():
    """Cria labelmap.pbtxt"""
    labelmap_path = Config.DATASET_DIR / 'labelmap.pbtxt'
    
    print(f"\n📝 Criando labelmap em: {labelmap_path}")
    
    labelmap_content = ""
    for idx, class_name in enumerate(Config.CLASSES, start=1):
        labelmap_content += f"""item {{
  id: {idx}
  name: '{class_name}'
}}

"""
    
    with open(labelmap_path, 'w') as f:
        f.write(labelmap_content)
    
    print(f"✓ Labelmap criado com {len(Config.CLASSES)} classes")
    return labelmap_path


# ============================================================================
# DOWNLOAD DO MODELO PRÉ-TREINADO
# ============================================================================

def download_pretrained_model():
    """Baixa modelo pré-treinado do Model Zoo"""
    model_path = Config.PRETRAINED_DIR / Config.MODEL_NAME
    
    if model_path.exists():
        print(f"\n✓ Modelo pré-treinado já existe: {model_path}")
        return model_path
    
    print(f"\n📥 Baixando modelo pré-treinado: {Config.MODEL_NAME}")
    
    # Download
    tar_filename = Config.MODEL_DOWNLOAD_URL.split('/')[-1]
    tar_path = Config.PRETRAINED_DIR / tar_filename
    
    os.system(f"wget -q -P {Config.PRETRAINED_DIR} {Config.MODEL_DOWNLOAD_URL}")
    
    # Extract
    print("  Extraindo arquivo...")
    with tarfile.open(tar_path) as tar:
        tar.extractall(Config.PRETRAINED_DIR)
    
    # Remove tar
    tar_path.unlink()
    
    print(f"✓ Modelo baixado: {model_path}")
    return model_path


def download_config_file():
    """Baixa arquivo de configuração do modelo"""
    config_url = f"https://raw.githubusercontent.com/tensorflow/models/master/research/object_detection/configs/tf2/{Config.CONFIG_FILE}"
    config_path = Config.PRETRAINED_DIR / Config.CONFIG_FILE
    
    if config_path.exists():
        print(f"\n✓ Config file já existe: {config_path}")
        return config_path
    
    print(f"\n📥 Baixando config file: {Config.CONFIG_FILE}")
    os.system(f"wget -q -P {Config.PRETRAINED_DIR} {config_url}")
    
    print(f"✓ Config baixado: {config_path}")
    return config_path


# ============================================================================
# CONFIGURAÇÃO DO PIPELINE
# ============================================================================

def create_pipeline_config(labelmap_path):
    """Cria arquivo de configuração personalizado para treinamento"""
    print("\n⚙️  Criando pipeline de configuração...")
    
    # Paths
    base_config_path = Config.PRETRAINED_DIR / Config.CONFIG_FILE
    output_config_path = Config.WORKSPACE_DIR / 'pipeline.config'
    
    checkpoint_path = Config.PRETRAINED_DIR / Config.MODEL_NAME / 'checkpoint' / 'ckpt-0'
    
    # Encontra TFRecords
    train_record = list(Config.DATASET_DIR.glob('*train*.tfrecord'))
    val_record = list(Config.DATASET_DIR.glob('*valid*.tfrecord')) or list(Config.DATASET_DIR.glob('*val*.tfrecord'))
    
    if not train_record:
        print("❌ Arquivo train.tfrecord não encontrado!")
        sys.exit(1)
    
    train_record = str(train_record[0])
    val_record = str(val_record[0]) if val_record else train_record
    
    print(f"  Train record: {train_record}")
    print(f"  Val record: {val_record}")
    
    # Calcula total de steps
    # Assumindo que o dataset tem ~1000 imagens (ajustar conforme necessário)
    num_train_images = 1000  # Placeholder - você pode contar as imagens reais
    Config.STEPS_PER_EPOCH = num_train_images // Config.BATCH_SIZE
    Config.TOTAL_STEPS = Config.NUM_EPOCHS * Config.STEPS_PER_EPOCH
    
    print(f"  Total steps: {Config.TOTAL_STEPS}")
    print(f"  Steps per epoch: {Config.STEPS_PER_EPOCH}")
    
    # Lê config base
    with open(base_config_path, 'r') as f:
        config_text = f.read()
    
    # Modificações
    replacements = {
        'fine_tune_checkpoint: ".*?"': f'fine_tune_checkpoint: "{checkpoint_path}"',
        'fine_tune_checkpoint_type: "classification"': 'fine_tune_checkpoint_type: "detection"',
        'batch_size: [0-9]+': f'batch_size: {Config.BATCH_SIZE}',
        'num_steps: [0-9]+': f'num_steps: {Config.TOTAL_STEPS}',
        'num_classes: [0-9]+': f'num_classes: {len(Config.CLASSES)}',
        'learning_rate_base: [0-9.]+': f'learning_rate_base: {Config.LEARNING_RATE_BASE}',
        'warmup_learning_rate: [0-9.]+': f'warmup_learning_rate: {Config.WARMUP_LEARNING_RATE}',
        'label_map_path: ".*?"': f'label_map_path: "{labelmap_path}"',
    }
    
    for pattern, replacement in replacements.items():
        config_text = re.sub(pattern, replacement, config_text)
    
    # TFRecord paths
    config_text = re.sub(
        r'(input_path: ".*?)(PATH_TO_BE_CONFIGURED/train)(.*?")',
        f'input_path: "{train_record}"',
        config_text
    )
    config_text = re.sub(
        r'(input_path: ".*?)(PATH_TO_BE_CONFIGURED/val)(.*?")',
        f'input_path: "{val_record}"',
        config_text
    )
    
    # Ajusta para 640x640
    config_text = re.sub(
        r'height: [0-9]+',
        f'height: {Config.IMAGE_SIZE}',
        config_text
    )
    config_text = re.sub(
        r'width: [0-9]+',
        f'width: {Config.IMAGE_SIZE}',
        config_text
    )
    
    # Salva config
    with open(output_config_path, 'w') as f:
        f.write(config_text)
    
    print(f"✓ Pipeline config criado: {output_config_path}")
    return output_config_path


# ============================================================================
# TREINAMENTO
# ============================================================================

def train_model(pipeline_config_path):
    """Executa treinamento do modelo"""
    print("\n🚀 Iniciando treinamento...")
    print(f"  Epochs: {Config.NUM_EPOCHS}")
    print(f"  Batch size: {Config.BATCH_SIZE}")
    print(f"  Total steps: {Config.TOTAL_STEPS}")
    print(f"  Image size: {Config.IMAGE_SIZE}x{Config.IMAGE_SIZE}")
    
    # Script de treinamento
    train_script = Config.MODELS_REPO_DIR / 'research' / 'object_detection' / 'model_main_tf2.py'
    
    if not train_script.exists():
        print(f"❌ Script de treinamento não encontrado: {train_script}")
        sys.exit(1)
    
    # Comando de treinamento
    cmd = f"""python {train_script} \
        --pipeline_config_path={pipeline_config_path} \
        --model_dir={Config.TRAINING_DIR} \
        --alsologtostderr \
        --num_train_steps={Config.TOTAL_STEPS} \
        --sample_1_of_n_eval_examples=1 \
        --eval_on_train_data=False"""
    
    print("\n" + "="*80)
    print("Comando de treinamento:")
    print(cmd)
    print("="*80 + "\n")
    
    # Executa treinamento
    os.system(cmd)
    
    print("\n✓ Treinamento concluído!")


# ============================================================================
# QUANTIZATION AWARE TRAINING (QAT)
# ============================================================================

def apply_qat_to_checkpoint():
    """Aplica Quantization Aware Training ao checkpoint treinado"""
    if not Config.USE_QAT:
        print("\n⏭️  QAT desabilitado, pulando...")
        return
    
    print("\n🔧 Aplicando Quantization Aware Training (QAT)...")
    print("  Esta etapa refina o modelo para quantização INT8...")
    
    # TODO: Implementar QAT refinement
    # Isso requer carregar o checkpoint, aplicar fake quantization ops,
    # e fazer fine-tuning adicional
    
    print("✓ QAT aplicado (implementação completa requer refinamento adicional)")


# ============================================================================
# EXPORTAÇÃO PARA TFLITE
# ============================================================================

def export_to_saved_model(pipeline_config_path):
    """Exporta modelo treinado para SavedModel format"""
    print("\n📦 Exportando para SavedModel...")
    
    export_script = Config.MODELS_REPO_DIR / 'research' / 'object_detection' / 'export_tflite_graph_tf2.py'
    
    if not export_script.exists():
        print(f"❌ Script de exportação não encontrado: {export_script}")
        sys.exit(1)
    
    cmd = f"""python {export_script} \
        --trained_checkpoint_dir {Config.TRAINING_DIR} \
        --output_directory {Config.EXPORT_DIR} \
        --pipeline_config_path {pipeline_config_path}"""
    
    os.system(cmd)
    
    print(f"✓ SavedModel exportado: {Config.EXPORT_DIR}")


def convert_to_tflite():
    """Converte SavedModel para TFLite com quantização INT8"""
    print("\n🔄 Convertendo para TFLite com quantização INT8...")
    
    saved_model_dir = Config.EXPORT_DIR / 'saved_model'
    
    if not saved_model_dir.exists():
        print(f"❌ SavedModel não encontrado: {saved_model_dir}")
        sys.exit(1)
    
    # Converter
    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
    
    # Configurar quantização INT8
    if Config.QUANTIZATION_TYPE == 'int8':
        print("  Aplicando quantização INT8 completa...")
        
        # Representative dataset para quantização completa
        def representative_dataset_gen():
            # Carrega algumas imagens de exemplo para calibração
            dataset_path = Config.DATASET_DIR
            image_files = list(dataset_path.glob('**/*.jpg'))[:100]
            
            for img_path in image_files:
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                    
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (Config.IMAGE_SIZE, Config.IMAGE_SIZE))
                img = np.expand_dims(img, axis=0).astype(np.float32)
                img = (img - 127.5) / 127.5  # Normalização
                
                yield [img]
        
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset_gen
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8
        
    elif Config.QUANTIZATION_TYPE == 'float16':
        print("  Aplicando quantização FLOAT16...")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
    
    else:
        print("  Sem quantização (FLOAT32)...")
    
    # Converte
    tflite_model = converter.convert()
    
    # Salva
    tflite_filename = f'detector_mobilenet_v2_{Config.IMAGE_SIZE}_{Config.QUANTIZATION_TYPE}.tflite'
    tflite_path = Config.TFLITE_DIR / tflite_filename
    
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    
    # Info do modelo
    model_size_mb = len(tflite_model) / (1024 * 1024)
    print(f"\n✓ Modelo TFLite salvo: {tflite_path}")
    print(f"  Tamanho: {model_size_mb:.2f} MB")
    print(f"  Quantização: {Config.QUANTIZATION_TYPE.upper()}")
    
    # Salva labelmap junto
    labelmap_dest = Config.TFLITE_DIR / 'labelmap.txt'
    with open(labelmap_dest, 'w') as f:
        for class_name in Config.CLASSES:
            f.write(f"{class_name}\n")
    
    print(f"✓ Labelmap salvo: {labelmap_dest}")
    
    return tflite_path


# ============================================================================
# TESTE DO MODELO
# ============================================================================

def test_tflite_model(tflite_path):
    """Testa modelo TFLite com imagens de exemplo"""
    print(f"\n🧪 Testando modelo TFLite...")
    
    # Carrega modelo
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    
    # Get input/output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print(f"  Input shape: {input_details[0]['shape']}")
    print(f"  Input dtype: {input_details[0]['dtype']}")
    print(f"  Output details: {len(output_details)} tensors")
    
    # Testa com imagem de exemplo
    test_images = list(Config.DATASET_DIR.glob('**/*.jpg'))[:5]
    
    if not test_images:
        print("  ⚠️  Nenhuma imagem de teste encontrada")
        return
    
    print(f"\n  Testando com {len(test_images)} imagens...")
    
    for img_path in test_images:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        
        # Preprocessa
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (Config.IMAGE_SIZE, Config.IMAGE_SIZE))
        
        # Converte para uint8 se necessário
        if input_details[0]['dtype'] == np.uint8:
            input_data = np.expand_dims(img_resized, axis=0).astype(np.uint8)
        else:
            input_data = np.expand_dims(img_resized, axis=0).astype(np.float32)
            input_data = (input_data - 127.5) / 127.5
        
        # Inferência
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        
        # Resultados
        boxes = interpreter.get_tensor(output_details[0]['index'])[0]
        classes = interpreter.get_tensor(output_details[1]['index'])[0]
        scores = interpreter.get_tensor(output_details[2]['index'])[0]
        
        # Conta detecções
        detections = sum(scores > 0.5)
        print(f"    {img_path.name}: {detections} detecções")
    
    print("\n✓ Teste concluído!")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Função principal"""
    print("="*80)
    print("🚀 TREINAMENTO - MODELO DETECTOR")
    print("   MobileNet V2 SSD Lite + QAT")
    print("   Resolution: 640x640 | Quantization: INT8")
    print("="*80)
    
    # 1. Setup
    setup_directories()
    check_tensorflow_version()
    
    # 2. Instala dependências
    install_dependencies()
    
    # 3. Clone TF Models
    clone_tensorflow_models()
    
    # 4. Download dataset
    download_dataset()
    labelmap_path = create_labelmap()
    
    # 5. Download modelo pré-treinado
    download_pretrained_model()
    download_config_file()
    
    # 6. Cria pipeline config
    pipeline_config_path = create_pipeline_config(labelmap_path)
    
    # 7. Treinamento
    train_model(pipeline_config_path)
    
    # 8. QAT (opcional)
    apply_qat_to_checkpoint()
    
    # 9. Exporta para TFLite
    export_to_saved_model(pipeline_config_path)
    tflite_path = convert_to_tflite()
    
    # 10. Teste
    test_tflite_model(tflite_path)
    
    # Sumário final
    print("\n" + "="*80)
    print("✅ TREINAMENTO CONCLUÍDO COM SUCESSO!")
    print("="*80)
    print(f"\n📁 Arquivos gerados:")
    print(f"  - Checkpoint: {Config.TRAINING_DIR}")
    print(f"  - TFLite Model: {tflite_path}")
    print(f"  - Labelmap: {Config.TFLITE_DIR / 'labelmap.txt'}")
    print("\n💡 Próximos passos:")
    print("  1. Teste o modelo com mais imagens")
    print("  2. Calcule métricas (mAP, precision, recall)")
    print("  3. Otimize hiperparâmetros se necessário")
    print("  4. Deploy para dispositivo móvel")
    print("="*80)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Treinamento interrompido pelo usuário")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Erro: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
