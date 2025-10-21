#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pipeline Completo de Reconhecimento de Placas
Integra Detector + Vision OCR para reconhecimento end-to-end
"""

import os
import sys
import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import json

# ============================================================================
# CONFIGURAÇÃO
# ============================================================================

class PipelineConfig:
    """Configurações do pipeline"""
    
    BASE_DIR = Path(__file__).parent
    
    # Modelos
    DETECTOR_MODEL = BASE_DIR / 'workspace_detector' / 'tflite_model' / 'detector_mobilenet_v2_640_int8.tflite'
    DETECTOR_LABELS = BASE_DIR / 'workspace_detector' / 'tflite_model' / 'labelmap.txt'
    
    OCR_MODEL = BASE_DIR / 'workspace_vision_ocr' / 'tflite_model' / 'vision_ocr_mobilenet_v2_160_int8.tflite'
    OCR_LABELS = BASE_DIR / 'workspace_vision_ocr' / 'tflite_model' / 'labelmap.txt'
    
    # Thresholds
    DETECTOR_CONFIDENCE = 0.5
    OCR_CONFIDENCE = 0.6
    
    # Tamanhos
    DETECTOR_SIZE = 640
    OCR_SIZE = 160


# ============================================================================
# CLASSES
# ============================================================================

# Labels Detector
PLATE_TYPES = [
    'placa carro',
    'placa carro mercosul',
    'placa moto',
    'placa moto mercosul'
]

# Labels Vision OCR (36 classes)
OCR_CHARS = [
    '00', '01', '02', '03', '04', '05', '06', '07', '08', '09',
    'A', 'B', 'C', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
    'M', 'N', 'O', 'Q', 'R', 'S', 'T', 'V', 'W', 'X', 'Y', 'Z', 'i'
]


# ============================================================================
# PIPELINE DE RECONHECIMENTO
# ============================================================================

class PlateRecognitionPipeline:
    """Pipeline completo: Detector + OCR"""
    
    def __init__(self):
        """Inicializa pipeline"""
        print("🚀 Inicializando Pipeline de Reconhecimento de Placas...")
        
        # Verificar modelos
        if not PipelineConfig.DETECTOR_MODEL.exists():
            print(f"❌ Modelo Detector não encontrado: {PipelineConfig.DETECTOR_MODEL}")
            print("   Execute: python train_detector_mobilenet_qat.py")
            sys.exit(1)
        
        if not PipelineConfig.OCR_MODEL.exists():
            print(f"❌ Modelo OCR não encontrado: {PipelineConfig.OCR_MODEL}")
            print("   Execute: python train_vision_ocr_mobilenet_qat.py")
            sys.exit(1)
        
        # Carregar Detector
        print("  📦 Carregando Detector Model...")
        self.detector = tf.lite.Interpreter(model_path=str(PipelineConfig.DETECTOR_MODEL))
        self.detector.allocate_tensors()
        self.detector_input = self.detector.get_input_details()
        self.detector_output = self.detector.get_output_details()
        print(f"     ✓ Detector carregado ({PipelineConfig.DETECTOR_MODEL.stat().st_size / 1024 / 1024:.1f} MB)")
        
        # Carregar OCR
        print("  📦 Carregando Vision OCR Model...")
        self.ocr = tf.lite.Interpreter(model_path=str(PipelineConfig.OCR_MODEL))
        self.ocr.allocate_tensors()
        self.ocr_input = self.ocr.get_input_details()
        self.ocr_output = self.ocr.get_output_details()
        print(f"     ✓ OCR carregado ({PipelineConfig.OCR_MODEL.stat().st_size / 1024 / 1024:.1f} MB)")
        
        print("✅ Pipeline pronto!\n")
    
    def detect_plate(self, image: np.ndarray) -> Optional[Dict]:
        """
        Estágio 1: Detectar placa na imagem
        
        Args:
            image: Imagem BGR (OpenCV)
        
        Returns:
            Dict com informações da placa ou None
        """
        h, w = image.shape[:2]
        
        # Preprocessar
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(image_rgb, (PipelineConfig.DETECTOR_SIZE, PipelineConfig.DETECTOR_SIZE))
        
        if self.detector_input[0]['dtype'] == np.uint8:
            input_data = np.expand_dims(img_resized, axis=0).astype(np.uint8)
        else:
            input_data = np.expand_dims(img_resized, axis=0).astype(np.float32)
            input_data = (input_data - 127.5) / 127.5
        
        # Inferência
        self.detector.set_tensor(self.detector_input[0]['index'], input_data)
        self.detector.invoke()
        
        # Resultados
        boxes = self.detector.get_tensor(self.detector_output[0]['index'])[0]
        classes = self.detector.get_tensor(self.detector_output[1]['index'])[0]
        scores = self.detector.get_tensor(self.detector_output[2]['index'])[0]
        
        # Pegar melhor detecção
        best_idx = np.argmax(scores)
        
        if scores[best_idx] < PipelineConfig.DETECTOR_CONFIDENCE:
            return None
        
        # Converter coordenadas
        ymin, xmin, ymax, xmax = boxes[best_idx]
        xmin, ymin = int(xmin * w), int(ymin * h)
        xmax, ymax = int(xmax * w), int(ymax * h)
        
        # Garantir coordenadas válidas
        xmin, ymin = max(0, xmin), max(0, ymin)
        xmax, ymax = min(w, xmax), min(h, ymax)
        
        # Recortar placa
        plate_crop = image_rgb[ymin:ymax, xmin:xmax]
        
        if plate_crop.size == 0:
            return None
        
        plate_type = PLATE_TYPES[int(classes[best_idx])] if int(classes[best_idx]) < len(PLATE_TYPES) else "Unknown"
        
        return {
            'crop': plate_crop,
            'type': plate_type,
            'confidence': float(scores[best_idx]),
            'bbox': (xmin, ymin, xmax, ymax)
        }
    
    def read_characters(self, plate_image: np.ndarray) -> Optional[Dict]:
        """
        Estágio 2: OCR na placa recortada
        
        Args:
            plate_image: Imagem da placa RGB
        
        Returns:
            Dict com texto e caracteres ou None
        """
        # Preprocessar para 160x160
        img_resized = cv2.resize(plate_image, (PipelineConfig.OCR_SIZE, PipelineConfig.OCR_SIZE))
        
        if self.ocr_input[0]['dtype'] == np.uint8:
            input_data = np.expand_dims(img_resized, axis=0).astype(np.uint8)
        else:
            input_data = np.expand_dims(img_resized, axis=0).astype(np.float32)
            input_data = (input_data - 127.5) / 127.5
        
        # Inferência
        self.ocr.set_tensor(self.ocr_input[0]['index'], input_data)
        self.ocr.invoke()
        
        # Resultados
        boxes = self.ocr.get_tensor(self.ocr_output[0]['index'])[0]
        classes = self.ocr.get_tensor(self.ocr_output[1]['index'])[0]
        scores = self.ocr.get_tensor(self.ocr_output[2]['index'])[0]
        
        # Filtrar caracteres
        chars = []
        for i, score in enumerate(scores):
            if score > PipelineConfig.OCR_CONFIDENCE:
                class_id = int(classes[i])
                if class_id < len(OCR_CHARS):
                    char = OCR_CHARS[class_id]
                    ymin, xmin, ymax, xmax = boxes[i]
                    x_center = (xmin + xmax) / 2
                    
                    chars.append({
                        'char': char,
                        'x': float(x_center),
                        'y': float((ymin + ymax) / 2),
                        'confidence': float(score),
                        'bbox': (float(xmin), float(ymin), float(xmax), float(ymax))
                    })
        
        if not chars:
            return None
        
        # Ordenar da esquerda para direita
        chars.sort(key=lambda c: c['x'])
        
        # Montar texto
        plate_text = ''.join([c['char'] for c in chars])
        
        return {
            'text': plate_text,
            'characters': chars,
            'num_chars': len(chars)
        }
    
    def recognize(self, image_path: str, save_visualization: bool = True) -> Optional[Dict]:
        """
        Pipeline completo de reconhecimento
        
        Args:
            image_path: Caminho da imagem
            save_visualization: Se True, salva imagem com visualização
        
        Returns:
            Dict com resultados completos ou None
        """
        print(f"📸 Processando: {image_path}")
        
        # Carregar imagem
        image = cv2.imread(image_path)
        if image is None:
            print(f"  ❌ Erro ao carregar imagem")
            return None
        
        # Estágio 1: Detectar placa
        print("  1️⃣ Detectando placa...")
        plate_detection = self.detect_plate(image)
        
        if plate_detection is None:
            print("  ❌ Nenhuma placa detectada")
            return None
        
        print(f"  ✓ Placa detectada: {plate_detection['type']}")
        print(f"    Confiança: {plate_detection['confidence']:.2%}")
        print(f"    Bounding box: {plate_detection['bbox']}")
        
        # Estágio 2: OCR
        print("  2️⃣ Lendo caracteres...")
        ocr_result = self.read_characters(plate_detection['crop'])
        
        if ocr_result is None:
            print("  ⚠️  Nenhum caractere detectado")
            plate_text = "???"
            num_chars = 0
        else:
            plate_text = ocr_result['text']
            num_chars = ocr_result['num_chars']
            print(f"  ✓ Placa lida: {plate_text}")
            print(f"    Caracteres: {num_chars}")
        
        # Resultado completo
        result = {
            'plate_text': plate_text,
            'plate_type': plate_detection['type'],
            'detector_confidence': plate_detection['confidence'],
            'bbox': plate_detection['bbox'],
            'num_characters': num_chars,
            'ocr_result': ocr_result
        }
        
        # Visualização
        if save_visualization:
            self._save_visualization(image, result, image_path)
        
        return result
    
    def _save_visualization(self, image: np.ndarray, result: Dict, original_path: str):
        """Salva imagem com visualização"""
        img_vis = image.copy()
        
        # Desenhar bounding box
        xmin, ymin, xmax, ymax = result['bbox']
        cv2.rectangle(img_vis, (xmin, ymin), (xmax, ymax), (0, 255, 0), 3)
        
        # Texto
        label = f"{result['plate_text']} ({result['plate_type']})"
        cv2.putText(img_vis, label, (xmin, ymin - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # Salvar
        output_path = Path(original_path).stem + '_recognized.jpg'
        cv2.imwrite(output_path, img_vis)
        print(f"  💾 Visualização salva: {output_path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Função principal"""
    print("="*80)
    print("🚀 PIPELINE DE RECONHECIMENTO DE PLACAS")
    print("   Detector + Vision OCR")
    print("="*80 + "\n")
    
    # Verificar argumentos
    if len(sys.argv) < 2:
        print("❌ Uso: python inference_pipeline.py <caminho_da_imagem>")
        print("\nExemplo:")
        print("  python inference_pipeline.py test_image.jpg")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    if not Path(image_path).exists():
        print(f"❌ Imagem não encontrada: {image_path}")
        sys.exit(1)
    
    # Criar pipeline
    pipeline = PlateRecognitionPipeline()
    
    # Processar
    result = pipeline.recognize(image_path, save_visualization=True)
    
    # Resultado
    print("\n" + "="*80)
    if result:
        print("✅ RECONHECIMENTO CONCLUÍDO")
        print("="*80)
        print(f"📋 Placa: {result['plate_text']}")
        print(f"🚗 Tipo: {result['plate_type']}")
        print(f"📊 Confiança Detector: {result['detector_confidence']:.2%}")
        print(f"🔤 Caracteres Detectados: {result['num_characters']}")
        
        # Salvar JSON
        json_path = Path(image_path).stem + '_result.json'
        with open(json_path, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"💾 Resultado salvo: {json_path}")
    else:
        print("❌ FALHA NO RECONHECIMENTO")
        print("   Nenhuma placa detectada na imagem")
    
    print("="*80)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrompido pelo usuário")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Erro: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
