#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de Comparação - Detector vs Vision Models
Compara performance, velocidade e acurácia dos dois modelos treinados
"""

import os
import sys
import time
import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path
import json
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ============================================================================
# CONFIGURAÇÃO
# ============================================================================

class ComparisonConfig:
    """Configurações para comparação"""
    
    BASE_DIR = Path(__file__).parent
    
    # Modelos
    DETECTOR_MODEL = BASE_DIR / 'workspace_detector' / 'tflite_model' / 'detector_mobilenet_v2_640_int8.tflite'
    DETECTOR_LABELS = BASE_DIR / 'workspace_detector' / 'tflite_model' / 'labelmap.txt'
    
    VISION_MODEL = BASE_DIR / 'workspace_vision' / 'tflite_model' / 'vision_mobilenet_v2_640_int8.tflite'
    VISION_LABELS = BASE_DIR / 'workspace_vision' / 'tflite_model' / 'labelmap.txt'
    
    # Test images
    TEST_DIR = BASE_DIR / 'test_images'
    
    # Output
    OUTPUT_DIR = BASE_DIR / 'comparison_results'
    
    # Thresholds
    CONFIDENCE_THRESHOLD = 0.5
    IOU_THRESHOLD = 0.5
    
    # Number of iterations for speed test
    SPEED_TEST_ITERATIONS = 100


# ============================================================================
# UTILITÁRIOS
# ============================================================================

def load_labels(label_path: Path) -> List[str]:
    """Carrega labels do arquivo"""
    with open(label_path, 'r') as f:
        return [line.strip() for line in f.readlines()]


def load_tflite_model(model_path: Path):
    """Carrega modelo TFLite"""
    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    return interpreter


def preprocess_image(image: np.ndarray, target_size: Tuple[int, int], dtype=np.uint8) -> np.ndarray:
    """Preprocessa imagem para inferência"""
    img_resized = cv2.resize(image, target_size)
    input_data = np.expand_dims(img_resized, axis=0)
    
    if dtype == np.uint8:
        return input_data.astype(np.uint8)
    else:
        input_data = input_data.astype(np.float32)
        return (input_data - 127.5) / 127.5


def run_inference(interpreter, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Executa inferência"""
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Input
    input_shape = input_details[0]['shape'][1:3]
    input_dtype = input_details[0]['dtype']
    
    # Preprocess
    input_data = preprocess_image(image, tuple(input_shape), input_dtype)
    
    # Inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    
    # Output
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]
    classes = interpreter.get_tensor(output_details[1]['index'])[0]
    scores = interpreter.get_tensor(output_details[2]['index'])[0]
    
    return boxes, classes, scores


def filter_detections(boxes, classes, scores, threshold=0.5):
    """Filtra detecções por threshold de confiança"""
    valid_indices = scores > threshold
    return boxes[valid_indices], classes[valid_indices], scores[valid_indices]


# ============================================================================
# COMPARAÇÃO DE VELOCIDADE
# ============================================================================

def benchmark_speed(model_path: Path, test_image: np.ndarray, iterations: int = 100) -> Dict:
    """Benchmarka velocidade de inferência"""
    print(f"\n⏱️  Benchmarking: {model_path.name}")
    
    interpreter = load_tflite_model(model_path)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    input_shape = tuple(input_details[0]['shape'][1:3])
    input_dtype = input_details[0]['dtype']
    
    # Warmup
    for _ in range(10):
        input_data = preprocess_image(test_image, input_shape, input_dtype)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
    
    # Benchmark
    times = []
    for i in range(iterations):
        input_data = preprocess_image(test_image, input_shape, input_dtype)
        
        start = time.perf_counter()
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        end = time.perf_counter()
        
        times.append((end - start) * 1000)  # Convert to ms
    
    results = {
        'mean_ms': np.mean(times),
        'std_ms': np.std(times),
        'min_ms': np.min(times),
        'max_ms': np.max(times),
        'median_ms': np.median(times),
        'fps': 1000 / np.mean(times)
    }
    
    print(f"  Média: {results['mean_ms']:.2f} ms ({results['fps']:.1f} FPS)")
    print(f"  Min/Max: {results['min_ms']:.2f} / {results['max_ms']:.2f} ms")
    
    return results


# ============================================================================
# COMPARAÇÃO DE DETECÇÕES
# ============================================================================

def compare_detections(image_path: Path) -> Dict:
    """Compara detecções dos dois modelos em uma imagem"""
    print(f"\n🔍 Comparando detecções: {image_path.name}")
    
    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"  ❌ Erro ao carregar imagem")
        return None
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Load models
    detector = load_tflite_model(ComparisonConfig.DETECTOR_MODEL)
    vision = load_tflite_model(ComparisonConfig.VISION_MODEL)
    
    detector_labels = load_labels(ComparisonConfig.DETECTOR_LABELS)
    vision_labels = load_labels(ComparisonConfig.VISION_LABELS)
    
    # Run inference
    det_boxes, det_classes, det_scores = run_inference(detector, image_rgb)
    vis_boxes, vis_classes, vis_scores = run_inference(vision, image_rgb)
    
    # Filter
    det_boxes, det_classes, det_scores = filter_detections(
        det_boxes, det_classes, det_scores, ComparisonConfig.CONFIDENCE_THRESHOLD
    )
    vis_boxes, vis_classes, vis_scores = filter_detections(
        vis_boxes, vis_classes, vis_scores, ComparisonConfig.CONFIDENCE_THRESHOLD
    )
    
    print(f"  Detector: {len(det_scores)} detecções")
    print(f"  Vision: {len(vis_scores)} detecções")
    
    return {
        'image_path': image_path,
        'image': image_rgb,
        'detector': {
            'boxes': det_boxes,
            'classes': det_classes,
            'scores': det_scores,
            'labels': detector_labels
        },
        'vision': {
            'boxes': vis_boxes,
            'classes': vis_classes,
            'scores': vis_scores,
            'labels': vision_labels
        }
    }


# ============================================================================
# VISUALIZAÇÃO
# ============================================================================

def visualize_comparison(comparison_data: Dict, save_path: Path):
    """Visualiza comparação lado a lado"""
    image = comparison_data['image']
    h, w = image.shape[:2]
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Detector
    ax1 = axes[0]
    ax1.imshow(image)
    ax1.set_title(f"Detector Model ({len(comparison_data['detector']['scores'])} detections)", 
                  fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    for box, cls, score in zip(comparison_data['detector']['boxes'],
                                comparison_data['detector']['classes'],
                                comparison_data['detector']['scores']):
        ymin, xmin, ymax, xmax = box
        xmin, ymin, xmax, ymax = int(xmin * w), int(ymin * h), int(xmax * w), int(ymax * h)
        
        rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                 linewidth=2, edgecolor='lime', facecolor='none')
        ax1.add_patch(rect)
        
        label = f"{comparison_data['detector']['labels'][int(cls)]}: {score:.2f}"
        ax1.text(xmin, ymin - 5, label, color='lime', fontsize=10,
                bbox=dict(facecolor='black', alpha=0.7, pad=2))
    
    # Vision
    ax2 = axes[1]
    ax2.imshow(image)
    ax2.set_title(f"Vision Model ({len(comparison_data['vision']['scores'])} detections)", 
                  fontsize=14, fontweight='bold')
    ax2.axis('off')
    
    for box, cls, score in zip(comparison_data['vision']['boxes'],
                                comparison_data['vision']['classes'],
                                comparison_data['vision']['scores']):
        ymin, xmin, ymax, xmax = box
        xmin, ymin, xmax, ymax = int(xmin * w), int(ymin * h), int(xmax * w), int(ymax * h)
        
        rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                 linewidth=2, edgecolor='cyan', facecolor='none')
        ax2.add_patch(rect)
        
        label = f"{comparison_data['vision']['labels'][int(cls)]}: {score:.2f}"
        ax2.text(xmin, ymin - 5, label, color='cyan', fontsize=10,
                bbox=dict(facecolor='black', alpha=0.7, pad=2))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Salvo: {save_path}")


# ============================================================================
# RELATÓRIO
# ============================================================================

def generate_report(speed_results: Dict, detection_results: List[Dict], output_path: Path):
    """Gera relatório de comparação"""
    print(f"\n📊 Gerando relatório...")
    
    report = {
        'speed_comparison': speed_results,
        'detection_summary': {
            'total_images': len(detection_results),
            'detector': {
                'total_detections': sum(len(r['detector']['scores']) for r in detection_results),
                'avg_detections_per_image': np.mean([len(r['detector']['scores']) for r in detection_results]),
                'avg_confidence': np.mean([np.mean(r['detector']['scores']) if len(r['detector']['scores']) > 0 else 0 
                                          for r in detection_results])
            },
            'vision': {
                'total_detections': sum(len(r['vision']['scores']) for r in detection_results),
                'avg_detections_per_image': np.mean([len(r['vision']['scores']) for r in detection_results]),
                'avg_confidence': np.mean([np.mean(r['vision']['scores']) if len(r['vision']['scores']) > 0 else 0 
                                          for r in detection_results])
            }
        }
    }
    
    # Save JSON
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"  ✓ Relatório salvo: {output_path}")
    
    # Print summary
    print("\n" + "="*80)
    print("📈 RESUMO DA COMPARAÇÃO")
    print("="*80)
    
    print("\n⏱️  Velocidade de Inferência:")
    print(f"  Detector: {speed_results['detector']['mean_ms']:.2f} ms ({speed_results['detector']['fps']:.1f} FPS)")
    print(f"  Vision:   {speed_results['vision']['mean_ms']:.2f} ms ({speed_results['vision']['fps']:.1f} FPS)")
    
    faster = 'Detector' if speed_results['detector']['mean_ms'] < speed_results['vision']['mean_ms'] else 'Vision'
    speedup = max(speed_results['detector']['mean_ms'], speed_results['vision']['mean_ms']) / \
              min(speed_results['detector']['mean_ms'], speed_results['vision']['mean_ms'])
    print(f"  🏆 Mais rápido: {faster} ({speedup:.2f}x)")
    
    print("\n🎯 Detecções:")
    print(f"  Detector: {report['detection_summary']['detector']['total_detections']} detecções " +
          f"({report['detection_summary']['detector']['avg_detections_per_image']:.1f} por imagem)")
    print(f"  Vision:   {report['detection_summary']['vision']['total_detections']} detecções " +
          f"({report['detection_summary']['vision']['avg_detections_per_image']:.1f} por imagem)")
    
    print("\n🎲 Confiança Média:")
    print(f"  Detector: {report['detection_summary']['detector']['avg_confidence']:.3f}")
    print(f"  Vision:   {report['detection_summary']['vision']['avg_confidence']:.3f}")
    
    print("\n💡 Recomendações:")
    if speed_results['detector']['fps'] > speed_results['vision']['fps']:
        print("  • Detector é mais rápido - ideal para real-time")
    else:
        print("  • Vision é mais rápido - ideal para real-time")
    
    if report['detection_summary']['detector']['total_detections'] > report['detection_summary']['vision']['total_detections']:
        print("  • Detector detecta mais objetos - melhor recall")
    else:
        print("  • Vision detecta mais objetos - melhor recall")
    
    print("="*80)


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Função principal"""
    print("="*80)
    print("🔬 COMPARAÇÃO: DETECTOR vs VISION")
    print("   MobileNet V2 SSD Lite Models")
    print("="*80)
    
    # Verificar modelos
    if not ComparisonConfig.DETECTOR_MODEL.exists():
        print(f"❌ Modelo Detector não encontrado: {ComparisonConfig.DETECTOR_MODEL}")
        print("   Execute train_detector_mobilenet_qat.py primeiro")
        sys.exit(1)
    
    if not ComparisonConfig.VISION_MODEL.exists():
        print(f"❌ Modelo Vision não encontrado: {ComparisonConfig.VISION_MODEL}")
        print("   Execute train_vision_mobilenet_qat.py primeiro")
        sys.exit(1)
    
    # Criar output dir
    ComparisonConfig.OUTPUT_DIR.mkdir(exist_ok=True)
    
    # 1. Speed benchmark
    print("\n" + "="*80)
    print("1️⃣  BENCHMARK DE VELOCIDADE")
    print("="*80)
    
    # Criar imagem de teste dummy
    test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    
    speed_results = {
        'detector': benchmark_speed(ComparisonConfig.DETECTOR_MODEL, test_image, 
                                   ComparisonConfig.SPEED_TEST_ITERATIONS),
        'vision': benchmark_speed(ComparisonConfig.VISION_MODEL, test_image,
                                 ComparisonConfig.SPEED_TEST_ITERATIONS)
    }
    
    # 2. Detection comparison
    print("\n" + "="*80)
    print("2️⃣  COMPARAÇÃO DE DETECÇÕES")
    print("="*80)
    
    # Procurar imagens de teste
    if not ComparisonConfig.TEST_DIR.exists():
        print(f"\n⚠️  Diretório de teste não existe: {ComparisonConfig.TEST_DIR}")
        print("   Criando diretório... Adicione imagens de teste lá.")
        ComparisonConfig.TEST_DIR.mkdir(exist_ok=True)
        test_images = []
    else:
        test_images = list(ComparisonConfig.TEST_DIR.glob('*.jpg')) + \
                     list(ComparisonConfig.TEST_DIR.glob('*.png'))
    
    if not test_images:
        print("\n⚠️  Nenhuma imagem de teste encontrada")
        print(f"   Adicione imagens em: {ComparisonConfig.TEST_DIR}")
        detection_results = []
    else:
        print(f"\nEncontradas {len(test_images)} imagens de teste")
        
        detection_results = []
        for img_path in test_images:
            result = compare_detections(img_path)
            if result:
                detection_results.append(result)
                
                # Visualizar
                vis_path = ComparisonConfig.OUTPUT_DIR / f"comparison_{img_path.stem}.png"
                visualize_comparison(result, vis_path)
    
    # 3. Generate report
    print("\n" + "="*80)
    print("3️⃣  GERANDO RELATÓRIO")
    print("="*80)
    
    report_path = ComparisonConfig.OUTPUT_DIR / 'comparison_report.json'
    generate_report(speed_results, detection_results, report_path)
    
    print("\n✅ Comparação concluída!")
    print(f"📁 Resultados salvos em: {ComparisonConfig.OUTPUT_DIR}")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Comparação interrompida")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Erro: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
