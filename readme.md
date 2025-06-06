### Conforme conversado com Professor Raphael fui autorizado a usar o OpenCV no meu projeto ao invés do mediapipe.

Enricco Rossi de Souza Carvalho Miranda - RM551717
Gabriel Marquez Trevisan - RM99227
Samuel Ramos de Almeida - RM99134

#  AlertaEnergia — Detecção de Colisões entre Veículos 

Sistema simples de rastreamento de objetos usando centróides, com foco na **detecção de colisões entre carros** em vídeos.

Este projeto utiliza caixas delimitadoras (bounding boxes) para acompanhar objetos ao longo de quadros e identificar colisões com base em interseções.

## 📌 Funcionalidades

- Rastreamento de múltiplos veículos usando centróides
- Detecção automática de colisões por sobreposição de caixas
- Atribuição única de ID para cada veículo
- Reset automático de IDs após certo número de quadros ausentes

## 🛠 Requisitos

Instale as dependências com:

```bash
pip install -r requirements.txt

### 🔽 Baixar pesos do YOLOv4-tiny

Para utilizar a detecção de veículos, baixe os arquivos de pesos e configuração do **YOLOv4-tiny**:

- 🔗 [yolov4-tiny.weights](https://pjreddie.com/media/files/yolov4-tiny.weights)
- 🔗 [yolov4-tiny.cfg](https://github.com/AlexeyAB/darknet/blob/master/cfg/yolov4-tiny.cfg)

Salve ambos os arquivos em uma pasta diretório do projeto 

