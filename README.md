# Projeto de Rastreamento de Objetos com Python e OpenCV <img src="imagens\readme\Opencv.png" alt="Alt text" width="40"/> + <img src="imagens\readme\Python.svg.png" alt="Alt text" width="40"/>

Este repositório contém implementações de diversos algoritmos de rastreamento de objetos utilizando a biblioteca OpenCV. O projeto é dividido em diferentes pastas, cada uma explorando um conjunto específico de algoritmos.

## Algoritmos de Rastreamento

### 🚶‍♂️ 0_Single_Tracking
Nesta pasta, encontram-se implementações dos seguintes algoritmos de rastreamento, focados em identificar e seguir um único elemento no vídeo:

- **BOOSTING**
- **MIL**
- **KCF**
- **TLD**
- **MEDIANFLOW**
- **MOSSE**
- **CSRT**

### 👥 1_Multi_Tracking
Aqui, estão implementações dos mesmos algoritmos da pasta `0_Single_Tracking`, adaptados para identificar e rastrear múltiplos elementos no vídeo.

### 🔄 2_Goturn
Dedicada exclusivamente ao algoritmo Goturn, que utiliza técnicas de aprendizado profundo (Deep Learning) para realizar o rastreamento de objetos, ajustando seu modelo à medida que o objeto se move.

### 🎯 3_MeanShift
Implementação do algoritmo Meanshift, que utiliza histogramas para encontrar a região de máxima similaridade em cada quadro, permitindo o acompanhamento eficiente do objeto.

### 🔄🔍 4_CAMShift
Apresenta o algoritmo CAMShift (Continuously Adaptive Mean Shift), uma extensão do Meanshift que ajusta continuamente o tamanho e a orientação da janela de rastreamento.

### 🌐👁️ 5_Optical_Flow
Esta seção aborda dois tipos de métodos de Optical Flow:
- **Sparse Optical Flow:** Rastreamento baseado em pontos de interesse específicos no vídeo.
- **Dense Optical Flow:** Rastreamento de fluxo óptico em toda a imagem, permitindo uma compreensão mais abrangente do movimento.

### 🔄🔍👀 6_Detection
Explora a combinação de detecção e rastreamento no vídeo, proporcionando maior robustez na identificação e rastreamento de objetos em movimento.

## Benefícios da Abordagem de Detecção e Rastreamento
A combinação de detecção e rastreamento oferece benefícios significativos, permitindo que o sistema:
- Detecte novos objetos no vídeo.
- Rastreie objetos conhecidos de forma mais eficiente, especialmente em cenários desafiadores.
- Adapte-se a mudanças na aparência do objeto ao longo do tempo.

Essa abordagem híbrida aumenta a robustez e a precisão do sistema de rastreamento em diferentes situações, tornando-o mais adequado para aplicações práticas.

Sinta-se à vontade para explorar cada pasta para obter detalhes específicos sobre a implementação de cada algoritmo.
