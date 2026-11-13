# TERCERA PRÁCTICA CALIFICADA - COMPUTACIÓN GRÁFICA
<p align="center">
  <img width="833" height="1015" alt="LOGO_UNI" src="https://github.com/user-attachments/assets/172ef688-84e2-46aa-9c60-c4d199e9dfdb" />
</p>

# Restauración y Super-Resolución de Video de la Televisión Peruana (2000–2010) con SRGAN


Implementación de una red generativa adversaria de super-resolución (**SRGAN — Super-Resolution Generative Adversarial Network**) desarrollada en **TensorFlow/Keras** y ejecutada en **Google Colab**, orientada a restaurar y escalar material audiovisual de archivo de la televisión peruana de la década del 2000 (resoluciones SD comprimidas como **302x240** a 185 kbps) hacia calidad **HD (1208x960, escala x4)**, preservando la pista de audio original.

---

##  Objetivos del Proyecto

### Objetivo General
Desarrollar e implementar un sistema de super-resolución y restauración de video  basado en una **Red Generativa Antagónica de Super-Resolución (SRGAN)**, capaz de mejorar la calidad visual y reducir los artefactos de compresión de material audiovisual antiguo con degradación tipo televisión analógica.

### Objetivos Específicos
- **Simular un *pipeline* de degradación realista:** Construir un proceso de preprocesamiento sobre el dataset **CelebA-HQ** que replique las fallas de transmisiones antiguas (desenfoque gaussiano, submuestreo y compresión JPEG agresiva) para entrenar el modelo frente a condiciones reales.
- **Implementar una arquitectura SRGAN personalizada:** Diseñar un generador residual con capas de reescalado sub-píxel (`PixelShuffle`) y un discriminador convolucional entrenado mediante una función de pérdida híbrida (pérdida perceptual con **VGG19**, pérdida adversarial con *label smoothing* y pérdida a nivel de píxel `MAE`).
- **Optimizar la inferencia sobre secuencias de video:** Procesar videos completos por micro-lotes en GPU y reintegrar de forma automatizada la pista de audio original mediante `FFmpeg` sin pérdida perceptible de calidad.
- **Evaluar cuantitativa y visualmente los resultados:** Contrastar el desempeño del modelo frente a la interpolación bicúbica tradicional mediante las métricas **PSNR** y **SSIM**, complementadas con comparativas lado a lado e inspecciones con acercamiento (*zoom*).
---
##  Requisitos y Dependencias

El cuaderno está diseñado para ejecutarse en **Google Colab** con aceleración por GPU activada:
- `tensorflow` / `keras`
- `opencv-python` (`cv2`)
- `scikit-image` (`psnr`, `ssim`)
- `kagglehub`
- `matplotlib`
- `ffmpeg`

## Modelos Utilizados y su Rol en el Proyecto

| Componente | Modelo / Arquitectura | Configuración en el Código | Función Específica en el Video de TV Perú |
| :--- | :--- | :--- | :--- |
| **Generador (G)** | **SRResNet** (Red Residual) | 8 bloques residuales + 2 capas `PixelShuffle(scale=2)` + entrada `(None, None, 3)` | Toma cada fotograma rectangular de **302x240**, limpia los cuadros de compresión de los 185 kbps y lo reconstruye a **1208x960** sin deformar el formato 4:3. |
| **Discriminador (D)** | **CNN Clasificadora** | 7 capas `Conv2D` (`64` a `512` filtros) + `BatchNormalization` + `LeakyReLU(0.2)` | Evalúa parches de **128x128** durante el entrenamiento para penalizar al Generador si los rostros o bordes se ven plásticos o borrosos. |
| **Extractor Perceptual** | **VGG19** (Pre-entrenada) | Pesos de ImageNet congelados (`trainable=False`), salida en `block5_conv4` | Compara rasgos faciales profundos (ojos, cabello, contornos) aprendidos de **CelebA-HQ** para proyectar nitidez real sobre los conductores y actores del video. |

---

##  Estructura del Flujo de Ejecución

1. **Entrenamiento (`Celda 1`):**
   - Descarga automática de 3,000 imágenes de `badasstechie/celebahq-resized-256x256` vía `kagglehub`.
   - Entrena el modelo y exporta los pesos en `/content/srgan_tv_peru_gen.keras`.
2. **Inferencia y Ensamblaje de Audio (`Celda 2`):**
   - Requiere subir el archivo `/content/video_original.mp4`.
   - Genera `resultado_final_srgan.mp4` (video restaurado con audio) y `comparativa_lado_a_lado.mp4` (Izquierda: Bicúbico | Derecha: SRGAN).
3. **Análisis Visual con Zoom (`Celda 3`):**
   - Extrae un fotograma en `t = 70s` y genera `comparativa_fotograma_zoom.png` con acercamiento en la región central.

---
##  Explicación de Bloques Clave del Código

El flujo de trabajo del cuaderno está dividido en seis etapas principales:

### 1. Preparación de Datos y Simulación de Degradación
Antes de iniciar el entrenamiento, el script descarga una muestra de 3,000 imágenes en alta definición del dataset **CelebA-HQ** mediante `kagglehub` y construye un pipeline de degradación sintética en tiempo real:

```python
img_blur = cv2.GaussianBlur(hr_crop, (5, 5), 1.1)
lr_img = cv2.resize(img_blur, (LR_SIZE, LR_SIZE), interpolation=cv2.INTER_AREA)
encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), random.randint(30, 65)]
_, encimg = cv2.imencode('.jpg', lr_img, encode_param)
lr_img = cv2.imdecode(encimg, 1)
```
- **¿Qué hace?** Toma recortes de alta resolución (`128x128`) y les aplica secuencialmente desenfoque óptico (`GaussianBlur`), reducción de escala $\times 4$ (`32x32` con `INTER_AREA`) y compresión JPEG agresiva en memoria (calidad aleatoria entre `30` y `65`).
- **Propósito:** Generar pares `(Imagen Degradada, Imagen Original)` que imiten los artefactos de transmisión de televisión analógica para que la red aprenda a revertirlos.

---

### 2. Construcción de las Redes Neuronales (Arquitectura SRGAN)
Se definen los componentes estructurales del modelo mediante Keras:

```python
@tf.keras.utils.register_keras_serializable()
class PixelShuffle(layers.Layer):
    def __init__(self, scale, **kwargs):
        super().__init__(**kwargs)
        self.scale = scale

    def call(self, inputs):
        return tf.nn.depth_to_space(inputs, self.scale)
```
- **Capa `PixelShuffle`:** Capa personalizada registrada (`@register_keras_serializable`) que utiliza `tf.nn.depth_to_space` para reorganizar canales en píxeles espaciales, ampliando la imagen $\times 4$ sin generar artefactos de cuadrícula.
- **Generador (`build_generator`):** Emplea 8 bloques residuales (*Skip Connections*) que conservan la estructura base de la escena mientras predicen los detalles de alta frecuencia.
- **Discriminador (`build_discriminator`):** Red convolucional de 7 bloques con *strides* alternados que actúa como clasificador binario para distinguir entre fotogramas reales y reconstruidos.

---

### 3. Función de Pérdida Perceptual y Bucle de Entrenamiento
En lugar de un entrenamiento estándar, se implementa un bucle personalizado con `tf.GradientTape()` dividido en dos fases:

```python
vgg = VGG19(weights="imagenet", include_top=False, input_shape=(HR_SIZE, HR_SIZE, 3))
vgg.trainable = False
feature_extractor = Model(inputs=vgg.input, outputs=vgg.get_layer("block5_conv4").output)

# Dentro de train_step():
real_labels = tf.ones((batch_size, 1)) * 0.9
percep_loss = tf.reduce_mean(tf.square(vgg_real - vgg_fake))
adv_loss = bce(tf.ones_like(fake_pred), fake_pred)
pixel_loss = mae(hr, fake_hr)
g_loss = 0.02 * percep_loss + 3e-3 * adv_loss + 1e-2 * pixel_loss
```
- **Fase 1 (Pre-entrenamiento - 10 épocas):** Optimiza únicamente el generador mediante error absoluto medio (`MAE`) para estabilizar colores y formas base.
- **Fase 2 (Entrenamiento Adversarial - 30 épocas):** Integra la red **VGG19** preentrenada (congelada en `block5_conv4`) para medir similitud de texturas profundas. Combina pérdida perceptual, adversarial y de píxel aplicando *label smoothing* (`0.9`) para evitar el colapso del discriminador, exportando finalmente el modelo en `srgan_tv_peru_gen.keras`.

---

### 4. Procesamiento e Inferencia de Video por Lotes
Carga el generador entrenado y procesa el archivo de entrada (`video_original.mp4`):

```python
batch_tensor = np.array(batch_frames, dtype=np.float32) / 255.0
sr_batch = generator.predict(batch_tensor, verbose=0)
sr_batch = np.clip(sr_batch * 255.0, 0, 255).astype(np.uint8)
```
- **Inferencia eficiente:** Normaliza los fotogramas al rango `[0, 1]` y los agrupa en micro-lotes (`BATCH_INFER = 4`) para aprovechar el paralelismo de la GPU sin saturar la memoria VRAM.
- **Evaluación en tiempo real:** Genera en paralelo una versión escalada por interpolación bicúbica para calcular periódicamente (cada 30 fotogramas) las métricas de referencia **PSNR** y **SSIM**.

---

### 5. Post-procesamiento de Audio y Exportación (`FFmpeg`)
Dado que OpenCV exporta flujos de video sin sonido, el sistema invoca `ffmpeg` mediante `subprocess` al finalizar la inferencia:

```python
subprocess.run([
    "ffmpeg", "-y", "-i", temp_sr_video, "-i", VIDEO_INPUT,
    "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
    "-map", "0:v:0", "-map", "1:a:0?", "-c:a", "aac", "-shortest", FINAL_OUTPUT
], check=True)
```
- **Video final con audio:** Fusiona el video restaurado con la pista de audio original (`-map 0:v:0 -map 1:a:0?`), codificando en H.264 con calidad visualmente sin pérdidas (`-crf 18`).
- **Comparativa sincronizada:** Utiliza el filtro `hstack` en un segundo comando para generar un video dividido lado a lado entre la interpolación bicúbica (izquierda) y el resultado SRGAN (derecha).

---

### 6. Inspección Visual con Recorte y Zoom
Herramienta de auditoría visual ubicada en la última celda del cuaderno:

```python
cap.set(cv2.CAP_PROP_POS_FRAMES, int(70 * fps))
# Recorte central de 250x250 píxeles sobre el fotograma reescalado
crop_sr = sr_frame[cy-125:cy+125, cx-125:cx+125]
crop_bic = bic_frame[cy-125:cy+125, cx-125:cx+125]
```
- **¿Qué hace?** Salta directamente al fotograma correspondiente al segundo `70`, lo procesa con ambos métodos y extrae una ventana central (`250x250` píxeles).
- **Propósito:** Generar una matriz comparativa (`comparativa_fotograma_zoom.png`) que permite evaluar en detalle la recuperación de bordes y texturas finas frente al suavizado tradicional.

---

## Archivos Generados en `/content/`

| Archivo | Descripción |
| :--- | :--- |
| `srgan_tv_peru_gen.keras` | Pesos entrenados del Generador SRGAN listos para inferencia. |
| `resultado_final_srgan.mp4` | Video restaurado en HD (**1208x960**) con su pista de audio original. |
| `comparativa_lado_a_lado.mp4` | Video split-screen: *Izquierda (Escalado Bicúbico)* vs. *Derecha (SRGAN)*. |
| `comparativa_fotograma_zoom.png` | Captura estática de alta resolución con acercamiento (zoom) en rostros y detalles. |

---
##  Resultados y Métricas

Evaluación realizada sobre un video de prueba de **2,134 fotogramas** reescalado de `302x240` a `1208x960` ($\times 4$), tomando la interpolación bicúbica como línea base:

- **PSNR Promedio:** `26.69 dB`
- **SSIM Promedio:** `0.8738`

### Evidencia del Proceso

### Entrenamiento
<p align="center">
<img width="921" height="548" alt="image" src="https://github.com/user-attachments/assets/80409a36-10b1-4e4a-8b71-69caaf360ffb" />
  
<img width="606" height="144" alt="image" src="https://github.com/user-attachments/assets/7788d762-a688-471d-83b7-623bdfaaf205" />
</p>

### Inferencia y Ensamblaje de Audio
<p align="center">
<img width="460" height="214" alt="image" src="https://github.com/user-attachments/assets/9965477e-602b-40ee-ae55-9ba2e8973fdd" />
</p>

### Análisis Visual muestra de fotograma
<p align="center">
<img width="894" height="735" alt="image" src="https://github.com/user-attachments/assets/4fbddc85-0980-4d18-862b-f67c5fb4dafc" />
</p>
*(Nota: En modelos SRGAN, valores moderados de PSNR/SSIM reflejan la generación de texturas de alta frecuencia más nítidas frente al desenfoque característico de la interpolación bicúbica).*

---

## Conclusiones y Análisis de Resultados

- **Limpieza visual (*Denoising*):** El modelo **SRGAN** elimina de forma consistente el granulado analógico y los artefactos de compresión JPEG, logrando transiciones de color más limpias y uniformes en comparación con la interpolación bicúbica tradicional.
- **Equilibrio entre fidelidad y nitidez:** Las métricas obtenidas (**PSNR: 26.69 dB** y **SSIM: 0.8738** en 2,134 fotogramas) junto con la función de pérdida perceptual (**VGG19**) confirman que el sistema conserva la estructura y colores originales sin generar deformaciones ni parpadeos entre cuadros.
- **Limitación por dominio de entrenamiento:** Dado que la red se entrenó exclusivamente con rostros en primer plano (**CelebA-HQ**), la mejora es notable en la limpieza de señal pero más sutil en planos generales, movimiento rápido y texto, donde prioriza el suavizado sobre el afilado de bordes.
- **Flujo viable y escalable:** El procesamiento por lotes en GPU y la integración automatizada de audio con `FFmpeg` demuestran que es posible restaurar y reescalar ($\times 4$) video de archivo de manera eficiente en Google Colab.
- **Trabajo futuro:** Entrenar con datasets de escenas variadas (como **DIV2K**), aumentar la profundidad de la red y añadir consistencia temporal entre fotogramas para mejorar el detalle en fondos y elementos no faciales.

## Créditos
- **Video original utilizado para inferencia:** Fragmento extraído del programa televisivo *"El Especial del Humor"* — [Enlace en YouTube](https://youtu.be/koB9PA-bucQ).
- **Dataset de entrenamiento:** [CelebA-HQ Resized](https://www.kaggle.com/datasets/badasstechie/celebahq-resized-256x256) vía Kaggle.

> *Nota: El material audiovisual enlazado se empleó únicamente con fines académicos y de demostración experimental para la evaluación del modelo de super-resolución.*

--- 
<p align="center">
José Manuel Montalvo Espinoza - Facultad de Ciencias de la Universidad Nacional de Ingeniería
</p>
