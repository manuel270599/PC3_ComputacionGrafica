# Restauración y Super-Resolución de Video de la Televisión Peruana (2000–2010) con SRGAN

Implementación de una red generativa adversaria de super-resolución (**SRGAN — Super-Resolution Generative Adversarial Network**) desarrollada en **TensorFlow/Keras** y ejecutada en **Google Colab**, orientada a restaurar y escalar material audiovisual de archivo de la televisión peruana de la década del 2000 (resoluciones SD comprimidas como **302x240** a 185 kbps) hacia calidad **HD (1208x960, escala x4)**, preservando la pista de audio original.

---

## Objetivo del Proyecto

Adaptar una arquitectura SRGAN diseñada originalmente para imágenes estáticas cuadradas de rostros (`CelebA-HQ`) hacia un pipeline completo de **restauración de video de archivo**, resolviendo:
1. **Baja resolución espacial:** Escalado **x4** desde **302x240** (240p) hasta **1208x960** (960p HD).
2. **Degradaciones propias de TV antigua:** Eliminación de bloques de compresión (macrobloques MPEG/FLV), ruido analógico de cintas VHS y desenfoque por baja tasa de bits.
3. **Preservación de identidad y audio:** Reconstrucción de rasgos faciales mediante pérdida perceptual (`VGG19`) y reensamblaje de audio sincronizado con `FFmpeg`.

---

## Modelos Utilizados y su Rol en el Proyecto

| Componente | Modelo / Arquitectura | Configuración en el Código | Función Específica en el Video de TV Perú |
| :--- | :--- | :--- | :--- |
| **Generador (G)** | **SRResNet** (Red Residual) | 8 bloques residuales + 2 capas `PixelShuffle(scale=2)` + entrada `(None, None, 3)` | Toma cada fotograma rectangular de **302x240**, limpia los cuadros de compresión de los 185 kbps y lo reconstruye a **1208x960** sin deformar el formato 4:3. |
| **Discriminador (D)** | **CNN Clasificadora** | 8 capas `Conv2D` (`64` a `512` filtros) + `BatchNormalization` + `LeakyReLU(0.2)` | Evalúa parches de **128x128** durante el entrenamiento para penalizar al Generador si los rostros o bordes se ven plásticos o borrosos. |
| **Extractor Perceptual** | **VGG19** (Pre-entrenada) | Pesos de ImageNet congelados (`trainable=False`), salida en `block5_conv4` | Compara rasgos faciales profundos (ojos, cabello, contornos) aprendidos de **CelebA-HQ** para proyectar nitidez real sobre los conductores y actores del video. |

---

## Flujo de Trabajo para Mejorar el Video

1. **Preparación de Datos (`CelebA-HQ` -> Pares Sintéticos):** Se toman 3,000 fotos HD de rostros (`CelebA-HQ`), se recortan parches de **128x128** (HR) y se degradan artificialmente a **32x32** (LR) simulando el deterioro de la TV antigua.
2. **Entrenamiento en 2 Fases:**
   * **Fase 1 (10 épocas):** El Generador aprende la estructura y colores base minimizando el error absoluto (`MAE`).
   * **Fase 2 (30 épocas):** Se activa el Discriminador y `VGG19` para recuperar texturas finas y nitidez en los bordes.
3. **Inferencia por Lotes sobre el Video:** Se lee `video_original.mp4` (**302x240**), se agrupa en lotes de 4 fotogramas y el Generador predice los fotogramas restaurados en **1208x960**.
4. **Ensamblaje de Audio y Evaluación:** `FFmpeg` une los fotogramas HD con el audio original del clip, generando el video final, una comparativa lado a lado (*Bicúbico vs. SRGAN*) y las métricas **PSNR** y **SSIM**.

---

## Explicación de Líneas Clave del Código

A continuación se detallan las líneas más significativas de cada etapa del pipeline en Google Colab:

### 1. Pipeline de Datos sin Desbordamiento de RAM (`tf.data`)
Cargar miles de imágenes de `CelebA-HQ` en memoria RAM provoca que Google Colab se reinicie. Para evitarlo, las imágenes se leen y procesan por lotes bajo demanda desde el disco:

```python
dataset = tf.data.Dataset.from_tensor_slices(image_paths)
dataset = dataset.shuffle(2000).map(tf_parse, num_parallel_calls=tf.data.AUTOTUNE)
dataset = dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
```
- `from_tensor_slices(image_paths)`: Crea un flujo ligero que solo almacena las rutas de texto de las imágenes, no los píxeles.
- `.shuffle(2000)`: Mezcla aleatoriamente los ejemplos en cada época para evitar que la red memorice el orden del dataset.
- `.map(tf_parse, num_parallel_calls=tf.data.AUTOTUNE)`: Aplica el recorte y la degradación en paralelo usando todos los núcleos de CPU disponibles.
- `.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)`: Agrupa las imágenes en lotes de 16 y prepara el siguiente lote en memoria mientras la GPU T4 entrena con el lote actual.

---

### 2. Simulación de la Degradación del Video de Archivo

```python
blurred = cv2.GaussianBlur(hr_crop, (5, 5), 1.1)
lr_crop = cv2.resize(blurred, (LR_SIZE, LR_SIZE), interpolation=cv2.INTER_AREA)
quality = np.random.randint(30, 65)
_, encimg = cv2.imencode('.jpg', lr_crop, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
```
- **Para qué sirve:** Replica los defectos exactos del archivo de TV peruana (desenfoque analógico + compresión fuerte a 185 kbps) sobre las fotos HD de `CelebA-HQ`, enseñando a la red a limpiar bloques pixelados antes de ampliar.

---

### 3. Arquitectura del Generador: Bloques Residuales y `PixelShuffle`
El Generador utiliza aprendizaje residual para conservar la información original del fotograma y solo predecir los detalles faltantes:

```python
def res_block(x):
    skip = x
    x = layers.Conv2D(64, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.PReLU(shared_axes=[1, 2])(x)
    x = layers.Conv2D(64, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    return layers.Add()([skip, x])
```
- `skip = x` y `layers.Add()([skip, x])`: **Conexión residual (Skip Connection)**. Permite que los gradientes fluyan sin desvanecerse y evita que el generador altere los colores o la estructura base del video.
- `layers.PReLU(shared_axes=[1, 2])`: Función de activación *Parametric ReLU*, que aprende la pendiente para valores negativos (crucial para preservar detalles finos en sombras y texturas).

Para aumentar el tamaño de la imagen sin crear patrones de cuadrícula (*checkerboard artifacts*), se usa **Sub-Pixel Convolution**:

```python
@tf.keras.utils.register_keras_serializable()
class PixelShuffle(layers.Layer):
    def __init__(self, scale=2, **kwargs):
        super().__init__(**kwargs)
        self.scale = scale

    def call(self, inputs):
        return tf.nn.depth_to_space(inputs, self.scale)
```
- `tf.nn.depth_to_space(inputs, 2)`: Reorganiza un tensor de `256` canales de baja resolución en `64` canales con el **doble de ancho y alto (x2)**. Al ejecutar este bloque dos veces seguidas, se obtiene la ampliación final de **x4** (**302x240 -> 1208x960**).
- `inputs = layers.Input(shape=(None, None, 3))`: Al dejar el alto y ancho como `None`, la red entrenada con parches de **32x32** puede recibir fotogramas completos rectangulares de **302x240** durante la inferencia.

---

### 4. Entrenamiento en 2 Fases y Función de Pérdida Perceptual
Entrenar una GAN desde cero suele ser inestable. Por ello el código divide el aprendizaje en dos etapas:

```python
# Fase 1: Pre-entrenamiento supervisado
gen.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss='mae')
gen.fit(dataset, epochs=EPOCHS_PRETRAIN)
```
- `loss='mae'`: Error Absoluto Medio (L1). Estabiliza el Generador durante las primeras 10 épocas para que produzca imágenes limpias y coherentes antes de enfrentarlo al Discriminador.

En la **Fase 2 (Adversarial)**, se calcula la pérdida compuesta dentro de `train_step`:

```python
d_loss = 0.5 * (bce(tf.ones_like(real_logits) * 0.9, real_logits) + bce(tf.zeros_like(fake_logits), fake_logits))

hr_vgg = vgg(applications.vgg19.preprocess_input((hr_imgs + 1.0) * 127.5))
sr_vgg = vgg(applications.vgg19.preprocess_input((sr_imgs + 1.0) * 127.5))

perceptual_loss = mse(hr_vgg, sr_vgg) * 0.02
adversarial_loss = bce(tf.ones_like(fake_logits), fake_logits) * 3e-3
pixel_loss = tf.reduce_mean(tf.abs(hr_imgs - sr_imgs)) * 1e-2

g_loss = perceptual_loss + adversarial_loss + pixel_loss
```
- `tf.ones_like(real_logits) * 0.9`: **Label Smoothing**. Suaviza la etiqueta real de `1.0` a `0.9` para evitar que el Discriminador se vuelva demasiado estricto y bloquee el aprendizaje del Generador.
- `hr_vgg` y `sr_vgg`: Pasa tanto la imagen HD real como la generada a través de `VGG19` (desnormalizando previamente de `[-1, 1]` a `[0, 255]`).
- `perceptual_loss`: Compara las activaciones internas de `VGG19` (`block5_conv4`) mediante MSE. Esto obliga al modelo a reconstruir **bordes nítidos, ojos y texturas naturales** en lugar de promediar píxeles.
- `adversarial_loss`: Penaliza al Generador si el Discriminador logra detectar que la imagen es falsa.
- `pixel_loss`: Mantiene la fidelidad cromática respecto a la imagen original.

---

### 5. Inferencia por Lotes y Sincronización de Audio (`FFmpeg`)
OpenCV procesa únicamente la pista visual, eliminando el sonido. En el script de procesamiento se realiza la inferencia acelerada y se reincorpora el audio original:

```python
preds = generator(batch_np, training=False).numpy()
sr_rgb = np.clip((preds[i] + 1.0) * 127.5, 0, 255).astype(np.uint8)
```
- `training=False`: Desactiva la actualización de estadísticas en `BatchNormalization`, garantizando estabilidad de color entre fotogramas consecutivos.
- `np.clip((preds[i] + 1.0) * 127.5, 0, 255)`: Convierte la salida de la capa `tanh` (rango `[-1, 1]`) al espacio estándar de imagen RGB de 8 bits (`[0, 255]`).

Finalmente, `FFmpeg` empaqueta el video restaurado con la pista de audio intacta:

```python
subprocess.run([
    "ffmpeg", "-y", "-i", TEMP_VIDEO, "-i", INPUT_VIDEO,
    "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
    "-c:a", "aac", "-map", "0:v:0", "-map", "1:a:0?", FINAL_OUTPUT
], check=True)
```
- `-c:v libx264 -crf 18`: Codifica el video resultante en H.264 de alta calidad visual (`crf 18` es prácticamente sin pérdida perceptible).
- `-map 0:v:0 -map 1:a:0?`: Toma el flujo de video del archivo generado por la SRGAN (`0:v:0`) y copia la pista de audio original del archivo de entrada (`1:a:0`), manteniendo la sincronización exacta.

---

## Archivos Generados en `/content/`

| Archivo | Descripción |
| :--- | :--- |
| `srgan_tv_peru_gen.keras` | Pesos entrenados del Generador SRGAN listos para inferencia. |
| `resultado_final_srgan.mp4` | Video restaurado en HD (**1208x960**) con su pista de audio original. |
| `comparativa_lado_a_lado.mp4` | Video split-screen: *Izquierda (Escalado Bicúbico)* vs. *Derecha (SRGAN)*. |
| `comparativa_fotograma_zoom.png` | Captura estática de alta resolución con acercamiento (zoom) en rostros y detalles. |

---

## Evaluación de Resultados (Métricas y Análisis)

Para validar el desempeño del modelo en la exposición se emplean dos enfoques complementarios:

1. **Evaluación Cuantitativa (`PSNR` y `SSIM`):**
   * **PSNR (Peak Signal-to-Noise Ratio):** Mide la fidelidad píxel a píxel en decibelios (dB). En modelos SRGAN, el PSNR suele ser moderado porque la red sintetiza texturas de alta frecuencia que no existían en el archivo comprimido original.
   * **SSIM (Structural Similarity Index):** Evalúa la preservación estructural, luminancia y contraste percibido entre fotogramas.
2. **Evaluación Cualitativa (Visual):**
   * Comparación directa mediante `comparativa_lado_a_lado.mp4` y `comparativa_fotograma_zoom.png`, evidenciando la reducción de bloques de compresión JPEG/MPEG y la definición de contornos faciales frente a la interpolación bicúbica clásica.
