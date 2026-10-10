# Model Card · Fashion-MNIST CNN

## Modelo y versión

Se desarrollaron y evaluaron dos modelos de clasificación de imágenes: una red neuronal densa (baseline) y una red neuronal convolucional (CNN), utilizando TensorFlow 2.21.0 y Keras 3.15.1.

El entrenamiento se realizó durante 8 épocas para ambos modelos, con el propósito de comparar su desempeño predictivo y costo computacional bajo las condiciones establecidas en el laboratorio U02.LAB07.

El modelo CNN entrenado se almacenó en `models/best_cnn.keras`.

## Uso previsto

El modelo está diseñado para clasificar imágenes de prendas de vestir pertenecientes a las diez categorías del conjunto de datos Fashion-MNIST.

Su finalidad es educativa y experimental, permitiendo comprender el funcionamiento de las redes neuronales convolucionales, evaluar métricas de clasificación y analizar la relación entre precisión predictiva y eficiencia computacional.

## Usos fuera de alcance

El modelo no está diseñado para aplicaciones médicas, vigilancia, reconocimiento de personas, identificación de objetos en entornos reales ni clasificación de imágenes comerciales sin validación adicional.

Tampoco debe utilizarse para tomar decisiones automatizadas de alto impacto.

## Dataset y particiones

Se utilizó el conjunto de datos Fashion-MNIST, compuesto por 70,000 imágenes en escala de grises, de resolución 28 × 28 píxeles, distribuidas en diez categorías de prendas de vestir.

Los datos se dividieron de la siguiente manera:

- **Entrenamiento:** 54,000 imágenes.
- **Validación:** 6,000 imágenes.
- **Prueba:** 10,000 imágenes.

El conjunto de validación corresponde a las últimas 6,000 imágenes del conjunto original de entrenamiento, mientras que el conjunto de prueba se mantuvo separado para la evaluación final.

Ambos modelos utilizaron las mismas particiones, lo que permite realizar una comparación bajo condiciones equivalentes.

Para favorecer la reproducibilidad, se estableció una semilla aleatoria de 42 en Python, NumPy y TensorFlow.

## Preprocesamiento

Las imágenes originales de Fashion-MNIST contienen valores de intensidad de píxeles entre 0 y 255.

Mediante la función `normalize_images`, estos valores se convierten al tipo de dato `float32` y se dividen entre 255, obteniendo valores normalizados en el intervalo de 0 a 1.

Posteriormente, se incorpora una dimensión adicional correspondiente al canal de color, transformando cada imagen de `(28, 28)` a `(28, 28, 1)`.

El programa también valida que las imágenes tengan las dimensiones esperadas, que coincidan con la cantidad de etiquetas y que sus valores sean finitos y estén dentro del intervalo permitido.

Este procedimiento garantiza que los datos tengan una representación consistente antes del entrenamiento de los modelos.

## Métricas globales y por clase

**Comparación global**

| Indicador | Modelo denso | CNN |
|---|---:|---:|
| F1 macro | 0.8638 | 0.7893 |
| Parámetros | 50,890 | 19,466 |
| Tiempo de entrenamiento | 13.49 s | 137.28 s |
| Inferencia por imagen | 0.0722 ms | 0.2079 ms |

**Resultados por clase de la CNN**

| Clase | Precisión | Recall | F1-score |
|---|---:|---:|---:|
| Camiseta/top | 0.645 | 0.807 | 0.717 |
| Pantalón | 0.988 | 0.932 | 0.959 |
| Suéter | 0.705 | 0.642 | 0.672 |
| Vestido | 0.753 | 0.843 | 0.796 |
| Abrigo | 0.661 | 0.663 | 0.662 |
| Sandalia | 0.950 | 0.885 | 0.916 |
| Camisa | 0.500 | 0.374 | 0.428 |
| Zapatilla deportiva | 0.855 | 0.952 | 0.901 |
| Bolso | 0.917 | 0.937 | 0.927 |
| Botín | 0.935 | 0.896 | 0.915 |

La CNN alcanzó una exactitud global aproximada de 79.3 % sobre las 10,000 imágenes de prueba.

La categoría con menor recall fue Camisa, con 37.4 %, mientras que Pantalón presentó el mayor F1-score, con 95.9 %.

## Comparación de costo

**Hardware:** CPU, Windows 11, Python 3.12.15. No se detectó GPU disponible para TensorFlow.

**Parámetros:**
- Modelo denso: 50,890.
- CNN: 19,466.

**Tiempo de entrenamiento:**
- Modelo denso: 13.49 segundos.
- CNN: 137.28 segundos.

**Tiempo de inferencia por imagen:**
- Modelo denso: 0.0722 milisegundos.
- CNN: 0.2079 milisegundos.

Desde la perspectiva de Green AI, el modelo denso presentó una mejor relación entre desempeño predictivo y tiempo de ejecución en esta configuración experimental.

Aunque la CNN utilizó aproximadamente 61.7 % menos parámetros, su entrenamiento fue alrededor de 10.2 veces más lento y su inferencia aproximadamente 2.9 veces más lenta.

Por tanto, los resultados obtenidos no justifican seleccionar la CNN únicamente por tratarse de una arquitectura especializada en imágenes.

## Limitaciones y riesgos

El modelo fue entrenado con imágenes de baja resolución y un conjunto de datos limitado a diez categorías de prendas de vestir.

Se identificaron dificultades para diferenciar prendas con características visuales similares, especialmente camisas, camisetas, suéteres y abrigos.

El desempeño observado en Fashion-MNIST no garantiza resultados equivalentes en imágenes reales, con diferentes condiciones de iluminación, fondos, resoluciones o tipos de prendas.

Además, los resultados corresponden a una configuración experimental de ocho épocas, por lo que no representan necesariamente el desempeño máximo alcanzable por cada arquitectura.

## Supervisión y monitoreo

Se recomienda supervisar periódicamente las métricas globales y por clase, especialmente el recall de las categorías que presentan mayores dificultades.

Antes de utilizar el modelo en un contexto diferente, sería necesario evaluar su desempeño con datos representativos del nuevo dominio, analizar los errores, verificar la estabilidad de las predicciones y considerar ajustes de arquitectura o entrenamiento.

El modelo debe mantenerse bajo supervisión humana y no utilizarse como único mecanismo de decisión en aplicaciones reales.