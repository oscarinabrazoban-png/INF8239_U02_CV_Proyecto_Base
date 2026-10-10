# INF-8239 · Unidad 02 · Proyecto de visión computacional

**Asignatura:** Ciencia de Datos II  
**Laboratorio:** U02.LAB07 · CNN reproducible con Fashion-MNIST  
**Autor académico:** Edwin Ramón José Nolasco

## Objetivo

Entrenar y comparar una red neuronal densa (baseline) y una red neuronal convolucional (CNN) utilizando Fashion-MNIST, evaluando su desempeño predictivo, costo computacional, errores de clasificación y reproducibilidad.

## Ejecución en CPU

```bash
uv python install 3.12
uv sync --extra cpu
uv run python scripts/check_runtime.py
uv run python scripts/train_cv.py --epochs 8
```

## Ejecución en GPU

La ruta GPU se utiliza solamente en WSL2/Linux con NVIDIA correctamente configurada:

```bash
uv sync --extra gpu
```

## Resultados experimentales

Los modelos fueron entrenados durante ocho épocas utilizando CPU en Windows 11.

| Indicador | Modelo denso | CNN |
|---|---:|---:|
| F1 macro | 0.8638 | 0.7893 |
| Parámetros | 50,890 | 19,466 |
| Entrenamiento (s) | 13.49 | 137.28 |
| Inferencia (ms/imagen) | 0.0722 | 0.2079 |

En esta ejecución, el modelo denso obtuvo un mejor desempeño predictivo y menores tiempos de entrenamiento e inferencia, mientras que la CNN utilizó menos parámetros.

## Evidencias generadas

- `reports/cv_metrics.json`: métricas de desempeño y costo.
- `reports/confusion_cnn.png`: matriz de confusión de la CNN.
- `reports/cnn_errors.png`: ejemplos de clasificación incorrecta.
- `models/best_cnn.keras`: modelo CNN guardado.
- `MODEL_CARD.md`: documentación de desempeño, limitaciones, riesgos y usos previstos.

## Pruebas automatizadas

Para verificar los contratos técnicos del proyecto:

```bash
uv run pytest -q
```

Resultado obtenido:

**3 pruebas aprobadas en 9.28 segundos.**

## Limitaciones

Fashion-MNIST es un conjunto de datos didáctico. Los resultados obtenidos no garantizan un desempeño equivalente en imágenes reales ni en otros dominios.

El experimento se realizó con ocho épocas y una configuración específica, por lo que las conclusiones se limitan a estas condiciones.