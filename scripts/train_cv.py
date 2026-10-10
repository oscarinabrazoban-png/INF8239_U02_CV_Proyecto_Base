
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from time import perf_counter

# Configurar determinismo antes de importar TensorFlow.
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    f1_score,
)

from inf8239_u02_cv.data import normalize_images, validate_images
from inf8239_u02_cv.models import build_cnn, build_dense


ROOT = Path(__file__).resolve().parents[1]
SEED = 42

CLASS_NAMES = [
    "Camiseta/top",
    "Pantalón",
    "Suéter",
    "Vestido",
    "Abrigo",
    "Sandalia",
    "Camisa",
    "Zapatilla",
    "Bolso",
    "Botín",
]


def configure_seed() -> None:
    """Fija las semillas para favorecer la reproducibilidad."""
    random.seed(SEED)
    np.random.seed(SEED)
    tf.keras.utils.set_random_seed(SEED)


def compile_model(model):
    """Compila los modelos con la configuración original."""
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def save_learning_curves(history, name: str, reports: Path) -> None:
    """Guarda curvas de accuracy y loss."""
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))

    epochs = range(1, len(history.history["loss"]) + 1)

    axes[0].plot(
        epochs,
        history.history["accuracy"],
        marker="o",
        label="Entrenamiento",
    )
    axes[0].plot(
        epochs,
        history.history["val_accuracy"],
        marker="o",
        label="Validación",
    )
    axes[0].set_title(f"Accuracy - {name}")
    axes[0].set_xlabel("Época")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(
        epochs,
        history.history["loss"],
        marker="o",
        label="Entrenamiento",
    )
    axes[1].plot(
        epochs,
        history.history["val_loss"],
        marker="o",
        label="Validación",
    )
    axes[1].set_title(f"Loss - {name}")
    axes[1].set_xlabel("Época")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    figure.tight_layout()
    figure.savefig(
        reports / f"curves_{name}.png",
        dpi=170,
        bbox_inches="tight",
    )
    plt.close(figure)


def save_confusion_matrix(
    y_true,
    predictions,
    name: str,
    reports: Path,
) -> None:
    """Guarda la matriz de confusión de cada modelo."""
    matrix = confusion_matrix(
        y_true,
        predictions,
        labels=list(range(10)),
    )

    figure, axis = plt.subplots(figsize=(9, 8))
    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=list(range(10)),
    )
    display.plot(
        ax=axis,
        cmap="Blues",
        values_format="d",
        colorbar=False,
    )
    axis.set_title(f"Matriz de confusión - {name}")
    figure.tight_layout()
    figure.savefig(
        reports / f"confusion_{name}.png",
        dpi=170,
        bbox_inches="tight",
    )
    plt.close(figure)


def save_error_examples(
    images,
    y_true,
    predictions,
    name: str,
    reports: Path,
) -> None:
    """Guarda hasta 16 imágenes clasificadas incorrectamente."""
    wrong = np.where(predictions != y_true)[0][:16]

    figure, axes = plt.subplots(4, 4, figsize=(9, 9))

    for axis in axes.ravel():
        axis.axis("off")

    for axis, index in zip(axes.ravel(), wrong):
        axis.imshow(images[index].squeeze(), cmap="gray")
        axis.set_title(
            f"Real: {y_true[index]} | Pred.: {predictions[index]}",
            fontsize=9,
        )

    figure.suptitle(f"Errores de clasificación - {name}")
    figure.tight_layout()
    figure.savefig(
        reports / f"errors_{name}.png",
        dpi=170,
        bbox_inches="tight",
    )
    plt.close(figure)


def measure_inference(model, x_test, repeats: int = 5) -> dict:
    """Mide cinco ejecuciones de inferencia sobre el conjunto de prueba."""
    # Calentamiento: reduce el efecto de la primera llamada.
    model.predict(x_test[:128], verbose=0)

    times_ms = []

    for _ in range(repeats):
        start = perf_counter()
        model.predict(x_test, verbose=0)
        elapsed = perf_counter() - start

        times_ms.append(
            float(1000 * elapsed / len(x_test))
        )

    return {
        "inference_repeats": repeats,
        "inference_times_ms_per_image": times_ms,
        "inference_mean_ms_per_image": float(np.mean(times_ms)),
        "inference_std_ms_per_image": float(np.std(times_ms)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=8)
    args = parser.parse_args()

    configure_seed()

    reports = ROOT / "reports"
    models_dir = ROOT / "models"

    reports.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    # Cargar el conjunto Fashion-MNIST.
    (x_train_all, y_train_all), (x_test, y_test) = (
        tf.keras.datasets.fashion_mnist.load_data()
    )

    # Normalizar píxeles a [0, 1] y agregar dimensión de canal.
    x_train_all = normalize_images(x_train_all)
    x_test = normalize_images(x_test)

    # Partición fija:
    # Entrenamiento: 54,000 imágenes
    # Validación:     6,000 imágenes
    # Prueba:        10,000 imágenes
    x_train = x_train_all[:-6000]
    y_train = y_train_all[:-6000]

    x_valid = x_train_all[-6000:]
    y_valid = y_train_all[-6000:]

    validate_images(x_train, y_train)
    validate_images(x_valid, y_valid)
    validate_images(x_test, y_test)

    metrics = {
        "experiment": {
            "dataset": "Fashion-MNIST",
            "seed": SEED,
            "epochs_requested": args.epochs,
            "batch_size": 128,
            "train_samples": len(x_train),
            "validation_samples": len(x_valid),
            "test_samples": len(x_test),
            "normalization": "float32 / 255.0",
            "class_names": CLASS_NAMES,
            "tensorflow_version": tf.__version__,
            "gpu_devices": [
                device.name
                for device in tf.config.list_physical_devices("GPU")
            ],
        }
    }

    for name, factory in {
        "dense": build_dense,
        "cnn": build_cnn,
    }.items():

        print(f"\n{'=' * 50}")
        print(f"Entrenando modelo: {name}")
        print(f"{'=' * 50}")

        # Reiniciar semillas antes de crear cada modelo.
        configure_seed()

        model = compile_model(factory(tf))

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=2,
                restore_best_weights=True,
            )
        ]

        if name == "cnn":
            callbacks.append(
                tf.keras.callbacks.ModelCheckpoint(
                    filepath=models_dir / "best_cnn.keras",
                    monitor="val_loss",
                    save_best_only=True,
                )
            )

        # Entrenamiento y tiempo.
        start = perf_counter()

        history = model.fit(
            x_train,
            y_train,
            validation_data=(x_valid, y_valid),
            epochs=args.epochs,
            batch_size=128,
            callbacks=callbacks,
            verbose=2,
        )

        train_seconds = float(perf_counter() - start)

        # Curvas de entrenamiento.
        save_learning_curves(history, name, reports)

        # Predicción sobre el conjunto de prueba.
        start = perf_counter()
        probabilities = model.predict(x_test, verbose=0)
        inference_seconds = perf_counter() - start

        predictions = probabilities.argmax(axis=1)

        # Guardar modelo y medir tamaño real en disco.
        model_path = models_dir / f"{name}_final.keras"
        model.save(model_path)

        model_size_mb = float(
            model_path.stat().st_size / (1024 ** 2)
        )

        # Repetir inferencia para estimar promedio y variación.
        inference_metrics = measure_inference(
            model,
            x_test,
            repeats=5,
        )

        # Métricas detalladas por clase.
        report = classification_report(
            y_test,
            predictions,
            labels=list(range(10)),
            target_names=CLASS_NAMES,
            output_dict=True,
            zero_division=0,
        )

        # Historial de entrenamiento.
        training_history = {
            key: [float(value) for value in values]
            for key, values in history.history.items()
        }

        metrics[name] = {
            "f1_macro": float(
                f1_score(
                    y_test,
                    predictions,
                    average="macro",
                )
            ),
            "test_accuracy": float(
                np.mean(predictions == y_test)
            ),
            "parameters": int(model.count_params()),
            "train_seconds": train_seconds,
            "inference_ms_per_image": float(
                1000 * inference_seconds / len(x_test)
            ),
            "model_size_mb": model_size_mb,
            "model_file": str(model_path.relative_to(ROOT)),
            "epochs_completed": len(history.history["loss"]),
            "classification_report": report,
            "history": training_history,
            **inference_metrics,
        }

        # Matriz de confusión y ejemplos de errores.
        save_confusion_matrix(
            y_test,
            predictions,
            name,
            reports,
        )

        save_error_examples(
            x_test,
            y_test,
            predictions,
            name,
            reports,
        )

        print(f"\nReporte de clasificación: {name}")
        print(
            classification_report(
                y_test,
                predictions,
                labels=list(range(10)),
                target_names=CLASS_NAMES,
                digits=3,
                zero_division=0,
            )
        )

        print(
            json.dumps(
                {
                    "f1_macro": metrics[name]["f1_macro"],
                    "test_accuracy": metrics[name]["test_accuracy"],
                    "parameters": metrics[name]["parameters"],
                    "train_seconds": train_seconds,
                    "model_size_mb": model_size_mb,
                    "inference_mean_ms_per_image":
                        inference_metrics["inference_mean_ms_per_image"],
                    "inference_std_ms_per_image":
                        inference_metrics["inference_std_ms_per_image"],
                },
                indent=2,
            )
        )

    # Comparación de ambos modelos.
    metrics["comparison"] = {
        "f1_macro_difference_cnn_minus_dense": float(
            metrics["cnn"]["f1_macro"] - metrics["dense"]["f1_macro"]
        ),
        "training_time_ratio_cnn_over_dense": float(
            metrics["cnn"]["train_seconds"]
            / metrics["dense"]["train_seconds"]
        ),
        "parameter_ratio_cnn_over_dense": float(
            metrics["cnn"]["parameters"]
            / metrics["dense"]["parameters"]
        ),
        "size_ratio_cnn_over_dense": float(
            metrics["cnn"]["model_size_mb"]
            / metrics["dense"]["model_size_mb"]
        ),
    }

    # Guardar todos los resultados en formato JSON.
    metrics_path = reports / "cv_metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\nExperimento finalizado correctamente.")
    print(f"Métricas: {metrics_path}")
    print(f"Gráficos: {reports}")
    print(f"Modelos: {models_dir}")


if __name__ == "__main__":
    main()
