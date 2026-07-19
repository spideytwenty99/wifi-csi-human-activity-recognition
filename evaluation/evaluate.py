import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

'''
evaluate.py macht: 
1. Confusion Matrix -> zeigt welche Klassen verwechselt werden
2. Accuracy Balkendiagramm -> visueller Vergleich der 3 Modelle
3. Laufzeit Diagramm -> Training vs. Inference Zeit
4. Classification Report -> Precision, Recall, F1 pro Klasse
'''

def plot_confusion_matrix(y_test, predictions, model_name, class_names):
    """
    Plottet eine Confusion Matrix für ein Modell.
    
    y_test       -> echte Labels
    predictions  -> vorhergesagte Labels
    model_name   -> Name des Modells (für Titel)
    class_names  -> Liste der Klassennamen
    """
    cm = confusion_matrix(y_test, predictions)
    
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names
    )
    
    fig, ax = plt.subplots(figsize=(6, 5)) # plot
    disp.plot(ax=ax, cmap='Blues', colorbar=False) #blaue farbskala
    
    ax.set_title(f'Confusion Matrix — {model_name}', 
                 fontsize=13, fontweight='bold')
    
    plt.tight_layout() # autmatisch abstaende zwischen elementen anpassen
    plt.savefig(f'confusion_matrix_{model_name}.png', dpi=150) # speichert plot als png
    plt.close() # schliesst plot nach speichern
    
    print(f"    Confusion Matrix gespeichert: "
          f"confusion_matrix_{model_name}.png")
    


#############################
# Modelvergleich plot
#############################

def plot_model_comparison(results):
    """
    Plottet Accuracy und Laufzeit aller Modelle nebeneinander.
    Balkendiagramm pro Model
    results -> Dictionary mit Modellnamen und Ergebnissen
    """
    models    = list(results.keys())
    accuracies = [results[m]['accuracy'] * 100 for m in models]
    train_times = [results[m]['training_time'] for m in models]
    inf_times   = [results[m]['inference_time'] for m in models]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    colors = ['steelblue', 'seagreen', 'tomato']
    # -> 3 Plots nebeneinander (1 Zeile, 3 Spalten)
    # -> axes[0], axes[1], axes[2] für jeden einzelne
    
    # Plot 1 — Accuracy
    axes[0].bar(models, accuracies, color=colors)
    axes[0].set_title('Accuracy (%)', fontweight='bold')
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].set_ylim(0, 110)
    for i, v in enumerate(accuracies):
        axes[0].text(i, v + 1, f'{v:.1f}%', 
                     ha='center', fontweight='bold')
    axes[0].grid(axis='y', alpha=0.3)
    
    # Plot 2 — Training Time
    axes[1].bar(models, train_times, color=colors)
    axes[1].set_title('Trainingszeit (s)', fontweight='bold')
    axes[1].set_ylabel('Sekunden')
    for i, v in enumerate(train_times): # schreibt genauen Wert ueber jeden balken
        axes[1].text(i, v + 0.01, f'{v:.4f}s', 
                     ha='center', fontweight='bold', fontsize=9)
    axes[1].grid(axis='y', alpha=0.3)
    
    # Plot 3 — Inference Time
    axes[2].bar(models, inf_times, color=colors)
    axes[2].set_title('Inferenzzeit (s)', fontweight='bold')
    axes[2].set_ylabel('Sekunden')
    for i, v in enumerate(inf_times):
        axes[2].text(i, v + 0.0001, f'{v:.4f}s', 
                     ha='center', fontweight='bold', fontsize=9)
    axes[2].grid(axis='y', alpha=0.3)
    
    plt.suptitle('Modellvergleich: ' + ' vs. '.join(models), # uebertitel fuer alle plots zusammen
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("    Vergleichsplot gespeichert: model_comparison.png")



#########################
# classification report
#########################

def print_classification_report(y_test, predictions, 
                                  model_name, class_names):
    """
    Gibt Precision, Recall und F1-Score pro Klasse aus.
    Precision -> von allen die als Klasse X vorhergesagt wurden, wie viele waren wirklich X?
    Recall -> von allen echten Klasse X Samples, wie viele wurden korrekt erkannt?
    F1-Score -> Mittelwert aus Precision und Recall
    """
    print(f"\n  Classification Report — {model_name}")
    print("-" * 50)
    print(classification_report(
        y_test, predictions,
        target_names=class_names
    ))



######################## 
# Hauptfunktion
########################

def evaluate_all(results_with_predictions, y_test, class_names):
    """
    Führt alle Auswertungen für alle Modelle durch.
    
    results_with_predictions -> Dictionary mit Ergebnissen 
                               und Vorhersagen pro Modell
    """
    print("\n[Evaluation] Starte Auswertung...")
    
    # Confusion Matrix pro Modell
    print("\n  Confusion Matrices:")
    for model_name, data in results_with_predictions.items():
        plot_confusion_matrix(
            y_test,
            data['predictions'],
            model_name,
            class_names
        )
        print_classification_report(
            y_test,
            data['predictions'],
            model_name,
            class_names
        )
    
    # Vergleichsplot
    print("\n  Modellvergleich:")
    plot_model_comparison(results_with_predictions)
    
    print("\n[Evaluation] Fertig.")

