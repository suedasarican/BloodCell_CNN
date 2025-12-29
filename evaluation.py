import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.metrics import confusion_matrix, classification_report
from model import BloodCellAI
from data_loader import load_bloodmnist

def evaluate_and_plot():
    _, _, (x_test, y_test) = load_bloodmnist("bloodmnist.npz")
    model = BloodCellAI()
    classes = ["Nötrofil", "Eozinofil", "Basofil", "Lenfosit", "Monosit", "Immature", "Eritroblast", "Trombosit"]
    
    try:
        data = np.load('trained_weights.npy', allow_pickle=True).item()
        model.layers[0].w, model.layers[0].b = data['c1w'], data['c1b']
        model.layers[3].w, model.layers[3].b = data['c2w'], data['c2b']
        model.layers[6].w, model.layers[6].b = data['d1w'], data['d1b']
        model.layers[8].w, model.layers[8].b = data['d2w'], data['d2b']
        print("Ağırlıklar yüklendi.")
    except Exception as e:
        print(f"Hata: Ağırlıklar yüklenemedi: {e}")
        return

    y_pred = []
    for i in range(0, len(x_test), 100):
        probs = model.forward(x_test[i:i+100])
        y_pred.extend(np.argmax(probs, axis=1))
    
    print("\nPERFORMANS METRİKLERİ")
    print(classification_report(y_test, y_pred, target_names=classes))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Hücre Sınıflandırma Karmaşıklık Matrisi')
    plt.ylabel('Gerçek Sınıf')
    plt.xlabel('Tahmin Edilen Sınıf')
    plt.savefig('confusion_matrix.png')
    plt.show()

    try:
        with open('train_history.pkl', 'rb') as f:
            history = pickle.load(f)
        
        epochs = range(1, len(history['train_loss']) + 1)
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(epochs, history['train_loss'], 'r-o', label='Eğitim Kaybı (Loss)')
        plt.title('Hata Oranı (Loss)')
        plt.xlabel('Epoch'); plt.grid(True); plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(epochs, history['val_acc'], 'g-s', label='Doğrulama Başarısı')
        plt.title('Doğruluk Oranı (Accuracy %)')
        plt.xlabel('Epoch'); plt.grid(True); plt.legend()

        plt.tight_layout()
        plt.savefig('learning_curves.png')
        plt.show()
        print(" Grafikler 'learning_curves.png' olarak kaydedildi.")
    except FileNotFoundError:
        print("'train_history.pkl' bulunamadı, eğitim grafikleri çizilemiyor.")

if __name__ == "__main__":
    evaluate_and_plot()