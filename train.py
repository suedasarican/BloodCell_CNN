from model import BloodCellAI
from data_loader import load_bloodmnist
import numpy as np
import pickle

def calculate_accuracy(probs, labels):
    preds = np.argmax(probs, axis=1)
    return np.mean(preds == labels)

def train_model():
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_bloodmnist("bloodmnist.npz")
    model = BloodCellAI()
    
    epochs = 15
    batch_size = 128
    learning_rate = 0.001 
    
    train_losses = []
    train_accuracies = []
    val_accuracies = []
    
    best_val_acc = 0
    print(f"CNN Eğitimi Başlıyor")

    for epoch in range(epochs):
        perm = np.random.permutation(len(x_train))
        x_train, y_train = x_train[perm], y_train[perm]
        
        epoch_loss = 0
        epoch_acc = 0
        num_batches = len(x_train) // batch_size
        
        for i in range(0, len(x_train), batch_size):
            x_batch = x_train[i:i+batch_size].copy()
            y_batch = y_train[i:i+batch_size]
            
            if np.random.rand() > 0.5: x_batch = np.flip(x_batch, axis=3)
            if np.random.rand() > 0.5: x_batch = np.flip(x_batch, axis=2)
            
            probs = model.forward(x_batch)
            
            m = y_batch.shape[0]
            loss = -np.sum(np.log(probs[range(m), y_batch] + 1e-10)) / m
            epoch_loss += loss
            epoch_acc += calculate_accuracy(probs, y_batch)
            
            grad = probs.copy()
            grad[range(m), y_batch] -= 1
            grad /= m
            
            for layer in reversed(model.layers):
                grad = layer.backward(grad, learning_rate)
        
        val_probs = model.forward(x_val)
        val_acc = calculate_accuracy(val_probs, y_val)
        
        avg_train_loss = epoch_loss / (num_batches + 1)
        avg_train_acc = (epoch_acc / num_batches) * 100
        
        train_losses.append(avg_train_loss)
        train_accuracies.append(avg_train_acc)
        val_accuracies.append(val_acc * 100)
        
        print(f"Epoch {epoch+1:2d} | Loss: {avg_train_loss:.4f} | Doğrulama Başarısı: %{val_acc*100:.2f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            weights = {
                'c1w': model.layers[0].w, 'c1b': model.layers[0].b,
                'c2w': model.layers[3].w, 'c2b': model.layers[3].b,
                'd1w': model.layers[6].w, 'd1b': model.layers[6].b,
                'd2w': model.layers[8].w, 'd2b': model.layers[8].b
            }
            np.save('trained_weights.npy', weights)

    # --- EĞİTİM SONUNDA GEÇMİŞİ KAYDET ---
    history = {
        'train_loss': train_losses,
        'train_acc': train_accuracies,
        'val_acc': val_accuracies
    }
    with open('train_history.pkl', 'wb') as f:
        pickle.dump(history, f)
    print("\nEğitim geçmişi 'train_history.pkl' olarak kaydedildi.")
    # -------------------------------------

    test_probs = model.forward(x_test)
    test_acc = calculate_accuracy(test_probs, y_test)
    print(f"Final Test Başarısı: %{test_acc*100:.2f}")

    return model

if __name__ == "__main__":
    train_model()