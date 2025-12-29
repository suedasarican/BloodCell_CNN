# # BloodCell-AI: Pure NumPy CNN Implementation

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/SuedaNur/BloodCell-AI)

Projenin temel amacı, herhangi bir derin öğrenme kütüphanesi (**PyTorch, TensorFlow, Keras vb.**) kullanmadan, sadece **NumPy** kütüphanesi ve matematiksel formüller kullanılarak sıfırdan bir Evrişimli Sinir Ağı (CNN) mimarisi inşa etmek ve kan hücrelerini sınıflandırmaktır.

## Proje Özellikleri

Bu projede kullanılan tüm katmanlar ve algoritmalar el yordamı ile kodlanmıştır:

* **Sıfırdan CNN Mimarisi:** `Conv2D`, `MaxPool`, `LeakyReLU`, `Dense` (Fully Connected), `Softmax` katmanları.
* **İleri Matematiksel Optimizasyon:** Python döngülerinin yavaşlığını aşmak için **`im2col` (Image to Column)** algoritması ile vektörize edilmiş konvolüsyon işlemi.
* **Optimizer:** Momentum ve RMSProp tabanlı **Adam Optimizer** $(\beta_1=0.9, \beta_2=0.999)$ implementasyonu.
* **Backpropagation:** Tüm katmanlar için gradyan hesaplamaları ve ağırlık güncellemeleri manuel türev alma yöntemleriyle yazılmıştır.
* **Numerically Stable Softmax:** Üstel işlem taşmalarını önlemek için kararlı softmax fonksiyonu.
* **Veri Artırma (Data Augmentation):** Eğitim sırasında rastgele yatay/dikey çevirme (Random Flip).

## Veri Seti: BloodMNIST

Projede, tıbbi görüntü analizinde standart bir benchmark olan **BloodMNIST** veri seti kullanılmıştır.

* **Görüntü Boyutu:** $28 \times 28$ piksel (RGB)
* **Toplam Görüntü:** 17,092
* **Sınıflar (8 Adet):** Nötrofil, Eozinofil, Basofil, Lenfosit, Monosit, Immature (Olgunlaşmamış), Eritroblast, Trombosit.

## Performans Sonuçları

Model, 15 Epoch sonunda aşağıdaki başarı oranlarına ulaşmıştır:

* **Test Doğruluğu (Accuracy):** **%88.48**
* **En Yüksek Doğrulama Başarısı:** **%91.88** (Epoch 13)

### Sınıf Bazlı Performans Tablosu

| Hücre Türü | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- |
| **Nötrofil** | 0.74 | 0.89 | 0.81 |
| **Eozinofil** | 0.96 | 0.97 | 0.97 |
| **Basofil** | 0.95 | 0.91 | 0.93 |
| **Lenfosit** | 0.77 | 0.80 | 0.79 |
| **Monosit** | 0.88 | 0.92 | 0.90 |
| **Immature** | 0.91 | 0.66 | 0.77 |
| **Eritroblast** | 0.95 | 0.96 | 0.96 |
| **Trombosit** | 1.00 | 1.00 | 1.00 |

## Kurulum

Projeyi yerel makinenizde çalıştırmak için aşağıdaki adımları izleyin:

1.  **Depoyu klonlayın:**
    ```bash
    git clone https://github.com/suedasarican/BloodCell_CNN.git
    cd BloodCell_CNN
    ```

2.  **Gereksinimleri yükleyin:**
    Modelin kendisi sadece NumPy kullanır, ancak veri işleme ve görselleştirme için yardımcı kütüphaneler gereklidir:
    ```bash
    pip install numpy opencv-python matplotlib seaborn scikit-learn gradio
    ```

## Kullanım

### 1. Modeli Eğitmek
Modeli sıfırdan eğitmek için `train.py` dosyasını çalıştırın. Veri seti otomatik olarak indirilecek ve eğitim başlayacaktır.

```bash
python train.py
```
Eğitim tamamlandığında ağırlıklar trained_weights.npy olarak, eğitim geçmişi train_history.pkl olarak kaydedilir.

### 2. Değerlendirme (Evaluation)
Eğitilmiş modelin metriklerini ve karmaşıklık matrisini görmek için:
```bash
python evaluation.py
```
### 3. Canlı Demo (Hugging Face Spaces)
Kurulum yapmadan modeli tarayıcınız üzerinden hemen test etmek için aşağıdaki bağlantıyı kullanabilirsiniz:

-> [BloodCell-AI Canlı Demo](https://huggingface.co/spaces/SuedaNur/BloodCell-AI)

### 4. Yerel Arayüz ile Test Etmek
Gradio arayüzünü kendi bilgisayarınızda başlatmak için:
```bash
python serve.py
```
### Geliştirici
Ad Soyad: Süeda Nur Sarıcan

Öğrenci No: 23120205031

Ders: Derin Öğrenme (Deep Learning)
