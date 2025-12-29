import gradio as gr
import numpy as np
import cv2
import os
from model import BloodCellAI

model = BloodCellAI()

classes = [
    "Nötrofil", "Eozinofil", "Basofil",
    "Lenfosit", "Monosit", "Immature",
    "Eritroblast", "Trombosit"
]

name_map = {
    "Neutrophil": "Nötrofil",
    "Eosinophil": "Eozinofil",
    "Basophil": "Basofil",
    "Lymphocyte": "Lenfosit",
    "Monocyte": "Monosit",
    "Immature": "Immature",
    "Erythroblast": "Eritroblast",
    "Platelet": "Trombosit"
}

try:
    data = np.load("trained_weights.npy", allow_pickle=True).item()
    model.layers[0].w, model.layers[0].b = data["c1w"], data["c1b"]
    model.layers[3].w, model.layers[3].b = data["c2w"], data["c2b"]
    model.layers[6].w, model.layers[6].b = data["d1w"], data["d1b"]
    model.layers[8].w, model.layers[8].b = data["d2w"], data["d2b"]
    print("Model başarıyla yüklendi.")
except Exception as e:
    print(f"Model yüklenemedi: {e}")

def analyze_cell(img_path):
    if img_path is None:
        return None, "Görsel yüklenmedi."

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = cv2.resize(img, (28, 28))
    img = img.transpose(2, 0, 1) / 255.0
    img = np.expand_dims(img, axis=0)

    probs = model.forward(img)[0]
    pred_idx = np.argmax(probs)
    pred_label = classes[pred_idx]

    file_name = os.path.basename(img_path)
    actual_en = file_name.split("_")[0]
    actual_tr = name_map.get(actual_en)

    if actual_tr is None:
        result_text = f"Gerçek etiket bulunamadı | Tahmin: {pred_label}"
    elif actual_tr == pred_label:
        result_text = f"DOĞRU! (Gerçek: {actual_tr}, Tahmin: {pred_label})"
    else:
        result_text = f"YANLIŞ! (Gerçek: {actual_tr}, Tahmin: {pred_label})"

    prob_dict = {classes[i]: float(probs[i]) for i in range(len(classes))}
    return prob_dict, result_text

base_dir = os.path.dirname(os.path.abspath(__file__))
example_path = os.path.join(base_dir, "test_samples")

def load_examples():
    examples = []
    if os.path.exists(example_path):
        for f in os.listdir(example_path):
            if f.lower().endswith((".png", ".jpg", ".jpeg")):
                examples.append([os.path.join(example_path, f)])
    return examples

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# BloodCell-AI: Akıllı Hücre Tanılama")
    gr.Markdown(
        "Bir hücre görseli yükleyin veya örneklerden seçin. "
        "**Sistem tahmini, dosya adındaki gerçek değer ile otomatik karşılaştırır.**"
    )

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(
                label="Hücre Görüntüsü",
                type="filepath",
                sources=["upload"]  
            )
            analyze_btn = gr.Button("Analiz Et", variant="primary")

        with gr.Column():
            output_probs = gr.Label(
                label="Sınıf Olasılıkları",
                num_top_classes=3
            )
            output_text = gr.Textbox(
                label="Karşılaştırma Sonucu",
                interactive=False
            )

    analyze_btn.click(
        fn=analyze_cell,
        inputs=input_image,
        outputs=[output_probs, output_text]
    )

    gr.Examples(
        examples=load_examples(),
        inputs=input_image,
        outputs=[output_probs, output_text],
        fn=analyze_cell,
        cache_examples=False,
        label="Örnekler"
    )

if __name__ == "__main__":
    demo.launch()
