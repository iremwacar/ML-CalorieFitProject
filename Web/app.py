from flask import Flask, render_template, request
from joblib import load
import pandas as pd

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Formdan gelen verileri al
    enerji = float(request.form['enerji'])
    yag = float(request.form['yag'])
    doymus_yag = float(request.form['doymus_yag'])
    karbonhidrat = float(request.form['karbonhidrat'])
    seker = float(request.form['seker'])
    lif = float(request.form['lif'])
    protein = float(request.form['protein'])

    # Özellik isimleri
    feature_names = ['Enerji_kcal', 'Yağ_g', 'Doymuş_Yağ_g', 'Karbonhidrat_g', 'Şeker_g', 'Lif_g', 'Protein_g']
    
    # Tahmin girdilerini oluştur
    input_data = pd.DataFrame({
        'Enerji_kcal': [enerji],
        'Yağ_g': [yag],
        'Doymuş_Yağ_g': [doymus_yag],
        'Karbonhidrat_g': [karbonhidrat],
        'Şeker_g': [seker],
        'Lif_g': [lif],
        'Protein_g': [protein]
    }, columns=feature_names)

    # Modeli yükle ve tahmin yap
    model = load("..\eniyi.joblib")
    prediction = model.predict(input_data)

    # Tahminin tam sayı kısmını al, noktadan sonra kalan kısmı sil
    predicted_value = round(prediction[0])  # Tam sayıya yuvarla
    predicted_str = str(predicted_value)  # Tam sayıyı string'e dönüştür
    
    # Noktadan önceki ilk rakamı silmek için
    result = predicted_str[1:]  # İlk karakteri (rakamı) sil

    # Sonucu result.html sayfasında göster
    return render_template('result.html', prediction=result)

if __name__ == "__main__":
    app.run(debug=True)
