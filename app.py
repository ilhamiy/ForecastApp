from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
from fbprophet import Prophet

app = Flask(__name__)

def train_prophet_model(df):
    df = df.rename(columns={'time': 'ds', 'value': 'y'})
    model = Prophet()
    model.fit(df)
    return model

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Excel dosyasını yükleme
        uploaded_file = request.files['file']
        if uploaded_file.filename != '':
            df = pd.read_excel(uploaded_file)
            # Modeli eğitme
            model = train_prophet_model(df)
            # Geleceği tahmin etme
            future = model.make_future_dataframe(periods=365)
            forecast = model.predict(future)
            # Tahminleri görselleştirme
            fig = model.plot(forecast)
            plt.savefig('static/forecast_plot.png')  # Tahmin grafiğini kaydetme
            # Tahminleri Excel dosyasına kaydetme
            forecast.to_excel('static/forecast.xlsx')
            return render_template('result.html')
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
