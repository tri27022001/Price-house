from flask import Flask, render_template, request, redirect, url_for
import joblib
import numpy as np
import os

app = Flask(__name__)

# Đường dẫn đến file pkl cho các mô hình
model_paths = {
    'model1': 'elastic_net_model.pkl',
    'model2': 'GB_model.pkl',
    'model3': 'huber_model.pkl',
    'model4': 'LN_model.pkl',
    'model5': 'ridge_model.pkl',
    'model6': 'ridge_model.pkl'
}

model_descriptions = {}

# Đọc nội dung từ các tệp văn bản trong thư mục 'comment'
for model_name in model_paths.keys():
    file_path = os.path.join('comment', f"{model_name}.txt")
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            # Chuyển mỗi dòng thành một phần tử trong danh sách
            model_descriptions[model_name] = file.read().splitlines()
    except FileNotFoundError:
        model_descriptions[model_name] = [f"No description available for {model_name}."]

# Định nghĩa trang chủ
@app.route('/', methods=['GET', 'POST'])
def index():
    show_model_image = False
    if request.method == 'POST':
        # Lấy mô hình được chọn từ giao diện
        selected_model = request.form['model']
        # Tải mô hình từ file pkl
        model = joblib.load(model_paths[selected_model])
        #tải file chuẩn hóa input
        with open('scaler.pkl', 'rb') as scaler_file:
            scaler = joblib.load(scaler_file)

        model_description = model_descriptions.get(selected_model, 'No description available.')
        # Lấy dữ liệu đầu vào từ giao diện
        input_features = []
        input_values = {}
        for i in range(1, 17):
            feature_name = f'feature{i}'
            input_value = request.form.get(feature_name, ' ')
            input_values[feature_name] = input_value

            if not input_value:
                # Nếu input trống, hiển thị thông báo và không thực hiện dự đoán
                return render_template('index.html', predicted_price=None, message="Please enter all input!")

            input_features.append(float(input_value))
        # Chuẩn hóa đầu vào bằng scaler
        new_house_scaled = scaler.transform([input_features])
        # Dự đoán giá nhà
        predicted_price = model.predict(new_house_scaled)
        show_model_image = True 
        model_image_filename = f"{selected_model}.png"
        return render_template('index.html', predicted_price=predicted_price[0], show_model_image=show_model_image, model_image=model_image_filename, input_values=input_values, model_description=model_description)
    return render_template('index.html', predicted_price=None, show_model_image=show_model_image)


# Chuyển đến trang dashboard
@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

if __name__ == '__main__':
    app.run(debug=True)
