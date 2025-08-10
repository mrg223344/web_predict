from flask import Flask, render_template, request, session
import pandas as pd
import joblib

app = Flask(__name__)
app.secret_key = 'your-secret-key'  # 用于会话管理

# 加载模型
try:
    model = joblib.load('svm_model.pkl')
except Exception as e:
    print(f"模型加载失败: {e}")
    model = None  # 实际应用中应处理模型加载失败的情况

# 特征映射（中英文对照）
feature_map = {
    'Age': '年龄',
    'educational level': '文化程度',
    'Regular exercise': '规律运动',
    'DCCN': '糖尿病慢性并发症数量',
    'Malnutrition': '营养不良',
    'Depressive': '抑郁'
}

# 预先创建反向映射字典
reverse_feature_map = {v: k for k, v in feature_map.items()}

# 特征选项映射（中英文对照）
feature_options = {
    '年龄': {
        '60-69岁': 0,
        '70-79岁': 1,
        '80岁及以上': 2
    },
    '文化程度': {
        '小学及以下': 0,
        '初中': 1,
        '高中/中专/技校': 2,
        '大专及以上': 3
    },
    '规律运动': {
        '否': 0,
        '是': 1
    },
    '糖尿病慢性并发症数量': {
        '<2个': 0,
        '≥2个': 1
    },
    '营养不良': {
        '否': 0,
        '是': 1
    },
    '抑郁': {
        '否': 0,
        '是': 1
    }
}

# 风险等级文本映射（中英文对照）
risk_level_map = {
    'zh-CN': {
        'low': '低风险',
        'moderate': '中风险',
        'high': '高风险'
    },
    'en': {
        'low': 'Low Risk',
        'moderate': 'Moderate Risk',
        'high': 'High Risk'
    }
}


@app.route('/', methods=['GET'])
def home():
    current_lang = session.get('current_lang', 'zh-CN')
    return render_template('index.html',
                           features=feature_map.values(),
                           feature_options=feature_options,
                           current_lang=current_lang)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 保存当前语言到session
        current_lang = request.form.get('current_lang', 'zh-CN')
        session['current_lang'] = current_lang

        # 获取表单数据
        data = request.form.to_dict()
        if 'current_lang' in data:
            del data['current_lang']  # 移除语言字段，防止干扰模型输入

        # 转换为模型输入格式
        input_data = {}
        for chinese_name, chinese_value in data.items():
            english_name = reverse_feature_map[chinese_name]
            input_data[english_name] = feature_options[chinese_name][chinese_value]

        # 模型预测
        if model:
            features_order = ['Age', 'educational level', 'Regular exercise',
                              'DCCN', 'Malnutrition', 'Depressive']
            df = pd.DataFrame([input_data], columns=features_order)
            proba = model.predict_proba(df)[0][1] * 100  # 转换为百分比
        else:
            # 模型加载失败时返回默认值
            proba = 50.0
            print("警告: 使用默认预测值，模型未成功加载")

        # 确定风险等级
        if proba < 50:
            risk_level = 'low'
        elif proba < 70:
            risk_level = 'moderate'
        else:
            risk_level = 'high'

        return render_template('result.html',
                               risk_probability=f"{proba:.1f}%",
                               input_data=data,
                               current_lang=current_lang,
                               risk_level=risk_level,
                               risk_level_text=risk_level_map[current_lang][risk_level])

    except Exception as e:
        print(f"预测错误: {e}")
        return render_template('error.html',
                               error_message=str(e),
                               current_lang=session.get('current_lang', 'zh-CN'))


if __name__ == '__main__':
    app.run(debug=True, port=5000)