import requests
import json
import time
from datetime import datetime

# لیست API‌های مدل‌ها
api_endpoints = [
    "https://arvancloudai.ir/gateway/models/Qwen3-Coder-480b-A35B-Instruct/NPoUR5bGxvSvcjrBexerUsgdqO9KMJDeYVRQB8n31HNxLIosf4acOj9597uMDJpeU1PRo1jU4YuvBakdK3u_UNNnHd6WRbyVYX1P9pDZml1prZFcVL-trLvpVYhlc-oBRj2tSD-11VyVLfujxyGaVVZc8pawaKyagzNFFktCeDrzSu-Jmqp6EBNsfkzKUJYYYS8-mL7qa5jP_awVfmdmSN3ky5tbNdH4NzBjehhuXDmEG4G2wIT18H7RYz-8WlnseeZPNnkvX9O4RiIRkAXVdawAFpuY60aGu5jh6IBGG2wBRQ/v1/chat/completions",
    "https://arvancloudai.ir/gateway/models/GPT-OSS-120B/2vzNZIDUZO7Qo-NcBDgNBBTkFo_WK0EG7v3zY7v0sFkLsAZAoXua2ogrhf_mt6k1RkDK4PFe1_3_eI2tK6KX4pDC_TWqRjixm6lvY7LlWs8cl3HvuXMaQtpTN7QAOB5thVwkoNheYqZBFtWe7_8VaaU6zHwFzaHDa_NMDtB3F5STXc1nlzZ-TQUMAp4Was-9mcG9ZJM5LQzsmyRe8oPqzs9POkP5AomkqoikMvsBN4WXA7jz-EyKxi2Ol-cXfw/v1/chat/completions",
    "https://arvancloudai.ir/gateway/models/DeepSeek-R1-Distill-Qwen-32b/yc9WMKmzdd7IRN8jI5Z3dBgcUa2sRW8RaTDiXtS3Smm345yEtTGzSEKYvxwlut8HV28MSdHltwCbhmng8yS9RvnLobpmJq_cas0XcMl7rLNNwiucXbeUNV6maLhkKnoVFxkdSS3uyYijjrbfe3ux1jZ3iUcGqLXvxiiRIxm8lF6QmTX_zEVLkMy6DpkGEyRDl06B-bJvtPd8dHzoh3oZEVET5bLY4k6a5-L37_PmrFW8j9X7qKkrKL2cU5t72NA4nJP5953hLFse1pWvwgjdwz4G0AlMpLQr5JSAg28r/v1/chat/completions",
    "https://arvancloudai.ir/gateway/models/GPT-5/_7wmPfrbXtRLRZK4tEYkmy4JTiGS7sYtRAL2C3plSsBBoO9jRNTIk3AutqxWbohCfNdfTJMunIpCQmuh2i8X1gHqqfyYs3-XNVzDAlcteeOWGDKo7DHr2kjxhStSuK18xTkRg47g7_eOnenucfU_Y6iy7-BkF5NMsaBE5PEn-bFHuimUF13gnNLm1E3QY0Q0jEtzNpNYhkkuAVWJ76oeqglOFZc4wZ9vh9K2KLKqylQ/v1/chat/completions"
]

# توکن API
api_key = "REDACTED"

# سوال
question = "چرا آسمان آبی است؟"

# داده‌های درخواست
payload = {
    "model": "Qwen3-Coder-480b-A35B-Instruct",  # می‌توانید این را برای هر مدل تغییر دهید
    "messages": [{"role": "user", "content": question}],
    "max_tokens": 300,
    "temperature": 0.7
}

# لیست نتایج
results = []

# تست هر API
for i, endpoint in enumerate(api_endpoints):
    try:
        print(f"\n--- در حال تست API {i+1} ({endpoint.split('/')[-2]}) ---")
        
        headers = {
            "Authorization": f"apikey {api_key}",
            "Content-Type": "application/json"
        }
        
        response = requests.post(endpoint, headers=headers, data=json.dumps(payload), timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            content = result.get("choices", [{}])[0].get("message", {}).get("content", "خطا: محتوا پیدا نشد")
            results.append({
                "model": endpoint.split('/')[-2],
                "status": "OK",
                "content": content[:200] + "..." if len(content) > 200 else content,
                "time": datetime.now().strftime("%H:%M:%S")
            })
            print(f"✅ موفق: {content[:100]}...")
        else:
            results.append({
                "model": endpoint.split('/')[-2],
                "status": f"خطا ({response.status_code})",
                "content": response.text,
                "time": datetime.now().strftime("%H:%M:%S")
            })
            print(f"❌ خطا: {response.status_code} - {response.text}")
            
    except Exception as e:
        results.append({
            "model": endpoint.split('/')[-2],
            "status": "خطا (خطا در اتصال)",
            "content": str(e),
            "time": datetime.now().strftime("%H:%M:%S")
        })
        print(f"❌ خطا: {e}")

# نمایش نتایج نهایی
print("\n" + "="*60)
print("📊 نتایج نهایی:")
print("="*60)
for r in results:
    print(f"Model: {r['model']}")
    print(f"Status: {r['status']}")
    print(f"Time: {r['time']}")
    print(f"Content: {r['content']}")
    print("-" * 50)
