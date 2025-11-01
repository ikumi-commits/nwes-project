
import google.generativeai as genai

# 直接 APIキーを渡す
genai.configure(api_key="AIzaSyD_M80QOdUIWm7e7nFhQo9NiE5MZuYngps")

# 利用可能なモデル一覧を取得
models = genai.list_models()

for m in models:
    print(m.name)
