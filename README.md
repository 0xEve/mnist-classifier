# MNIST digit classifier — Flask + PyTorch

نفس بنية الموديل اللي دربناه في Colab (`784 → 256 → 128 → 64 → 10`)، متغلف في موقع بسيط: بترفع صورة رقم، والـ backend (Flask) بيحمّل الموديل المدرّب ويرجّعلك التوقع.

## هيكل المشروع

```
mnist-classifier/
├── app.py                 # Flask app + endpoint التوقع
├── model.py                # نفس كلاس الموديل بتاعك بالظبط
├── model_weights.pth        # لازم تضيفه إنت (الخطوة اللي تحت)
├── requirements.txt
├── vercel.json              # لو هتنشره على Vercel
├── templates/index.html
└── static/style.css
```

## 1. جيب أوزان الموديل بتاعك من Colab

بعد التدريب، في Colab اعمل:

```python
torch.save(model.state_dict(), "model_weights.pth")
```

نزّل الملف الناتج، وحطه في جذر المشروع (نفس مكان `app.py`) — الاسم لازم يكون بالظبط `model_weights.pth`.

## 2. شغّله محليًا

```bash
pip install -r requirements.txt
python app.py
```

افتح `http://127.0.0.1:5000` في المتصفح.

## 3. النشر أونلاين

**Vercel** (زي المثال اللي بعتهولي): `vercel.json` جاهز في المشروع. بس خد بالك: `torch` مكتبة كبيرة (مئات الـ MB)، والباقة المجانية من Vercel بتحدد حجم الـ serverless function — ممكن يفشل الـ deploy بسبب الحجم.

لو حصل كده، البدائل دي بتتعامل مع تطبيقات Flask/PyTorch الكاملة براحة أكتر:
- **Render** (خطة مجانية، أسهل حل لتطبيق Flask كامل)
- **Railway**
- **Hugging Face Spaces** (مصمم أصلاً لاستضافة موديلات ML)

كلهم بياخدوا نفس الملفات دي من غير أي تعديل تقريبًا.
