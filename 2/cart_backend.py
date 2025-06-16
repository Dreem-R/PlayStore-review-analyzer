from flask import Flask, request, jsonify
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from flask_cors import CORS
import google.generativeai as genai

GMAIL_ADDRESS = "rkiri1209@gmail.com"
GMAIL_APP_PASSWORD = "ioxfwyhxonxcrsze"
GEMINI_API_KEY = "AIzaSyBb4FH7WLVBS40jzvTUA9RhaFAKk_ZHLcg"

genai.configure(api_key=GEMINI_API_KEY)

app = Flask(__name__)
CORS(app)

def generate_funny_message(user_name, product_name, product_link):
    prompt = f"""
Write a short, funny, and persuasive email to {user_name} who abandoned '{product_name}' in their cart.
Make it playful and clever. Include this exact sentence at the end:
"Click here to grab it now: {product_link}"
Keep it under 150 words.
"""
    try:
        model = genai.GenerativeModel("gemini-1.5-pro")
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print("❌ Gemini error:", e)
        return (
            f"Hey {user_name}, you left '{product_name}' behind! Don’t worry, it misses you too 😉.\n"
            f"Click here to grab it now: {product_link}"
        )


def send_email(user_name, to_email, item):
    message = generate_funny_message(user_name, item['name'], item.get('link', '#'))
    msg = MIMEMultipart()
    msg['From'] = GMAIL_ADDRESS
    msg['To'] = to_email
    msg['Subject'] = f"🛒 Still thinking about {item['name']}?"

    msg.attach(MIMEText(message, 'plain'))

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(GMAIL_ADDRESS, GMAIL_APP_PASSWORD)
            server.sendmail(GMAIL_ADDRESS, to_email, msg.as_string())
            print(f"✅ Email sent to {user_name} ({to_email})")
    except Exception as e:
        print(f"❌ Email failed for {to_email}: {e}")

@app.route('/track', methods=['POST'])
def track_cart():
    data = request.json
    name = data.get("name")
    email = data.get("email")
    items = data.get("items", [])
    if not name or not email or not items:
        return jsonify({"error": "Missing required fields"}), 400

    for item in items:
        send_email(name, email, item)

    return jsonify({"status": "emails_sent", "count": len(items)})

@app.route('/')
def home():
    return "🛒 Cart Recovery Server is Running"

if __name__ == '__main__':
    app.run(debug=True)
