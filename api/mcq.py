from http.server import BaseHTTPRequestHandler
import json
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

model = None
tokenizer = None

def load_model():
    global model, tokenizer
    model = T5ForConditionalGeneration.from_pretrained('t5-small')
    tokenizer = T5Tokenizer.from_pretrained('t5-small')

class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = json.loads(self.rfile.read(content_length))
        
        if model is None:
            load_model()
        
        # Your existing generation logic here
        input_text = f"generate questions: {post_data['text']}"
        input_ids = tokenizer.encode(input_text, return_tensors="pt")
        outputs = model.generate(input_ids)
        questions = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps({"questions": questions}).encode())