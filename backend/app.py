from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
import nltk
import pke
import string
from nltk.corpus import stopwords, wordnet
from collections import defaultdict
import random
from fpdf import FPDF
import io
import logging

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, resources={
    r"/generate": {"origins": "*"},
    r"/generate-pdf": {"origins": "*"}
})

# Download NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Initialize models
try:
    logger.info("Loading T5 model...")
    model = T5ForConditionalGeneration.from_pretrained('ramsrigouthamg/t5_squad_v1')
    tokenizer = T5Tokenizer.from_pretrained('ramsrigouthamg/t5_squad_v1')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Model loading failed: {str(e)}")
    raise

def get_synonyms(word):
    synonyms = set()
    try:
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonym = lemma.name().replace('_', ' ').lower()
                if synonym != word and len(synonym.split()) == 1:
                    synonyms.add(synonym)
        return list(synonyms)[:3]
    except:
        return []

def get_keywords(text):
    try:
        # Preprocess text - remove special chars and normalize
        text = ''.join(char for char in text if char.isalnum() or char in (' ', '.', '?', '!'))
        
        # Initialize extractor
        extractor = pke.unsupervised.MultipartiteRank()
        
        # Load document with enhanced processing
        extractor.load_document(
            input=text,
            language='en',
            normalization=None,  # Disable PKE's default normalization
            spacy_model='en_core_web_sm'  # Use better linguistic processing
        )
        
        # Candidate selection with more inclusive POS tags
        pos = {'NOUN', 'PROPN', 'ADJ', 'VERB'}
        stoplist = list(string.punctuation) + stopwords.words('english')
        extractor.candidate_selection(pos=pos, stoplist=stoplist)
        
        # More robust candidate weighting
        extractor.candidate_weighting(
            alpha=1.1,
            threshold=0.65,  # Lower threshold for more candidates
            method='average'
        )
        
        # Get best candidates with minimum length check
        keywords = [
            kw[0] for kw in extractor.get_n_best(n=15)
            if len(kw[0]) >= 3  # Minimum 3-character keywords
        ]
        
        # Fallback if no keywords found
        if not keywords:
            # Simple noun extraction fallback
            words = nltk.word_tokenize(text)
            pos_tags = nltk.pos_tag(words)
            keywords = [word for word, pos in pos_tags 
                       if pos in ('NN', 'NNS', 'NNP', 'NNPS') and len(word) >= 3]
            keywords = list(set(keywords))[:10]  # Deduplicate and limit
            
        return keywords
        
    except Exception as e:
        print(f"Keyword extraction error: {str(e)}")
        # Final fallback - return first few nouns
        words = nltk.word_tokenize(text)
        return [word for word in words if word[0].isupper()][:5]  # Capitalized words

def generate_distractors(answer, context, count=3):
    try:
        distractors = get_synonyms(answer.split()[0].lower())
        if len(distractors) < count:
            keywords = [kw for kw in get_keywords(context) 
                       if kw.lower() != answer.lower() and len(kw.split()) == 1]
            distractors.extend(keywords[:count-len(distractors)])
        distractors = list(set(d.capitalize() for d in distractors if d))
        random.shuffle(distractors)
        return distractors[:count]
    except:
        return [f"Option {i+1}" for i in range(count)]

def get_question(context, answer):
    try:
        text = f"context: {context} answer: {answer}"
        encoding = tokenizer.encode_plus(
            text, 
            max_length=384, 
            truncation=True, 
            return_tensors="pt"
        ).to(device)
        
        outs = model.generate(
            **encoding,
            max_length=72,
            num_beams=5,
            early_stopping=True
        )
        question = tokenizer.decode(outs[0], skip_special_tokens=True)
        return question.replace("question:", "").strip()
    except Exception as e:
        logger.error(f"Question generation failed: {str(e)}")
        return ""

def generate_pdf(mcqs):
    try:
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        
        for i, mcq in enumerate(mcqs, 1):
            pdf.multi_cell(0, 10, f"Q{i}: {mcq['question']}", 0, 'L')
            
            for j, option in enumerate(mcq['options']):
                if option == mcq['answer']:
                    pdf.set_text_color(0, 128, 0)  # Green for correct answer
                pdf.cell(0, 10, f"   {chr(65+j)}) {option}", 0, 1)
                pdf.set_text_color(0, 0, 0)  # Reset to black
            
            pdf.ln(5)
        
        return pdf
    except Exception as e:
        logger.error(f"PDF generation failed: {str(e)}")
        raise

@app.route('/generate', methods=['POST'])
def generate_mcqs():
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400

        text = data['text'].strip()
        if len(text) < 50:
            return jsonify({'error': 'Text too short (min 50 characters)'}), 400

        options_count = min(int(data.get('options_count', 4)), 5)
        keywords = get_keywords(text)
        
        if not keywords:
            return jsonify({'error': 'Could not extract keywords from text'}), 400

        mcqs = []
        for answer in keywords[:10]:  # Limit to 10 questions
            question = get_question(text, answer)
            if not question:
                continue
                
            distractors = generate_distractors(answer, text, options_count-1)
            options = list({answer.capitalize(), *[d for d in distractors if d]})
            
            if len(options) < 2:
                continue
                
            random.shuffle(options)
            mcqs.append({
                'question': question,
                'answer': answer.capitalize(),
                'options': options[:options_count]
            })

        if not mcqs:
            return jsonify({'error': 'Generated 0 questions (try more detailed text)'}), 400

        return jsonify({'mcqs': mcqs})

    except Exception as e:
        logger.error(f"Generation error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/generate-pdf', methods=['POST'])
def generate_pdf_endpoint():
    try:
        data = request.get_json()
        if not data or 'mcqs' not in data:
            return jsonify({'error': 'No questions provided'}), 400

        pdf = generate_pdf(data['mcqs'])
        pdf_bytes = pdf.output(dest='S').encode('latin1')
        
        response = make_response(pdf_bytes)
        response.headers['Content-Type'] = 'application/pdf'
        response.headers['Content-Disposition'] = 'attachment; filename=questions.pdf'
        return response

    except Exception as e:
        logger.error(f"PDF endpoint error: {str(e)}")
        return jsonify({'error': 'Failed to generate PDF'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)