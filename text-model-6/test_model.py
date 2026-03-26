import json
import pickle
import numpy as np
import re
import math
import sys
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Layer
import spacy

# Define AttentionLayer (copy from model.py)
class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(shape=(input_shape[-1], input_shape[-1]),
                                 initializer='random_normal',
                                 trainable=True)
        self.b = self.add_weight(shape=(input_shape[-1],),
                                 initializer='zeros',
                                 trainable=True)
        self.u = self.add_weight(shape=(input_shape[-1], 1),
                                 initializer='random_normal',
                                 trainable=True)
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs):
        score = tf.nn.tanh(tf.tensordot(inputs, self.W, axes=1) + self.b)
        attention_weights = tf.nn.softmax(tf.tensordot(score, self.u, axes=1), axis=1)
        context_vector = attention_weights * inputs
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights

# Suppress tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Load resources
print("Loading model and resources...")
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)
with open('response_map.json', 'r', encoding='utf-8') as f:
    resp2idx = json.load(f)
    distinct_responses = [None] * len(resp2idx)
    for r, i in resp2idx.items():
        distinct_responses[int(i)] = r
with open('intent_map.json', 'r', encoding='utf-8') as f:
    intent2idx = json.load(f)
    distinct_intents = [None] * len(intent2idx)
    for i, idx in intent2idx.items():
        distinct_intents[int(idx)] = i
with open('metadata.pkl', 'rb') as f:
    saved_data = pickle.load(f)
    metadata_aug = saved_data['metadata']
    original_data = saved_data['original_data']

# Load model
final_model = tf.keras.models.load_model('best_model.keras', custom_objects={'AttentionLayer': AttentionLayer})
print("Model loaded successfully.")

# Get OOV index
oov_index = tokenizer.word_index[tokenizer.oov_token]

# Helper functions (same as in model.py)
def evaluate_math_expression(expr):
    try:
        expr = expr.replace('×', '*').replace('x', '*').replace('÷', '/').replace('^', '**')
        expr = re.sub(r"[^0-9+\-*/.()^ ]", "", expr)
        if expr:
            result = eval(expr, {'__builtins__': None}, {'math': math})
            return float(result) if isinstance(result, (int, float)) else result
    except:
        return None

def contains_math_expression(text):
    patterns = [
        r'\d+\s*[\+\-\*/x×÷]\s*\d+',
        r'cuanto es (.*)\?',
        r'calcula (.*)',
        r'resultado de (.*)',
        r'\d+\s*\^\s*\d+',
        r'raiz cuadrada de \d+',
        r'\d+\s*!'
    ]
    return any(re.search(p, text.lower()) for p in patterns)

def normalize_text(text):
    nlp = spacy.load("es_core_news_sm")
    nlp.max_length = 1000000
    doc = nlp(text.lower())
    tokens = [t.lemma_ for t in doc if t.is_alpha or t.is_digit]
    return ' '.join(tokens)

def generate_response(user_text):
    # Verificar si es una expresión matemática
    if contains_math_expression(user_text):
        expr = re.search(r'(?:cuanto es|calcula|resultado de)\s*(.*?)\??$', user_text.lower())
        math_expr = expr.group(1) if expr else user_text
        result = evaluate_math_expression(math_expr)
        if result is not None:
            if isinstance(result, float):
                result = int(result) if result.is_integer() else round(result, 4)
            return f"El resultado de {math_expr} es {result}"
        return "No pude calcular esa expresión matemática. ¿Podrías formularla de otra manera?"
         
    # Procesar con el modelo de NLP
    user_seq = tokenizer.texts_to_sequences([normalize_text(user_text)])[0]
    if not user_seq:
        return "Lo siento, no te entendí."
         
    # Verificar ratio de palabras desconocidas
    oov_ratio = sum(1 for i in user_seq if i == oov_index) / len(user_seq)
    if oov_ratio > 0.4:
        return "Como asistente virtual, solo puedo ayudarte con informacion relacionada a saferide y transporte seguro. ¿Puedes formular tu pregunta de otra manera?"
         
    # Hacer predicción
    # Get the expected sequence length from the model's input shape
    expected_seq_length = final_model.input_shape[1]  # Assuming shape (None, seq_length)
    print(f"Expected sequence length from model: {expected_seq_length}")
    pad = pad_sequences([user_seq], maxlen=expected_seq_length, padding='post')
    print(f"Padded sequence shape: {pad.shape}")
    preds = final_model.predict(pad, verbose=0)
    response_pred = preds[0][0]  # Primera salida es la de respuesta
    intent_pred = preds[1][0]    # Segunda salida es la de intent
    top_response_index = np.argmax(response_pred)
    top_response = distinct_responses[top_response_index] if top_response_index < len(distinct_responses) else None
    top_intent_index = np.argmax(intent_pred)
    top_intent = distinct_intents[top_intent_index] if top_intent_index < len(distinct_intents) else None
         
    if top_response is None:
        return "Lo siento, no te entendí."
         
    # Buscar en los datos originales para incluir ejemplos y patterns (frases similares)
    for conv in original_data:
        if conv.get('completion', '').strip() == top_response.strip():
            response_text = top_response
            return response_text

    return top_response

# Test cases
test_inputs = [
    "Hola",
    "¿Cómo puedo solicitar un viaje?",
    "Gracias",
    "Buenos dias",
    "¿Qué es SafeRide?",
    "Como me registro?",
    "¿Cuánto es 2+2?",
    "¿Puedo chatear con el conductor?",
    "Adiós",
    "Quiero saber sobre los planes"
]

print("\n=== Testing model ===")
for inp in test_inputs:
    response = generate_response(inp)
    print(f"Input: {inp}")
    print(f"Response: {response}\n")