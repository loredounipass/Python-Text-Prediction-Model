# -*- coding: utf-8 -*-
import json
import re
import numpy as np
import unicodedata
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, BatchNormalization, Bidirectional, Input, RepeatVector, Concatenate, Layer
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, LearningRateScheduler
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import KFold
import time
import sys
import pickle
from collections import OrderedDict
import spacy
import os
import math

# Cargar modelo spaCy
nlp = spacy.load("es_core_news_sm")
memory = OrderedDict()

# Configuracion
# Set FAST_RUN=True for quick testing (no GPU)
FAST_RUN = False
if FAST_RUN:
    VOCAB_SIZE = 300
    EMBEDDING_DIM = 64
    MAX_LEN = 100
    NUM_NEURONS = 100
    EPOCHS = 60
    BATCH_SIZE = 32
    INITIAL_LR = 1e-3
    DROPOUT_RATE = 0.3
    L2_RATE = 1e-4
    VALIDATION_SPLIT = 0.2
else:
    VOCAB_SIZE = 400    # Reducir el tamaño del vocabulario para evitar sobreajuste
    EMBEDDING_DIM = 128    # Reducir la dimensión de los embeddings
    MAX_LEN = 20    # Reducir la longitud máxima de las secuencias basado en análisis de datos
    NUM_NEURONS = 64    # Reducir el número de neuronas en la capa LSTM
    EPOCHS = 60    # Aumentar el número de épocas según especificación del usuario
    BATCH_SIZE = 32    # Reducir el tamaño del batch para mejor generalización
    INITIAL_LR = 1e-3    # Mantener la tasa de aprendizaje inicial
    DROPOUT_RATE = 0.5    # Aumentar la tasa de dropout para reducir sobreajuste
    L2_RATE = 1e-6    # Reducir aún más la regularización L2
    VALIDATION_SPLIT = 0.2    # Reducir ligeramente el porcentaje de validación
    KFOLDS = 5    # Reducir el número de particiones para validación cruzada

# Capa de atención personalizada
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
        # inputs.shape = (batch, time_steps, hidden_size)
        score = tf.nn.tanh(tf.tensordot(inputs, self.W, axes=1) + self.b)
        attention_weights = tf.nn.softmax(tf.tensordot(score, self.u, axes=1), axis=1)
        context_vector = attention_weights * inputs
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights

# Funciones auxiliares
def warmup_scheduler(epoch, lr):
    if epoch < 5:
        return lr + (INITIAL_LR - 1e-5) / 5
    return lr

def normalize_text(text):
    # Aumentar el límite de tokens que procesa spaCy
    nlp.max_length = 1000000
    doc = nlp(text.lower())
    # Mantener más tokens para preservar el significado
    tokens = [t.lemma_ for t in doc if t.is_alpha or t.is_digit]
    return ' '.join(tokens)

def augment_texts(texts, completions, metadata_list, intent_list):
    augmented_texts = []
    augmented_completions = []
    augmented_metadata = []
    augmented_intents = []
      
    # Spanish synonyms for augmentation (basic set)
    synonyms = {
        'hola': ['buenas', 'saludos', 'que tal'],
        'gracias': ['muchas gracias', 'mil gracias', 'te lo agradezco'],
        'viaje': ['ride', 'recorrido', 'traslado'],
        'conductor': ['chofer', 'driver', 'operador'],
        'pasajero': ['cliente', 'usuario', 'riders'],
        'seguro': ['protegido', 'resguardado', 'cuidado'],
        'app': ['aplicación', 'plataforma', 'sistema'],
        'ayuda': ['soporte', 'asistencia', 'auxilio'],
        'problema': ['incidente', 'inconveniente', 'dificultad'],
        'mapa': ['navegación', 'ruta', 'guía']
    }
    
    for i, t in enumerate(texts):
        words = t.lower().split()
        if len(words) > 0:
            # Technique 1: Random word deletion (10% chance per word)
            if np.random.random() < 0.3 and len(words) > 2:
                words = [w for w in words if np.random.random() > 0.1]
            
            # Technique 2: Random word swapping (10% chance)
            if np.random.random() < 0.2 and len(words) > 2:
                idx1, idx2 = np.random.choice(len(words), 2, replace=False)
                words[idx1], words[idx2] = words[idx2], words[idx1]
            
            # Technique 3: Synonym replacement (15% chance per word)
            new_words = []
            for word in words:
                if word in synonyms and np.random.random() < 0.15:
                    new_words.append(np.random.choice(synonyms[word]))
                else:
                    new_words.append(word)
            words = new_words
            
            # Technique 4: Add random Spanish words occasionally
            if np.random.random() < 0.1:
                extra_words = ['por favor', 'gracias', 'hola', 'ayuda', 'información']
                insert_pos = np.random.randint(0, len(words) + 1)
                words = words[:insert_pos] + [np.random.choice(extra_words)] + words[insert_pos:]
        
        augmented_texts.append(' '.join(words))
        augmented_completions.append(completions[i])
        augmented_metadata.append(metadata_list[i])
        augmented_intents.append(intent_list[i])
         
    # Also add some variations by changing case and adding punctuation
    for i, t in enumerate(texts[:len(texts)//2]):  # Only augment half to avoid too much noise
        # Add variation with punctuation
        if np.random.random() < 0.3:
            augmented_texts.append(t + '.')
            augmented_completions.append(completions[i])
            augmented_metadata.append(metadata_list[i])
            augmented_intents.append(intent_list[i])
        
        # Add variation with different casing
        if np.random.random() < 0.3:
            augmented_texts.append(t.upper())
            augmented_completions.append(completions[i])
            augmented_metadata.append(metadata_list[i])
            augmented_intents.append(intent_list[i])
    
    return texts + augmented_texts, completions + augmented_completions, metadata_list + augmented_metadata, intent_list + augmented_intents

# Cargar datos
with open('data.json', 'r', encoding='utf-8') as f:
    data = json.load(f)['conversations']

expanded_prompts = []
expanded_completions = []
expanded_metadata = []
expanded_intents = []  # Nueva lista para intents

# Procesar todo el dataset incluyendo metadata completa
for conv in data:
    completion = conv['completion'].strip()
    intent = conv.get('intent', '')
         
    # Crear metadata completa para cada conversación
    metadata = OrderedDict([
        ('prompt', conv.get('prompt', '')),
        ('completion', completion),
        ('intent', intent),
        #('pattern', conv.get('pattern', [])),
        ('task', conv.get('task', '')),
        ('meaning', conv.get('meaning', '')),
        ('examples', conv.get('examples', []))
    ])
         
    # Agregar prompt principal
    expanded_prompts.append(conv['prompt'])
    expanded_completions.append(completion)
    expanded_metadata.append(metadata.copy())
    expanded_intents.append(intent)
         
    # Agregar todos los patterns
    for pattern in conv['pattern']:
        expanded_prompts.append(pattern)
        expanded_completions.append(completion)
        expanded_metadata.append(metadata.copy())
        expanded_intents.append(intent)

# Normalizar textos
prompts = [normalize_text(p) for p in expanded_prompts]

# Aplicar augmentación de datos
prompts_aug, completions_aug, metadata_aug, intents_aug = augment_texts(prompts, expanded_completions, expanded_metadata, expanded_intents)
print(f"Dataset expandido: {len(prompts_aug)} ejemplos de entrenamiento")
print(f"Intents únicos: {set(intents_aug)}")

# Tokenización
tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token='<OOV>')
tokenizer.fit_on_texts(prompts_aug)
oov_index = tokenizer.word_index[tokenizer.oov_token]
seqs = tokenizer.texts_to_sequences(prompts_aug)
padded_inputs = pad_sequences(seqs, maxlen=MAX_LEN, padding='post')

# Preparar salidas para respuestas
distinct_responses = sorted(set(completions_aug))
resp2idx = {r: i for i, r in enumerate(distinct_responses)}
y_indices = np.array([resp2idx[c] for c in completions_aug])
y_onehot = to_categorical(y_indices, num_classes=len(distinct_responses))
print(f"Respuestas únicas: {len(distinct_responses)}")

# Preparar salidas para intents
distinct_intents = sorted(set(intents_aug))
intent2idx = {i: idx for idx, i in enumerate(distinct_intents)}
y_intent_indices = np.array([intent2idx[i] for i in intents_aug])
y_intent_onehot = to_categorical(y_intent_indices, num_classes=len(distinct_intents))
print(f"Intents únicos: {len(distinct_intents)}")

# Guardar tokenizer, mapeo de respuestas, mapeo de intents y metadata
# Exportación: Guarda el objeto Tokenizer para su uso posterior en inferencia
with open('tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)
# Exportación: Guarda el mapeo de respuestas a índices para la inferencia
with open('response_map.json', 'w', encoding='utf-8') as f:
    json.dump(resp2idx, f, ensure_ascii=False, indent=2)
# Exportación: Guarda el mapeo de intents a índices para la inferencia
with open('intent_map.json', 'w', encoding='utf-8') as f:
    json.dump(intent2idx, f, ensure_ascii=False, indent=2)
# Exportación: Guarda la metadata aumentada y los datos originales
with open('metadata.pkl', 'wb') as f:
    pickle.dump({
        'metadata': metadata_aug,
        'distinct_responses': distinct_responses,
        'distinct_intents': distinct_intents,
        'original_data': data
    }, f)
print("Archivos de configuración guardados")

# Modelo con atención y multitask
def build_model():
    inputs = Input(shape=(MAX_LEN,), name='input_layer')
    x = Embedding(VOCAB_SIZE, EMBEDDING_DIM, input_length=MAX_LEN, mask_zero=True, name='embedding_layer')(inputs)
    x = Bidirectional(LSTM(NUM_NEURONS, return_sequences=True, kernel_regularizer=l2(L2_RATE)), name='bilstm_1')(x)
    x = Dropout(DROPOUT_RATE)(x)
    x = BatchNormalization()(x)
    
    # Capa de atención
    attn_output, attn_weights = AttentionLayer(name='attention_layer')(x)
    
    # Rama 1: predicción de respuesta
    x1 = Dense(128, activation='relu', kernel_regularizer=l2(L2_RATE), name='dense_response_1')(attn_output)
    x1 = Dropout(DROPOUT_RATE)(x1)
    x1 = Dense(len(distinct_responses), activation='softmax', name='response_output')(x1)
    
    # Rama 2: predicción de intent
    x2 = Dense(128, activation='relu', kernel_regularizer=l2(L2_RATE), name='dense_intent_1')(attn_output)
    x2 = Dropout(DROPOUT_RATE)(x2)
    x2 = Dense(len(distinct_intents), activation='softmax', name='intent_output')(x2)
    
    model = Model(inputs=inputs, outputs=[x1, x2])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=INITIAL_LR),
        loss={'response_output': 'categorical_crossentropy', 'intent_output': 'categorical_crossentropy'},
        loss_weights={'response_output': 1.0, 'intent_output': 0.5},  # Dar más peso a la tarea principal
        metrics={'response_output': 'accuracy', 'intent_output': 'accuracy'}
    )
    return model

callbacks = [
    # Exportación: Guarda el mejor modelo durante el entrenamiento
    ModelCheckpoint('best_model.keras', save_best_only=True, monitor='val_loss', verbose=1),
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=3, min_lr=1e-6, verbose=1),
    LearningRateScheduler(warmup_scheduler)
]

# Entrenamiento final
print("Iniciando entrenamiento...")
final_model = build_model()
history = final_model.fit(
    padded_inputs, {'response_output': y_onehot, 'intent_output': y_intent_onehot},     
    validation_split=VALIDATION_SPLIT,
    epochs=EPOCHS,     
    batch_size=BATCH_SIZE,     
    callbacks=callbacks,     
    verbose=2)

# Exportación: Guarda el modelo final después del entrenamiento
final_model.save('chatbot_model_final.keras')
print("Modelo entrenado y guardado")

# -------------------- INTERACCION --------------------
# Funciones matemáticas
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

# Cargar recursos entrenados
print("Cargando modelo entrenado...")
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

# Cargar modelo entrenado
final_model = tf.keras.models.load_model('best_model.keras', custom_objects={'AttentionLayer': AttentionLayer})

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
    pad = pad_sequences([user_seq], maxlen=MAX_LEN, padding='post')
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

            # Incluir primer ejemplo si existe (para cualquier intent)
            # if conv.get('examples'):
            #     example = conv['examples'][0]
            #     response_text = f"{response_text}\n\nEjemplo:\n{example}"

            # Incluir patterns / frases similares si existen
            # patterns = conv.get('pattern', []) or conv.get('patterns', [])
            # if patterns:
            #     # mostrar hasta 5 patrones para no saturar la respuesta
            #     show = patterns[:5]
            #     patterns_text = '\n'.join(f"- {p}" for p in show)
            #     response_text = f"{response_text}\n\nFrases similares:\n{patterns_text}"

            return response_text

    return top_response

def simulate_typing(text, delay=0.03):
    for c in text:
        sys.stdout.write(c)
        sys.stdout.flush()
        time.sleep(delay)
    print()

# 💬 Interacción por consola
if __name__ == "__main__":
    print("🤖 Chatbot entrenado con dataset completo")
    print("Escribí algo (o 'salir' para terminar):")
         
    while True:
        user_input = input("\n👤 Vos: ")
        if user_input.lower() in ['salir', 'exit', 'quit']:
            print("🤖 Pura vida, hasta luego.")
            break
                 
        response = generate_response(user_input)
        print("🤖 Bot: ", end="")
        simulate_typing(response)