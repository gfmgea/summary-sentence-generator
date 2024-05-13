import nltk
import random
import string

# Baixando os recursos do NLTK
nltk.download('punkt')
nltk.download('stopwords')

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

# Leitura do arquivo de texto
with open('C:\\Faroeste Caboclo.txt', 'r', encoding='utf-8') as file:
    corpus = file.read()

# Pré-processamento do corpus
def preprocess_text(text):
    sentences = sent_tokenize(text)  # Segmentação por sentenças
    words = word_tokenize(text)      # Segmentação por unidades (Tokenização)
    
    # Capitalização e remoção de stopwords
    words = [word.lower() for word in words if word.isalnum() and word.lower() not in stopwords.words('english')]
    
    return ' '.join(words)

# Aplicando o pré-processamento ao corpus
preprocessed_corpus = preprocess_text(corpus)

# Treinamento do modelo de linguagem MLE
def train_mle_model(corpus):
    bigrams = list(nltk.bigrams(corpus.split()))  # Utilizando bigramas
    model = nltk.ConditionalFreqDist(bigrams)
    return model

# Treinando o modelo MLE
mle_model = train_mle_model(preprocessed_corpus)

# Gerando novas sentenças usando o modelo treinado
def generate_sentence(model, seed_word, length=10):
    sentence = seed_word.lower()
    for _ in range(length - 1):
        next_word = model[sentence.split()[-1]].max()
        sentence += f' {next_word}'
    return sentence.capitalize()

# Escolhendo uma palavra aleatória do corpus como semente
seed_word = random.choice(preprocessed_corpus.split())

# Gerando uma nova sentença
new_sentence = generate_sentence(mle_model, seed_word, length=10)

# Exibindo a nova sentença gerada
print("Nova frase gerada:", new_sentence)
