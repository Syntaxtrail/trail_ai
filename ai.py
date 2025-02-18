import random
import re
import json
import time
from difflib import SequenceMatcher
from datetime import datetime
import os
import math


TRANSFORMER_AVAILABLE = False
try:
    import torch
    from transformers import AutoModel, AutoTokenizer, pipeline
    TRANSFORMER_AVAILABLE = True
    print("Transformer library detected! Enhanced capabilities available.")
except ImportError:
    print("Transformer library not detected. Using basic similarity methods.")
    print("For enhanced capabilities, install transformers with: pip install transformers torch")

MAX_LINE_LENGTH = 512
MAX_QUESTIONS = 1000
MATCH_THRESHOLD = 0.55
CONTEXT_HISTORY_SIZE = 8
LOGS_DIRECTORY = "conversation_logs"
USER_PROFILE_PATH = "user_profile.json"
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

if not os.path.exists(LOGS_DIRECTORY):
    os.makedirs(LOGS_DIRECTORY)

class SimpleTFIDF:
    def __init__(self):
        self.vectorizer = None
        self.tfidf_matrix = None
        self.idf = {}
        self.vocab = {}
        self.documents = []
    
    def _tokenize(self, text):
        return [w.lower() for w in re.findall(r'\b\w+\b', text) if w.lower() not in self.stop_words and len(w) > 2]
    
    def fit_transform(self, documents):
        self.documents = documents
        self.stop_words = {'a', 'an', 'the', 'and', 'or', 'but', 'is', 'are', 'was', 'were', 'be', 'been',
                    'to', 'of', 'in', 'for', 'with', 'by', 'about', 'against', 'between', 'into'}
        
        doc_count = {}
        for i, doc in enumerate(documents):
            tokens = self._tokenize(doc)
            for token in set(tokens):
                doc_count[token] = doc_count.get(token, 0) + 1
        
        N = len(documents)
        self.vocab = {token: idx for idx, token in enumerate(doc_count.keys())}
        self.idf = {token: math.log((N + 1) / (count + 1)) + 1 for token, count in doc_count.items()}
        
        self.tfidf_matrix = [self._calculate_tfidf(doc) for doc in documents]
        
        return self.tfidf_matrix
    
    def _calculate_tfidf(self, document):
        tokens = self._tokenize(document)
        if not tokens:
            return [0] * len(self.vocab)
        
        tf = {}
        for token in tokens:
            tf[token] = tf.get(token, 0) + 1
        
        for token in tf:
            tf[token] /= len(tokens)
        
        tfidf_vector = [0] * len(self.vocab)
        for token, freq in tf.items():
            if token in self.vocab:
                idx = self.vocab[token]
                tfidf_vector[idx] = freq * self.idf.get(token, 0)
        
        return tfidf_vector
    
    def transform(self, documents):
        if isinstance(documents, str):
            documents = [documents]
        
        return [self._calculate_tfidf(doc) for doc in documents]


class TransformerSimilarity:
    def __init__(self, model_name=MODEL_NAME):
        if not TRANSFORMER_AVAILABLE:
            raise ImportError("Transformer libraries not available")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.embeddings = None
        self.documents = []
    
    def _mean_pooling(self, model_output, attention_mask):

        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def get_embedding(self, text):

        encoded_input = self.tokenizer(text, padding=True, truncation=True, max_length=128, return_tensors='pt')
        encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}
        

        with torch.no_grad():
            model_output = self.model(**encoded_input)
        
        
        embedding = self._mean_pooling(model_output, encoded_input['attention_mask'])
        
        
        embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
        
        return embedding.cpu()
    
    def fit_transform(self, documents):
        self.documents = documents
        
        batch_size = 32
        all_embeddings = []
        
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i+batch_size]
            embeddings = self.get_embedding(batch_docs)
            all_embeddings.append(embeddings)
        
        self.embeddings = torch.cat(all_embeddings, dim=0)
        return self.embeddings
    
    def transform(self, documents):
        if isinstance(documents, str):
            documents = [documents]
        
        return self.get_embedding(documents)


def cosine_similarity(vec1, vec2):
    if TRANSFORMER_AVAILABLE and isinstance(vec1, torch.Tensor) and isinstance(vec2, torch.Tensor):
        return torch.nn.functional.cosine_similarity(vec1, vec2).item()
    else:
     
        dot_product = sum(a*b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a*a for a in vec1))
        norm2 = math.sqrt(sum(b*b for b in vec2))
        
        if norm1 == 0 or norm2 == 0:
            return 0
        
        return dot_product / (norm1 * norm2)


class UserProfile:
    def __init__(self):
        self.data = self._load_profile()
        self.sentiment_analyzer = None
        
    def _load_profile(self):
        try:
            with open(USER_PROFILE_PATH, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {
                "favorite_topics": {},
                "conversation_count": 0,
                "total_interactions": 0,
                "last_seen": None,
                "sentiment_history": []
            }
    
    def save_profile(self):
        with open(USER_PROFILE_PATH, 'w') as f:
            json.dump(self.data, f, indent=2)
    
    def update_interaction(self, user_input):
        self.data["total_interactions"] += 1
        self.data["last_seen"] = datetime.now().isoformat()
        
        keywords = extract_keywords(user_input)
        for keyword in keywords:
            self.data["favorite_topics"][keyword] = self.data["favorite_topics"].get(keyword, 0) + 1
        
        self.save_profile()
    
    def get_favorite_topics(self, limit=3):
        topics = sorted(self.data["favorite_topics"].items(), key=lambda x: x[1], reverse=True)
        return [topic for topic, count in topics[:limit]]
    
    def is_returning_user(self):
        return self.data["conversation_count"] > 0
    
    def start_conversation(self):
        self.data["conversation_count"] += 1
        self.save_profile()
    
    def analyze_sentiment(self, text):
        try:
        
            if TRANSFORMER_AVAILABLE and self.sentiment_analyzer is not None:
                result = self.sentiment_analyzer(text)
                if result[0]['label'] == 'POSITIVE':
                    sentiment = "positive"
                elif result[0]['label'] == 'NEGATIVE':
                    sentiment = "negative"
                else:
                    sentiment = "neutral"
            else:
                
                positive_words = {'good', 'great', 'excellent', 'awesome', 'happy', 'love', 'like', 'thanks', 'thank'}
                negative_words = {'bad', 'terrible', 'awful', 'hate', 'dislike', 'poor', 'angry', 'sad', 'frustrated'}
                
                words = set(extract_keywords(text.lower()))
                pos_count = len(words.intersection(positive_words))
                neg_count = len(words.intersection(negative_words))
                
                sentiment = "positive" if pos_count > neg_count else "negative" if neg_count > pos_count else "neutral"
        except Exception as e:
            print(f"Error in sentiment analysis: {e}")
            sentiment = "neutral"
        
        self.data["sentiment_history"].append(sentiment)
        if len(self.data["sentiment_history"]) > 20:
            self.data["sentiment_history"] = self.data["sentiment_history"][-20:]
        
        self.save_profile()
        return sentiment
    
    def load_sentiment_analyzer(self):
        if TRANSFORMER_AVAILABLE:
            try:
                self.sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", 
                                                 device=0 if torch.cuda.is_available() else -1)
                return True
            except Exception as e:
                print(f"Could not load sentiment analyzer: {e}")
        return False


def load_training_data(filename):
    questions, responses = [], []
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            for line in file:
                if len(questions) >= MAX_QUESTIONS:
                    break
                parts = line.strip().split('|', 1)
                if len(parts) == 2:
                    questions.append(parts[0].strip().lower())
                    responses.append(parts[1].strip())
    except FileNotFoundError:
        print("Error: Training data file not found.")
    except IOError:
        print("Error: Could not read the training data file.")
    return questions, responses


def save_user_interaction(user_input, ai_response, context):
    today = datetime.now().strftime("%Y-%m-%d")
    log_file = os.path.join(LOGS_DIRECTORY, f"{today}.jsonl")
    
    try:
        with open(log_file, "a", encoding='utf-8') as file:
            record = {
                "timestamp": datetime.now().isoformat(),
                "user_input": user_input,
                "ai_response": ai_response,
                "context_used": bool(context.get("recent_questions"))
            }
            file.write(json.dumps(record) + "\n")
    except IOError:
        print("Warning: Could not save interaction to history file")


def similarity_score(str1, str2):
    return SequenceMatcher(None, str1, str2).ratio()


def extract_keywords(input_str):
    stop_words = {'a', 'an', 'the', 'and', 'or', 'but', 'is', 'are', 'was', 'were', 'be', 'been',
                 'to', 'of', 'in', 'for', 'with', 'by', 'about', 'against', 'between', 'into',
                 'through', 'during', 'before', 'after', 'above', 'below', 'from', 'up', 'down',
                 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there',
                 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most',
                 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than',
                 'too', 'very', 'can', 'will', 'just', 'should', 'now'}
    words = re.findall(r'\b\w+\b', input_str.lower())
    return [word for word in words if word not in stop_words and len(word) > 2]


def find_best_match(input_str, similarity_obj, questions, conversation_context):
    if TRANSFORMER_AVAILABLE and isinstance(similarity_obj, TransformerSimilarity):
        input_embed = similarity_obj.transform(input_str.lower())
        
       
        similarity_scores = []
        for i in range(len(similarity_obj.embeddings)):
            question_embed = similarity_obj.embeddings[i:i+1]
            score = torch.nn.functional.cosine_similarity(input_embed, question_embed).item()
            similarity_scores.append(score)
    else:

        input_vector = similarity_obj.transform(input_str.lower())[0]
        similarity_scores = [cosine_similarity(input_vector, similarity_obj.tfidf_matrix[i]) 
                             for i in range(len(similarity_obj.tfidf_matrix))]
    
    candidates = [(score, idx) for idx, score in enumerate(similarity_scores) if score >= MATCH_THRESHOLD]
    candidates.sort(reverse=True)
    
    if candidates and candidates[0][0] >= MATCH_THRESHOLD:
        return candidates[0][1]
    
    if candidates and "recent_questions" in conversation_context:
        context_enhanced_candidates = []
        for score, idx in candidates:
            context_boost = 0
            for i, recent_q in enumerate(conversation_context["recent_questions"]):
                recency_weight = 1.0 - (i * 0.15)
                
                if TRANSFORMER_AVAILABLE and isinstance(similarity_obj, TransformerSimilarity):
                    recent_embed = similarity_obj.transform(recent_q.lower())
                    question_embed = similarity_obj.embeddings[idx:idx+1]
                    context_similarity = torch.nn.functional.cosine_similarity(recent_embed, question_embed).item()
                else:
                    recent_vector = similarity_obj.transform(recent_q.lower())[0]
                    question_vector = similarity_obj.transform(questions[idx].lower())[0]
                    context_similarity = cosine_similarity(recent_vector, question_vector)
                
                context_boost += context_similarity * recency_weight * 0.25
            
            enhanced_score = score + context_boost
            if enhanced_score >= MATCH_THRESHOLD:
                context_enhanced_candidates.append((enhanced_score, idx))
        
        if context_enhanced_candidates:
            context_enhanced_candidates.sort(reverse=True)
            return context_enhanced_candidates[0][1]
    
    return -1


def small_talk(input_str, user_profile):
    small_talk_responses = {
        "hello": [
            "Hey there! How can I assist you today?", 
            "Hello! What can I help you with?", 
            "Hi! How can I make your day better?"
        ],
        "hi": [
            "Hello! How's your day going?",
            "Hi there! What can I do for you?",
            "Hey! Nice to chat with you!"
        ],
        "hey": [
            "Hello! What's on your mind?",
            "Hey there! How can I help?",
            "Hi! I'm here to assist you!"
        ],
        "how are you": [
            "I'm doing well, thanks for asking! How about you?", 
            "I'm functioning perfectly! How are you feeling today?", 
            "All systems are go! How's your day been?"
        ],
        "what's your name": [
            "I'm SyntaxTrail AI. Nice to meet you!", 
            "You can call me SyntaxTrail AI. What should I call you?", 
            "I'm SyntaxTrail AI, your virtual assistant."
        ],
        "what can you do": [
            "I can answer questions, provide information, and chat with you!", 
            "I'm here to assist with information, have conversations, and help with various queries.", 
            "I can help answer your questions, engage in conversation, and provide assistance when needed."
        ],
        "who created you": [
            "I was created by SafwanGanz from the SyntaxTrail team.", 
            "My creator is SafwanGanz from SyntaxTrail.", 
            "I'm a product of SafwanGanz at SyntaxTrail."
        ],
        "thank you": [
            "You're welcome! Is there anything else I can help with?",
            "Happy to help! Let me know if you need anything else.",
            "My pleasure! Any other questions?"
        ],
        "thanks": [
            "You're welcome! Anything else on your mind?",
            "No problem at all! What else can I do for you?",
            "Glad I could help! Feel free to ask more questions."
        ],
    }
    
    input_str = input_str.lower().strip()
    
    if input_str in ["hello", "hi", "hey"] and user_profile.is_returning_user():
        favorite_topics = user_profile.get_favorite_topics()
        if favorite_topics:
            personalized_responses = [
                f"Welcome back! I remember you were interested in {', '.join(favorite_topics)}. How can I help today?",
                f"Great to see you again! Last time we talked about {favorite_topics[0]}. What's on your mind now?",
                f"Hello! Based on our previous chats, would you like to talk more about {favorite_topics[0]}?"
            ]
            return random.choice(personalized_responses)
    
    for key in small_talk_responses:
        if input_str == key:
            return random.choice(small_talk_responses[key])
    
    for key in small_talk_responses:
        if key in input_str:
            return random.choice(small_talk_responses[key])
    
    return None


def generate_fallback_response(context, user_input, user_profile):
    general_fallbacks = [
        "I'm not quite sure about that. Could you provide more details?",
        "I don't have enough information to answer that properly. Can you rephrase or clarify?",
        "I'm still learning and don't have an answer for that yet. Is there something else I can help with?",
        "That's an interesting question! Unfortunately, I don't have a good answer right now.",
        "I'm not certain I understood correctly. Could you try asking in a different way?"
    ]
    
    recent_sentiments = user_profile.data["sentiment_history"][-3:] if user_profile.data["sentiment_history"] else []
    if recent_sentiments and all(s == "negative" for s in recent_sentiments):
        return "I notice you might be frustrated. I'm sorry I'm not able to help as well as you'd like. Perhaps we can try a different approach or topic?"
    
    if context.get("recent_questions"):
        input_keywords = extract_keywords(user_input)
        most_related_question = None
        highest_similarity = 0
        
        for recent_q in context["recent_questions"]:
            recent_keywords = extract_keywords(recent_q)
            similarity = similarity_score(user_input, recent_q)
            if similarity > highest_similarity and similarity > 0.3:
                highest_similarity = similarity
                most_related_question = recent_q
        
        if most_related_question:
            return f"I'm not sure about that specific question. But we were talking about '{most_related_question}' - would you like to continue discussing that?"
    
    favorite_topics = user_profile.get_favorite_topics()
    if favorite_topics and random.random() < 0.3:
        return f"I'm not sure how to answer that. Based on our previous conversations, would you like to talk more about {favorite_topics[0]}?"
    
    return random.choice(general_fallbacks)


def respond(input_str, questions, responses, context, similarity_obj, user_profile):
    user_profile.update_interaction(input_str)
    user_profile.analyze_sentiment(input_str)
    
    if "recent_questions" not in context:
        context["recent_questions"] = []
    if "recent_responses" not in context:
        context["recent_responses"] = []
    
    if "input_history" not in context:
        context["input_history"] = []
    context["input_history"].insert(0, input_str)
    if len(context["input_history"]) > CONTEXT_HISTORY_SIZE:
        context["input_history"].pop()
    
    small_talk_response = small_talk(input_str, user_profile)
    if small_talk_response:
        response = small_talk_response
    else:
        index = find_best_match(input_str, similarity_obj, questions, context)
        
        if index != -1:
            response = responses[index]
            context["recent_questions"].insert(0, questions[index])
            if len(context["recent_questions"]) > CONTEXT_HISTORY_SIZE:
                context["recent_questions"].pop()
        else:
            response = generate_fallback_response(context, input_str, user_profile)
    
    context["recent_responses"].insert(0, response)
    if len(context["recent_responses"]) > CONTEXT_HISTORY_SIZE:
        context["recent_responses"].pop()
    
    typing_delay = min(0.1 * len(response) / 40, 2.0)
    time.sleep(typing_delay)
    
    print(f"SyntaxTrail AI: {response}")
    
    save_user_interaction(input_str, response, context)
    
    return response


def show_help():
    print("\n--- SyntaxTrail AI Help ---")
    print("1. Type 'hello' to start the conversation.")
    print("2. Ask me any question you'd like to discuss.")
    print("3. Type 'bye' or 'exit' to end our conversation.")
    print("4. Type 'help' to see this menu again.")
    print("5. You can ask about my capabilities by typing 'what can you do'")
    print("6. Feel free to share feedback on my responses!")
    print("7. Type 'stats' to see conversation statistics.")
    print("-------------------------")


def show_stats(context, user_profile):
    print("\n--- Conversation Statistics ---")
    session_duration = round((time.time() - context["session_start"]) / 60, 1)
    print(f"Current session duration: {session_duration} minutes")
    print(f"Total conversations: {user_profile.data['conversation_count']}")
    print(f"Total interactions: {user_profile.data['total_interactions']}")
    
    favorite_topics = user_profile.get_favorite_topics(5)
    if favorite_topics:
        print(f"Your favorite topics: {', '.join(favorite_topics)}")
    
    sentiment_count = {"positive": 0, "negative": 0, "neutral": 0}
    for sentiment in user_profile.data["sentiment_history"]:
        sentiment_count[sentiment] += 1
    print(f"Your recent mood: {max(sentiment_count, key=sentiment_count.get)}")
    print("-----------------------------")


def initialize_similarity_model(questions):
    if TRANSFORMER_AVAILABLE:
        try:
            print("Loading transformer model. This may take a moment...")
            transformer = TransformerSimilarity()
            transformer.fit_transform(questions)
            print("Transformer model loaded successfully!")
            return transformer
        except Exception as e:
            print(f"Error loading transformer model: {e}")
            print("Falling back to TF-IDF similarity...")
    

    print("Using TF-IDF similarity model")
    tfidf = SimpleTFIDF()
    tfidf.fit_transform(questions)
    return tfidf


def main():
    print("\n" + "="*50)
    print("  SyntaxTrail AI - Your Intelligent Assistant")
    print("="*50)
    
    print("\nInitializing...")
    questions, responses = load_training_data("train_data.dat")
    if not questions:
        print("No training data found. Using only built-in responses.")
    
    print("Loading NLP models...")
    similarity_obj = initialize_similarity_model(questions)
    
    print("Loading user profile...")
    user_profile = UserProfile()
    if TRANSFORMER_AVAILABLE:
        if user_profile.load_sentiment_analyzer():
            print("Enhanced sentiment analysis loaded.")
        else:
            print("Using basic sentiment analysis.")
    else:
        print("Using basic sentiment analysis.")
    
    user_profile.start_conversation()
    
    context = {"session_start": time.time()}
    
    if user_profile.is_returning_user():
        last_seen = user_profile.data["last_seen"]
        if last_seen:
            last_seen = datetime.fromisoformat(last_seen)
            days_since_last_visit = (datetime.now() - last_seen).days
            
            if days_since_last_visit < 1:
                print("\nWelcome back! It's great to see you again today.")
            elif days_since_last_visit == 1:
                print("\nWelcome back after a day! How have you been?")
            else:
                print(f"\nWelcome back! It's been {days_since_last_visit} days since we last chatted.")
        else:
            print("\nWelcome back! It seems like this is your first visit.")
    else:
        print("\nWelcome to SyntaxTrail AI! I'm here to chat and answer your questions.")
    
    print("Type 'help' for assistance or just say hello to begin.")
    
    while True:
        user_input = input("\nYou: ").strip()
        
        if not user_input:
            print("SyntaxTrail AI: I didn't catch that. Could you please try again?")
            continue
        
        if user_input.lower() in ["bye", "exit", "quit", "goodbye"]:
            session_duration = round((time.time() - context["session_start"]) / 60, 1)
            print(f"SyntaxTrail AI: Goodbye! We've chatted for {session_duration} minutes. Hope to see you again soon!")
            break
        elif user_input.lower() == "help":
            show_help()
        elif user_input.lower() == "stats":
            show_stats(context, user_profile)
        else:
            respond(user_input, questions, responses, context, similarity_obj, user_profile)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nSyntaxTrail AI: Session terminated. Have a nice day!")
    except Exception as e:
        print(f"\n\nAn unexpected error occurred: {e}")
        print("SyntaxTrail AI has encountered a problem and needs to restart.")
        with open(os.path.join(LOGS_DIRECTORY, "error_log.txt"), "a") as f:
            f.write(f"{datetime.now().isoformat()}: {str(e)}\n")
