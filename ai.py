import random
import re
import json
import time
from difflib import SequenceMatcher

MAX_LINE_LENGTH = 512
MAX_QUESTIONS = 200
MATCH_THRESHOLD = 0.55  
CONTEXT_HISTORY_SIZE = 5

def load_training_data(filename):
    questions = []
    responses = []
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            for line in file:
                if len(questions) >= MAX_QUESTIONS:
                    break
                parts = line.strip().split('|', 1)
                if len(parts) == 2:
                    questions.append(parts[0].strip().lower())
                    responses.append(parts[1].strip())
        print(f"Loaded {len(questions)} question-response pairs")
    except FileNotFoundError:
        print("Error: Training data file not found.")
    except IOError:
        print("Error: Could not read the training data file.")
    return questions, responses

def save_user_interaction(user_input, ai_response):
    try:
        with open("interaction_history.jsonl", "a", encoding='utf-8') as file:
            record = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "user_input": user_input,
                "ai_response": ai_response
            }
            file.write(json.dumps(record) + "\n")
    except IOError:
        print("Warning: Could not save interaction to history file")

def similarity_score(str1, str2):

    return SequenceMatcher(None, str1, str2).ratio()

def semantic_similarity(words1, words2):
    
    if not words1 or not words2:
        return 0
    

    weighted_words1 = {w: 0.5 + min(len(w) / 10, 0.5) for w in words1}
    weighted_words2 = {w: 0.5 + min(len(w) / 10, 0.5) for w in words2}
    

    intersection_score = 0
    for w1 in words1:
        if w1 in words2:
            intersection_score += weighted_words1[w1]
        else:

            for w2 in words2:
                if (w1.startswith(w2) or w2.startswith(w1)) and min(len(w1), len(w2)) > 3:
                    intersection_score += weighted_words1[w1] * 0.7  
                    break
    
    total_weight1 = sum(weighted_words1.values())
    total_weight2 = sum(weighted_words2.values())
    
    if total_weight1 == 0 or total_weight2 == 0:
        return 0
    
    
    return intersection_score / ((total_weight1 + total_weight2) / 2)

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

def find_best_match(input_str, questions, conversation_context):
    best_score = 0
    best_idx = -1
    candidates = []
    

    normalized_input = input_str.lower().strip()
    for idx, question in enumerate(questions):
        normalized_question = question.lower().strip()
        

        if normalized_input == normalized_question:
            return idx
        

        direct_score = similarity_score(normalized_input, normalized_question)
        

        input_keywords = extract_keywords(normalized_input)
        question_keywords = extract_keywords(normalized_question)
        semantic_score = semantic_similarity(input_keywords, question_keywords)
        
        
        combined_score = (direct_score * 0.4) + (semantic_score * 0.6)
        
        if combined_score >= MATCH_THRESHOLD:
            candidates.append((combined_score, semantic_score, idx))
    

    candidates.sort(reverse=True)
    

    if candidates and candidates[0][0] >= MATCH_THRESHOLD:
        return candidates[0][2]
    

    if candidates and "recent_questions" in conversation_context:
        context_enhanced_candidates = []
        for score, sem_score, idx in candidates:
            context_boost = 0
            
            
            for i, recent_q in enumerate(conversation_context["recent_questions"]):
                recency_weight = 1.0 - (i * 0.2)
                recent_keywords = extract_keywords(recent_q)
                question_keywords = extract_keywords(questions[idx])
                context_similarity = semantic_similarity(recent_keywords, question_keywords)
                context_boost += context_similarity * recency_weight * 0.3
            

            enhanced_score = score + context_boost
            if enhanced_score >= MATCH_THRESHOLD:
                context_enhanced_candidates.append((enhanced_score, sem_score, idx))
        

        if context_enhanced_candidates:
            context_enhanced_candidates.sort(reverse=True)
            return context_enhanced_candidates[0][2]
    

    return -1

def small_talk(input_str):
    small_talk_responses = {
        "hello": ["Hey there! How can I assist you today?", 
                 "Hello! What can I help you with?", 
                 "Hi! How can I make your day better?"],
        "hi": ["Hello! How's your day going?",
              "Hi there! What can I do for you?",
              "Hey! Nice to chat with you!"],
        "hey": ["Hello! What's on your mind?",
               "Hey there! How can I help?",
               "Hi! I'm here to assist you!"],
        "how are you": ["I'm doing well, thanks for asking! How about you?", 
                       "I'm functioning perfectly! How are you feeling today?", 
                       "All systems are go! How's your day been?"],
        "what's your name": ["I'm SyntaxTrail AI. Nice to meet you!", 
                           "You can call me SyntaxTrail AI. What should I call you?", 
                           "I'm SyntaxTrail AI, your virtual assistant."],
        "what can you do": ["I can answer questions, provide information, and chat with you!", 
                          "I'm here to assist with information, have conversations, and help with various queries.", 
                          "I can help answer your questions, engage in conversation, and provide assistance when needed."],
        "who created you": ["I was created by SafwanGanz from the SyntaxTrail team.", 
                          "My creator is SafwanGanz from SyntaxTrail.", 
                          "I'm a product of SafwanGanz at SyntaxTrail."],
        "thank you": ["You're welcome! Is there anything else I can help with?",
                    "Happy to help! Let me know if you need anything else.",
                    "My pleasure! Any other questions?"],
        "thanks": ["You're welcome! Anything else on your mind?",
                  "No problem at all! What else can I do for you?",
                  "Glad I could help! Feel free to ask more questions."],
    }
    
    input_str = input_str.lower().strip()
    

    for key in small_talk_responses:
        if input_str == key:
            return random.choice(small_talk_responses[key])
    

    for key in small_talk_responses:
        if key in input_str:
            return random.choice(small_talk_responses[key])
    

    return None

def generate_fallback_response(context, user_input):
    general_fallbacks = [
        "I'm not quite sure about that. Could you provide more details?",
        "I don't have enough information to answer that properly. Can you rephrase or clarify?",
        "I'm still learning and don't have an answer for that yet. Is there something else I can help with?",
        "That's an interesting question! Unfortunately, I don't have a good answer right now.",
        "I'm not certain I understood correctly. Could you try asking in a different way?"
    ]
    
    # Check if we can suggest a related topic from conversation history
    if context.get("recent_questions"):
        input_keywords = extract_keywords(user_input)
        most_related_question = None
        highest_similarity = 0
        
        for recent_q in context["recent_questions"]:
            recent_keywords = extract_keywords(recent_q)
            similarity = semantic_similarity(input_keywords, recent_keywords)
            if similarity > highest_similarity and similarity > 0.3:
                highest_similarity = similarity
                most_related_question = recent_q
        
        if most_related_question:
            return f"I'm not sure about that specific question. But we were talking about '{most_related_question}' - would you like to continue discussing that?"

    return random.choice(general_fallbacks)

def respond(input_str, questions, responses, context):

    if "recent_questions" not in context:
        context["recent_questions"] = []
    if "recent_responses" not in context:
        context["recent_responses"] = []
    

    if "input_history" not in context:
        context["input_history"] = []
    context["input_history"].insert(0, input_str)
    if len(context["input_history"]) > CONTEXT_HISTORY_SIZE:
        context["input_history"].pop()
    
    
    small_talk_response = small_talk(input_str)
    if small_talk_response:
        response = small_talk_response
    else:

        index = find_best_match(input_str, questions, context)
        
        if index != -1:
            response = responses[index]

            context["recent_questions"].insert(0, questions[index])
            if len(context["recent_questions"]) > CONTEXT_HISTORY_SIZE:
                context["recent_questions"].pop()
        else:
            response = generate_fallback_response(context, input_str)
    

    context["recent_responses"].insert(0, response)
    if len(context["recent_responses"]) > CONTEXT_HISTORY_SIZE:
        context["recent_responses"].pop()
    
    
    typing_delay = min(0.1 * len(response) / 40, 2.0)
    time.sleep(typing_delay)
    
    print(f"SyntaxTrail AI: {response}")
    

    save_user_interaction(input_str, response)
    
    return response

def show_help():
    print("\n--- SyntaxTrail AI Help ---")
    print("1. Type 'hello' to start the conversation.")
    print("2. Ask me any question you'd like to discuss.")
    print("3. Type 'bye' or 'exit' to end our conversation.")
    print("4. Type 'help' to see this menu again.")
    print("5. You can ask about my capabilities by typing 'what can you do'")
    print("6. Feel free to share feedback on my responses!")
    print("-------------------------")

def main():
    print("\n" + "="*50)
    print("  SyntaxTrail AI - Your Intelligent Assistant")
    print("="*50)
    
    print("\nInitializing...")
    questions, responses = load_training_data("train_data.dat")
    if not questions:
        print("No training data found. Using only built-in responses.")
    
    print("Warming up neural networks...")
    time.sleep(1)
    print("Loading conversational models...")
    time.sleep(0.5)
    print("Ready!")
    
    context = {"session_start": time.time()}
    print("\nWelcome to SyntaxTrail AI! I'm here to chat and answer your questions.")
    print("Type 'help' for assistance or just say hello to begin.")
    
    while True:
        user_input = input("\nYou: ")
        user_input = user_input.strip()
        
        if not user_input:
            print("SyntaxTrail AI: I didn't catch that. Could you please try again?")
            continue
        
        if user_input.lower() in ["bye", "exit", "quit", "goodbye"]:
            session_duration = round((time.time() - context["session_start"]) / 60, 1)
            print(f"SyntaxTrail AI: Goodbye! We've chatted for {session_duration} minutes. Hope to see you again soon!")
            break
        elif user_input.lower() == "help":
            show_help()
        else:
            respond(user_input, questions, responses, context)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nSyntaxTrail AI: Session terminated. Have a nice day!")
    except Exception as e:
        print(f"\n\nAn unexpected error occurred: {e}")
        print("SyntaxTrail AI has encountered a problem and needs to restart.")
