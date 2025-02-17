import random
import re
from difflib import get_close_matches

MAX_LINE_LENGTH = 512
MAX_QUESTIONS = 20

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
    except FileNotFoundError:
        print("Error: Training data file not found.")
    except IOError:
        print("Error: Could not read the training data file.")
    return questions, responses

def trim_spaces(input_str):
    return input_str.strip()

def extract_keywords(input_str):
    
    return [word for word in re.findall(r'\b\w+\b', input_str.lower()) if len(word) > 2]

def find_best_match(input_str, questions):
    user_keywords = set(extract_keywords(input_str))
    matches = []
    for q in questions:
        question_keywords = set(extract_keywords(q))
        score = len(user_keywords.intersection(question_keywords)) / len(user_keywords.union(question_keywords))
        matches.append((score, q))
    matches.sort(reverse=True)
    if matches and matches[0][0] > 0.4:  # Arbitrary threshold for match
        return questions.index(matches[0][1])
    return -1

def small_talk(input_str):
    small_talk_responses = {
        "hello": ["Hey! How can I help?", "Hello! What's up?", "Hi there!"],
        "how are you": ["I'm doing well, thanks for asking! How about you?", "I'm functioning great! How are you?", "All systems are go! How about you?"],
        "what's your name": ["I'm SyntaxTrail AI. Nice to meet you!", "You can call me SyntaxTrail AI.", "I'm SyntaxTrail AI."],
        "what can you do": ["I can answer questions, provide information, and chat with you!", "I'm here to assist with any information you need.", "I can help answer your queries."],
        "who created you": ["I was created by SafwanGanz from the SyntaxTrail team.", "My creator is SafwanGanz from SyntaxTrail.", "I'm a product of SafwanGanz at SyntaxTrail."],
    }
    input_str = input_str.lower()
    for key in small_talk_responses:
        if key in input_str:
            return random.choice(small_talk_responses[key])
    return None

def respond(input_str, questions, responses, context):
    index = find_best_match(input_str, questions)
    if index != -1:
        response = responses[index]
        context["previous_question"] = questions[index]
        print(f"SyntaxTrail AI: {response}")
    elif small_talk_response := small_talk(input_str):
        print(f"SyntaxTrail AI: {small_talk_response}")
    else:
        if "previous_question" in context:
            print(f"SyntaxTrail AI: I'm not sure about that. Let's talk about '{context['previous_question']}' again.")
        else:
            print("SyntaxTrail AI: I'm not sure how to answer that. Can you rephrase or ask something else?")

def show_help():
    print("\n--- SyntaxTrail AI Help ---")
    print("1. Type 'hello' to start the conversation.")
    print("2. Type 'how are you' to check my status.")
    print("3. Type 'bye' to exit the conversation.")
    print("4. Type 'help' to see this help menu again.")
    print("-------------------------")

def main():
    questions, responses = load_training_data("train_data.dat")
    if not questions:
        print("No training data found. Exiting...")
        return

    context = {}
    print("\nWelcome to SyntaxTrail AI! Type 'help' for assistance.")
    
    while True:
        user_input = input("\nYou: ")
        user_input = trim_spaces(user_input).lower()

        if user_input == "bye":
            print("SyntaxTrail AI: Goodbye! See you later.")
            break
        elif user_input == "help":
            show_help()
        else:
            respond(user_input, questions, responses, context)

if __name__ == "__main__":
    main()
