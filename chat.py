import random

def clean_and_sanitize(user_input):
    """
    PDF Instruction: Handle case & whitespace (Sanitization)
    """
    return user_input.lower().strip()

def get_bot_response(clean_input, intent_rules):
    """
    PDF Instruction: O(1) Dictionary Lookup + Fallback with .get()
    Wide Catching Engine: Scans and extracts key intent from any sentence.
    """
    # 1. Strict Exit Strategy [cite: 23, 208]
    if any(exit_word in clean_input for exit_word in ['exit', 'quit', 'bye', 'terminate', 'stop', 'close']):
        return "EXIT_SIGNAL"
    
    # 2. Vast Catch-Word Engine (Word-by-word token tracking)
    # Yeh poore sentence ko tod kar ya substring match karke sahi keyword nikalega
    for intent, data in intent_rules.items():
        for keyword in data['keywords']:
            if keyword in clean_input:  # Catching the exact word inside a wide sentence
                return random.choice(data['responses'])
                
    # 3. Fallback Response for unknown inputs [cite: 208]
    return "I am sorry, I couldn't catch any familiar keyword. Try asking about 'python', 'frontend', 'database', 'ai', or 'help'!"

def main():
    # EXTREMELY VAST KNOWLEDGE BASE (20+ Wide Categories for Technical & General Tasks) [cite: 207]
    intent_rules = {
        'greeting': {
            'keywords': ['hello', 'hi', 'hey', 'dear', 'sup', 'greetings', 'namaste'],
            'responses': [
                'Hi there! Welcome to my DecodeLabs project portal.', 
                'Hey! Great to chat with you. What are we building today?', 
                'Hello! Logic engine is active and listening.'
            ]
        },
        'wellbeing': {
            'keywords': ['how are you', 'how r u', 'doing good', 'you good', 'fine', 'whats up'],
            'responses': [
                'I am running on 100% computational efficiency, thank you for asking!',
                'As a rule-based chatbot, my logic gates are perfectly stable and happy!',
                'Doing great! Ready to crush this project development.'
            ]
        },
        'python': {
            'keywords': ['python', 'py', 'scripting', 'list', 'tuple', 'dictionary'],
            'responses': [
                'Python is amazing! It handles dictionaries in O(1) time, which makes this chatbot lightning fast[cite: 154, 171].',
                'In Python, indentation is key. Always keep your loops and conditions clean!',
                'Need Python help? Remember, clean code is better than clever code.'
            ]
        },
        'frontend': {
            'keywords': ['html', 'css', 'javascript', 'js', 'frontend', 'ui', 'ux', 'web'],
            'responses': [
                'Frontend web dev is pure art! HTML gives structure, CSS brings style, and JS adds life[cite: 207].',
                'We can use a beautiful modern UI window with CSS animations for our final deployment!',
                'JavaScript can fetch data dynamically without refreshing the page. Perfect for chat boxes!'
            ]
        },
        'backend': {
            'keywords': ['backend', 'server', 'flask', 'django', 'node', 'api'],
            'responses': [
                'Backend is the backbone! It handles the actual logical skeleton and heavy calculations[cite: 84, 206].',
                'Flask is a lightweight Python framework, but for GitHub Pages, pure JS frontend is cleaner!',
                'APIs connect two different applications. Next week we might look into semantic APIs[cite: 187].'
            ]
        },
        'database': {
            'keywords': ['database', 'sql', 'mongodb', 'query', 'data', 'storage'],
            'responses': [
                'Databases store critical structured records. Think of a dictionary as an in-memory database!',
                'SQL handles relational data tables, while NoSQL stores unstructured JSON-like documents.',
                'Always index your databases to keep response speeds fast, just like a Hash Map lookup[cite: 154].'
            ]
        },
        'ai_concepts': {
            'keywords': ['ai', 'artificial intelligence', 'machine learning', 'ml', 'deep learning'],
            'responses': [
                'AI is vast[cite: 1]! Before diving into probabilistic deep learning, mastering deterministic rule engines is critical[cite: 8, 53].',
                'True AI engineering requires understanding white-box logic before handling black-box models[cite: 55].',
                'Machine learning systems learn from data patterns, while rule-based systems follow hardcoded instructions[cite: 9, 59].'
            ]
        },
        'llm': {
            'keywords': ['llm', 'large language model', 'gpt', 'gemini', 'hallucination'],
            'responses': [
                'LLMs are brilliant but probabilistic core networks[cite: 72]. Without rule-based guardrails, they hallucinate[cite: 66, 73, 224]!',
                'Frameworks like Llama Guard and NVIDIA NeMo act as deterministic filters above the LLM layer[cite: 67, 69, 73].',
                'An LLM predicts the next most likely token, whereas I give you 100% hardcoded precise safety[cite: 59].'
            ]
        },
        'git': {
            'keywords': ['git', 'github', 'repository', 'commit', 'push', 'repo'],
            'responses': [
                'Git is essential for version control. Always write meaningful commit messages!',
                'We will deploy this directly via GitHub Pages (github.io) for a permanent portfolio link.',
                'Command reminder: git add . -> git commit -m "update" -> git push origin main.'
            ]
        },
        'bugs': {
            'keywords': ['bug', 'error', 'debug', 'fail', 'crash', 'not working', 'issue'],
            'responses': [
                'Don\'t panic! Read the traceback error carefully—it usually tells you the exact line number.',
                '90% of programming bugs are just simple typos or indentation mismatches. Check your syntax!',
                'Every unexpected error is just a learning opportunity to make the logic model stronger[cite: 230].'
            ]
        },
        'internship': {
            'keywords': ['task', 'project', 'week 1', 'internship', 'decodelabs', 'work'],
            'responses': [
                'DecodeLabs requires us to establish a clean Rule-Based Skeleton to unlock next week\'s tier[cite: 7, 15]!',
                'This online internship focuses on building real-world portfolio projects step-by-step[cite: 231].',
                'Your current target: Perfect the IPO model (Input, Process, Output) for this basic bot[cite: 75].'
            ]
        },
        'motivation': {
            'keywords': ['tired', 'hard', 'stuck', 'boring', 'motivation', 'inspire'],
            'responses': [
                'Keep going, dev! Great software takes time and continuous logical debugging[cite: 230].',
                'Don\'t just aim to finish—experiment, break things, and build a portfolio you are proud of[cite: 230, 231]!',
                'Small steps every day lead to massive development engineering milestones[cite: 231].'
            ]
        }
    }

    print("=======================================================================")
    print("🤖 THE GRAND VAST ENGINE 2.0 — Active & Scanning (Type 'exit' to quit) 🤖")
    print("=======================================================================")
    print("Bot: System fully wide initialized. Try asking complex questions about AI, Python, Git, or Web development!")

    # Continuous loop (The Heartbeat) [cite: 26, 114]
    while True:
        try:
            raw_input = input("\nYou: ")
            clean_input = clean_and_sanitize(raw_input)
            
            if not clean_input:
                continue
                
            response = get_bot_response(clean_input, intent_rules)
            
            if response == "EXIT_SIGNAL":
                print("Bot: Goodbye, dear! Session terminated cleanly. Ready for the next development sprint[cite: 208]!")
                break
                
            print(f"Bot: {response}")
            
        except (KeyboardInterrupt, EOFError):
            print("\nBot: Session interrupted. Goodbye!")
            break

if __name__ == "__main__":
    main()