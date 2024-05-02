import os
import nltk
nltk.download('punkt')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import json
import numpy as np
from tensorflow import keras
import random
import streamlit as st
from gtts import gTTS
import datetime  # Import datetime module

# Get an object
lemmatizer = WordNetLemmatizer()

# Load your model
model = keras.models.load_model('model.h5')

# Load intents data
data_file = open('intents.json').read()
intents = json.loads(data_file)

words=[]
classes = []
documents = []
ignore_words = ['?', '!']

for intent in intents['intents']:
    for pattern in intent['patterns']:
        # take each word and tokenize it
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        # adding documents
        documents.append((w, intent['tag']))
        # adding classes to our class list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))
classes = sorted(list(set(classes)))

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words)
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s:
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))

def predict_class(sentence, model):
    # filter out predictions below a threshold
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def get_books_by_tag(tag):
    # Get all books for a specific tag from intents data
    tag_books = []
    for intent in intents['intents']:
        if intent['tag'].lower() == tag.lower():
            tag_books.extend(intent['responses'])
           
    return tag_books

def get_highest_rated_book(tag):
    # Get all books for a specific tag from intents data
    tag_books = []
    for intent in intents['intents']:
        if intent['tag'].lower() == tag.lower():
            tag_books.extend(intent['responses'])  # Assuming 'responses' contain dictionaries
           
    # Ensure tag_books is a list of dictionaries
    if isinstance(tag_books, list):
        # Filter out items without a 'Rate' key
        rated_books = [book for book in tag_books if 'Rate' in book]
        
        if rated_books:
            # Sort the books by the 'Rate' key in descending order
            sorted_books = sorted(rated_books, key=lambda x: x['Rate'], reverse=True)
            return sorted_books[0]  # Return the highest rated book
        else:
            return None  # No books with a 'Rate' key found
    else:
        return None  # Invalid data structure for tag_books


def get_next_book(tag, current_index):
    # Get the next book for a specific tag based on the current index
    tag_books = get_books_by_tag(tag)
    return tag_books[current_index] if current_index < len(tag_books) else None


def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']== tag):
            response = random.choice(i['responses'])
            other_books = i.get('other_books', [])
            return response, other_books

def format_response(response):
    # Define prompts for the user
    prompts = [
        "Let me know if you want another book",
        "Do you want another suggestion",
        "Would you like to explore more books"
    ]

    highest_rated_prompts = [
        "  That book is considered one of the best in its category, with glowing reviews and high ratings.",
        "That book has earned a stellar reputation within its category, frequently receiving top ratings and recommendations."
   
    ]

    if response:
        if isinstance(response, str):
            formatted_response = ""
            for line in response.split(':'):
                formatted_response += f"{line.strip()}: \n"
            return formatted_response.strip()
        elif isinstance(response, dict):
            formatted_response = ""
            if 'Image' in response:
                formatted_response += f"<img src='{response['Image']}' alt='Book Cover' style='width: 100px; height:150px;'><br>"

            formatted_response += f"<b>Book:</b> {response['Book']}<br>"
            formatted_response += f"<b>Author:</b> {response['Author']}<br>"
            formatted_response += f"<b>Feedback:</b> {response['Feedback']}<br>"
            formatted_response += f"<b>Rate:</b> {response['Rate']}<br>"
            formatted_response += f"<b>Published Year:</b> {response['Published Year']}<br>"
            # Increment the book counter
            st.session_state.book_counter = st.session_state.get('book_counter', 0) + 1
            # Check if the response is about the second book
            if st.session_state.get('book_counter', 0) == 2 and 'Rate' in response and 'highest_book_message_sent' not in st.session_state:
                # Select a random prompt for the highest-rated book
                highest_rated_prompt = random.choice(highest_rated_prompts)
                category = intent['tag']
                formatted_response = f"Certainly! in {category} {highest_rated_prompt}<br><br>" + formatted_response
                
                # Set the flag to indicate that the message has been sent
                st.session_state.highest_book_message_sent = True
            # Select a random prompt for the user
            prompt = random.choice(prompts)
            category = intent['tag']
            formatted_response += f"<br>{prompt} in {category}\n"
           
            return formatted_response
    else:
        return "I'm sorry, I didn't understand that."

# Streamlit GUI
## Streamlit GUI
def display_initial_message():
    # Check if the initial message has been displayed
    if not st.session_state.get('initial_message_displayed', False):
        # Define the initial message with suggestions
        st.markdown("<h1 style='text-align: center; display: flex; align-items: center;'>How can I help you today? <img src='https://img.freepik.com/vector-premium/bot-chat-decir-hola-robots-programados-hablar-clientes-linea_68708-622.jpg?w=826' style='height: 70px;width:70px; margin-left: 5px;margin-bottom:10px;' /></h1>", unsafe_allow_html=True)


        # Define the suggestions as clickable buttons
        suggestions = [
            
            # Add more suggestions as needed
        ]
        for suggestion in suggestions:
            if st.button(suggestion):
                # If the user clicks on a suggestion, handle the corresponding action or response
                handle_suggestion(suggestion)
        # Set the flag to indicate that the initial message has been displayed
        st.session_state.initial_message_displayed = True

#
# Display the initial message
display_initial_message()


# File path for chat history
chat_history_file = "chat_history.txt"

# Ensure chat history file exists
if not os.path.exists(chat_history_file):
    with open(chat_history_file, "w") as file:
        file.write("")

# Clear chat history if it's a new session
if "new_session" not in st.session_state:
    st.session_state.new_session = False

    with open(chat_history_file, "w") as file:
        file.write("")




# Chat History
chat_history = st.empty()
import streamlit as st


# Initialize session state
if 'current_book_index' not in st.session_state:
    st.session_state.current_book_index = 0

# Define colors for user and chatbot messages
user_color = '#9744d1'
chatbot_color = '#e5e2e9'

# Streamlit GUI

# Initialize session state
if 'current_book_index' not in st.session_state:
    st.session_state.current_book_index = 0


# File path for chat history
chat_history_file = "chat_history.txt"

# Ensure chat history file exists
if not os.path.exists(chat_history_file):
    with open(chat_history_file, "w") as file:
        file.write("")

# Clear chat history if it's a new session
if "new_session" not in st.session_state:
    st.session_state.new_session = False

    with open(chat_history_file, "w") as file:
        file.write("")

# Chat History
chat_history = st.empty()

# Initialize response variable
response = None


# File path for chat history
chat_history_dir = "chat_history"


# Ensure chat history directory exists
if not os.path.exists(chat_history_dir):
    os.makedirs(chat_history_dir)

# Function to save conversation to a file
def save_conversation(conversation):
    current_date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_name = os.path.join(chat_history_dir, f"conversation_{current_date}.txt")
    with open(file_name, "w") as file:
        file.write(conversation)
        


# Function to list all conversation files in the sidebar
def list_conversations():
    conversation_files = [f for f in os.listdir(chat_history_dir) if f.startswith("conversation_")]
    return conversation_files



# Ensure chat history directory exists
if not os.path.exists(chat_history_dir):
    os.makedirs(chat_history_dir)

def load_conversation(selected_conversation):
    file_path = os.path.join(chat_history_dir, selected_conversation)
    with open(file_path, "r") as file:
        conversation_lines = file.readlines()

    formatted_conversation = ""
    for line in conversation_lines:
        if line.startswith("You:"):
            formatted_conversation += f"<div style='display:flex; align-items:center;'><img src='https://cdn-icons-png.flaticon.com/128/15486/15486795.png' style='vertical-align:middle; margin-right:5px; height: 20px; width: 20px;'><div style='background-color:{user_color}; padding:10px; margin-bottom:12px; border-radius:15px;'> {line}</div></div>"
        else:
             formatted_conversation += f"<div style='display:flex; align-items:flex-start; ;'><img src='https://cdn-icons-png.flaticon.com/128/11189/11189417.png'' style='vertical-align:middle; margin-right:5px; height: 20px; width: 20px;'><div style='background-color:{chatbot_color}; padding:10px; margin-bottom:12px; border-radius:15px;'> {line}</div></div>"
    st.markdown(formatted_conversation, unsafe_allow_html=True)


# Function to begin a new session
def begin_new_session():
    # Clear session state variables
    st.session_state.clear()
    # Set the flag to indicate a new session
    st.session_state.new_session = True
    # Clear chat history file
    chat_history_file = "chat_history.txt"
    with open(chat_history_file, "w") as file:
        file.write("")
    # Display initial message
    display_initial_message()

# Add "New chat" button to the sidebar
if st.sidebar.button("New chat"):
    begin_new_session()


# Streamlit GUI

# Streamlit GUI

# Streamlit GUI

# Streamlit GUI

# Sidebar to display conversation history titles
st.sidebar.title("Conversation History")

# List conversation files in the sidebar
conversation_files = list_conversations()
if conversation_files:
    st.sidebar.markdown("### Conversations:")
    for conversation_file in conversation_files:
        if st.sidebar.button(conversation_file):
            load_conversation(conversation_file)
else:
    st.sidebar.markdown("No conversation history available.")



# CSS style to adjust the position of the submit button
st.markdown("""
    <style>
        .submit-button-container {
            display: flex;
            align-items: flex-end;
            height: 4vh; /* Adjust the height as needed */
        }
    </style>
""", unsafe_allow_html=True)
# Streamlit GUI

import random
# Streamlit GUI
# Import necessary libraries
import random
import streamlit as st

# Suggestion buttons
suggestions = [
    "Can you recommend a book?",
    "Tell me about reading habits",
    "How can I improve my reading speed?",
    "Can you do a brief of the book?"
]
suggestion_clicked = st.session_state.get('suggestion_clicked', False)


# Group all suggestion buttons together
if not suggestion_clicked:
    suggestion_button_group = st.columns(len(suggestions))
    for i, col in enumerate(suggestion_button_group):
        if col.button(suggestions[i], key=f"button_{i}"):
            st.session_state.suggestion_clicked = True 
            user_input = suggestions[i]
            tag_input = user_input.lower()
            if tag_input in ['can you recommend a book?', 'i\'m looking for a book', 'recommend me something to read']:
                response = random.choice(intents['intents'][3]['responses'])  # 'book_search' intent
            elif tag_input in ['tell me about reading habits']:
                response = random.choice(intents['intents'][9]['responses'])  # 'reading_habits' intent
            elif tag_input in ['how can i improve my reading speed?']:
                response = random.choice(intents['intents'][13]['responses'])
            elif tag_input in ['can you do a brief of the book?']:
                response = random.choice(intents['intents'][14]['responses'])
            else:
                response = "I'm sorry, I didn't understand that."
            # If no matching pattern is found, display the default response
            if not response:
                response = "I'm sorry, I didn't understand that."
            chatbot_response_text = format_response(response)
            # Append user question and chatbot response to conversation
            st.session_state.conversation += f"You: {user_input}\n"
            st.session_state.conversation += f"{chatbot_response_text}\n"
            # Append user question and chatbot response to chat history with styling
            with open(chat_history_file, "a") as file:
                formatted_response = format_response(response)
                # User message with grey background
                file.write(f"<div style='display:flex; align-items:center;'>")
                file.write(f"<img src='https://cdn-icons-png.flaticon.com/128/15486/15486795.png' style='vertical-align:middle; margin-right:5px; height: 20px; width: 20px;'>")
                file.write(f"<div style='background-color:{user_color}; padding:10px; margin-bottom:15px; border-radius:15px;'> You: {user_input}</div>")
                file.write(f"</div>")
                # Chatbot response with blue background and small icon
                file.write(f"<div style='display:flex; align-items:flex-start;'>")
                file.write(f"<img src='https://cdn-icons-png.flaticon.com/128/11189/11189417.png' style='vertical-align:middle; margin-right:5px; height: 20px; width: 20px;'>")
                file.write(f"<div style='background-color:{chatbot_color}; padding:10px; margin-bottom:12px; border-radius:15px;'> {formatted_response}</div>")
                file.write(f"</div>")


# Initialize conversation in session state
if "conversation" not in st.session_state:
    st.session_state.conversation = ""


# User Input and Chat
with st.form(key='chat_form'):
    user_input_col, submit_button_col = st.columns([4, 1])
    
    with user_input_col:
        user_input = st.text_input("You:", placeholder="Type message here....")
        
        # Hide suggestion buttons if user starts typing
        if user_input:
            st.session_state.suggestion_clicked = True

    with submit_button_col:
        # Custom container for the submit button to apply CSS
        st.markdown('<div class="submit-button-container">', unsafe_allow_html=True)
        submit_button = st.form_submit_button(label='➤')
        st.markdown('</div>', unsafe_allow_html=True)

    # Define tag_input here to ensure it's always defined
    tag_input = user_input.lower()
    
    if submit_button and user_input.strip() != "":
        response = None
        if tag_input in ['hi', 'hello', 'hey', 'greetings', 'good morning', 'good afternoon', 'good evening']:
            response = random.choice(intents['intents'][0]['responses'])  # 'greeting' 
        elif tag_input in ['goodbye', 'bye', 'see you later', 'adios', 'take care']:
            response = random.choice(intents['intents'][1]['responses'])  # 'goodbye' intent
        elif tag_input in ['thanks', 'thank you', 'thanks a lot', 'appreciate it', 'thank you so much']:
            response = random.choice(intents['intents'][2]['responses'])  # 'thanks' intent
        elif tag_input in ['can you recommend a book?', 'i\'m looking for a book', 'recommend me something to read']:
            response = random.choice(intents['intents'][3]['responses'])  # 'book_search' intent
        elif tag_input in ['can you tell me about author j.k. rowling?', 'who is author j.k. rowling?', 'recommend books by j.k. rowling']:
            response = random.choice(intents['intents'][4]['responses'])  # 'author_info' intent for J.K. Rowling
        elif tag_input in ['can you tell me about author stephen king?', 'who is author stephen king?', 'recommend books by stephen king']:
            response = random.choice(intents['intents'][5]['responses'])  # 'author_info' intent for Stephen King
        elif tag_input in ['can you tell me about author agatha christie?', 'who is authoragatha christie? ', 'recommend books by agatha christie']:
            response = random.choice(intents['intents'][6]['responses'])  # 'author_info' intent for Agatha Christie
        elif 'dan brown' in tag_input:
            response = random.choice(intents['intents'][7]['responses'])
        elif 'importance of books' in tag_input:
            response = random.choice(intents['intents'][8]['responses'])
        elif 'reading habit' in tag_input:
            response = random.choice(intents['intents'][9]['responses'])
        elif 'reading spots' in tag_input:
            response = random.choice(intents['intents'][10]['responses'])
        elif 'book for beginners' in tag_input:
            response = random.choice(intents['intents'][11]['responses'])
        elif 'personal library' in tag_input:
            response = random.choice(intents['intents'][12]['responses'])
        elif 'reading speed' in tag_input:
            response = random.choice(intents['intents'][13]['responses'])
        elif 'brief of the book' in tag_input:
            response = random.choice(intents['intents'][14]['responses'])
        elif 'write a book' in tag_input:
            response = random.choice(intents['intents'][15]['responses'])
        elif 'types of category' in tag_input:
            response = random.choice(intents['intents'][16]['responses'])
        elif 'feel boring' in tag_input:
            response = random.choice(intents['intents'][17]['responses'])
        elif 'gone girl' in tag_input:
            response = random.choice(intents['intents'][18]['responses'])
        elif 'the hobbit' in tag_input:
            response = random.choice(intents['intents'][19]['responses'])
        elif 'the great gatsby' in tag_input:
            response = random.choice(intents['intents'][20]['responses'])
        else:
            # Check if the user input contains any pattern in the intents file
            for intent in intents['intents']:
                for pattern in intent['patterns']:
                    if pattern.lower() in tag_input:  # Match if any pattern is contained within the user input
                        response = get_highest_rated_book(intent['tag'])
                        st.session_state.current_book_index += 1
                        if response:
                            next_book = get_next_book(intent['tag'], st.session_state.current_book_index)
                            if next_book:
                                response = next_book
                            else:
                                response = "No more books found for this category."
                        break
                if response:
                    break

        # If no matching pattern is found, display the default response
        if not response:
            response = "I'm sorry, I didn't understand that."

        chatbot_response_text = format_response(response)
        
        # Append user question and chatbot response to conversation
        st.session_state.conversation += f"You: {user_input}\n"
        st.session_state.conversation += f"{chatbot_response_text}\n"
    
        # Append user question and chatbot response to chat history with styling
        with open(chat_history_file, "a") as file:
            formatted_response = format_response(response)
            # User message with grey background
            file.write(f"<div style='display:flex; align-items:center;'>")
            file.write(f"<img src='https://cdn-icons-png.flaticon.com/128/15486/15486795.png' style='vertical-align:middle; margin-right:5px; height: 20px; width: 20px;'>")
            file.write(f"<div style='background-color:{user_color}; padding:10px; margin-bottom:15px; border-radius:15px;'> You: {user_input}</div>")
            file.write(f"</div>")
          # Chatbot response with blue background and small icon
            file.write(f"<div style='display:flex; align-items:flex-start;'>")
            file.write(f"<img src='https://cdn-icons-png.flaticon.com/128/11189/11189417.png' style='vertical-align:middle; margin-right:5px; height: 20px; width: 20px;'>")
            file.write(f"<div style='background-color:{chatbot_color}; padding:10px; margin-bottom:12px; border-radius:15px;'> {formatted_response}</div>")
            file.write(f"</div>")




    # Read chat history and display
    with open(chat_history_file, "r") as file:
        chat_history_text = file.read()

    # Render chat history with HTML if it's not empty
    if chat_history_text.strip() != "":
        chat_history.markdown(chat_history_text, unsafe_allow_html=True)