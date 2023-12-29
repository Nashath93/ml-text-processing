from flask import Flask, render_template, request
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense
from sklearn.model_selection import train_test_split
import numpy as np
from app import app

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/check_msg', methods=['POST'])
def check_message():
    message_type = ""
    if request.method == "POST":
        message_type = process_text(request.form.get('message'))

    return render_template('index.html', alert=message_type['alert'], msg_type=message_type['msg_type'])


def process_text(new_text):
    # Example dataset
    texts = [
        "Please create a new user account for me.",
        "I need assistance with onboarding process.",
        "Can you provide access to the report section?",
        "Requesting to add a new feature to the system.",
        "Need help setting up my profile.",
        "Asking for offboarding assistance.",
        "Can you generate a performance report for my team?",
        "Requesting access to the training materials.",
        "I'd like to request a software upgrade.",
        "Asking for help with the registration process.",
        "Requesting support to configure my email account.",
        "Can you guide me through the process of updating my password?",
        "Asking for assistance in integrating the new tool into our workflow.",
        "Requesting access to the latest version of the software.",
        "I need help with setting up a recurring report.",
        "Can you provide instructions on how to request time off?",
        "Unable to log in to the system, getting an error message.",
        "System seems to be down, unable to access any features.",
        "Experiencing slow internet connection issues.",
        "Can't connect to the VPN, getting a connection error.",
        "Encountering an error when trying to submit a form.",
        "My account seems to be locked, unable to access any resources.",
        "The application is crashing repeatedly.",
        "Can't open the latest report file, getting a file corruption error.",
        "Experiencing difficulty in accessing shared files.",
        "Unable to send emails, getting a server error.",
        "Encountering issues with printing documents.",
        "Experiencing frequent timeouts when accessing the database.",
        "Can't access the website, getting a page not found error.",
        "The software is not responding to my inputs.",
        "Encountering display issues with the user interface.",
        "Unable to download attachments from emails."
    ]

    labels = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    # Tokenize the text
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)

    # Pad sequences to a fixed length
    max_length = max(len(seq) for seq in sequences)
    padded_sequences = pad_sequences(sequences, maxlen=max_length)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(padded_sequences, labels, test_size=0.2, random_state=42)

    # Convert labels to arrays
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    # Define the model
    vocab_size = len(tokenizer.word_index) + 1
    num_classes = 2  # Number of classes (positive and negative)

    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=64, input_length=max_length))
    model.add(LSTM(1000))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

    # Make predictions on new data
    new_sequences = tokenizer.texts_to_sequences([new_text])
    new_padded_sequences = pad_sequences(new_sequences, maxlen=max_length)

    predictions = model.predict(new_padded_sequences)

    # Convert the prediction to "positive" or "negative"
    sentiment = "Positive" if predictions[0, 1] > 0.5 else "Negative"
    confidence = predictions[0, 1] if sentiment == "Positive" else 1 - predictions[0, 1]
    print(f"Text: {new_text} | Sentiment: {sentiment} | Confidence: {confidence:.4f}")

    data = {
        'alert': "primary" if sentiment == "Positive" else "danger",
        'msg_type': "Request" if sentiment == "Positive" else "Incident"
    }
    
    return data
    