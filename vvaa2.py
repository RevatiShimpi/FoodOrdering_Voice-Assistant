import speech_recognition as sr
from gtts import gTTS
from base64 import b64encode
from io import BytesIO
import os
import time
import datetime
import numpy as np
import pickle
from gradio import Audio, Interface, Textbox
from speech_recognition import AudioFile, Recognizer
from transformers import BertTokenizer
import torch.nn as nn
import torch
from pymongo import MongoClient
from bson import ObjectId



MONGO_URI = 'mongodb://localhost:27017'  # Update with your MongoDB URI
DB_NAME = 'orders'

class BERT_Arch(nn.Module):
   def __init__(self, bert):      
       super(BERT_Arch, self).__init__()
       self.bert = bert 
      
       # dropout layer
       self.dropout = nn.Dropout(0.2)
      
       # relu activation function
       self.relu =  nn.ReLU()
       # dense layer
       self.fc1 = nn.Linear(768,512)
       self.fc2 = nn.Linear(512,256)
       self.fc3 = nn.Linear(256,5)
       #softmax activation function
       self.softmax = nn.LogSoftmax(dim=1)
       #define the forward pass
   def forward(self, sent_id, mask):
      #pass the inputs to the model  
      cls_hs = self.bert(sent_id, attention_mask=mask)[0][:,0]
      
      x = self.fc1(cls_hs)
      x = self.relu(x)
      x = self.dropout(x)
      
      x = self.fc2(x)
      x = self.relu(x)
      x = self.dropout(x)
      # output layer
      x = self.fc3(x)
   
      # apply softmax activation
      x = self.softmax(x)
      return x


def stt(audio: object, language: str) -> str:
    r = Recognizer()
    try:
        with AudioFile(audio) as source:
            audio_data = r.record(source)
            text = r.recognize_google(audio_data, language=language)
        return text
    except sr.UnknownValueError as e:
        print(f"Speech Recognition could not understand audio: {str(e)}")
        return ""
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {str(e)}")
        return ""



def tts(text: str, language: str) -> object:
    
    return gTTS(text=text, lang=language, slow=False)


def tts_to_bytesio(tts_object: object) -> bytes:
    """Converts tts object to bytes.
    Args:
        tts_object (object): audio object obtained from gtts
    Returns:
        bytes: audio bytes
    """
    bytes_object = BytesIO()
    tts_object.write_to_fp(bytes_object)
    bytes_object.seek(0)
    return bytes_object.getvalue()


def html_audio_autoplay(bytes: bytes) -> object:
    
    b64 = b64encode(bytes).decode()
    html = f"""
    <audio controls autoplay>
    <source src="data:audio/wav;base64,{b64}" type="audio/wav">
    </audio>
    """
    return html
class ChatBot():
    model_path = 'C:\\Users\\hp\\Desktop\\BOTSOMETHINGFROMNET\\model1.pickle'

    
    def __init__(self, name, model_path,  mongo_uri, db_name, collection_name):
        print("----- Starting up", name, "-----")
        self.name = name
        self.model = self.load_model(model_path)
        self.user_orders = []
        self.mongo_client = MongoClient(mongo_uri)
        self.db = self.mongo_client[db_name]
        self.collection = self.db[collection_name]

    
    def load_model(self, model_path):
        try:
            with open(model_path, 'rb') as model_file:
                model = pickle.load(model_file)
            return model
        except Exception as e:
            print("Error loading the model:", str(e))
            return None
   

    def detect_intent(self, text):
        if self.model is not None:
           
            tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)

            
            sent_id = inputs["input_ids"]
            mask = inputs["attention_mask"]

            # Load the BERT model
            bert_model = torch.hub.load('huggingface/pytorch-transformers', 'model', 'bert-base-uncased')

            # Create an instance of your BERT_Arch model
            model = BERT_Arch(bert_model)

            # Make a prediction
            preds = model(sent_id, mask)

            
            intent = torch.argmax(preds, dim=1).item()
            return intent
        else:
            return "Sorry, the model couldn't be loaded."

    def text_to_speech(self, text):
        if text is not None:
            print("Dev --> ", text)
            speaker = gTTS(text=text, lang="en", slow=False)
            speaker.save("res.mp3")
            statbuf = os.stat("res.mp3")
            mbytes = statbuf.st_size / 1024
            duration = mbytes / 200
            os.system('start res.mp3')
            time.sleep(int(50 * duration))
            os.remove("res.mp3")
        else:
            print("Dev --> No text to speak")
    
    def delete_recent_order(self):
        # Delete the most recent order from MongoDB
        if self.collection.count_documents({}) > 0:
            recent_order = self.collection.find().sort("timestamp", -1).limit(1)[0]
            self.collection.delete_one({"_id": recent_order["_id"]})
            return "Your, order has been cancelled. Do visit again!!"
        else:
            return "There are no orders to delete."
    def handle_user_input(self, user_input):
        
        # Log user input to conversation history
        res = ""
        intent=-1
        if any(i in user_input for i in ["thank", "thanks"]):
            res = np.random.choice(["Welcome, do order again from us!"])

        elif any(i in user_input for i in ["yourself","hello"]):
            res = np.random.choice(["I am a food-ordering voice assistant! What would you like to order?"])

        elif any(i in user_input for i in ["exit", "close"]):
            res = np.random.choice(["Tata", "Have a good day", "Bye", "Goodbye", "Hope to meet soon", "Peace out!"])
        elif any(i in user_input for i in ["delete", "cancel"]):
            res = self.delete_recent_order()
        elif any(i in user_input for i in ["orders", "review", "my order"]):
            res = self.review_orders()
            
        

        # Recognize user intent using the DistilBERT-based model
        else:
            # Check if the user is requesting the menu
            if any(i in user_input for i in ["menu", "options", "list","dry"]):
                # Provide the menu options
                menu = "Here are our menu options:\n1. Pizza\n2. Burger 2\n3. Fries 3\n4. Cheese Pizza 4\n5. Sausages\n 6. Pasta"
                res = menu
            if any(i in user_input for i in ["drink","drinking"]):
                # Provide the menu options
                menu = "Here are our menu options:\n1. Fanta\n2. Mango Juice 2\n3. Cold Coffee 3\n4. Espresso 4\n5. Milkshake"
                res = menu
            # Check if the user is requesting to review their orders
            
                
            
            else:
                intent = self.detect_intent(user_input)
                if intent == 1:
                    res = "Sure I will add your order, would you like to add more?"
                    # Log user order
                    self.user_orders.append(user_input)
                elif intent == 2:
                    res = "I have added the items in file."
                    self.user_orders.append(user_input)
                elif intent == 3:
                    res = "That's a great choice, should I consider this final?"
                    self.user_orders.append(user_input)
                elif intent == 4:
                    res = "Your order will be ready in 20 mins"
                    self.user_orders.append(user_input)
                elif intent == 5:
                    res = "Okay, I will make the quantity according to the specified number of people"
                    self.user_orders.append(user_input)

        order_result = self.log_order_to_mongo(user_input,intent)
        print("Order logged to MongoDB with ID:", order_result)
        return res 
    

    def log_order_to_mongo(self, order,intent):
        # Log the user's order to MongoDB with date and time
        
        if intent in [1, 2, 3, 4, 5]:
            recent_order = self.collection.find().sort("timestamp", -1).limit(1)

            if recent_order.count() > 0:
               
                recent_order = recent_order[0]
                updated_order = {
                    "$set": {
                        "order": order,
                        "timestamp": datetime.datetime.now(),
                    }
                }
                result = self.collection.update_one({"_id": recent_order["_id"]}, updated_order)
                return result.modified_count
            else:
                
                order_data = {
                    "order": order,
                    "timestamp": datetime.datetime.now(),
                }
                result = self.collection.insert_one(order_data)
                return result.inserted_id
        else:
            return None
       
        

    def review_orders(self):
    # Retrieve the most recent order from MongoDB
        recent_order = self.collection.find().sort("timestamp", -1).limit(1)

        if recent_order.count() > 0:
            # If there is a recent order, provide a summary
            order_summary = f"Your order is:\n{recent_order[0]['order']}"
            #self.text_to_speech(order_summary)
            return order_summary
        else:
            # If there are no orders in the collection, inform the user
            no_orders_message = "You haven't placed any orders yet."
            self.text_to_speech(no_orders_message)
            return no_orders_message



    def summarize_orders(self):
        # Generate a summary of the user's orders
        if not self.user_orders:
            return "You haven't placed any orders yet."

        summary = "Here's a summary of your orders:\n"
        for i, order in enumerate(self.user_orders, start=1):
            summary += f"{i}. {order}\n"

        return summary
    

    
def main(audio: object):
    desired_language = 'en'

    mongo_uri = 'mongodb://localhost:27017'
    db_name = 'customer_order'
    collection_name = 'fastfoodorders'
    
    # Create an instance of the ChatBot class
    chat_bot = ChatBot(name="DEV", model_path="C:\\Users\\hp\\Desktop\\VA2\\model1.pickle" , mongo_uri=mongo_uri, db_name=db_name, collection_name=collection_name)

    # Use the instance to call the methods
    user_speech_text = stt(audio, desired_language)
    intent = chat_bot.detect_intent(user_speech_text)
    bot_response_en = chat_bot.handle_user_input(user_speech_text)
    bot_voice = tts(bot_response_en, desired_language)
    bot_voice_bytes = tts_to_bytesio(bot_voice)
    html = html_audio_autoplay(bot_voice_bytes)
    return user_speech_text, bot_response_en, html


Interface(
    fn=main,
    title="Kookie: Your Fast Food Voice Assistant",
    inputs=[
        Audio(
            source="microphone",
            type="filepath",
        ),
    ],
    outputs=[
        Textbox(label="Customer: "),
        Textbox(label="Kookie: "),
        "html",
    ],
    live=True,
    allow_flagging="never",
).launch()

    

