import speech_recognition as sr
import streamlit as st
import pickle

rec=sr.Recognizer()

import speech_recognition as sr

def predict_sentiment(text, model, vectoriser):
  list_test = [text]
  X_test  = vectoriser.transform(list_test)
  # prediction 
  st.write("Predicting")
  Y_pred = model.predict(X_test)

  if (Y_pred[0]):
    st.success("positive")
  else:
    st.error("negative")

# Load the model
with open('sentiment_analysis_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

with open('vectorizer.pkl', 'rb') as file:
    loaded_vectorizer = pickle.load(file)


def transcribe_audio():
    # Create a recognizer object
    my_micro = sr.Recognizer()

    # Use a context manager (with statement) to handle resources properly
    with sr.Microphone() as source:
         # Adjust for ambient noise
        # sr.adjust_for_ambient_noise(source)
        text = ""
        # Use Google Speech Recognition to transcribe the audio
        try:
            st.write("Say something...")
            audio = my_micro.listen(source)
            text = rec.recognize_google(audio)
            st.write(f"You said: {text}")
        except sr.UnknownValueError:
            st.error("Unable to recognize speech.")
        except sr.RequestError as e:
            st.error(f"Error: {e}")
        st.write("Listening")
        predict_sentiment(text, loaded_model, loaded_vectorizer)


  
# this is the main function in which we define our webpage 
def main():
      # giving the webpage a title
    st.title("Speech sentiment analysis")
      
    # here we define some of the front end elements of the web page like 
    # the font and background color, the padding and the text to be displayed
    html_temp = """
    <div style ="background-color:pink;padding:13px;margin-bottom:20px;">
        <h1 style ="color:black;text-align:center;">Tell us how you feel ?</h1>
    </div>
    """
      
    # this line allows us to display the front end aspects we have 
    # defined in the above code
    st.markdown(html_temp, unsafe_allow_html = True)
      
    # the following lines create text boxes in which the user can enter 
    # the data required to make the prediction
    st.write("Click the button and start speaking.")

    # Create a button to start the speech recognition
    if st.button("Start Listening"):
        # Call the speech recognition function
        transcribe_audio()
     
if __name__=='__main__':
    main()
    

    