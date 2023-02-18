import pickle
import streamlit as st



model=pickle.load(open("spam.pkl","rb"))
cv=pickle.load(open("vectorizer.pkl","rb"))


def main():
    st.title("SMS Spam Classification App")
    st.subheader("Build with Streamlit & Python \n by Rohan Tyagi")
    msg=st.text_input("Enter a Text: ")
    if st.button("Predict"):
        data=[msg]
        vect=cv.transform(data).toarray()
        prediction=model.predict(vect)
        result=prediction[0]
        if result==1:
            st.error("SPAM !!!")
        else:
            st.success("Not Spam")


main()