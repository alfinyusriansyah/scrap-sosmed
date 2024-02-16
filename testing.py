import streamlit as st

def main():
    st.subheader("new")
    name = st.text_input("Enter your name")
    if st.button("Send"):
        st.write(name)
    if st.checkbox("bye"):
        st.write("I see")
    if st.checkbox("welcome"):
        st.write("you see")

main()