
import app1
import app2
import app3
import app4
import streamlit as st
PAGES = {
    "Home": app1,
    "Landmark Detection": app2,
    "About US": app3,
    "Contact US":app4
}
st.sidebar.title('Menu')
selection = st.sidebar.radio("Go to", list(PAGES.keys()))
page = PAGES[selection]
page.app()
