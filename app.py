import streamlit as st
from src import predictbyvalues
from src import predictbyfile

def main():
    html_temp = """
    <div style="background-color:DodgerBlue;padding:10px">
    <h2 style="color:white;text-align:center;">Healthcare Provider Fraud Predictor </h2>
    </div>
    <br>
    """

    PAGES = {
    "Predict By Values": predictbyvalues,
    "Predict By File": predictbyfile
    }
    st.sidebar.title('Navigation')
    selection = st.sidebar.radio("Go to", list(PAGES.keys()))
    page = PAGES[selection]
    page.main()

if __name__ == '__main__' :
    main()
