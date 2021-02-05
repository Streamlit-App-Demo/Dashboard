# Setup:
import awesome_streamlit as ast
import os
import streamlit as st

# Images:
from PIL import Image
import requests

# Page:
import des_statistics
import home
import ml_interpretations

# To disable PyplotGlobalUseWarning
st.set_option('deprecation.showPyplotGlobalUse', False)

# Selection Pages:
PAGES = {
    "Home": home,
    "Descriptive Statistics": des_statistics,
    "Machine Learning": ml_interpretations,
}

# Title and image
THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
my_file = os.path.join(THIS_FOLDER, "../fps_dashboard/Assets/fp_logo.png")
img = Image.open(my_file)
st.image(img, width=900)


def main():
    # Select Page from navigation
    st.sidebar.title("Navigation")
    selection = st.sidebar.radio("Go to", list(PAGES.keys()))

    page = PAGES[selection]
    # creates a loading message for each page
    # also takes page value and writes it in streamlit format
    with st.spinner(f"loading {selection} ..."):
        ast.shared.components.write_page(page)

    # About us button for labs30 fps ds team:
    about_button = st.sidebar.button("About us")
    if about_button:
        st.sidebar.markdown("## Labs30 DS Team B:")
        st.sidebar.markdown("### Erle Granger - "
            '<a href="https://github.com/ilEnzio"><img src="https://www.flaticon.com/svg/vstatic/svg/1051/1051275.svg?token=exp=1612227945~hmac=11c169068df735133dfb37861f582fec" width=22></a>',
            unsafe_allow_html=True
            )
        st.sidebar.markdown("### Lester Gomez - "
            '<a href="https://github.com/machine-17"><img src="https://www.flaticon.com/svg/vstatic/svg/1051/1051275.svg?token=exp=1612227945~hmac=11c169068df735133dfb37861f582fec" width=22></a>',
            unsafe_allow_html=True
            )
        st.sidebar.markdown("### Robert Giuffre - "
            '<a href="https://github.com/rgiuffre90"><img src="https://www.flaticon.com/svg/vstatic/svg/1051/1051275.svg?token=exp=1612227945~hmac=11c169068df735133dfb37861f582fec" width=22></a>',
            unsafe_allow_html=True
            )


if __name__ == "__main__":
    main()
