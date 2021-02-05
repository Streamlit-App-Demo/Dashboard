# Setup:
import awesome_streamlit as ast
import streamlit as st

# Weather API library
# import pyowm
# from pyowm import OWM
# from pyowm.utils import config
# from pyowm.utils import timestamps

# Weather API to display temperature and overall conditions.
# To be used in the future for homepage text

# owm = pyowm.OWM('2b9a0d21ce45b6c1dfd756f0f892047a')
# mgr = owm.weather_manager()


def write():
    st.title("Welcome!")
    st.subheader("Please select from Navigation to begin.")

#   API link code and queries to the weather database:
#     observation = mgr.weather_at_place('Spokane,US')
#     w = observation.weather
#     temp_dict_fahrenheit = w.temperature('fahrenheit')
#     st.markdown("### Today's Weather")
#     st.write(f"Temperature High {temp_dict_fahrenheit['temp_max']} °F,
#              Low {temp_dict_fahrenheit['temp_min']} °F")
#     st.write(f"Current weather outlook: {w.detailed_status}")
