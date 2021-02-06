# Description
- We worked on a student project for The Family Promise of Spokane Organization, a US-based nonprofit organization based in Spokane, WA. They are an organization that helps homeless families as well as preventing families from becoming homeless. They provide shelter for families as well as rental assitance. For the full description visit their website [Family Promise of Spokane](https://www.familypromiseofspokane.org/)
- The dashboard we made was meant for the supervisors and caseworkers to make informed decisions.
- This repository main purpose is to demo the streamlit app dashboard portion only. The only contributors are below.

# Contributors
| [Erle Granger II](https://github.com/ilEnzio) | [Lester Gomez](https://github.com/machine-17) | [Robert Giuffre](https://github.com/rgiuffre90) |

# How to run streamlit app:
1. `git clone https://github.com/Streamlit-App-Demo/Dashboard.git`
2. Install packages using `pipenv install` or `pipenv shell`
3. Type `streamlit run fps_dashboard/fps_app.py` to run dashboard

## Demo:
- Quick video about the app: [Youtube](https://youtu.be/MIn8YVSNczk)

### Known bugs:
- From `ml_interpretations.py`
1. Posx and posy should be finite values. Text and fig scaling issues.
2. In shap section: (matplotlib = True) is not yet supported for force plots with multiple samples! Example: Pick [Personal ID 53716]
3. Segmentation fault. Sometimes it crashes.

- From `des_statistics.py`
1. Basic Enrollment Stats: if the date range changes, and if within those observations the number of possible genders changes, the colors indicating gender in the legend may change.
2. Exit Outcome Facet Chart: if the date range selected by the user does not contain at least one example of each exit outcome, the chart will fail to render.
3. Exit Outcome Facet Chart: if the user zooms the 3 charts will become "unmoored" - the x axis moves.

# Libraries:
![pandas](https://img.shields.io/badge/-pandas-blue)
![matplotlib](https://img.shields.io/badge/-matplotlib-blue)
![altair](https://img.shields.io/badge/-altair-red)
![scikit-learn](https://img.shields.io/badge/-scikit--learn-blue)
![catboost](https://img.shields.io/badge/-catboost-yellow)
![xgboost](https://img.shields.io/badge/-xgboost-blue)
![eli5](https://img.shields.io/badge/-eli5-blue)
![pdpbox](https://img.shields.io/badge/-pdpbox-blue)
![shap](https://img.shields.io/badge/-shap-blueviolet)
![streamlit](https://img.shields.io/badge/-streamlit-red)

# Linked Repositories
[Family Promise of Spokane Data Science](https://github.com/Lambda-School-Labs/family-promise-spokane-ds-b) 
