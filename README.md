### Known bugs:
- From `ml_interpretations.py`
1. Posx and posy should be finite values. Text and fig scaling issues.
2. Shap - matplotlib = True is not yet supported for force plots with multiple samples! Example: Pick [Personal ID 53716]
3. Segmentation fault. Sometimes it crashes.

- From `des_statistics.py`
1. Basic Enrollment Stats - if the date range changes, and if within those observations the number of possible genders changes, the colors indicating gender in the legend may change. 
2. Exit Outcome Facet Chart.  If the date range selected by the user does not contain at least one example of each exit outcome, the chart will fail to render.
3. Exit Outcome Facet Chart.  if the user zooms the 3 charts will become "unmoored" - the x axis moves.

# Family Promise of Spokane

You can find the deployed project at [Family Promise of Spokane](https://family-profile-styling-emily.d3hmwb1bmjh3u1.amplifyapp.com//).

# Description

The Family Promise of Spokane  Organization is a US-based nonprofit organization based in Spokane, WA. They are an organization that helps homeless families as well as preventing families from becoming homeless. They provide shelter for families as well as rental assitance. For the full description visit their website [Family Promise of Spokane](https://www.familypromiseofspokane.org/)
# Contributors

| [Dominick Bruno]() | [Robert Giuffre](https://github.com/rgiuffre90) | [Sara Cearc](https://github.com/cearac-sara) |
| :---: | :---: | :---: | 
| [<img src="https://ca.slack-edge.com/ESZCHB482-W012JHQU86N-44142d528214-512" width = "200" />]() | [<img src="https://avatars.githubusercontent.com/u/69161174?s=460&u=24e9833995e636841ff7fed6a3c5535c96d4949b&v=4" width = "200" />](https://github.com/rgiuffre90) | [<img src="https://avatars.githubusercontent.com/u/67298892?s=460&u=2c4c90762496cd094b55c68e75230fb0d13217b5&v=4" width = "200" />](https://github.com/cearac-sara) | 
| TPL | Data Scientist | Web Developer |  
|[<img src="https://github.com/favicon.ico" width="15"> ]() | [<img src="https://github.com/favicon.ico" width="15"> ](https://github.com/rgiuffre90) | [<img src="https://github.com/favicon.ico" width="15"> ](https://github.com/cearac-sara) | 
| [ <img src="https://static.licdn.com/sc/h/al2o9zrvru7aqj8e1x2rzsrca" width="15"> ](https://www.linkedin.com/in/dbruno93/) | [ <img src="https://static.licdn.com/sc/h/al2o9zrvru7aqj8e1x2rzsrca" width="15"> ](https://www.linkedin.com/in/robert-giuffre/) | [ <img src="https://static.licdn.com/sc/h/al2o9zrvru7aqj8e1x2rzsrca" width="15"> ](https://www.linkedin.com/in/sara-cearc/) |
| [Lester Gomez](https://github.com/machine-17) | [Erle Granger II](https://github.com/ilEnzio) | [Mudesir Suleyman](https://github.com/mudesir) |
| [<img src="https://avatars.githubusercontent.com/u/68140323?s=400&u=249894a3f4684124a2fe6b050fba8ab255af842b&v=4" width = "200" />](https://github.com/machine-17) | [<img src="https://ca.slack-edge.com/ESZCHB482-W015P64MU5B-108d53177582-512" width = "200" />](https://github.com/ilEnzio) | [<img src="https://ca.slack-edge.com/ESZCHB482-W012R4C0T44-d512b3c6174c-512" width = "200" />](https://github.com/mudesir) | 
| Data Scientist | Data Engineer | Data Scientist |  
|[<img src="https://github.com/favicon.ico" width="15"> ](https://github.com/machine-17) | [<img src="https://github.com/favicon.ico" width="15"> ](https://github.com/ilEnzio) | [<img src="https://github.com/favicon.ico" width="15"> ](https://github.com/mudesir) | [<img src="https://github.com/favicon.ico" width="15"> ]() |
| [ <img src="https://static.licdn.com/sc/h/al2o9zrvru7aqj8e1x2rzsrca" width="15"> ](https://www.linkedin.com/in/lg17/) | [ <img src="https://static.licdn.com/sc/h/al2o9zrvru7aqj8e1x2rzsrca" width="15"> ](https://www.linkedin.com/in/erle-granger-a7b231/) | [ <img src="https://static.licdn.com/sc/h/al2o9zrvru7aqj8e1x2rzsrca" width="15"> ](https://www.linkedin.com/in/mudesir-suleyman-a0b891190/) |           
| [Suzanne Cabral](https://github.com/suzannecabral) | [Andrew Muto]() | [Breson Whorely]() |
| [<img src="https://avatars.githubusercontent.com/u/25539417?s=400&u=e48fba22ff44e3b615a393ca394ed864ef41e141&v=4" width = "200" />]() | [<img src="https://ca.slack-edge.com/ESZCHB482-W012R4ECJ1J-bbb0dcc461ae-512" width = "200" />]() | [<img src="https://https://ca.slack-edge.com/ESZCHB482-W017UE8LQP6-c55573e09a38-512" width = "200" />]() | 
| Web Developer | Web Developer | Web Developer |  
|[<img src="https://github.com/favicon.ico" width="15"> ](https://github.com/suzannecabral) | [<img src="https://github.com/favicon.ico" width="15"> ]() | [<img src="https://github.com/favicon.ico" width="15"> ]() | 
| [ <img src="https://static.licdn.com/sc/h/al2o9zrvru7aqj8e1x2rzsrca" width="15"> ](https://www.linkedin.com/in/suzanne-cabral/) | [ <img src="https://static.licdn.com/sc/h/al2o9zrvru7aqj8e1x2rzsrca" width="15"> ]() | [ <img src="https://static.licdn.com/sc/h/al2o9zrvru7aqj8e1x2rzsrca" width="15"> ](https://www.linkedin.com/in/brenson-w/) |
             

<br>          

<br>
<br>

![fastapi](https://img.shields.io/badge/fastapi-0.60.1-blue)
![pandas](https://img.shields.io/badge/pandas-1.1.0-blueviolet)

![uvicorn](https://img.shields.io/badge/uvicorn-0.11.8-ff69b4)
![python-dotenv](https://img.shields.io/badge/python--dotenv-0.14.0-green)

![scikit-learn](https://img.shields.io/badge/scikit--learn-0.23.2-yellow)
![psycopg2](https://img.shields.io/badge/psycopg2--2.8.6-informational)
![fastapi-utils](https://img.shields.io/badge/fastapi--utils-0.2.1-informational)



# Deployed Product
[Data Science API](http://a-labs29-family-promise.eba-syir5yx3.us-east-1.elasticbeanstalk.com/) 


# Linked Repositories
[Family Promise of Spokane Data Science](https://github.com/Lambda-School-Labs/family-promise-spokane-ds-b) 


# Getting Started

### User Flows

[Trello Board](https://trello.com/b/5J8xmgZo/labs30-family-promise-b)

Our team is developing a digital intake form for families in Family Promise Shelter. This intake form is a replacement for the paper form currently being filled by guests of the shelter. All the Data is given by the guests. We have a multi-class random forest model that takes the guest data and predicts their exit destination. The possible Exit Destination are as follow. 
- Permanent Exit
- Temporary Exit
- Emergency Shelter
- Transitional Housing
- Unknown/Other

### Key Features

- Supervisors can Create guest profile 
- Case Manager can view guest profile 
- Case Manager can view the exit destination prediction
- Guest can be flagged for mis conduct
- notes can be added to guest's profile. 
- Guest can view their own profile. 
### Environment Variables

In order for the app to function correctly, the user must set up their own environment variables. There should be a .env file containing the following:

    *  DB_URL  - Postgres database credentials



### Content Licenses

| Image Filename | Source / Creator | License                                                                      |
| -------------- | ---------------- | ---------------------------------------------------------------------------- |
| Nopic.yet      | INSERT NAME      | [MIT](input githandle here)                             |

### Installation Instructions

Use requirements.txt to create virtual enviornment locally.
In order to run the dashboard app, use streamlit run fps_app.py

#### Scripts

    Get AWS credentials
    
    Get your AWS access keys
    
    Install AWS Command Line Interface
    
    * aws configure -> configures AWS CLI
    * pip install pipx -> installs pipx
    * pipx install awsebcli -> installs AWS Elastic BeanStalk CLI
[Follow AWS EB docs](https://docs.labs.lambdaschool.com/data-science/tech/aws-elastic-beanstalk)
    
    Then use the EB CLI:
    
    * git add --all 
    * git commit -m "Your commit message" 
    * eb init -p docker YOUR-APP-NAME --region us-east-1 
    * eb create YOUR-APP-NAME 
    * eb open 
    
    Then use AWS Route 53 to set up a domain name with HTTPS for your DS API
    
    Redeploy:
    
    * git commit ... 
    * docker build ... 
    * docker push ... 
    * eb deploy 
    * eb open 

# Tech Stack

### Data Science API built using:

- Python
- Docker
- FastAPI
- AWS Elastic Beanstalk
- PostgreSQL
- Pipenv

### Data Science Dashboard built using:

- Python
- Pipenv
- Streamlit
- Awesome-Streamlit

### Why we made our tech stack decisions:

1. FastAPI
- Wanted to gain insight to AWS
- Docker and Pipenv makes environments easier
- FastAPI has been gaining traction over Flask
- SQL queries are better structured 

2. Dashboard App
- Streamlit is much more lightweight than other libraries (dash, flask)
- No front-end code required
- interprets all standardized python plot libaries 

### Libraries used:
- FastAPI
- sci-kit-learn
- Pandas
- psycopg2

# User Flow

### Data Science API

We are receiving a POST request with the member id from the web team and using that id to query the database, choose the features used to create the classification model, predict the exit destination and returning a prediction for the exit destination along with the top features in JSON format. 

# Architecture Chart
![Architecture](https://github.com/Lambda-School-Labs/family-promise-spokane-ds-a/raw/main/architecture_diagram2.png)




# End Points
/predict: send POST request to this endpoint with the member_id value. 

# Issues
* Newest Model not implemented Due to the following issues. 
*  Some Features needed for the new predictive model is not on the back-end Database
* Some Features on the new Predictive Model differs in data format and data type from the back-end database. 
* Streamlit App not linked to backend API
```
# This Code Snippet Shows where and how to fix issues
def set_variables(member_id):
#current date for Days/ Years calculations
  today_date = datetime.date.today()
  results_dict = {} # Dictionary to hold the results value to be transformed to df
  
  query = 'SELECT * FROM members,families where members.id = {} and families.id = members.family_id'.format(member_id)
  results = dbmanage(uri,query)
#repeated columns need to be used more than once. 
  enroll_date = results['date_of_enrollment']
  date_of_birth = datetime.datetime.strptime(
      results['demographics']['DOB'], '%m-%d-%Y').date()
# Above Code is working - 
# Below Commented Out Code needs to be fixed.
# Not Commented out code, working fine. 
#sets variables from the db results
#Project_Name (Probably Take out from model. 
 # results_dict['project_name'] = 'family promise' # This might be removed from last model 
#Relationship to head of household
  results_dict['relationship'] = results['demographics']['relationship']
#Case Members
  results_dict['case_members'] = results['case_members']
#enroll date
  results_dict['enroll_date'] = enroll_date
#Exit Date (not yet)(need table ? ) 
  # results_dict['exit_date'] = 'exit date' #not in table. needs to be added. 
  # Social SEcurity Quality( will need table (not yet))
  # results_dict['social_security_quality']  = 'Full SSN Given' #PlaceHOlder
#age at enrollment
  results_dict['age_at_enrollment'] = int((enroll_date - date_of_birth).days / 365.2425)
# Race
  results_dict['race'] = results['demographics']['race']
#ethnicity
  results_dict['ethnicity'] = results['demographics']['ethnicity']
# Gender
  results_dict['gender'] = results['demographics']['gender']
#veteran Status
  results_dict['veteran_status'] = results['gov_benefits']['veteran_services']
#disabilities at time of entry 
  results_dict['physical_disabilities'] = results['barriers']['physical_disabilities']
#living situation (current location)
  results_dict['living_situation'] = results['homeless_info']['current_location']
#length of stay
  results_dict['length_of_stay'] = results['length_of_stay']
#homeless start date 
  results_dict['homless_start_date'] = datetime.datetime.strptime(
      results['homeless_info']['homeless_start_date'], '%d-%b-%Y').date()
# Length of time homeless aprox start
  results_dict['length_of_time_homeless'] = results['homeless_info']['total_len_homeless'] 
  results_dict['time_homeless_last_years'] = results['homeless_info']['total_len_homeless']
  # results_dict['total_month_homeless_last_year'] = 4 #PlaceHolder. nOt in Database
# Last Permanent Address
 # results_dict['last_permanent_address'] = results['last_permanent_address'] # database does not have same data format as ds dataset. #needs to be fixed. 
  results_dict['state'] = (results['last_permanent_address'].split(" "))[-3] 
  results_dict['zip'] = (results['last_permanent_address'].split(" "))[-2]
# Enrollment length  
  results_dict['enrollment_length'] = int((today_date - enroll_date).days) # Days Enrolled in project. 
  #Housing Status 
  # results_dict['housing_status'] = 'homeless ' # Place HOlder #Not in backend database. 
  results_dict['covered_by_insurance'] = results['insurance']['has_insurance']
#Domestic Violence 
#  results_dict['domestic_violence'] = results['domestic_violence_info']['fleeing_dv'] #Probably needs chang from boolean to string on backend
#Household Type
  results_dict['household_type'] = results['household_type']
# Last Grade Completed   
  results_dict['last_grade_completed'] = results['schools']['highest_grade_completed']
# School Enrolled Status  
 # results_dict['school_status'] = results['schools']['enrolled_status'] #Not the same data format. 

# # Following columns either need to be added to the backend Databse or Removed before training the new model. 
#    'Employed Status'
#    'Why Not Employed',
#    'Count of Bed Nights (Housing Check-ins)'
#    'Date of Last ES Stay (Beta)'
#    'Date of First ES Stay (Beta)'
#    'Income Total at Entry'
#    'Income Total at Exit'
#    'Non-Cash Benefit Count'
#     'Non-Cash Benefit Count at Exit',

  results_dict['barrier_count'] = 0 
  
  for item in results['barriers'].values():
    if item == True:
      results_dict['barrier_count'] += 1
#Following column not in database 
  # 'Under Years Old' # Probably need to remove from model. ? . 
#Health Issues      
 # results_dict['chronic_health_issues'] = results['barriers']['chronic_health_issues'] # different format backend from dataset. boolean/string. 

#  results_dict['mental_illness'] = results['barriers']['mental_illness'] # different data format backend from dataset boolean/string.
  #The following columns are not on the backend database
  # 'CaseChildren'
  # 'CaseAdults'

  # The following columns need to be condensed to one in data science side . exist on backend as ['health_insurance_type']
   
  # 'Other Public'
  # 'State Funded'
  # 'Indian Health Services (IHS)'
  # 'Other'
  # 'Combined Childrens HealthInsurance/Medicaid'
  # 'Medicaid'
  # 'Medicare'
  # 'State Children's health Insurance S-CHIP'
  # 'Veteran's Administration Medical Services'
  # 'Health Insurance obtained through COBRA'
  # 'Private - Employer'
  # 'Private'
  # 'Private - Individual'

# Following Columns are not on the backend database 
 # 'Earned Income',
 # 'Unemployment Insurance'
 # 'Supplemental Security Income',
 # 'Social Security Disability  Income'
 # 'VA Disability Compensation'
 # 'Private Disability Income'
 # 'Workers Compensation'
 # 'TANF',
 # 'General Assistance'
 # 'Child Support'
 # 'Other Income' 
  

 # results_dict['current_age'] = int((today_date - date_of_birth).days / 365.2425 #used on previous model. not used on this model. 
 #     )


   ```
**If you are having an issue with the existing project code, please submit a bug report under the following guidelines:**

- Check first to see if your issue has already been reported.
- Check to see if the issue has recently been fixed by attempting to reproduce the issue using the latest master branch in the repository.
- Create a live example of the problem.
- Submit a detailed bug report including your environment & browser, steps to reproduce the issue, actual and expected outcomes, where you believe the issue is originating from, and any potential solutions you have considered.


# Support
Robert, Lester, Erle on slack

### Contributing

When contributing to this repository, please first discuss the change you wish to make via issue, email, or any other method with the owners of this repository before making a change.

Describe what you have changed in this repo as a team
Provide examples and descriptions of components, how props are handled, where to find these changes, database tables, models, etc.

### Feature Requests

We would love to hear from you about new features which would improve this app and further the aims of our project. Please provide as much detail and information as possible to show us why you think your new feature should be implemented.

### Pull Requests

If you have developed a patch, bug fix, or new feature that would improve this app, please submit a pull request. It is best to communicate your ideas with the developers first before investing a great deal of time into a pull request to ensure that it will mesh smoothly with the project.

Remember that this project is licensed under the MIT license, and by submitting a pull request, you agree that your work will be, too.

#### Pull Request Guidelines

- Ensure any install or build dependencies are removed before the end of the layer when doing a build.
- Update the README.md with details of changes to the interface, including new plist variables, exposed ports, useful file locations and container parameters.
- Ensure that your code conforms to our existing code conventions and test coverage.
- Include the relevant issue number, if applicable.
- You may merge the Pull Request in once you have the sign-off of two other developers, or if you do not have permission to do that, you may request the second reviewer to merge it for you.

### Attribution

These contribution guidelines have been adapted from [this good-Contributing.md-template](https://gist.github.com/PurpleBooth/b24679402957c63ec426).

### Documentation
[Front End](https://github.com/Lambda-School-Labs/family-promise-spokane-fe-a/blob/main/README.md)

[Back End](https://github.com/Lambda-School-Labs/family-promise-spokane-be-a/blob/main/README.md)

[Data Science](https://github.com/Lambda-School-Labs/family-promise-spokane-ds-a/blob/main/README.md)