## EMPLOYEE ATTRITION CLASSIFICATION
->This project aims to predict employee attrition using various features related to employees' demographics, job satisfaction, performance, and company characteristics. By analyzing these factors, the model can help HR departments to identify employees who are likely to leave the company and take proactive measures to retain them.

## Prerequisites
 Python 3.x
 Pip (Python package installer)
 AWS CLI
 AWS Elastic Beanstalk CLI

### Table of Contents
1.Project Overview
2.Features
3.Installation
4.Usage
5.Model Training
6.Prediction
7.Results
8.Contributing
9.License 


## Project Overview
Employee attrition, or turnover, can be a significant issue for companies, leading to high costs for recruiting and training new employees. This project leverages machine learning techniques to predict whether an employee is likely to leave the company based on various features.

## Features
* Demographic Information: Gender, Education Level, Marital Status, Number of Dependents.
* Job-related Information: Years at Company, Job Level, Job Satisfaction, Performance Rating, Number of Promotions, Overtime, Distance from Home, Remote Work.
*Company-related Information: Work-Life Balance, Company Size, Company Reputation, Employee Recognition.

## Installation
->To run this project locally, you need to have Python and the necessary libraries installed.
1.Clone the repo.;
        git clone https://github.com/steve601/employeeAttrition.git
        cd employeeAttrition
2.Create a virtual environment and activate it:
        python -m venv venv
        source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
3.Install the required packages:
        pip install -r requirements.txt

## Usage
=>Load the Model,then run the emp.py file on localhost then perform your prediction

## Deployment to AWS Elastic Beanstalk
1.Initialize Elastic Beanstalk:
      eb init
  Follow the prompts to configure your application. Choose the appropriate region and platform (Python).
2.Create an Elastic Beanstalk environment:
      eb create flask-env
3.Deploy the application:
      eb deploy
4.Access the application: After deployment, access your application using the URL provided by Elastic Beanstalk.

## Configuration File: .ebextensions/python.config
=>Ensure your configuration file is set up correctly:
       option_settings:
         aws:elasticbeanstalk:container:python:
           WSGIPath: emp:app
## Contributing
Contributions are welcome!

