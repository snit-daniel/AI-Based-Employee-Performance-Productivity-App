from flask import Flask, render_template, jsonify, redirect, url_for, request, session, flash
from flask_mail import Mail, Message
from chat import process_contact_message, answer_hr_question  # Import from chat.py
import re
import pandas as pd
import os
import plotly.express as px
import mysql.connector
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
from multiprocessing import get_start_method
import matplotlib
matplotlib.use('Agg')
#from sklearn.preprocessing import LabelEncoder
from flask_bcrypt import Bcrypt
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
import base64
from sklearn.preprocessing import LabelEncoder
from datetime import datetime


# Initialize start method
start_method = get_start_method()
print(f"Using start method: {start_method}")

# Load the pre-trained model
model_path = os.path.join(os.path.expanduser("~"), "DOWNLOADS", "Final trial", "random_forest_model.pkl")
try:
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        print("✅ Model loaded successfully!")
    else:
        print(f"❌ Model file not found at {model_path}")
        model = None
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

# Initialize Flask app
app = Flask(__name__)
app.secret_key = "your_secret_key"

# Initialize extensions
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = "login"




# Email Configuration
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = "snitdan17@gmail.com"  # Your Gmail address
app.config['MAIL_PASSWORD'] = "pfpa qwni sxjh natz"  # Your App Password or regular password
app.config['MAIL_DEFAULT_SENDER'] = "snitdan17@gmail.com"  # Default sender

# Initialize Flask-Mail
mail = Mail(app)

# Database connection
def get_db_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="snit",
        database="employee_performance"
    )

# Function to send an HR email
def send_hr_email(employee, status):
    try:
        subject = f"Alert: {status} Detected for Employee {employee['Name']}"
        body = f"""
        Dear HR,
        This is to notify you that employee {employee['Name']} (ID: {employee['Employee_ID']}) has a {status}.
        Performance Score: {employee['Performance_Score']}

        Please take necessary action.

        Best regards,
        Performance Monitoring System
        """
        msg = Message(
            subject=subject,
            sender=app.config['MAIL_DEFAULT_SENDER'],
            recipients=["snitdan17@gmail.com"]  # Send to yourself for testing
        )
        msg.body = body
        mail.send(msg)
        print(f"Email sent successfully to HR about {employee['Name']}")
        return True
    except Exception as e:
        print(f"Failed to send email: {str(e)}")
        return False

def check_and_notify_low_performance():
    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)

    low_performance_threshold = 2.5

    cursor.execute("SELECT * FROM employees WHERE Performance_Score < %s", (low_performance_threshold,))
    low_performers = cursor.fetchall()

    for employee in low_performers:
        send_hr_email(employee, status="Low Performance")

    cursor.close()
    connection.close()
    print(f"Checked and sent alerts at {datetime.now()}")




@app.route('/test-email')
def test_email():
    try:
        # Create a test employee
        test_employee = {
            'Name': 'Test Employee',
            'Employee_ID': '123',
            'Performance_Score': 2.0
        }
        
        # Send test email
        success = send_hr_email(test_employee, "Low Performance")
        
        if success:
            return "Test email sent successfully! Check your inbox (and spam folder)."
        else:
            return "Failed to send test email."
    except Exception as e:
        return f"Error: {str(e)}"

# User Class for Flask-Login
class User(UserMixin):
    def __init__(self, id, username, email):
        self.id = id
        self.username = username
        self.email = email

@login_manager.user_loader
def load_user(user_id):
    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)
    cursor.execute("SELECT * FROM users WHERE id = %s", (user_id,))
    user = cursor.fetchone()
    cursor.close()
    connection.close()
    if user:
        return User(user["id"], user["username"], user["email"])
    return None

# Helper function to get employee data
def get_employee_data(employee_id):
    connection = mysql.connector.connect(
        host="localhost",
            user="root",
            password="snit",
        database="employee_performance"
    )
    cursor = connection.cursor(dictionary=True)
    cursor.execute("SELECT * FROM employees WHERE Employee_ID = %s", (employee_id,))
    employee = cursor.fetchone()
    cursor.close()
    connection.close()
    return employee

# Helper function to send HR emails
def send_hr_email(employee, status):
    if not app.config['MAIL_USERNAME'] or not app.config['MAIL_PASSWORD']:
        print("Email configuration missing!")
        return
    
    subject = f"Employee Performance Alert: {employee['Employee_ID']}"
    body = f"""
    HR Team,

    Employee {employee['Employee_ID']} - {employee['Job_Title']} in {employee['Department']} has been classified as: {status}.

    Recommended Action: {status}

    Regards,
    AI Performance System
    """
    msg = Message(subject, sender=app.config['MAIL_USERNAME'], recipients=["snitdan17@gmail.com"])
    msg.body = body
    mail.send(msg)

# Routes
@app.route('/')
def index():
    return redirect(url_for('login'))

@app.route('/home')
@login_required
def home():
    return render_template('home.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')

        try:
            connection = get_db_connection()
            cursor = connection.cursor()
            cursor.execute("INSERT INTO users (username, email, password) VALUES (%s, %s, %s)", 
                         (username, email, hashed_password))
            connection.commit()
            cursor.close()
            connection.close()
            flash('Account created successfully! Please log in.', 'success')
            return redirect(url_for('login'))
        except Exception as e:
            flash('Error: ' + str(e), 'danger')

    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        connection = get_db_connection()
        cursor = connection.cursor(dictionary=True)
        cursor.execute("SELECT * FROM users WHERE email = %s", (email,))
        user = cursor.fetchone()
        cursor.close()
        connection.close()

        if user and bcrypt.check_password_hash(user['password'], password):
            login_user(User(user["id"], user["username"], user["email"]))
            session['user_id'] = user["id"]
            flash('Login successful!', 'success')
            return redirect(url_for('home'))
        else:
            flash('Invalid email or password', 'danger')
            return redirect(url_for('login'))

    return render_template('login.html')

@app.route('/logout')
def logout():
    logout_user()
    session.pop('user_id', None)
    flash('You have been logged out successfully.', 'info')
    return redirect(url_for('login'))

@app.route('/visualization')
def visualization():
    try:
        connection = get_db_connection()
        cursor = connection.cursor(dictionary=True)
        cursor.execute("SELECT * FROM employees")
        data = cursor.fetchall()
        df = pd.DataFrame(data)

        # Ensure columns are of the correct type
        df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
        df['Performance_Score'] = pd.to_numeric(df['Performance_Score'], errors='coerce')
        df['Monthly_Salary'] = pd.to_numeric(df['Monthly_Salary'], errors='coerce')
        df['Projects_Handled'] = pd.to_numeric(df['Projects_Handled'], errors='coerce')
        df['Years_At_Company'] = pd.to_numeric(df['Years_At_Company'], errors='coerce')

        # Drop rows with NaN values
        df = df.dropna(subset=['Age', 'Performance_Score', 'Monthly_Salary', 'Years_At_Company'])

        # Create the scatter plot for Performance Score vs Salary
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        ax1.scatter(df['Monthly_Salary'], df['Performance_Score'], alpha=0.5, color='blue')
        ax1.set_xlabel("Monthly Salary")
        ax1.set_ylabel("Performance Score")
        ax1.set_title("Performance Score vs Monthly Salary")
        
        # Save the scatter plot
        img1 = BytesIO()
        fig1.savefig(img1, format='png')
        img1.seek(0)
        scatter_plot_url = base64.b64encode(img1.getvalue()).decode('utf8')

        # Create the count plot for Projects Handled by Age
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        sns.countplot(x='Age', data=df[df['Projects_Handled'] == 1], palette="magma", ax=ax2)
        ax2.set_title("Projects Handled by Age")
        ax2.set_xlabel('Age')
        ax2.set_ylabel('Number of Projects Handled')
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')

        # Save the count plot
        img2 = BytesIO()
        fig2.savefig(img2, format='png')
        img2.seek(0)
        count_plot_url = base64.b64encode(img2.getvalue()).decode('utf8')

        # Create the pie chart for Department Distribution
        department_counts = df['Department'].value_counts()
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        ax3.pie(department_counts, labels=department_counts.index, autopct='%1.1f%%', 
               startangle=90, colors=sns.color_palette("Set3", len(department_counts)))
        ax3.axis('equal')
        ax3.set_title("Employee Distribution by Department")

        # Save the pie chart
        img3 = BytesIO()
        fig3.savefig(img3, format='png')
        img3.seek(0)
        pie_chart_url = base64.b64encode(img3.getvalue()).decode('utf8')

        # Create the line plot for Performance Score vs Years at the Company
        df_grouped = df.groupby('Years_At_Company').agg({'Performance_Score': 'mean'}).reset_index()
        fig5, ax5 = plt.subplots(figsize=(10, 6))
        ax5.plot(df_grouped['Years_At_Company'], df_grouped['Performance_Score'], marker='o', color='purple')
        ax5.set_xlabel("Years at Company")
        ax5.set_ylabel("Average Performance Score")
        ax5.set_title("Performance Score vs Years at the Company")

        # Save the line plot
        img5 = BytesIO()
        fig5.savefig(img5, format='png')
        img5.seek(0)
        line_chart_url = base64.b64encode(img5.getvalue()).decode('utf8')

        # Close the database connection
        cursor.close()
        connection.close()

        return render_template('visualization.html', 
                            scatter_plot_url=scatter_plot_url, 
                            count_plot_url=count_plot_url, 
                            pie_chart_url=pie_chart_url,
                            line_chart_url=line_chart_url)

    except Exception as e:
        return f"Error: {e}"


# Performance Prediction Route
@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    # Send email with all low performers when page is loaded (GET request)
    if request.method == 'GET':
        try:
            connection = get_db_connection()
            cursor = connection.cursor(dictionary=True)
            
            # Get all employees with performance < 2
            cursor.execute("SELECT * FROM employees WHERE Performance_Score < 2")
            low_performers = cursor.fetchall()
            
            if low_performers:
                # Prepare email content
                subject = f"Low Performers Alert - {datetime.now().strftime('%Y-%m-%d %H:%M')}"
                body = "The following employees have performance scores below 2:\n\n"
                
                for employee in low_performers:
                    body += (f"ID: {employee['Employee_ID']}, "
                            f"Score: {employee['Performance_Score']}, "
                            f"Department: {employee['Department']}\n")
                
                body += "\nPlease review their performance."
                
                # Send email
                msg = Message(
                    subject=subject,
                    sender=app.config['MAIL_DEFAULT_SENDER'],
                    recipients=["snitdan17@gmail.com"]  # Your email
                )
                msg.body = body
                mail.send(msg)
                
                flash('Low performers report sent to HR', 'info')
            
            cursor.close()
            connection.close()
            
        except Exception as e:
            print(f"Error sending low performers email: {str(e)}")
            flash('Failed to send low performers report', 'error')

    if request.method == 'POST':
        # Handle existing employee search
        if 'employee_id' in request.form:
            employee_id = request.form['employee_id']
            employee = get_employee_data(employee_id)

            if not employee:
                return render_template('prediction.html', error="Employee not found!")

            performance_score = employee.get('Performance_Score')

            if performance_score is None:
                return render_template('prediction.html', error="Performance score not available for this employee!")

            # Determine employee status
            if performance_score < 2:
                status = "At Risk of Layoff"
                recommendation = "Consider retraining or performance improvement plan."
            elif performance_score >= 4:
                status = "Promotion Recommended"
                recommendation = "Eligible for promotion based on performance."
            else:
                status = "Needs Motivation"
                recommendation = "Provide additional incentives or training."

            return render_template('prediction.html', employee=employee, 
                                   score=performance_score, status=status, 
                                   recommendation=recommendation)

        # Handle new employee prediction
        elif 'monthly_salary' in request.form:
            # Get form data
            form_data = request.form.to_dict()

            # Encode categorical features
            categorical_features = ['job_title', 'department', 'education', 'gender']
            le = LabelEncoder()
            for col in categorical_features:
                form_data[col] = le.fit_transform([form_data[col]])[0]

            # Map form data to the feature names used during training
            input_data = {
                "Monthly_Salary": float(form_data["monthly_salary"]),
                "Job_Title": form_data["job_title"],
                "Department": form_data["department"],
                "Sick_Days": int(form_data["sick_days"]),
                "Employee_Satisfaction_Score": float(form_data["employee_satisfaction_score"]),
                "Promotions": int(form_data["promotions"]),
                "Training_Hours": int(form_data["training_hours"]),
                "Team_Size": int(form_data["team_size"]),
                "Remote_Work_Frequency": float(form_data["remote_work_frequency"]),
                "Projects_Handled": int(form_data.get("projects_handled", 0)),  # Default value
                "Overtime_Hours": int(form_data.get("overtime_hours", 0)),  # Default value
                "Gender": form_data["gender"],
                "Work_Hours_Per_Week": int(form_data.get("work_hours_per_week", 40)),  # Default value
                "Education_Level": form_data["education"],
                "Years_At_Company": int(form_data.get("years_at_company", 0))  # Default value
            }

            # Convert input data to DataFrame
            input_df = pd.DataFrame([input_data])

            # Ensure the columns are in the correct order
            feature_columns = [
                "Monthly_Salary", "Job_Title", "Department", "Sick_Days",
                "Employee_Satisfaction_Score", "Promotions", "Training_Hours", "Team_Size",
                "Remote_Work_Frequency", "Projects_Handled", "Overtime_Hours", "Gender",
                "Work_Hours_Per_Week", "Education_Level", "Years_At_Company"
            ]
            input_df = input_df[feature_columns]

            # Make prediction
            prediction = model.predict(input_df)

            # Determine employee status and recommendation based on prediction
            if prediction[0] < 2:
                status = "At Risk of Layoff"
                recommendation = "Consider retraining or performance improvement plan."
            elif prediction[0] >= 4:
                status = "Promotion Recommended"
                recommendation = "Eligible for promotion based on performance."
            else:
                status = "Needs Motivation"
                recommendation = "Provide additional incentives or training."

            # Print variables for debugging
            print(f"Prediction: {prediction[0]}, Status: {status}, Recommendation: {recommendation}")

            # Return prediction result
            return render_template('prediction.html', prediction=prediction[0], 
                                   status=status, recommendation=recommendation)

    # Default GET request
    return render_template('prediction.html')


@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact', methods=['GET', 'POST'])
def contact():
    success = None
    error = None
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        message = request.form['message']

        try:
            # Process the message using chat.py logic
            chat_response = process_contact_message(name, email, message)

            # Send email to the designated recipient
            msg = Message(f"New message from {name}", 
                         sender=app.config['MAIL_USERNAME'], 
                         recipients=["recipient-email@example.com"])
            msg.body = f"Message: {message}\nFrom: {name} ({email})"
            mail.send(msg)

            # Set success message
            success = "Your message has been sent successfully."
        except Exception as e:
            # Set error message
            error = f"An error occurred: {str(e)}"

    return render_template('contact.html', success=success, error=error)

# Chat endpoint (delegates to chat.py)
@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    question = data['message']
    response = answer_hr_question(question)
    return jsonify({'response': response})

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=10000)
