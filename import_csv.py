import mysql.connector
import pandas as pd

# Connect to MySQL
connection = mysql.connector.connect(
    host='localhost',
    user='root',
    password='saron12',
    database='employee_performance'
)

cursor = connection.cursor()

# Read the CSV file
df = pd.read_csv('Extended_Employee_Performance_and_Productivity_Data.csv')

# Convert the 'Hire_Date' column to datetime and ensure it's in the correct format (YYYY-MM-DD HH:MM:SS)
df['Hire_Date'] = pd.to_datetime(df['Hire_Date'], errors='coerce')

# Iterate over the rows of the DataFrame
for index, row in df.iterrows():
    # Ensure Hire_Date is formatted as DATETIME
    hire_date = row['Hire_Date'].strftime('%Y-%m-%d %H:%M:%S') if pd.notna(row['Hire_Date']) else None
    
    # Prepare the data for insertion
    insert_values = (
        row['Employee_ID'], row['Department'], row['Gender'], row['Age'], row['Job_Title'],
        hire_date, row['Years_At_Company'], row['Education_Level'], row['Performance_Score'],
        row['Monthly_Salary'], row['Work_Hours_Per_Week'], row['Projects_Handled'],
        row['Overtime_Hours'], row['Sick_Days'], row['Remote_Work_Frequency'], row['Team_Size'],
        row['Training_Hours'], row['Promotions'], row['Employee_Satisfaction_Score'],
        row['Resigned']
    )
    
    # Insert into the table
    cursor.execute("""
        UPDATE employees SET 
            Department = %s, Gender = %s, Age = %s, Job_Title = %s, Hire_Date = %s,
            Years_At_Company = %s, Education_Level = %s, Performance_Score = %s, Monthly_Salary = %s,
            Work_Hours_Per_Week = %s, Projects_Handled = %s, Overtime_Hours = %s, Sick_Days = %s,
            Remote_Work_Frequency = %s, Team_Size = %s, Training_Hours = %s, Promotions = %s,
            Employee_Satisfaction_Score = %s, Resigned = %s
        WHERE Employee_ID = %s
    """, (*insert_values[1:], insert_values[0]))  # Update with the same insert values, but Employee_ID at the end


# Commit the transaction and close the connection
connection.commit()
cursor.close()
connection.close()

print("Data insertion completed successfully.")
