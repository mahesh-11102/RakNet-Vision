# RakNet-Vision
In today's interconnected world, users demand seamless mobile experiences. Dropped calls, slow videos, and unexpected charges are frustrating. Rakuten Mobile and similar providers need AI-powered tools to predict issues, offer data insights, suggest better connectivity, and ensure privacy, ultimately creating smoother mobile experiences. 

`Tech Stacks `: Python, TensorFlow, Android Studio(Java), SQL, Firebase, Docker 
## Introduction to Expense & Usage Forecaster:
Rakuten's *Expense & Usage Forecaster* serves as a robust solution designed to aid Rakuten Mobile users in effectively budgeting their telecom expenditures.By leveraging historical data and intricate usage patterns, this forecaster offers accurate predictions of prospective monthly expenses. But its utility doesn't end there; it further enriches user experience by recommending tailored data-saving suggestions.
##   Data Genesis and Features:
Central to this solution is the "rakuten_telecom_expense.csv", a hypothetical dataset that encompasses an array of vital features. It documents monthly data usage, call frequencies, call durations, usage stats for services like Rakuten Fashion and Rakuten Recipe, and users' streaming quality preferences. Calculations of the monthly expenses hinge on these data metrics, establishing the foundation for predictive modeling.

## Neural Network Modeling with TensorFlow: 
The heart of the forecaster is a Neural Network-based regression model powered by TensorFlow. This model, enriched with multiple dense layers interlaced with dropout layers, safeguards against overfitting, ensuring consistent and reliable predictions. Trained meticulously on the dataset, the model uses Mean Squared Error (MSE) as its primary loss metric. To offer a clearer insight into its performance, the Mean Absolute Error (MAE) serves as the evaluative metric. 
## Delivering Outcomes to Users:
Once operational, the model equips Rakuten users with precise forecasts of their impending monthly expenses, grounding its predictions in their historical usage data. Beyond this, the model's insights can be harnessed to deliver data-saving suggestions. Examples include advising users to adjust their Rakuten Fashion streaming quality or modulating their Rakuten Recipe usage to align with budget constraints. 
## A Glimpse into the Future:
As we envision the future of the Expense & Usage Forecaster, there's an array of enhancements on the horizon. There's ample room to integrate more features, refining the predictive accuracy. Moreover, a dedicated recommendation engine could be conceived, blending the model's forecasts with nuanced user

# Model Architecture and Data used
## Smart Connectivity Guide:
### Since the dataset for this model aren't available  publicly, created a hypothetical datastet and worked on it

### Architecture
````
Model: "sequential_39"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 lstm_20 (LSTM)              (None, 10, 100)           45200     
                                                                 
 dropout_71 (Dropout)        (None, 10, 100)           0         
                                                                 
 lstm_21 (LSTM)              (None, 50)                30200     
                                                                 
 dense_122 (Dense)           (None, 1)                 51        
                                                                 
=================================================================
Total params: 75,451
Trainable params: 75,451
Non-trainable params: 0
_________________________________________________________________
````
### Hypothetical Dataset: [`synthetic_telecom_data.csv`](https://github.com/mahesh-11102/RakNet-Vision/blob/main/synthetic_telecom_data.csv)
1. **Timestamped Data**: 
    - **Date and Time**: Generate a continuous series of timestamps. For instance, every minute for a year.

2. **Traffic Volume**:
    - **Data Rate**: Use a random function (with some constraints) to emulate network traffic volume. Consider adding daily or weekly patterns (e.g., higher traffic during evenings or weekends).
    - **Packet Rate**: Similarly, generate random values, ensuring they're in sync with the Data Rate (higher data rate should correlate with higher packet rates).

3. **Traffic Type**:
    - **Voice**: Randomly generate voice traffic volumes; perhaps show dips during non-peak hours.
    - **Text**: Random numbers for text traffic, possibly showing spikes during certain hours.
    - **Data**: Categorized data for different services. For instance, video streaming might be higher during evenings.

4. **Network Metrics**:
    - **Latency**: Generate values with slight fluctuations, but introduce occasional spikes to represent network congestion.
    - **Jitter**: Randomly generated values around a baseline, with occasional peaks.
    - **Packet Loss Rate**: Mostly low values with occasional spikes.
    - **Error Rates**: Introduce random errors sporadically.

5. **Infrastructure Information**:
    - **Cell Tower Data**: Choose a fixed number of towers and distribute the generated traffic among them.
    - **Network Topology**: This can be a static part of your dataset, indicating your hypothetical infrastructure.
    - **Hardware Status**: Generate random 'uptime' periods, occasionally introducing 'downtime'.

6. **Signal Information**:
    - **Signal Strength**: Fluctuate around a strong signal but introduce occasional weak signals.
    - **Noise Levels**: Randomly generate noise levels, higher noise could correlate with lower signal strength.

7. **User Data**:
    - **Active Users**: Fluctuate this value to show user activity, e.g., more users during the day and fewer at night.
    - **User Behavior**: Generate patterns, like more streaming during certain hours.

8. **External Factors**:
    - **Events**: Introduce occasional 'event days' with higher traffic.
    - **Weather Data**: Randomly assign weather conditions, and perhaps on 'stormy' days, show some deterioration in signal quality.

9. **Historical Incident Data**:
    - Randomly introduce past incidents, noting the duration and cause.

10. **Service Level Agreements (SLAs)**:
    - This could be a static part of your dataset, indicating your hypothetical SLA standards.
   
## AI-Powered Network Maintenance & Outage Predictor:
### Architectture

```
Model: "sequential_43"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 dense_136 (Dense)           (None, 256)               512       
                                                                 
 dropout_79 (Dropout)        (None, 256)               0         
                                                                 
 dense_137 (Dense)           (None, 128)               32896     
                                                                 
 dropout_80 (Dropout)        (None, 128)               0         
                                                                 
 dense_138 (Dense)           (None, 64)                8256      
                                                                 
 dropout_81 (Dropout)        (None, 64)                0         
                                                                 
 dense_139 (Dense)           (None, 32)                2080      
                                                                 
 dense_140 (Dense)           (None, 1)                 33        
                                                                 
=================================================================
Total params: 43,777
Trainable params: 43,777
Non-trainable params: 0
_________________________________________________________________
```
### Hypothetical Dataset: [`hypothetical_telecom_data.csv`](https://github.com/mahesh-11102/RakNet-Vision/blob/main/hypothetical_telecom_data.csv)

1. **user_id**:
    - **Description**: Unique identifier assigned to each telecom user.
    - **Type**: Integer
    - **Example Value**: 1, 2, 3, ... 1000
    - **Purpose**: Helps in tracking individual users' behavior and monthly statistics.

2. **month_year**:
    - **Description**: Represents the month and year during which the data was collected.
    - **Type**: String
    - **Example Value**: "Jan_2023", "Feb_2023", ...
    - **Purpose**: Helps in tracking and analyzing data on a monthly basis.

3. **app1_data_used** to **app5_data_used**:
    - **Description**: The amount of data (in GB) used by a user on 5 different apps in the respective month. For simplicity, we've named them `app1`, `app2`, etc., but in a real-world scenario, they might be popular apps like YouTube, Netflix, Facebook, etc.
    - **Type**: Float
    - **Example Value**: 4.5, 2.3, ...
    - **Purpose**: To analyze individual app usage patterns, which can be utilized to make data-saving recommendations.

4. **total_data_used**:
    - **Description**: The total amount of data (in GB) used by a user in that month. It's the summation of data used across all apps and other background data usages.
    - **Type**: Float
    - **Example Value**: 20.5, 35.2, ...
    - **Purpose**: Key metric to predict monthly expenses and understand overall data consumption.

5. **monthly_expense**:
    - **Description**: The total amount of money (in a hypothetical currency) the user has been charged for their data and other telecom services in that month.
    - **Type**: Float
    - **Example Value**: 45.2, 60.5, ...
    - **Purpose**: The target variable for our predictive analytics model. Based on previous data usage, this model aims to predict future expenses.
  
## Predictive Network Maintenance:
### Architecture
```
Model: "sequential_40"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 dense_123 (Dense)           (None, 128)               768       
                                                                 
 dropout_72 (Dropout)        (None, 128)               0         
                                                                 
 dense_124 (Dense)           (None, 64)                8256      
                                                                 
 dropout_73 (Dropout)        (None, 64)                0         
                                                                 
 dense_125 (Dense)           (None, 32)                2080      
                                                                 
 dense_126 (Dense)           (None, 1)                 33        
                                                                 
=================================================================
Total params: 11,137
Trainable params: 11,137
Non-trainable params: 0
_________________________________________________________________
```
### Hypothetical Dataset: [`hypothetical_pred_telecom_data.csv`](https://github.com/mahesh-11102/RakNet-Vision/blob/main/hypothetical_pred_telecom_data.csv)

1. **total_data_used (in GB)**
    - Description: Represents the total data consumed by users in a specific cell or region for a particular day.
    - Range: 1-100 GB
    - Interpretation: A sudden surge in data usage might be a sign of an impending network issue or even a sign of more users joining the network. Conversely, a sudden drop might indicate network problems that prevent users from consuming data.

2. **daily_new_users**
    - Description: This indicates the number of new users added daily to the network in a specific region.
    - Range: 10-1000 users
    - Interpretation: A spike in new users can strain the network and might indicate potential future network congestion. It also provides insight into regions where marketing or expansion strategies are working.

3. **server_load (in percentage)**
    - Description: Represents the average server load for the day.
    - Range: 50%-100%
    - Interpretation: If the load is consistently above 90%, it suggests that the servers are almost at their capacity. This can lead to potential downtimes if not addressed. A higher server load also means a greater chance of overheating or other hardware failures.

4. **faulty_hardware_reports**
    - Description: This column represents the number of hardware components (like routers, switches, etc.) reported faulty in a day.
    - Range: Mostly in the range 0-10, based on a Poisson distribution with a mean of 3.
    - Interpretation: An increase in faulty hardware reports is an immediate red flag. It means more equipment is failing, which can directly impact network uptime and efficiency.

5. **signal_strength (on a scale of 1 to 5)**
    - Description: The average signal strength for the region. A value of 1 represents weak signal strength, while 5 represents very strong signal strength.
    - Range: 1-5
    - Interpretation: A decreasing trend in signal strength might indicate potential hardware issues or signal interferences. A high signal strength typically means a healthier network.

6. **network_issue (binary: 0 or 1)**
    - Description: Target variable. If there was a network issue the next day, the value is 1, otherwise 0.
    - Interpretation: This is a derived column based on certain conditions in the data (like server load above 90% or more than 5 faulty hardware reports). It represents whether the network faced any significant issues or not.

### Data Assumptions:

- Server Load: A consistent load of over 90% for several days is risky. This suggests that servers are close to their maximum capacity, which can cause slowdowns or outages.
- Faulty Hardware Reports: If there are more than 5 reports in a day, it's an alarming situation indicating increased chances of network issues.
- The data is for a specific region or cell. For a comprehensive analysis, data from multiple regions or cells would be combined.
- External factors like weather conditions, which can impact signal strengths and hardware health, are not considered in this hypothetical dataset.

Remember, this dataset is entirely hypothetical. In a real-world scenario, more columns and complex interdependencies might exist.


## Expense & Usage Forecaster:
### Architecture
```
Model: "sequential_41"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 dense_127 (Dense)           (None, 64)                640       
                                                                 
 dropout_74 (Dropout)        (None, 64)                0         
                                                                 
 dense_128 (Dense)           (None, 32)                2080      
                                                                 
 dropout_75 (Dropout)        (None, 32)                0         
                                                                 
 dense_129 (Dense)           (None, 16)                528       
                                                                 
 dense_130 (Dense)           (None, 1)                 17        
                                                                 
=================================================================
Total params: 3,265
Trainable params: 3,265
Non-trainable params: 0
_________________________________________________________________
```
### Hypothetical Dataset: [`rakuten_telecom_expense.csv`](https://github.com/mahesh-11102/RakNet-Vision/blob/main/rakuten_telecom_expense.csv)

1. **monthly_data_usage (in GB)**
    - Description: Represents the total data consumed by a specific user in a month.
    - Range: 0-50 GB
    - Interpretation: Higher data usage generally leads to higher monthly telecom expenses. Specific apps/services might use more data than others.

2. **number_of_calls**
    - Description: Total number of calls made by the user in the month.
    - Range: 0-300
    - Interpretation: A higher number of calls can increase monthly expenses. 

3. **call_duration (in hours)**
    - Description: Total hours spent on calls by the user in the month.
    - Range: 0-100 hours
    - Interpretation: Longer call durations can contribute to higher monthly expenses, especially if not on unlimited plans.

4. **rakuten_fashion_usage (in hours)**
    - Description: Time spent by the user on the Rakuten Fashion app.
    - Range: 0-50 hours
    - Interpretation: More time on the app might lead to more data consumption, directly impacting the monthly expense. Also provides insight into user preferences.

5. **rakuten_recipe_usage (in hours)**
    - Description: Time spent by the user on the Rakuten Recipe app.
    - Range: 0-50 hours
    - Interpretation: As with Rakuten Fashion, more time on this app indicates higher data usage and consequently higher monthly expenses.

6. **streaming_quality_preference (480p, 720p, 1080p, 4K)**
    - Description: The user's preferred streaming quality setting for video apps/services.
    - Interpretation: Higher streaming quality like 1080p or 4K consumes more data than 480p, leading to higher expenses if the user is on a limited data plan.

7. **monthly_expense (in USD)**
    - Description: The target variable, representing the user's total expense for the month on Rakuten Mobile services.
    - Range: 5-200 USD
    - Interpretation: A composite of all the usage factors along with the user's subscription plan.

### Data Assumptions:

- Monthly Data Usage: The relationship between data usage and monthly expenses might be nonlinear. For example, users on unlimited data plans might not see an increase in expenses after a certain threshold.
  
- Rakuten App Usages: Using Rakuten apps might come with certain benefits, like data-saving modes or discounts on data usage.

- Call Expenses: Not all calls might be charged the same. International calls or calls to other networks might have different tariffs.

- Streaming Quality: Users might have set their preferred streaming quality, but actual consumption can vary. For example, they might be watching at 480p even if they've set a preference for 1080p, depending on their network connection or device capabilities.

- External factors like promotional discounts, loyalty benefits, or additional services not captured in the dataset can also impact the monthly expense.

The dataset is entirely hypothetical. In a real-world scenario, the dataset might contain more features, more complex relationships between variables, and a variety of external influencing factors.
