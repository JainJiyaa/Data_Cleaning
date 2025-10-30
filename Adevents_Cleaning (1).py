#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns


# In[2]:


ad_events=pd.read_csv(r"C:\Users\jjiya\Downloads\Marketing Campaign\Marketing Campaign\ad_events.csv")
ad_events


# In[3]:


ad_events.user_id.unique()


# In[4]:


ad_events.info()


# In[5]:


ad_events.isnull().sum()


# In[6]:


ad_events.duplicated().sum()


# In[7]:


ad_events.describe(include="all")


# In[8]:


ad_events.columns


# **CHECKING THE OUTLIERS IN NUMERICAL COLUMNS:**

# In[9]:


ad_events.boxplot(column=['ad_id'])


# ## Performing EDA 

# 1. How many unique ads, users, and event types are there?

# In[10]:


# Count unique values
unique_ads = ad_events['ad_id'].nunique()
unique_users = ad_events['user_id'].nunique()
unique_event_types = ad_events['event_type'].nunique()

print(f"Unique Ads: {unique_ads}")
print(f"Unique Users: {unique_users}")
print(f"Unique Event Types: {unique_event_types}")


# 2. What are the most and least common event_types?

# In[11]:


event_counts = ad_events["event_type"].value_counts()
# Most and least common
most_common_event = event_counts.idxmax()
least_common_event = event_counts.idxmin()

print("\nMost Common Event Type:", most_common_event, "-", event_counts.max(), "occurrences")
print("Least Common Event Type:", least_common_event, "-", event_counts.min(), "occurrences")


# 3. How are events distributed across days of the week and times of day?

# In[15]:


# Cross-tab for day vs time of day
day_time_distribution = pd.crosstab(ad_events['day_of_week'], ad_events['time_of_day'])
print("Day vs Time of Day Distribution:\n", day_time_distribution)


# In[16]:


import seaborn as sns
import matplotlib.pyplot as plt

sns.heatmap(day_time_distribution, annot=True, fmt="d", cmap="YlGnBu")
plt.title("Event Distribution: Day vs Time of Day")
plt.xlabel("Time of Day")
plt.ylabel("Day of Week")
plt.show()


# 4. Convert timestamp to datetime — what’s the date range of the dataset?

# In[17]:


# Convert timestamp and check range
ad_events['timestamp'] = pd.to_datetime(ad_events['timestamp'])

start_date = ad_events['timestamp'].min()
end_date = ad_events['timestamp'].max()
duration = end_date - start_date

print(f"Date range: {start_date} → {end_date}")
print(f"Total duration: {duration}")


# 5. Which day or time of day sees the highest engagement (likes/shares)?

# In[18]:


# Step 1: Create new columns for analysis
ad_events['day_of_week'] = ad_events['timestamp'].dt.day_name()
ad_events['hour'] = ad_events['timestamp'].dt.hour

# Step 2: Filter engagement events only (likes + shares)
eng = ad_events[ad_events['event_type'].str.lower().isin(['like', 'share'])]

# Step 3: Find the most engaging day
top_day = eng['day_of_week'].value_counts().idxmax()

# Step 4: Find the most engaging hour
top_hour = eng['hour'].value_counts().idxmax()

print("Day with highest engagement:", top_day)
print("Hour with highest engagement:", top_hour)


# 6. How many events does each user perform on average?

# In[19]:


ad_events.groupby('user_id')['event_id'].count().mean()


# 7. Who are the top 10 most active users?

# In[27]:


top_users = ad_events['user_id'].value_counts().head(10)
print(top_users)


# 8. Are there users who interact with only one type of event?
# 

# In[30]:


# Find users who did only one event type
one_type = ad_events.groupby('user_id')['event_type'].nunique()

# Keep only those users
one_type_users = one_type[one_type == 1]

print("Users with only one type of event:")
print(one_type_users)


# 9. Which ads get the most interactions (likes/shares)?

# In[32]:


# Find which ads got most likes or shares
ads = ad_events[ad_events['event_type'].isin(['like', 'share'])]
top_ads = ads['ad_id'].value_counts().head(10)

print(top_ads)


# 10. Are some ads more popular on specific days or times?

# In[33]:


# Only likes and shares
ads = ad_events[ad_events['event_type'].isin(['like', 'share'])]

# Check by day
ads['day'] = ads['timestamp'].dt.day_name()
print(ads.groupby(['ad_id', 'day']).size())

# Check by hour
ads['hour'] = ads['timestamp'].dt.hour
print(ads.groupby(['ad_id', 'hour']).size())


# In[34]:


ad_events


# 11. Are there duplicate event_ids or duplicate rows?

# In[35]:


ad_events['event_id'].duplicated().sum()


# 12. Which users perform the most events overall?

# In[31]:


top_users = ad_events['user_id'].value_counts().head(5)

plt.bar(top_users.index, top_users.values)
plt.title('Top 5 Most Active Users')
plt.xlabel('User ID')
plt.ylabel('Number of Events')
plt.xticks(rotation=45)
plt.show()


# 13. How does user activity vary by day or time of day?

# In[40]:


ad_events['day'] = ad_events['timestamp'].dt.day_name()
ad_events['hour'] = ad_events['timestamp'].dt.hour

# Activity by day
by_day = ad_events['day'].value_counts()
plt.bar(by_day.index, by_day.values)
plt.title('User Activity by Day')
plt.xlabel('Day')
plt.ylabel('Number of Events')
plt.xticks(rotation=45)
plt.show()

# Activity by hour
by_hour = ad_events['hour'].value_counts().sort_index()
plt.plot(by_hour.index, by_hour.values, marker='o')
plt.title('User Activity by Hour')
plt.xlabel('Hour of Day')
plt.ylabel('Number of Events')
plt.show()


# 14. Ads with Most Interactions (Likes + Shares) 

# In[46]:


top_ads = ad_events['ad_id'].value_counts().head(10)

sns.barplot(x=top_ads.index, y=top_ads.values, palette='coolwarm')
plt.title('Top 10 Ads with Most Interactions')
plt.xlabel('Ad ID')
plt.ylabel('Total Interactions')
plt.xticks(rotation=45)
plt.show()


# 15. Most Active Users

# In[47]:


top_users = ad_events['user_id'].value_counts().head(10)

sns.barplot(x=top_users.index, y=top_users.values, palette='viridis')
plt.title('Top 10 Most Active Users')
plt.xlabel('User ID')
plt.ylabel('Number of Events')
plt.xticks(rotation=45)
plt.show()


# 16. User Activity by Day

# In[49]:


ad_events['day'] = ad_events['timestamp'].dt.day_name()

sns.countplot(data=ad_events, x='day', palette='coolwarm')
plt.title('User Activity by Day')
plt.xlabel('Day of Week')
plt.ylabel('Number of Events')
plt.xticks(rotation=45)
plt.show()


# 17. User Activity By Hour

# In[50]:


ad_events['hour'] = ad_events['timestamp'].dt.hour

sns.countplot(data=ad_events, x='hour', palette='crest')
plt.title('User Activity by Hour of Day')
plt.xlabel('Hour')
plt.ylabel('Number of Events')
plt.show()


# 18. Users With Only One Type of Event

# In[51]:


one_type = ad_events.groupby('user_id')['event_type'].nunique()
single_users = one_type[one_type == 1].reset_index()

sns.barplot(data=single_users, x='user_id', y='event_type', palette='magma')
plt.title('Users with Only One Type of Event')
plt.xlabel('User ID')
plt.ylabel('Unique Event Type Count')
plt.xticks(rotation=45)
plt.show()

