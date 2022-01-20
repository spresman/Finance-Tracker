import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import sklearn.metrics

df = pd.read_csv(r'C:/users/sam/downloads/MockActivity.csv', index_col=False)

deb_cred = df["Details"]

date = df["Posting Date"]

date_obj = [datetime.strptime(d, '%m/%d/%Y') for d in date]

today_month = datetime.today().month

# Sorts it with newer transactions last
date_obj = sorted(date_obj)

print(date_obj[0])

day_sincefirst = [(date_obj[i] - date_obj[0]).days for i,j in enumerate(date_obj)]

desc = df["Description"]

date_less_descr = [i[:len(i)-5] if i[len(i)-3] == '/' else i for i in desc]


labels = tuple(set(date_less_descr))
sizes = [date_less_descr.count(labels[i]) for i, j in enumerate(labels)]

fig1, ax1 = plt.subplots()
ax1.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
ax1.axis('equal')
ax1.set_title('Expenses by Frequency')


amount = df["Amount"]
expenses = [-1 * i if i < 0 else 0 for i in amount]

expensedict = {i: 0 for i in labels}

for i in expensedict:
    for q in range(len(expenses)):
        if i == date_less_descr[q]:
            expensedict[i] += expenses[q]

expenses_dateless = [expensedict[i] for i in labels]


fig2, ax2 = plt.subplots()
ax2.pie(expenses_dateless, labels=labels, autopct='%1.1f%%', startangle=90)
ax2.axis('equal')
ax2.set_title('Expenses by Amount')

type1 = df["Type"]

balance = df["Balance"]

x = day_sincefirst
y = balance


# Model returns an array of 0: regression coefficient, and 1: intercept
# Predicts x amount of days from the initial day
model = np.polyfit(x, y, 1)

pretty_model = ['{:.3f}'.format(model[i]) for i in range(2)]

print("--------------------------------------\n" +
      "Linear Regression for Initial -> Present")

print(f"y = {pretty_model[0]}x + {pretty_model[1]}")

predict = np.poly1d(model)

# R_2 value
r2 = '{:.3f}'.format(sklearn.metrics.r2_score(y, predict(x)))

print(f'R^2: {r2}'
      + "\n--------------------------------------")

x_r = range(0, x[-1])
y_r = predict(x_r)

fig3, ax3 = plt.subplots()
ax3.scatter(x, y)
ax3.plot(x_r, y_r, c = 'r')

ax3.set(xlabel='Num Days since Initial Day', ylabel='Balance')
ax3.set_title('Balance($) vs. Days')

plt.show()






