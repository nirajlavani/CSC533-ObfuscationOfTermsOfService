import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

'''
SCORING ALGO
    1) External, Thirdparty, webt: +1 | loctrack, usercontentanalysis: 0 | account rest, private: -1.5
    2) private: +1 | All else: -1/6
    3) account_rest: +1 | user_anal: 0 |webt, loct, ext, thirdp, priv = -1/5
    4) loct: +1 | ext, webt: 0 | user_anal, account_rest, thirdp, priv = -1/4
    5) user_anal: +1 | private: 0 | webt, loct, account_rest, thirdp, ext = -1/5
    6) External, Thirdparty, webt: +1 | loctrack, usercontentanalysis: 0 | account rest, private: -1.5
'''
def scoringAlgo(answers):

    # Score weight maps
    c1Map = {
        'User Data Collection from external sources': 1,
        'Private Messaging Analysis': -1.5,
        'Third-Parties': 1,
        'Account / Content Restrictions': -1.5,
        'Location Tracking': 0,
        'Web Tracking': 1,
        'User Content Analysis': 0
    }

    c2Map = {
        'User Data Collection from external sources': -1/6,
        'Private Messaging Analysis': 1,
        'Third-Parties': -1/6,
        'Account / Content Restrictions': -1/6,
        'Location Tracking': -1/6,
        'Web Tracking': -1/6,
        'User Content Analysis': -1/6
    }

    c3Map = {
        'User Data Collection from external sources': -1/5,
        'Private Messaging Analysis': -1/5,
        'Third-Parties': -1/5,
        'Account / Content Restrictions': 1,
        'Location Tracking': -1/5,
        'Web Tracking': -1/5,
        'User Content Analysis': 0,
    }

    c4Map = {
        'User Data Collection from external sources': 0,
        'Private Messaging Analysis': -1/4,
        'Third-Parties': -1/4,
        'Account / Content Restrictions': -1/4,
        'Location Tracking': 1,
        'Web Tracking': 0,
        'User Content Analysis': -1/4,
    }

    c5Map = {
        'User Data Collection from external sources': -1/5,
        'Private Messaging Analysis': 0,
        'Third-Parties': -1/5,
        'Account / Content Restrictions': -1/5,
        'Location Tracking': -1/5,
        'Web Tracking': -1/5,
        'User Content Analysis': 1,
    }

    c6Map = {
        'User Data Collection from external sources': 1,
        'Private Messaging Analysis': -1.5,
        'Third-Parties': 1,
        'Account / Content Restrictions': -1.5,
        'Location Tracking': 0,
        'Web Tracking': 1,
        'User Content Analysis': 0
    }
    # add all maps into an array for cleaner searching
    maps = [0, c1Map, c2Map, c3Map, c4Map, c5Map, c6Map]

    userScore = 0
    clauseNum = 1
    # score array to return score of each clause
    clauseScores = []
    # ClauseAns is the list of keywords chosen for a single clause
    for clauseAns in answers:
        # For each keyword for single clause, get value and append to clause score
        singleClauseScore = 0
        for keywordChoice in clauseAns:
            singleClauseScore += maps[clauseNum][keywordChoice]
        
        # Once all keywords of a single clause are scored, add to causeScores list, add clausescore to total, and repeat for next clause
        clauseScores.append(singleClauseScore)
        userScore += singleClauseScore
        clauseNum += 1
    return [userScore, clauseScores]

# Capture data from google form into dataframe
df = pd.read_csv('data.csv')

# Create a summary dataframe for important info
data = {
    'Education': [],
    'Gender': [],
    'OC1': [],
    'OC2': [],
    'OC3': [],
    'OC4': [],
    'OC5': [],
    'OC6': [],
    'SC1': [],
    'SC2': [],
    'SC3': [],
    'SC4': [],
    'SC5': [],
    'SC6': [],
    'Original TOS Score': [],
    'Original TOS Duration': [],
    'Simplified TOS Score': [],
    'Simplified TOS Duration': [],
    'Timed Delta': [],
    'Untimed Delta': [],
    'Preference': []
}


df_summary = pd.DataFrame(data)

# Counter variables to track deltas
positive_timed = 0
negative_timed = 0
neutral_timed = 0
positive = 0
negative = 0
neutral = 0
count = 0

# Cycle through each google form record and process their scores
for i, r in df.iterrows():
    count += 1

    edu = r["education"]
    gender = r["What is your gender?"]
    preference = r['Pref']

    OClause1_ans = r['OClause1'].split(', ')
    OClause2_ans = r['OClause2'].split(', ')
    OClause3_ans = r['OClause3'].split(', ')
    OClause4_ans = r['OClause4'].split(', ')
    OClause5_ans = r['OClause5'].split(', ')
    OClause6_ans = r['OClause6'].split(', ')
    ansOriginal = [OClause1_ans, OClause2_ans, OClause3_ans, OClause4_ans, OClause5_ans, OClause6_ans]
    scoringOrig = scoringAlgo(ansOriginal)
    rowScore_original = scoringOrig[0]
    originalClauseScores = scoringOrig[1]

    SClause1_ans = r['SClause1'].split(', ')
    SClause2_ans = r['SClause2'].split(', ')
    SClause3_ans = r['SClause3'].split(', ')
    SClause4_ans = r['SClause4'].split(', ')
    SClause5_ans = r['SClause5'].split(', ')
    SClause6_ans = r['SClause6'].split(', ')
    ansSimplified = [SClause1_ans, SClause2_ans, SClause3_ans, SClause4_ans, SClause5_ans, SClause6_ans]
    scoringSimp = scoringAlgo(ansSimplified)
    rowScore_simplified = scoringSimp[0]
    simplifiedClauseScores = scoringSimp[1]

    # Time taken
    time1 = pd.to_datetime(r['time1'], format='%I:%M:%S %p')
    time2 = pd.to_datetime(r['time2'], format='%I:%M:%S %p')
    time3 = pd.to_datetime(r['time3'], format='%I:%M:%S %p')
    
    ansOriginalDuration = int(str(time2 - time1).split(' ')[2].split(':')[1])
    if ansOriginalDuration == 0:
        ansOriginalDuration = 1
    rowScore_original_timed = rowScore_original / ansOriginalDuration

    ansSimplifiedDuration = int(str(time3 - time2).split(' ')[2].split(':')[1])
    if ansSimplifiedDuration == 0:
        ansSimplifiedDuration = 1
    rowScore_simplified_timed = rowScore_simplified / ansSimplifiedDuration

    print(f'\nNAME: {r["name"]} -------------timed----------------untimed------')
    # print(f'ORIGINAL   TOS SCORE: {round(rowScore_original_timed, 2)} with time {ansOriginalDuration}  |  Original untimed: {round(rowScore_original, 2)}')
    # print(f'SIMPLIFIED TOS SCORE: {round(rowScore_simplified_timed, 2)} with time {ansSimplifiedDuration}  |  Simplified untimed: {round(rowScore_simplified, 2)}')
    delta_timed = rowScore_simplified_timed - rowScore_original_timed
    delta = rowScore_simplified - rowScore_original
    
    # Count the number of positives, no-positives, and 
    if delta_timed > 0:
        positive_timed += 1
    elif delta_timed == 0:
        neutral_timed += 1
    else:
        negative_timed += 1

    if delta > 0:
        positive += 1
    elif delta == 0:
        neutral += 1
    else:
        negative += 1
    
    print(f'Delta_timed: {round(delta_timed, 2)}   Delta: {round(delta, 2)}')
    print(f'Preference: | {preference} |')
    
    # Add user entry to dataframe
    # df_summary.loc[len(df.index)] = [r["education"], r["What is your gender?"], 
    #     rowScore_original_timed, ansOriginalDuration, rowScore_simplified_timed, ansSimplifiedDuration,
    #     delta_timed, delta, r["Pref"]]

    entry = {
        'Education': edu,
        'Gender': gender,
        'OC1': originalClauseScores[0],
        'OC2': originalClauseScores[1],
        'OC3': originalClauseScores[2],
        'OC4': originalClauseScores[3],
        'OC5': originalClauseScores[4],
        'OC6': originalClauseScores[5],
        'SC1': simplifiedClauseScores[0],
        'SC2': simplifiedClauseScores[1],
        'SC3': simplifiedClauseScores[2],
        'SC4': simplifiedClauseScores[3],
        'SC5': simplifiedClauseScores[4],
        'SC6': simplifiedClauseScores[5],
        'Original TOS Score': round(rowScore_original_timed,3),
        'Original TOS Duration': ansOriginalDuration,
        'Simplified TOS Score': round(rowScore_simplified_timed,3),
        'Simplified TOS Duration': ansSimplifiedDuration,
        'Timed Delta': round(delta_timed,3),
        'Untimed Delta': round(delta,3),
        'Preference': preference,
    }

    df_summary = df_summary.append(entry, ignore_index=True)    

print(f'\n\ntotal {count} --   TIMED positives: {positive_timed}  {round(100*positive_timed/(count),2)}%,   no-positive: {neutral_timed}  {round(100*neutral_timed/(count),2)}%,   negatives: {negative_timed}  {round(100*negative_timed/count, 2)}%')
# print(f' total {count} -- UNTIMED positives: {positive}  {round(100*positive/(count),2)}%,   no-positive: {neutral}  {round(100*neutral/(count),2)}%,   negatives: {negative}  {round(100*negative/count, 2)}%')

# Create dataframe of users and their scores and deltas
print(f'\n\n\nDATA FRAME SUMMARY ------------------')
print(df_summary.loc[:,('Original TOS Score','Simplified TOS Score')].describe())
# for col in df.columns:
#     print(col)

# fig, axes = plt.subplots(nrows=2, ncols=1)

# plt.figure(figsize=(15,6))

def allScoresWithGender():
    plt.scatter(df_summary.loc[:,'Gender'], df_summary.loc[:,'Simplified TOS Score'], s=7, color="blue", label="SimplifiedTOS score")
    plt.scatter(df_summary.loc[:,'Gender'], df_summary.loc[:,'Original TOS Score'], s=7, color="red", label="OriginalTOS score")
    # df_summary.plot(x='Gender', y='Original TOS Score', kind='scatter', color="red", label="OriginalTOS score")
    plt.grid(linewidth=0.25)
    plt.title("Comprehension Scores of Terms of Service Formats by Gender")
    plt.xlabel("Gender")
    plt.ylabel("Comprehension Score")

def allScoresWithPreference():
    plt.scatter(df_summary.loc[:,'Preference'], df_summary.loc[:,'Simplified TOS Score'], s=7, color="blue", label="SimplifiedTOS score")
    plt.scatter(df_summary.loc[:,'Preference'], df_summary.loc[:,'Original TOS Score'], s=7, color="red", label="OriginalTOS score")
    # df_summary.plot(x='Gender', y='Original TOS Score', kind='scatter', color="red", label="OriginalTOS score")
    plt.grid(linewidth=0.25)
    plt.title("Comprehension Scores of Terms of Service Formats by Preference")
    plt.xlabel("Preference")
    plt.ylabel("Comprehension Score")

def allScoresBoxPlot():
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15,5))
    ax1.boxplot(df_summary[df_summary.columns[14]], whis=1000, autorange = True)
    ax2.boxplot(df_summary[df_summary.columns[16]], whis=1000, autorange = True)
    ax1.grid(True)
    ax2.grid(True)
    ax1.set_title("Original TOS Score")
    ax2.set_title("Simplified TOS Score")
    ax1.set_ylabel("Score")

def deltaBoxPlot():
    plt.boxplot(df_summary[df_summary.columns[18]], whis=1000, autorange = True)
    plt.grid(True)
    plt.title("Delta (improvement from original to simplified ToS) Box Plot")
    plt.ylabel("Score")

def wordsScoreMean():
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(15,5), sharey=True)
    clauseWordCount = ["18", "52", "27", "13", "90", "29"]
    clauseMeans = [df_summary['OC1'].mean(), df_summary['OC2'].mean(), df_summary['OC3'].mean(),
        df_summary['OC4'].mean(), df_summary['OC5'].mean(), df_summary['OC6'].mean()]
    
    ax[0,0].plot(clauseWordCount, clauseMeans, color="green", label="Original TOS score")
    ax[0,0].grid(True)
    ax[0,0].set_title("Clause Word Count vs. Original Clause Score Average")
    ax[0,0].set_ylabel("Score Average")
    plt.text(3,1,'text')


    clauseWordCount2 = ["11", "5", "12", "8", "15", "18"]
    clauseMeans2 = [df_summary['SC1'].mean(), df_summary['SC2'].mean(), df_summary['SC3'].mean(),
        df_summary['SC4'].mean(), df_summary['SC5'].mean(), df_summary['SC6'].mean()]
    
    ax[0,1].plot(clauseWordCount2, clauseMeans2, color="green", label="Simplified TOS score")
    ax[0,1].grid(True)
    ax[0,1].set_title("Clause Word Count vs. Simplified Clause Score Average")
  

    clauseWordCount3 = ["13", "18", "27", "29", "52", "90"]
    clauseMeans3 = [df_summary['OC4'].mean(), df_summary['OC1'].mean(), df_summary['OC3'].mean(),
        df_summary['OC6'].mean(), df_summary['OC2'].mean(), df_summary['OC5'].mean()]

    ax[1,0].plot(clauseWordCount3, clauseMeans3, color="purple", label="Simplified TOS score")
    ax[1,0].grid(True)
    ax[1,0].set_title("SORTED -- Clause Word Count vs. Original Clause Score Average")
    ax[1,0].set_ylabel("Score Average")

    clauseWordCount4 = ["5", "8", "11", "12", "15", "18"]
    clauseMeans4 = [df_summary['SC2'].mean(), df_summary['SC4'].mean(), df_summary['SC1'].mean(), df_summary['SC3'].mean(),
        df_summary['SC5'].mean(), df_summary['SC6'].mean()]

    ax[1,1].plot(clauseWordCount4, clauseMeans4, color="purple", label="Simplified TOS score")
    ax[1,1].grid(True)
    ax[1,1].set_title("SORTED -- Clause Word Count vs. Simplified Clause Score Average")

    fig.tight_layout()



# Box plots of each clause score original and Simplified
def eachClauseScore():
    fig, ax = plt.subplots(nrows=2, ncols=6, figsize=(15,5), sharey=True) # clausesList = ["3rd-party tracking", "private messaging", "account restrictions", "location tracking", "User content analysis", "DNT-ignoring web tracking"]

    fig.suptitle('Clause Score Box Plots')
    ax[0,0].set_title("Original Clause 1")
    ax[0,1].set_title("Original Clause 2")
    ax[0,2].set_title("Original Clause 3")
    ax[0,3].set_title("Original Clause 4")
    ax[0,4].set_title("Original Clause 5")
    ax[0,5].set_title("Original Clause 6")

    ax[1,0].set_title("Simplified Clause 1")
    ax[1,1].set_title("Simplified Clause 2")
    ax[1,2].set_title("Simplified Clause 3")
    ax[1,3].set_title("Simplified Clause 4")
    ax[1,4].set_title("Simplified Clause 5")
    ax[1,5].set_title("Simplified Clause 6")
    
    ax[0,0].boxplot(df_summary[df_summary.columns[2]], whis=1000, autorange = True)
    ax[0,1].boxplot(df_summary[df_summary.columns[3]], whis=1000, autorange = True)
    ax[0,2].boxplot(df_summary[df_summary.columns[4]], whis=1000, autorange = True)
    ax[0,3].boxplot(df_summary[df_summary.columns[5]], whis=1000, autorange = True)
    ax[0,4].boxplot(df_summary[df_summary.columns[6]], whis=1000, autorange = True)
    ax[0,5].boxplot(df_summary[df_summary.columns[7]], whis=1000, autorange = True)
    
    ax[1,0].boxplot(df_summary[df_summary.columns[8]], whis=1000, autorange = True)
    ax[1,1].boxplot(df_summary[df_summary.columns[9]], whis=1000, autorange = True)
    ax[1,2].boxplot(df_summary[df_summary.columns[10]], whis=1000, autorange = True)
    ax[1,3].boxplot(df_summary[df_summary.columns[11]], whis=1000, autorange = True)
    ax[1,4].boxplot(df_summary[df_summary.columns[12]], whis=1000, autorange = True)
    ax[1,5].boxplot(df_summary[df_summary.columns[13]], whis=1000, autorange = True)

    ax[0,0].grid(True)
    ax[0,1].grid(True)
    ax[0,2].grid(True)
    ax[0,3].grid(True)
    ax[0,4].grid(True)
    ax[0,5].grid(True)
    ax[1,0].grid(True)
    ax[1,1].grid(True)
    ax[1,2].grid(True)
    ax[1,3].grid(True)
    ax[1,4].grid(True)
    ax[1,5].grid(True)

    fig.tight_layout()

def timedDeltaComparedToOriginal():
    plt.scatter(df_summary.loc[:,'Original TOS Score'], df_summary.loc[:,'Timed Delta'], s=20, color="purple")
    plt.title("Delta (improvement) vs Original TOS Score")
    plt.xlabel("Original TOS Score")
    plt.ylabel("Difference between Simplified TOS Score and Original TOS Score")
    plt.grid(True) 

def OriginalVsSimplified():
    plt.scatter(df_summary.loc[:,'Original TOS Score'], df_summary.loc[:,'Simplified TOS Score'], s=20, color="purple")
    plt.title("Original ToS Score vs Simplified ToS Score")
    plt.xlabel("Original TOS Score")
    plt.ylabel("Simplified TOS Score")
    plt.grid(True) 

def hist():
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(15,5), sharex=True, sharey=True)
    ax1.hist(df_summary.loc[:,"Simplified TOS Score"].values, 50, facecolor='blue', alpha=0.75)
    ax2.hist(df_summary.loc[:,"Original TOS Score"].values, 50, facecolor='red', alpha=0.75)
    ax1.set_title("Simplified TOS Score")
    ax2.set_title("Original TOS Score")
    ax1.set_ylabel("counts")
    ax2.set_ylabel("counts")
    ax2.set_xlabel("Scores")
    ax1.grid(True)
    ax2.grid(True) 

print(df_summary.loc[:,'OC1':'OC6'].describe())
# eachClauseScore()
    
# timedDeltaComparedToOriginal()
# allScoresWithGender()
# allScoresBoxPlot()
# deltaBoxPlot()
eachClauseScore()
# OriginalVsSimplified()
# allScoresWithPreference()
# hist()
# wordsScoreMean()
plt.legend()
plt.show()

