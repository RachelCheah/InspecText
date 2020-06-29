from flask import Flask, flash, render_template, request, redirect, make_response
import pdfkit

import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from math import pi
#from IPython.core.display import display, HTML
from collections import Counter
from collections import OrderedDict
import re
import os
import datetime

app = Flask(__name__)
app.config['SECRET_KEY'] = 'inspectext'
file = '' 

##############################################################################################################

def process(textFile, n):
    def isNewMessage(line):
        NewLinePattern = "(\d+\/\d+\/\d+)(,)(\s)(\d+(:)\d+)(\s)(\w+)(\s)(-)(\s)(\w+)"
        if re.match(NewLinePattern,line):
            return True
        return False 

    def standardizeDate(str):
        return str.split('/')[2] + "-" + str.split('/')[1] + "-" + str.split('/')[0] 

    def standardizeTime(time):
        time = time.strip()
        if "pm" or "am" in time:
            if "am" in time:
                time = time.split(' ')[0]
                if len(time) == 4:
                    time = "0" + time
                return time
            if "pm" in time:
                time = time.split(' ')[0]
                time = str(int(time.split(':')[0]) + 12) + ":" + time.split(':')[1]
                return time
        return time

    def splitData(line):
        if isNewMessage(line):
            date = line.split(',')[0]
            date = standardizeDate(date)
            time = line.split(',')[1].split('-')[0]
            time = standardizeTime(time)
            author = line.split('-')[1].split(':')[0]
            author = author.strip()
            message = line.split(':')[2:]
            message = ''.join(map(str,message))
            message = message.strip()
            return date, time, author, message
        else:
            message = line
            return message

    parsedData= []

    with open(textFile, encoding="UTF-8") as fp:
    #    fp.readline() #skips first line which contains end-end encryption info
        date, time, author, message, hour = '', '', '', '', '' # initialise variables
        while True:
            line = fp.readline()
            if not line: #if the end of the file has been reached
                parsedData.append([date, time, author, message, hour])
                break 
            line = line.strip() #removes leading and trailing whitespace
            if isNewMessage(line):
                if message:
                    message = str(message)
                parsedData.append([date, time, author, message, hour])
                date, time, author, message = splitData(line)
                hour =  time.split(':')[0]
            else: 
                message += " " 
                message += splitData(line)

    # initializing a pandas framework:
    df = pd.DataFrame(parsedData, columns = ['Date', 'Time', 'Author', 'Message', 'Hour'])
    df = df.drop(df.index[0])

    ###########
    author1 = 'Raychur 🐺💞'
    author2 = 'Naeko D Chiobu 💙💦🦈🦈🦈'
    ###########

    #separate df for media only
    media_messages_df = df[df['Message'] == '<Media omitted>']
    #media_messages_df['Author'].value_counts()[author1]
    #media_messages_df['Author'].value_counts()[author2]
    #do we want to drop the media files??
    df= df.drop(media_messages_df.index)

    df = df.loc[(df['Author'] == author1)|(df['Author'] == author2)]

    author_value_counts = df['Author'].value_counts()

    df['Date'] = pd.to_datetime(df['Date'])
    df['Day_of_week'] = df['Date'].dt.day_name()

    author1_df =  df.loc[(df['Author'] == author1)]
    author2_df =  df.loc[(df['Author'] == author2)]
    author1_hour = author1_df['Hour'].value_counts()
    author2_hour = author2_df['Hour'].value_counts()

    def time_of_day_data():
        hours_dictionary = {}
        hours_dictionary['hourlist'] = ['Author 1', 'Author 2']
        for i in range(0,24):
            t_list = [0,0]
            j = str(i)
            if i<10:
                j = '0' + j
            if i == 0:
                j = '24'
            if j in author1_hour.index.tolist():
                t_list[0] = author1_hour.loc[j].item()
            if j in author2_hour.index.tolist():
                t_list[1] = author2_hour.loc[j].item()
            if i == 0:
                j = '00'
            hours_dictionary[j] = t_list
        return hours_dictionary

    def plot_time_of_day():
        time_of_day_df = pd.DataFrame(time_of_day_data())
        # number of variable
        categories=list(time_of_day_df)[1:]
        N = len(categories)
        
        # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
        angles = [n / float(N) * 2 * pi for n in range(N)]
        angles += angles[:1]

        # Initialise the spider plot
        ax = plt.subplot(111, polar=True)
        
        # If you want the first axis to be on top:
        ax.set_theta_offset(pi / 2)
        ax.set_theta_direction(-1)
        
        # Draw one axe per variable + add labels labels yet
        plt.xticks(angles[:-1], categories)
        
        # Draw ylabels
        ax.set_rlabel_position(0)
        plt.yticks([300,600,900], ["300","600","900"], color="grey", size=7)
        plt.ylim(0,1200)
            
        # Ind1
        values=time_of_day_df.loc[0].drop('hourlist').values.flatten().tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=1, linestyle='solid', label=author1)
        ax.fill(angles, values, 'b', alpha=0.1)
        
        # Ind2
        values=time_of_day_df.loc[1].drop('hourlist').values.flatten().tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=1, linestyle='solid', label=author2)
        ax.fill(angles, values, 'r', alpha=0.1)
        
        # Add legend
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

    # F I R S T #
    if n == 1:
        return plot_time_of_day()
    # # # # # # # 

    author1_day = author1_df['Day_of_week'].value_counts()
    author2_day = author2_df['Day_of_week'].value_counts()

    days_in_order = [ 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

    def day_of_week_data():
        day_dictionary = {}
        day_dictionary['Day'] = ['Author 1', 'Author 2']
        for dayname in days_in_order:
            t_list = [0,0]
            if dayname in author1_day.index.tolist():
                t_list[0] = author1_day.loc[dayname].item()
            if dayname in author2_day.index.tolist():
                t_list[1] = author2_day.loc[dayname].item()
            day_dictionary[dayname] = t_list
        return day_dictionary

    def plot_day_of_week():
        day_of_week_df = pd.DataFrame(day_of_week_data())
        # number of variable
        categories=list(day_of_week_df)[1:]
        N = len(categories)
        
        # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
        angles = [n / float(N) * 2 * pi for n in range(N)]
        angles += angles[:1]

        # Initialise the spider plot
        ax = plt.subplot(111, polar=True)
        
        # If you want the first axis to be on top:
        ax.set_theta_offset(pi / 2)
        ax.set_theta_direction(-1)
        
        # Draw one axe per variable + add labels labels yet
        plt.xticks(angles[:-1], categories)
        
        # Draw ylabels
        ax.set_rlabel_position(0)
        plt.yticks([800,1600,2400], ["800","1600","2400"], color="grey", size=7)
        plt.ylim(0,3200)
            
        # Ind1
        values=day_of_week_df.loc[0].drop('Day').values.flatten().tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=1, linestyle='solid', label=author1)
        ax.fill(angles, values, 'b', alpha=0.1)
        
        # Ind2
        values=day_of_week_df.loc[1].drop('Day').values.flatten().tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=1, linestyle='solid', label=author2)
        ax.fill(angles, values, 'r', alpha=0.1)
        
        # Add legend
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

    # S E C O N D #
    if n == 2:
        return plot_day_of_week()
    # # # # # # # #

    def timeline_data():
        timeline_dictionary = {}
        timeline_dictionary['date'] = ['Author 1', 'Author 2']
        for i in range(len(df)):
            t_list = [0,0]
            day,author = df.iloc[i,0], df.iloc[i,2]
            if day not in timeline_dictionary:
                timeline_dictionary[day] = t_list  
            t_list = timeline_dictionary[day]
            if author == author1:
                t_list[0] += 1
            if author == author2:
                t_list[1] += 1
            timeline_dictionary[day] = t_list
        return timeline_dictionary

    def timeline_data_2():
        timeline_dictionary_2 = {}
        timeline_dictionary_2['date'] = ['Author 1', 'Author 2']
        t_list = [0,0]
        for i in range(len(df)):
            day,author = df.iloc[i,0], df.iloc[i,2]
            if day not in timeline_dictionary_2:
                timeline_dictionary_2[day] = t_list  
    #            t_list = timeline_dictionary_2[day]
            if author == author1:
                t_list[0] += 1
            if author == author2:
                t_list[1] += 1
            tt_list = t_list.copy()
            timeline_dictionary_2[day] = tt_list
        return timeline_dictionary_2
    '''
    timeline_df_2 = pd.DataFrame(timeline_data_2())
    timeline_df_2 = timeline_df_2.T
    new_header = timeline_df_2.iloc[0]
    timeline_df_2 = timeline_df_2[1:]
    timeline_df_2.columns = new_header

    def plot_timeline_2():
        timeline_df_2.plot(figsize = (20,8))

    # T H I R D #
    if n == 3:
        return plot_timeline_2()
    # # # # # # #

    '''
    timeline_df = pd.DataFrame(timeline_data())
    timeline_df = timeline_df.T
    new_header = timeline_df.iloc[0]
    timeline_df = timeline_df[1:]
    timeline_df.columns = new_header

    def plot_timeline():
        timeline_df.plot(figsize = (20,8))

    # T H I R D #
    if n == 3:
        return plot_timeline()
    #############

    def top_words(df): 
        top_N = 40
        stopwords = nltk.corpus.stopwords.words('english')
        # RegEx for stopwords
        RE_stopwords = r'\b(?:{})\b'.format('|'.join(stopwords))
        #RE_stopwords.extend(['from', 'subject', 're', 'edu', 'use'])
        # replace '|'-->' ' and drop all stopwords
        words = (df.Message \
            .str.lower() \
                .replace([RE_stopwords], [''], regex=True) \
                    .str.cat(sep=' ') \
                        .split())

        words = [word for word in words if len(word) > 3]

        # generate DF out of Counter
        rslt = pd.DataFrame(Counter(words).most_common(top_N), \
            columns=['Word', 'Frequency']).set_index('Word')
        #rslt = rslt.drop(rslt[rslt['Word'].map(len) > 2].index)  
        #rslt = rslt[rslt['Word'].map(len) > 2]
        return rslt
    # F O U R T H #
    if n == 41:
        return top_words(author1_df)
    
    if n == 42:
        return top_words(author2_df)
    # # # # # # # #

    df_2 = top_words(author2_df)
    df_2.columns
    d = dict(zip(df_2.index, df_2.Frequency))

    wordcloud = WordCloud()
    wordcloud.generate_from_frequencies(frequencies=d)
    plt.figure()
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    # F I F T H #
    if n == 5:
        return plt.show()
    # # # # # # #

'''
def process(textFile):
    short = '' 
    for i in str(textFile):
        short += i
        if i == ']' or i == '\\':
            break
    return short
'''
####################################################################################################################

@app.route('/')
@app.route('/home')
def gobackhome():
    return render_template('index.html')
##########################################################################################################

@app.route('/start/')
def about():
    return render_template('start.html')

@app.route('/start/whatsapp/', methods=['POST', 'GET'])
def whatsapp():
    global file
    if request.method == "POST":
        file = request.files["fileWhatsapp"].read() #file is now the text file
        if file:
            return redirect('/output')
        else:
            flash("No file selected for uploading")
            redirect('/start/whatsapp/')
    return render_template('whatsapp.html')

@app.route('/start/telegram/', methods=['POST', 'GET'])
def telegram():
    global file
    if request.method == "POST":
        file = request.files["fileTelegram"].read() #file is now the text file
        if file:
            return redirect('/output')
        else:
            flash("No file selected for uploading")
            redirect('/start/telegram/')
    return render_template('telegram.html')

@app.route('/aims/')
def aims():
    return render_template('aims.html')

@app.route('/features/')
def features():
    return render_template('features.html')

@app.route('/customize/')
def customize():
    return render_template('customize.html')

@app.route('/output/')
def output():
    return render_template('file.html', one = process(file,1), two = process(file,2), \
        fourOne = process(file,41), fourTwo = process(file,42))
#    one = process(file,1) 
#    two = process(file,2)
#    three = process(file,3)
#    fourOne = process(file,41)
#    fourTwo = process(file,42)
#    five = process(file,5)
#     pdf = pdfkit.from_string(one, False) #False keeps the pdf in memory
#    response = make_response(pdf)
#    response.headers['Content-Type'] = 'application/pdf'
#    response.headers['Content-Disposition'] = 'inline; filename=inspecText.pdf'
#    return response
    
###################################################################################################################

if __name__ == '__main__':
    app.run(debug=True)