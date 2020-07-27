from flask import Flask, flash, render_template, request, redirect, make_response
from werkzeug.utils import secure_filename
import os
import pdfkit

import pandas as pd
import numpy as np
import emoji
import sys
import nltk
from nltk.corpus import stopwords
import colorsys
from wordcloud import WordCloud
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from math import pi
from PIL import Image, ImageOps
import random
import json
import csv
from pandas.io.json import json_normalize

from IPython.core.display import display, HTML
from collections import Counter
from collections import OrderedDict
import re
import datetime

import base64

from nltk.classify import NaiveBayesClassifier
from nltk.corpus import subjectivity
from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment.util import *
from nltk.sentiment.vader import SentimentIntensityAnalyzer


app = Flask(__name__)
app.config['SECRET_KEY'] = 'inspectext'

fileTelegram = ''
fileWhatsapp= ''

########################################################################################################################################

def telegramData(textFile):
    global df
    global author1
    global author2
    global author1_name
    global author2_name
    global startDate
    global endDate
    with open(textFile, encoding ="utf8") as data_file:    
        data = json.load(data_file)  
    df = json_normalize(data, 'messages', record_prefix='message_')
    df = df[df['message_text'].apply(type) == str]
    df = df[df['message_type']== 'message']
    df = df[['message_date', 'message_from','message_text']]
    df[['Date','Time']] = df.message_date.str.split("T",expand=True,)
    df.drop(columns=['message_date'],inplace=True)
    df.columns = ['Author','Message','Date','Time']
    df = df[['Date','Time','Author','Message']]
    df['Hour'] = df['Time']
    def getHour(time):
        return time.split(':')[0]
    df['Hour'] = df['Hour'].map(getHour)
    authors = df['Author'].unique()
    author1 = authors[0]
    author2 = authors[1]

    try:
        uchr = unichr  # Python 2
        if sys.maxunicode == 0xffff:
            # narrow build, define alternative unichr encoding to surrogate pairs
            # as unichr(sys.maxunicode + 1) fails.
            def uchr(codepoint):
                return (unichr(codepoint) if codepoint <= sys.maxunicode\
                    else unichr(codepoint - 0x010000 >> 10 | 0xD800) + unichr(codepoint & 0x3FF | 0xDC00))
    except NameError:
        uchr = chr  # Python 3

    # Unicode 11.0 Emoji Component map (deemed safe to remove)
    _removable_emoji_components = ((0x20E3, 0xFE0F), range(0x1F1E6, 0x1F1FF + 1), range(0x1F3FB, 0x1F3FF + 1), range(0x1F9B0, 0x1F9B3 + 1), range(0xE0020, 0xE007F + 1),)
    emoji_components = re.compile(u'({})'.format(u'|'.join([re.escape(uchr(c)) for r in _removable_emoji_components for c in r])), flags=re.UNICODE)

    def remove_emoji(text, remove_components=False):
        cleaned = emoji.get_emoji_regexp().sub(u'', text)
        if remove_components:
            cleaned = emoji_components.sub(u'', cleaned)
        return cleaned

    author1_name = remove_emoji(authors[0])
    author1_name = author1_name.strip()
    author2_name = remove_emoji(authors[1])
    author2_name = author2_name.strip()

    media_messages_df = df[df['Message'] == '<Media omitted>']
    df= df.drop(media_messages_df.index) #to drop the media files
    df = df.loc[(df['Author'] == author1)|(df['Author'] == author2)]    
    df['Date'] = pd.to_datetime(df['Date'])
    df['Day_of_week'] = df['Date'].dt.day_name()
    startDate,endDate = df.iloc[0,0], df.iloc[-1,0]

def whatsappData(textFile):
    global df
    global author1
    global author2
    global author1_name
    global author2_name
    global startDate
    global endDate
    def isNewMessage(line):
        result = False
        NewLinePattern = ["(\d+\/\d+\/\d+)(,)(\s)(\d+(:)\d+)(\s)(\w+)(\s)(-)(\s)(\w+)",\
            "(\d+\/\d+\/\d+)(,)(\s)(\d+(:)\d+)(\s)(-)(\s)(\w+)"]
        for x in NewLinePattern:
            if re.match(x,line):
                result = True
        return result

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

    with open(textFile, encoding ="utf-8") as fp:
        fp.readline() #skips first line which contains end-end encryption info
        date, time, author, message, hour = None, None, None, None, None # initialise variables
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
    df = pd.DataFrame(parsedData, columns = ['Date', 'Time', 'Author', 'Message', 'Hour'])
    df = df.drop(df.index[0])
    authors = df['Author'].unique()
    author1 = authors[0]
    author2 = authors[1]

    try:
        uchr = unichr  # Python 2
        if sys.maxunicode == 0xffff:
            # narrow build, define alternative unichr encoding to surrogate pairs
            # as unichr(sys.maxunicode + 1) fails.
            def uchr(codepoint):
                return (unichr(codepoint) if codepoint <= sys.maxunicode\
                    else unichr(codepoint - 0x010000 >> 10 | 0xD800) + unichr(codepoint & 0x3FF | 0xDC00))
    except NameError:
        uchr = chr  # Python 3

    # Unicode 11.0 Emoji Component map (deemed safe to remove)
    _removable_emoji_components = ((0x20E3, 0xFE0F), range(0x1F1E6, 0x1F1FF + 1), range(0x1F3FB, 0x1F3FF + 1), range(0x1F9B0, 0x1F9B3 + 1), range(0xE0020, 0xE007F + 1),)
    emoji_components = re.compile(u'({})'.format(u'|'.join([re.escape(uchr(c)) for r in _removable_emoji_components for c in r])), flags=re.UNICODE)

    def remove_emoji(text, remove_components=False):
        cleaned = emoji.get_emoji_regexp().sub(u'', text)
        if remove_components:
            cleaned = emoji_components.sub(u'', cleaned)
        return cleaned

    author1_name = remove_emoji(authors[0])
    author1_name = author1_name.strip()
    author2_name = remove_emoji(authors[1])
    author2_name = author2_name.strip()

    #separate df for media only
    media_messages_df = df[df['Message'] == '<Media omitted>']
    #to drop the media files
    df= df.drop(media_messages_df.index)
    df = df.loc[(df['Author'] == author1)|(df['Author'] == author2)]

    df['Date'] = pd.to_datetime(df['Date'])
    df['Day_of_week'] = df['Date'].dt.day_name()
    
    startDate,endDate = df.iloc[0,0], df.iloc[-1,0]

def plot():
    global df
    global author1
    global author2
    global startDate
    global endDate
    global author1_wpm
    global author2_wpm
    global more_words
    global author1_messages
    global author2_messages
    global more_messages
    #restricting range of data
    mask = (df['Date'] >= startDate) & (df['Date']<= endDate)
    df = df.loc[mask]

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
                j = '00'
            if j in author1_hour.index.tolist():
                t_list[0] = author1_hour.loc[j].item()
            if j in author2_hour.index.tolist():
                t_list[1] = author2_hour.loc[j].item()
            hours_dictionary[j] = t_list
        for x in hours_dictionary:
            if x =='hourlist':
                counter = 0
            elif int(hours_dictionary[x][0])> counter:
                counter = int(hours_dictionary[x][0])
            elif int(hours_dictionary[x][1])> counter:
                counter = int(hours_dictionary[x][1])
        return hours_dictionary, counter

    def roundup(x):
        return int(x) if x % 100 == 0 else int(x + 100 - x % 100)

    ### start of FIRST: time of day ###

    def plot_time_of_day():
        plt.style.use('fivethirtyeight')
        plt.style.use('bmh')

        plt.rcParams["font.family"] =  "Gabriola"
        plt.rcParams.update({'font.size': 16})
            
        tod_data,maxcount = time_of_day_data()  
        time_of_day_df = pd.DataFrame(tod_data)
            
        maxcount = roundup(maxcount) + 200
        a = roundup(maxcount/4)
        b = roundup(maxcount/2)
        c = roundup(3* maxcount/4)
        # No. of variable
        categories=list(time_of_day_df)[1:]
        N = len(categories)
        # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
        angles = [n / float(N) * 2 * pi for n in range(N)]
        angles += angles[:1]
        # Initialise the spider plot
        ax = plt.subplot(111, polar=True, label='time of day')
        # If you want the first axis to be on top:
        ax.set_theta_offset(pi / 2)
        ax.set_theta_direction(-1)
        # Draw one axe per variable + add labels labels yet
        plt.xticks(angles[:-1], categories, fontsize=16)
        # Draw ylabels
        ax.set_rlabel_position(0)
        plt.yticks([a,b,c], [str(a),str(b),str(c)], color="grey", size=12)
        plt.ylim(0,maxcount)

        # Ind1
        values=time_of_day_df.loc[0].drop('hourlist').values.flatten().tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=1, linestyle='solid', label=author1_name, color=author1_colour)
        ax.fill(angles, values, author1_colour, alpha=0.1)
        # Ind2
        values=time_of_day_df.loc[1].drop('hourlist').values.flatten().tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=1, linestyle='solid', label=author2_name, color=author2_colour)
        ax.fill(angles, values, author2_colour, alpha=0.1)
        # Add legend
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        plt.savefig(os.path.join("uploads", 'timeofday.png'), bbox_inches = 'tight')

    plot_time_of_day()

    ### end of FIRST: time of day ###

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
        for x in day_dictionary:
            if x =='Day':
                counter = 0
            else:
                temp = max(int(day_dictionary[x][0]),int(day_dictionary[x][1]))
                if temp > counter:
                    counter = temp
        return day_dictionary, counter

    ### start of SECOND: Day of week ###

    def plot_day_of_week():
        plt.style.use('fivethirtyeight') 
        plt.style.use('bmh')

        plt.rcParams["font.family"] =  "Gabriola"
        plt.rcParams.update({'font.size': 16})

        dow_data,maxcount = day_of_week_data() 
        day_of_week_df = pd.DataFrame(dow_data)
            
        maxcount = roundup(maxcount) + 200
        a = roundup(maxcount/4)
        b = roundup(maxcount/2)
        c = roundup(3* maxcount/4)
        # number of variable
        categories=list(day_of_week_df)[1:]
        N = len(categories)
        # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
        angles = [n / float(N) * 2 * pi for n in range(N)]
        angles += angles[:1]
        # Initialise the spider plot
        ax = plt.subplot(111, polar=True, label='day of week')
        # If you want the first axis to be on top:
        ax.set_theta_offset(pi / 2)
        ax.set_theta_direction(-1)
        # Draw one axe per variable + add labels labels yet
        plt.xticks(angles[:-1], categories, fontsize=16)

        for label,i in zip(ax.get_xticklabels(),range(0,len(angles))):
            angle_rad=angles[i]
            if angle_rad == 0:
                ha = 'center'
            elif angle_rad <= pi/2:
                ha= 'left'
            elif pi/2 < angle_rad <= pi:
                ha= 'left'
            elif pi < angle_rad <= (3*pi/2):
                ha= 'right'
            else:
                ha= 'right'

            label.set_horizontalalignment(ha)

        # Draw ylabels
        ax.set_rlabel_position(0)
        plt.yticks([a,b,c], [str(a),str(b),str(c)], color="grey", size=12)
        plt.ylim(0,maxcount)

        # Ind1
        values=day_of_week_df.loc[0].drop('Day').values.flatten().tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=1, linestyle='solid', label=author1_name, color=author1_colour)
        ax.fill(angles, values, author1_colour, alpha=0.1)
        # Ind2
        values=day_of_week_df.loc[1].drop('Day').values.flatten().tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=1, linestyle='solid', label=author2_name, color=author2_colour)
        ax.fill(angles, values, author2_colour, alpha=0.1)
        # Add legend
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        plt.savefig(os.path.join("uploads", 'dayofweek.png'), bbox_inches = 'tight')

    plot_day_of_week()

    ### end of SECOND: Day of week ###

    def timeline_data():
        timeline_dictionary = {}
        timeline_dictionary['date'] = ['Author1', 'Author2']
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

    timeline_df = pd.DataFrame(timeline_data())
    timeline_df = timeline_df.T
    new_header = timeline_df.iloc[0]
    timeline_df = timeline_df[1:]
    timeline_df.columns = new_header

    ### start of THIRD: timeline ###

    def plot_timeline():
        plt.style.use('fivethirtyeight')
        plt.rcParams["font.family"] =  "Gabriola"
        plt.rcParams.update({'font.size': 24})
        
        plt.figure(figsize = (20,8))
        plt.xlabel('Timeline', fontsize=30)
        

        ax1 = timeline_df.Author1.plot(color=author1_colour)
        ax2 = timeline_df.Author2.plot(color=author2_colour)

        ax1.xaxis.set_label_position('top')             
        ax1.legend([author1_name,author2_name],loc='upper right')
        plt.savefig(os.path.join("uploads", 'timeline.png'), bbox_inches = 'tight')  

    plot_timeline()

    ### end of THIRD: timeline ###
    
    def top_words(df): 
        top_N = 40
        stopwords = nltk.corpus.stopwords.words('english')
        # RegEx for stopwords
        RE_stopwords = r'\b(?:{})\b'.format('|'.join(stopwords))
        #RE_stopwords.extend(['from', 'subject', 're', 'edu', 'use'])
        # replace '|'-->' ' and drop all stopwords
        words = (df.Message\
            .str.lower()\
                .replace([RE_stopwords], [''], regex=True)\
                    .str.cat(sep=' ')\
                        .split())
        words = [word for word in words if len(word) > 3]
        # generate DF out of Counter
        rslt = pd.DataFrame(Counter(words).most_common(top_N),
                            columns=['Word', 'Frequency']).set_index('Word')
        return rslt

    def hex_to_rgb(hex):
        hex = hex.lstrip('#')
        hlen = len(hex)
        return tuple(int(hex[i:i + hlen // 3], 16) for i in range(0, hlen, hlen // 3))

    def rgb_to_hsl(r, g, b):
        r = float(r)
        g = float(g)
        b = float(b)
        high = max(r, g, b)
        low = min(r, g, b)
        h, s, l = ((high + low) / 2,)*3
        if high == low:
            h = 0.0
            s = 0.0
        else:
            d = high - low
            s = d / (2 - high - low) if l > 0.5 else d / (high + low)
            h = {
                r: (g - b) / d + (6 if g < b else 0),
                g: (b - r) / d + 2,
                b: (r - g) / d + 4,
            }[high]
            h /= 6
        return h, s, l

    a1_rgb = hex_to_rgb(author1_colour)
    a2_rgb = hex_to_rgb(author2_colour)

    a1_hlsva = rgb_to_hsl(a1_rgb[0]/255,a1_rgb[1]/255,a1_rgb[2]/255)
    a2_hlsva = rgb_to_hsl(a2_rgb[0]/255,a2_rgb[1]/255,a2_rgb[2]/255)

    a1_hlsva0 = round(a1_hlsva[0]*355)
    a1_hlsva1 = round(a1_hlsva[1]*100)
    a1_hlsva2 = round(a1_hlsva[2]*100)

    a2_hlsva0 = round(a2_hlsva[0]*355)
    a2_hlsva1 = round(a2_hlsva[1]*100)
    a2_hlsva2 = round(a2_hlsva[2]*100)
    
    ############################
    
    df_1 = top_words(author1_df)
    df_1.columns
    d = dict(zip(df_1.index, df_1.Frequency))

    plt.style.use('fivethirtyeight')
    plt.rcParams["font.family"] =  "Gabriola"
    plt.rcParams.update({'font.size': 16})
        
    fileloc = os.path.join("static", shape + '.jpg')
    mask = np.array(Image.open(fileloc))    
    wordcloud = WordCloud(background_color = '#F0F0F0', mask=mask, width = mask.shape[1], height = mask.shape[0])
    wordcloud.generate_from_frequencies(frequencies=d)
    plt.figure()

    def a1_color_func(word, font_size, position, orientation, random_state=None,**kwargs):
        return "hsl({0}, {1}%%, %d%%)".format(str(a1_hlsva0),str(a1_hlsva1)) % random.randint(60, 90)

    plt.imshow(wordcloud.recolor(color_func = a1_color_func), interpolation="bilinear")
    plt.axis("off")
    wordcloud.to_file(os.path.join("uploads", 'author1cloud.png'))

    df_2 = top_words(author2_df)
    df_2.columns
    d = dict(zip(df_2.index, df_2.Frequency))

    plt.style.use('fivethirtyeight')
    plt.rcParams["font.family"] =  "Gabriola"
    plt.rcParams.update({'font.size': 16})
    
    wordcloud = WordCloud(background_color = '#F0F0F0',mask=mask,width = mask.shape[1],height = mask.shape[0])
    wordcloud.generate_from_frequencies(frequencies=d)
    plt.figure()

    def a2_color_func(word, font_size, position, orientation, random_state=None,**kwargs):
        return "hsl({0}, {1}%%, %d%%)".format(str(a2_hlsva0),str(a2_hlsva1)) % random.randint(60, 90)
  
    plt.imshow(wordcloud.recolor(color_func = a2_color_func), interpolation="bilinear")
    plt.axis("off")
    wordcloud.to_file(os.path.join("uploads", 'author2cloud.png'))

    n_instances = 100
    subj_docs = [(sent, 'subj') for sent in subjectivity.sents(categories='subj')[:n_instances]]
    obj_docs = [(sent, 'obj') for sent in subjectivity.sents(categories='obj')[:n_instances]]
    train_subj_docs = subj_docs[:80]
    test_subj_docs = subj_docs[80:100]
    train_obj_docs = obj_docs[:80]
    test_obj_docs = obj_docs[80:100]
    training_docs = train_subj_docs+train_obj_docs
    testing_docs = test_subj_docs+test_obj_docs
    sentim_analyzer = SentimentAnalyzer()
    all_words_neg = sentim_analyzer.all_words([mark_negation(doc) for doc in training_docs])
    unigram_feats = sentim_analyzer.unigram_word_feats(all_words_neg, min_freq=4)
    len(unigram_feats)
    sentim_analyzer.add_feat_extractor(extract_unigram_feats, unigrams=unigram_feats)
    training_set = sentim_analyzer.apply_features(training_docs)
    test_set = sentim_analyzer.apply_features(testing_docs)
    trainer = NaiveBayesClassifier.train
    classifier = sentim_analyzer.train(trainer, training_set)
    '''
    for key,value in sorted(sentim_analyzer.evaluate(test_set).items()):
        print('{0}: {1}'.format(key, value))
    '''
    df['Message'].unique()[10:20]

    def sentiment(message):
        sid = SentimentIntensityAnalyzer()
        ss = sid.polarity_scores(message)
        return ss['compound']
    df["Sentiment"] = df.apply(lambda row : sentiment(row['Message']),axis = 1)

    def sentiment_data():
        sentiment_dictionary = {}
        sentiment_dictionary['date'] = ['Author1', 'Author2']
        for i in range(len(df)):
            t_list = [[0,0.0],[0,0.0]]
            month,author,sentiment = str(df.iloc[i,0]), df.iloc[i,2], df.iloc[i,6]
            if sentiment != 0.0:
                month = month.split('-')[0] + '-' + month.split('-')[1]
                if month not in sentiment_dictionary:
                    sentiment_dictionary[month] = t_list  
                t_list = sentiment_dictionary[month]
                if author == author1:
                    t_list[0][0] += 1
                    t_list[0][1] += sentiment
                if author == author2:
                    t_list[1][0] += 1
                    t_list[1][1] +=sentiment
                sentiment_dictionary[month] = t_list
            
        for x in sentiment_dictionary:
            if x != 'date':
                t_list = sentiment_dictionary[x]
                if t_list[0][0] != 0:
                    t_list[0] = float(t_list[0][1])/float(t_list[0][0])
                else:
                    t_list[0] = 0
                if t_list[1][0] != 0:
                    t_list[1] = float(t_list[1][1])/float(t_list[1][0])
                else:
                    t_list[1] = 0
                sentiment_dictionary[x] = t_list
        return sentiment_dictionary

    sentiment_df = pd.DataFrame(sentiment_data())
    sentiment_df = sentiment_df.T
    new_header = sentiment_df.iloc[0]
    sentiment_df = sentiment_df[1:]
    sentiment_df.columns = new_header

    def plot_sentiment():
        plt.style.use('fivethirtyeight')
        plt.rcParams["font.family"] =  "Gabriola"
        plt.rcParams.update({'font.size': 24})

        plt.figure(figsize = (20,8))
        plt.xlabel('Sentiment Analysis',fontsize =30)
        
        ax1 = sentiment_df.Author1.plot(color=author1_colour)
        ax2 = sentiment_df.Author2.plot(color=author2_colour)
        ax1.xaxis.set_label_position('top') 
        h1, l1 = ax1.get_legend_handles_labels()
            
        ax1.legend([author1_name,author2_name],loc='upper right')
        plt.savefig(os.path.join("uploads", 'sentiment.png'), bbox_inches = 'tight')

    plot_sentiment()

    #number of words
    def no_of_words(message):
        return len(message.split())
                
    df["WordCount"] = df.apply(lambda row : no_of_words(row['Message']),axis = 1)
    author1_df =  df.loc[(df['Author'] == author1)]
    author2_df =  df.loc[(df['Author'] == author2)]
    author1_wpm = author1_name + "'s average word per message is {:0.2f}".format(author1_df["WordCount"].mean())
    author2_wpm = author2_name + "'s average word per message is {:0.2f}".format(author2_df["WordCount"].mean())

    def who_sent_more_words():
        if author1_df["WordCount"].sum()> author2_df["WordCount"].sum():
            num = author1_df["WordCount"].sum()/author2_df["WordCount"].sum()
            num = num*100 -100
            return (author1_name + " sent {:0.0f}% more words than ".format(num)+ author2_name)
        elif author2_df["WordCount"].sum()> author1_df["WordCount"].sum():
            num = author2_df["WordCount"].sum()/author1_df["WordCount"].sum()
            num = num*100 -100
            return (author2_name + " sent {:0.0f}% more words than ".format(num)+ author1_name)
        else:
            return ("You both sent the same number of words somehow!")
    more_words = who_sent_more_words()

    days = "Number of days of texting: " + str(len(df["Date"].unique()))

    author1_messages = author1_name + " sent " + str(len(author1_df.index)) + " messages"
    author2_messages = author2_name + " sent " + str(len(author2_df.index)) + " messages"
    def who_sent_more():
        if len(author1_df.index)> len(author2_df.index):
            num = len(author1_df.index)/len(author2_df.index)
            return (author1_name + " sent {:0.2f} times more messages than ".format(num)+ author2_name)
        elif len(author2_df.index)> len(author1_df.index):
            num = len(author2_df.index)/len(author1_df.index)
            return (author2_name + " sent {:0.2f} times more messages than ".format(num)+ author1_name)
        else:
            return ("You both sent the same number of messages somehow!")
    more_messages = who_sent_more()

########################################################################################################################################

def process(textFile, author1, author2, startDate, endDate, colour1, colour2):
    prnt = "author1 = " + author1 + " author2 = " + author2 + " start date = " + str(startDate) + " end date = " + str(endDate) + " colour1 = " + colour1 + " colour2 = " + colour2
    #with open(textFile, encoding="UTF-8") as fp:
        #prnt += fp.readline()
    return prnt

########################################################################################################################################

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
    global fileWhatsapp
    global filename
    if request.method == "POST":
        fileWhatsapp = request.files["fileWhatsapp"] #file is now the text file
        if fileWhatsapp:
            filename = secure_filename(fileWhatsapp.filename)
            fileWhatsapp.save(os.path.join("uploads", filename))
            return redirect('/customize')
        else:
            flash("No file selected for uploading")
            return redirect('/start/whatsapp/')
    return render_template('whatsapp.html')

@app.route('/start/telegram/', methods=['POST', 'GET'])
def telegram():
    global fileTelegram
    global filename
    if request.method == "POST":
        fileTelegram = request.files["fileTelegram"] #file is now the text file
        if fileTelegram:
            filename = secure_filename(fileTelegram.filename)
            fileTelegram.save(os.path.join("uploads", filename))
            return redirect('/customize')
        else:
            flash("No file selected for uploading")
            return redirect('/start/telegram/')
    return render_template('telegram.html')

@app.route('/aims/')
def aims():
    return render_template('aims.html')

@app.route('/features/')
def features():
    return render_template('features.html')

@app.route('/features/visualize/')
def visualization():
    return render_template('visualisation.html')

@app.route('/features/customize/')
def customization():
    return render_template('customization.html')

@app.route('/customize/', methods=['POST', 'GET'])
def customize():
    global author1_name
    global author2_name
    global startDate
    global endDate
    global author1_colour
    global author2_colour
    global shape
    if fileTelegram:
        telegramData("uploads/" + filename)
    else:
        whatsappData("uploads/" + filename)

    if request.method == "POST":
        if request.form["contactOrNot"] == "no":
            if request.form["authorOne"] and request.form["authorTwo"]:
                author1_name = request.form["authorOne"]
                author2_name = request.form["authorTwo"]
            else:
                flash("Please fill up the name fields if you want to customize them")
                return redirect('/customize')
        if request.form["dateRange"] == "no":
            if request.form["startDate"] and request.form["endDate"]:
                if request.form["startDate"] < request.form["endDate"]:
                    startDate = request.form["startDate"]
                    endDate = request.form["endDate"]
                else:
                    flash("Please ensure that the start date is earlier than the end date")
                    return redirect('/customize')
            else:
                flash("Please fill up the date range if you want to customize them")
                return redirect('/customize')
        author1_colour = request.form["colour1"]
        author2_colour = request.form["colour2"]
        if author1_colour == author2_colour:
            flash("Please choose different colours for the two authors")
            return redirect('/customize')
        try:
            shape = request.form["shape"]
        except KeyError:
            flash("Please choose a shape for your wordcloud")
            return redirect('/customize')
        return redirect('/output')
    return render_template('customize.html', author1=author1_name, author2=author2_name, startDate=startDate, endDate=endDate)


@app.route('/output/')
def output():
    plot()
    '''
    pdf = pdfkit.from_string(process("uploads/" + filename, author1_name, author2_name, startDate, endDate, author1_colour, author2_colour), False) #for testing
    #pdf = pdfkit.from_file("templates/file.html", False) #generate pdf with a black rectangle
    #pdf = pdfkit.from_string(render_template('file.html', onehtml = base64.b64encode(open("uploads/timeline.png", "rb").read())), False) #cannot find file (because it's rendered)
    '''
    one = base64.b64encode(open("uploads/timeofday.png", "rb").read()).decode('utf-8')
    two = base64.b64encode(open("uploads/dayofweek.png", "rb").read()).decode('utf-8')
    three = base64.b64encode(open("uploads/timeline.png", "rb").read()).decode('utf-8')
    four = base64.b64encode(open("uploads/author1cloud.png", "rb").read()).decode('utf-8')
    five = base64.b64encode(open("uploads/author2cloud.png", "rb").read()).decode('utf-8')
    six = base64.b64encode(open("uploads/sentiment.png", "rb").read()).decode('utf-8')
    
    rendered_template = render_template('file.html', author1=author1_name, author2=author2_name,\
        colour1=author1_colour, colour2=author2_colour, \
            wpm1=author1_wpm, wpm2=author2_wpm, morewpm=more_words,\
                messages1=author1_messages, messages2=author2_messages, moremessages=more_messages,\
                    onehtml=one, twohtml=two, threehtml=three, fourhtml=four, fivehtml=five, sixhtml=six)
    pdf = pdfkit.from_string(rendered_template, False)
    
    response = make_response(pdf)
    response.headers['Content-Type'] = 'application/pdf'
    response.headers['Content-Disposition'] = 'inline; filename=inspecText.pdf'
    
    os.remove(os.path.join("uploads", filename))
    os.remove(os.path.join("uploads", 'timeline.png'))
    os.remove(os.path.join("uploads", 'timeofday.png'))
    os.remove(os.path.join("uploads", 'dayofweek.png'))
    os.remove(os.path.join("uploads", 'author1cloud.png'))
    os.remove(os.path.join("uploads", 'author2cloud.png'))
    os.remove(os.path.join("uploads", 'sentiment.png'))
    
    return response

###################################################################################################################

if __name__ == '__main__':
    app.run(debug=True)
