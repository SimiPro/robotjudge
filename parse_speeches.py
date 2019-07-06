#!/usr/bin/env python
# coding: utf-8

# In[57]:


import os
import json
import collections
import datetime
import calendar


# In[75]:


speaker_speech = collections.defaultdict(list)
js = None
# parse specific json
def parse_json(file_name):
    with open(file_name, 'r') as f_json:
        try:
            js_ = json.load(f_json)            
        except:
            return
        
        m = list(calendar.month_name).index(js_['header']['month'])
        date = datetime.datetime(int(js_['header']['year']), m, int(js_['header']['day']))
        date = str(date).split(' ')[0]
        for turn in js_['content']:
            if turn['turn'] >= 0 and turn['speaker'] != 'None' and turn['kind'] == 'speech':
                speaker_speech[turn['speaker']] = (date, turn['text'])
parse_json("/home/simi/projects/congressional-record/output/2018/CREC-2018-01-02/json/CREC-2018-01-02-pt1-PgE1767-2.json")


# In[ ]:





# In[76]:


base_folder = '/home/simi/projects/congressional-record/output'
years = os.listdir(base_folder)
for year in years:
    base_folder_year = f"{base_folder}/{year}"
    days = os.listdir(base_folder_year)
    for day in days:
        base_folder_day = f"{base_folder_year}/{day}/json"
        docs = os.listdir(base_folder_day)
        for doc in docs:
            doc_path_json = f"{base_folder_day}/{doc}"
            #print(doc_path_json)
            parse_json(doc_path_json)


# In[77]:


with open('speeches_by_speaker.json', 'w') as json_dump:
    json.dump(speaker_speech, json_dump)
print(f"dumped to: {os.curdir}/{json_dump.name}")


# In[78]:


num_speeches = sum(len(speeches) for date, speeches in speaker_speech.values())
num_speakers = len(speaker_speech.keys())
print(f"dumped {num_speeches} speeches by {num_speakers} speakers")


# In[ ]:





# In[ ]:




