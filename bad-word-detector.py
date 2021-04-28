import nltk
nltk.download('punkt')
import fasttext
import random
import sys
from pandas import *
import clean_tweet
import csv
import os

#cleaning the scraped tweets
finalOutput  = {}
score = {'BAD':1, 'HUMOUR': 0,'HATE': 0 ,'NA': 0}
tweets = clean_tweet.getDataFromCSV("scraped_data.csv")
cleanedTweet = []
for tweet in tweets:
    cleanedTweet.append(clean_tweet.tweet_cleaning_for_sentiment_analysis(tweet))

column = ["Letter"]

#Accessing the bad words file
df = pandas.read_csv("english.csv", names=column)
badwords = df.Letter.to_list()
badwords = set(badwords)
finalOutputList = []
#Labeling the tweets as Positive and Negative
for j in cleanedTweet:
    #bad word containting tweets in finalOutputList
    input_file = j.split()
    text = []
    for i in input_file:
        text.append(i)
    text_list = text
    allAbuse = []
    for sentence in text_list:
        if sentence in badwords:
            abuses = [i for i in sentence.lower().split()]
            allAbuse.append(abuses)
    if(len(allAbuse) != 0):
        finalOutput[j] = allAbuse
        finalOutputList.append(finalOutput)
label = ['Positive', 'Negative', 'Neutral']


#Labeling as Negative
finalLabelledData = []
positiveList = list(set(cleanedTweet)^(set(list(finalOutputList[0].keys()))))
for i in finalOutputList:
    for j in i:
        print('\n')
        negative = []
        negative.append('__label__NEGATIVE '+str(j))
        finalLabelledData.append(negative)
#Labeling as Postitive
for i in positiveList:
    positive = []
    positive.append('__label__POSITIVE '+str(i))
    finalLabelledData.append(positive)

#Adding the labeled in scra[ed_labeled_csv file]
random.shuffle(finalLabelledData)
csvFile = open('scraped_labeled_data.csv', 'w')
csvWriter = csv.writer(csvFile)
for i in finalLabelledData: 
    csvWriter.writerow(i)
#Upsampling the sample
def upsampling(input_file, output_file, ratio_upsmapling = 1):
    i = 0
    counts = {}
    dict_data_by_label = {}
    with open(input_file, 'r', newline='') as csvinfile: 
        csv_reader = csv.reader(csvinfile, delimiter=',', quotechar='"')
        for row in csv_reader:
            counts[row[0].split()[0]] = counts.get(row[0].split()[0], 0) + 1
            if not row[0].split()[0] in dict_data_by_label:
                dict_data_by_label[row[0].split()[0]]=[row[0]]
            else:
                dict_data_by_label[row[0].split()[0]].append(row[0])
            i=i+1
            if i%10000 ==0:
                print("read" + str(i))

    # finding the majority class
    majority_class=""
    count_majority_class=0
    for item in dict_data_by_label:
        if len(dict_data_by_label[item])>count_majority_class:
            majority_class= item
            count_majority_class=len(dict_data_by_label[item])  
    
    # upsampling the minority class
    data_upsampled=[]
    for item in dict_data_by_label:
        data_upsampled.extend(dict_data_by_label[item])
        if item != majority_class:
            items_added=0
            items_to_add = count_majority_class - len(dict_data_by_label[item])
            while items_added<items_to_add:
                data_upsampled.extend(dict_data_by_label[item][:max(0,min(items_to_add-items_added,len(dict_data_by_label[item])))])
                items_added = items_added + max(0,min(items_to_add-items_added,len(dict_data_by_label[item])))

    random.shuffle(data_upsampled)
    # WRITE ALL
    i=0

    with open(output_file, 'w') as txtoutfile:
        for row in data_upsampled:
            txtoutfile.write(row+ '\n' )
            i=i+1
            if i%10000 ==0:
                print("writer" + str(i))

upsampling('scraped_labeled_data.csv', 'uptweets.train')


training_data_path ='uptweets.train' 
#validation_data_path ='tweets.validation'
model_path =''
model_name="model-en"

#training using fastText
def train(inputText):
    print('Training start')
    try:
        hyper_params = {"lr": 0.01,
                        "epoch": 30,
                        "wordNgrams": 2,
                        "dim": 20}     
                               
        print(str(datetime.datetime.now()) + ' START=>' + str(hyper_params) )

        # Train the model.
        model = fasttext.train_supervised(input=training_data_path, **hyper_params)
        print("Model trained with the hyperparameter \n {}".format(hyper_params))

        # CHECK PERFORMANCE
        '''print(str(datetime.datetime.now()) + 'Training complete.' + str(hyper_params) )
        
        model_acc_training_set = model.test(training_data_path)
        model_acc_validation_set = model.test(validation_data_path)
        
        # DISPLAY ACCURACY OF TRAINED MODEL
        text_line = str(hyper_params) + ",accuracy:" + str(model_acc_training_set[1])  + ", validation:" + str(model_acc_validation_set[1]) + '\n' 
        print(text_line)'''
        


        #quantize a model to reduce the memory usage
        model.quantize(input=training_data_path, qnorm=True, retrain=True, cutoff=100000)
        
        print("Model is quantized!!")
        #model.save_model(os.path.join(model_path,model_name + ".ftz"))                
    
        ##########################################################################
        #
        #  TESTING PART
        #
        ##########################################################################            
        print ('validation')
        print('-----------------------------')

        x = model.predict([inputText],k=2)
        print(x)
    except Exception as e:
        print('Exception during training: ' + str(e) )


# Train your model.
x = input("Enter tweet: ")
train(str(x))
