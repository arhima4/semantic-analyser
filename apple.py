We hope you enjoyed the contest and we'd like to hear your feedback which can be mailed to at hackers@hackerrank.com.
We'd specifically appreciate inputs on what kind of challenges you'd like to see in this contest to enjoy it more.

As we will explain later in this editorial, we'd also like to allay fears that one needs to be a Machine Learning genius to attempt these challenges. That is so not true.

Approximately 150 contestants read the challenge and tried to compile a solution to this, and 113 made a final submission. A total of ~2100 submissions were made to this challenge indicating that the users who did try out the challenge made almost 20 submissions on an average. Moving on to the challenge.

Byte the Correct Apple

This is a typical word-sense disambiguation problem. Has the word Apple been used in the context of Apple Computers, or apple the fruit?

We provide training files of Wikipedia articles about both the "Apples". From that, you could build a simple language model to identify the kind of words in their surrounding, which could be used to guess which is which.

For instance if you see a sentence with Apple occurring in the same sentence as computer or company or shares or Steve Jobs or iPhones, chances are higher that the test refers to the computer-company. If you see "apple" occurring in the same chunk of text as diet, food, nutrition, fruit etc. chances are that the test refers to the fruit.

Then, there are punctuation and capitalization issues which hold hints in them. For instance, if you read about "apples" (i.e, plural form) it is most likely the fruit. If you read "Apple's" it is more likely the computer company. If you see Apple, with a capitalized A in the middle of a sentence it is typically the computer-company.

However, we've tried out best to ensure that contestants do not ace the challenge by simply breezing through some of these heuristics. So we've included a couple of tricky tests such as something about:

    Steve Jobs is a vegetarian who named the firm for his favorite fruit. After high school, he worked in an apple orchard.  
    The bitten apple is the logo of the well-known computer manufacturer.  
    The Apple Pie, with its two rounds of pastry enclosing slices of cinnamon sugared apples, is a favorite dessert in North America. (Mixed capitalization)
The winning solution - the first submission with the highest score - by Danielcmartin and use a simple modification of the most basic unigram model, i.e. a model based on a simple histogram of word counts. Apart from that, he uses a simple heuristic based on capitalization and plural forms. The code is clean and the logic is simple, clear but super effective. The reason we highlight this solution, is to demonstrate the fact, that even if you aren't a seasoned Machine Learning champ, your ability to come up with creative and smart ideas and heuristics and put them into code, can take you a long way in contests like these.

The next submission which attained the highest score, by Dimfin is just as clean, clear and effective. This solution attempts to identify the words which "signal" that Apple is being used in the context of computer-company or fruit, by creating a set of words,which are only found in the context of the fruit, and another set of words which are only found in the context of the computer-company.

The third submission with the highest score, uses the NLTK library, and well established techniques, involving stemming, tokenization and probabilistic classification. It also uses a bit of POS tagging ideas by segmenting out the nouns which act as critical signals. For instance, if we see 'peel' or 'fibre', chances are that we're talking about apple the fruit. A brilliant solution by aronaperauch.

The winning solution is pasted here. All submissions which scored 90 and above were truly great. You can view submissions from the leaderboard here.

Problem Setter's code:

# Enter your code here. Read input from STDIN. Print output to STDOUT
import operator
import re

fruit_lookup = {}
company_lookup = {}

N = int(raw_input())
with open('apple-fruit.txt','r') as fruit:
    for line in fruit:
        for word in re.split('\W+', line.upper()):
            if word not in fruit_lookup:
                fruit_lookup[word] = 1
            else:
                fruit_lookup[word] += 1

with open('apple-computers.txt','r') as company:
    for line in company:
        for word in re.split('\W+', line.upper()):
            if word not in company_lookup:
                company_lookup[word] = 1
            else:
                company_lookup[word] += 1

sorted_fruit = sorted(fruit_lookup.iteritems(), key=operator.itemgetter(1))    

#remove frequent and common words    
for word in sorted_fruit[:200]:
    if word in company_lookup:
        del company_lookup[word]
        del fruit_lookup[word]

for i in range(N):
    line = raw_input()
    #use simple heuristic as a first check otherwise do word based scoring
    if 'apple' in line or 'apples' in line or 'APPLES' in line.upper():
        print 'fruit'
        continue

    line = re.split('\W+', line.upper())
    fruit_score = 0
    company_score = 0

    for word in line:
        if word in fruit_lookup:
            fruit_score += 1
        if word in company_lookup:
            company_score += 1

    if company_score > fruit_score:
        print 'computer-company'
    else:
        print 'fruit' 