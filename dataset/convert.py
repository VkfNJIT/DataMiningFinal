import csv

def text_to_csv(text_file, csv_file, headers):
    with open(text_file, 'r') as infile, open(csv_file, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(headers)
        for line in infile:
            
            row = line.strip().split('\t')
            writer.writerow(row)

text_file = 'review_sentiment_labelled.txt'
csv_file = 'review_sentiment_labelled.csv'
headers = ['Sentence','Score']
text_to_csv(text_file, csv_file, headers)
