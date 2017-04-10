from csv import DictReader

class DataSet():
    def __init__(self, path="data", bodies="train_bodies.csv", stances="train_stances.csv"):
        self.path = path

        print("Reading dataset")

        self.stances = self.read(stances)
        articles = self.read(bodies)
        self.articles = dict()

        #make the body ID an integer value
        for s in self.stances:
            s['Body ID'] = int(s['Body ID'])

        #copy all bodies into a dictionary
        for article in articles:
            self.articles[int(article['Body ID'])] = article['articleBody']

        print("Total stances: " + str(len(self.stances)))
        print("Total bodies: " + str(len(self.articles)))

    def getBody(self, stanceDict):
        return self.articles[stanceDict['Body ID']]

    def read(self,filename):
        rows = []
        with open(self.path + "/" + filename, "r") as table:
            r = DictReader(table)

            for line in r:
                rows.append(line)
        return rows

def segmentize_dataset(dataset):
    headlines = []
    bodies = []
    classifications = []
    print ( 'Segmentizing Dataset...' )
    for stance in dataset.stances:
        headline = stance['Headline']
        headlines.append(headline)
        body = dataset.getBody(stance)
        bodies.append(body)
        classification = stance['Stance']
        classifications.append(classification)
    print ( 'Done.' )
    return (headlines, bodies, classifications)

def zip_segments(segments):
    headlines, bodies, classifications = segments
    return zip(headlines, bodies, classifications)
