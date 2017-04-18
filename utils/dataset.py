import csv, codecs, cStringIO, sys
from csv import DictReader
reload(sys);
sys.setdefaultencoding("utf8")



class UTF8Recoder:
    """
    Iterator that reads an encoded stream and reencodes the input to UTF-8
    """
    def __init__(self, f, encoding):
        self.reader = codecs.getreader(encoding)(f)

    def __iter__(self):
        return self

    def next(self):
        result = self.reader.next().encode("utf-8")
        return result.encode("ascii", "ignore")

class UnicodeDictReader:
    """
    A CSV reader which will iterate over lines in the CSV file "f",
    which is encoded in the given encoding.
    """

    def __init__(self, f, dialect=csv.excel, encoding="utf-8", **kwds):
        f = UTF8Recoder(f, encoding)
        self.reader = csv.reader(f, dialect=dialect, **kwds)
        self.header = self.reader.next()

    def next(self):
        row = self.reader.next()
        vals = [unicode(s, "utf-8") for s in row]
        return dict((self.header[x], vals[x]) for x in range(len(self.header)))

    def __iter__(self):
        return self


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
        with codecs.open(self.path + "/" + filename, "r", encoding="utf-8") as table:
            r = UnicodeDictReader(table)

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
