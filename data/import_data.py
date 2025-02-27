"""
Import sample data for classification engine
"""

import predictionio
import argparse

def import_events(client, file):
    f = open(file, 'r')
    count = 0
    print "Importing data..."
    for line in f:
        data = line.rstrip('\r\n').split(",")
        labelLen = len(data) - 120 # 120 features in this dataset
        features = data[-120:]
        labels = data[:labelLen]
        first_event_properties = {
            "labels": labels,
            "features": features,
        }
        client.create_event(
            event="$set",
            entity_type="item",
            entity_id="1",
            properties=first_event_properties
        )
        count += 1
    f.close()
    print "%s events are imported." % count

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Import sample data for classification engine")
    parser.add_argument('--access_key', default='invald_access_key')
    parser.add_argument('--url', default="http://localhost:7070")
    parser.add_argument('--file', default="./data/data.txt")

    args = parser.parse_args()
    print args

    client = predictionio.EventClient(
        access_key=args.access_key,
        url=args.url,
        threads=5,
        qsize=500)
    #libsvm_to_csv(client, args.file)
    libsvm_to_csv(client, args.file)