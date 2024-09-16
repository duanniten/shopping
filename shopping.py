import csv
import sys

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier as KNC

TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")

def is_weekend(isWeekend:bool):
    if isWeekend: return 1
    elif isWeekend=="False": return 0
    raise NameError("weekend type values problem")

def is_revenue(isRevenue:bool):
    if isRevenue: return 1
    elif isRevenue=="False": return 0
    raise NameError("Revenue type values problem")

def mouth_to_int(mes):
    # Mapeamento dos meses para inteiros de 0 a 11
    meses = {
        "Jan": 0,
        "Feb": 1,
        "Mar": 2,
        "Apr": 3,
        "May": 4,
        "Jun": 5,
        "Jul": 6,
        "Aug": 7,
        "Sep": 8,
        "Oct": 9,
        "Nov": 10,
        "Dec": 11
    }
    
    # Retorna o índice do mês ou uma mensagem de erro
    return meses.get(mes, "Invalid Mouth")

def new_or_return(client:str):
    if client == "Returning_Visitor" : return 1
    else : return 0


def load_data(filename):
    """
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).

    evidence should be a list of lists, where each list contains the
    following values, in order:
        - Administrative, an integer
        - Administrative_Duration, a floating point number
        - Informational, an integer
        - Informational_Duration, a floating point number
        - ProductRelated, an integer
        - ProductRelated_Duration, a floating point number
        - BounceRates, a floating point number
        - ExitRates, a floating point number
        - PageValues, a floating point number
        - SpecialDay, a floating point number
        - Month, an index from 0 (January) to 11 (December)
        - OperatingSystems, an integer
        - Browser, an integer
        - Region, an integer
        - TrafficType, an integer
        - VisitorType, an integer 0 (not returning) or 1 (returning)
        - Weekend, an integer 0 (if false) or 1 (if true)

    labels should be the corresponding list of labels, where each label
    is 1 if Revenue is true, and 0 otherwise.
    """
    with open(filename) as f:
        reader = csv.reader(f)
        next(reader)
        evidence = []
        labels = []
        for row in reader:
            evidence.append([
                    int(row[0]),
                    float(row[1]),
                    int(row[2]),
                    float(row[3]),
                    int(row[4]),
                    float(row[5]),
                    float(row[6]),
                    float(row[7]),
                    float(row[8]),
                    float(row[9]),
                    mouth_to_int(row[10]),
                    int(row[11]),
                    int(row[12]),
                    int(row[13]),
                    int(row[14]),
                    new_or_return(row[15]),
                    is_weekend(row[16])]
            )
            labels.append( is_revenue(row[17]))
            
    return (evidence, labels)

def train_model(evidence:list, labels:list):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    knc = KNC(n_neighbors=1)
    knc.fit(evidence, labels)
    return knc

def evaluate(labels:list, predictions:list):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificity).

    Assume each label is either a 1 (positive) or 0 (negative).

    `sensitivity` should be a floating-point value from 0 to 1
    representing the "true positive rate": the proportion of
    actual positive labels that were accurately identified.

    `specificity` should be a floating-point value from 0 to 1
    representing the "true negative rate": the proportion of
    actual negative labels that were accurately identified.
    """
    sensitivity = 0
    specificity = 0
    for y, y_predic in zip(labels, predictions):
        if y==1 and y_predic == 1:
            sensitivity += 1
        elif y==0 and y_predic == 0:
            specificity += 1
    sensitivity = sensitivity / labels.count(1)
    specificity = specificity / labels.count(0)

    return (sensitivity, specificity)


if __name__ == "__main__":
    main()
