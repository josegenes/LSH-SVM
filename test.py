import requests
import PyPDF2
import io
import re
import nltk
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.neighbors import NearestNeighbors
import numpy as np
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import GridSearchCV

# URL of the PDF file on the website
pdf_url = "https://clacso.redalyc.org/pdf/496/49615099018.pdf"
stemmer = PorterStemmer()

def download_pdf(url):
    response = requests.get(url)
    if response.status_code == 200:
        return response.content
    else:
        return None

class LSHSVM:
    def __init__(self, X, y, n_projections=100, n_buckets=40, eta1=0.001, eta2=0.005):
        self.X = X
        self.y = y
        self.n_projections = n_projections
        self.n_buckets = n_buckets
        self.eta1 = eta1
        self.eta2 = eta2
        self.projections = np.random.randn(self.n_projections, self.X.shape[1])
        self.hash_functions = []
        for i in range(self.n_buckets):
            hash_function = np.random.permutation(self.n_projections)
            self.hash_functions.append(hash_function)
    def train(self):
        buckets = []
        for i in range(self.n_buckets):
            bucket = []
            for j in range(len(self.X)):
                hash_value = self.hash_functions[i](self.projections[j])
                if hash_value == i:
                    bucket.append(j)
            buckets.append(bucket)
        selected_data_points = []
        for bucket in buckets:
            n_selected_data_points = int(self.eta1 * len(bucket))
            selected_data_points.append(np.random.choice(bucket, n_selected_data_points))
        self.svm = SVM()
        self.svm.train(self.X[selected_data_points], self.y[selected_data_points])

    def predict(self, X):
        new_projections = X @ self.projections
        new_hash_values = []
        for i in range(self.n_buckets):
            new_hash_values.append(np.argmin(np.linalg.norm(new_projections - self.projections[:, self.hash_functions[i]], axis=1)))
        selected_data_points = []
        for i in range(self.n_buckets):
            selected_data_points.append(self.X[buckets[new_hash_values[i]]])
        predictions = self.svm.predict(selected_data_points)
        return predictions

stemmer = PorterStemmer()

def extract_text_from_pdf(pdf_content):
    pdf_file = PyPDF2.PdfReader(io.BytesIO(pdf_content))
    text = ""
    pagespdf = len(pdf_file.pages)
    for page_num in range(pagespdf):
        page = pdf_file.pages[page_num]
        text += page.extract_text()
    return text

if __name__ == "__main__":
    contador = 0
    fallos = 0
    cantidad = input("Cantidad de articulos: ")
    cantidad = int(cantidad)
    ubicacion_PDF = int(49615090918)
    palabras = []
    contador = 0
    contador2 = 0
    y = [1, 1, 0, 0, 0, 0, 1, 0, 1, 0]
    while contador <= cantidad:
        ubicar = str(ubicacion_PDF+contador2)
        enlace = "https://clacso.redalyc.org/pdf/496/"+ubicar+".pdf"
        documento = download_pdf(enlace)
        if documento:
            texto = extract_text_from_pdf(documento)
            texto = re.sub(r'[^a-zA]', " ", text)
            texto = texto.lower()
            palabras.append(texto)
            contador = contador+1
            contador2 = contador2+1
    print("Encontró los artículos")

    paralelo = input("Paralelismo: Si o No?" )
    try:
        if paralelo == "Si":
            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.33)

            # Dimensionality reduction
            svd = TruncatedSVD(n_components=100)
            X_train = svd.fit_transform(X_train)
            X_test = svd.transform(X_test)

            # LSH Forest
            lshf = NearestNeighbors(n_neighbors=5, n_candidates=50, n_estimators=10, n_neighbors_to_return=5, radius=1.0,
                             radius_cutoff_ratio=0.9, n_jobs=-1)
            lshf.fit(X_train)
            def train_svm(X, y, params):
                clf = make_pipeline(LSHForestTransformer(lshf), SVC(**params))
                return clf.fit(X, y)
            param_grid = {'C': [0.1, 1, 10], 'gamma': [0.1, 1, 10]}
            best_params = GridSearchCV(clf, param_grid).fit(X_train, y_train).best_params_
            clfs = Parallel(n_jobs=-1)(delayed(train_svm)(X_train, y_train, param) for param in best_params)
            y_preds = [clf.predict(X_test) for clf in clfs]
            accuracies = [accuracy_score(y_test, y_pred) for y_pred in y_preds]
            best_accuracy_idx = accuracies.index(max(accuracies))
            best_clf = clfs[best_accuracy_idx]
            accuracy = accuracies[best_accuracy_idx]
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.33)
            svd = TruncatedSVD(n_components=100)
            X_train = svd.fit_transform(X_train)
            X_test = svd.transform(X_test)
            lshf = NearestNeighbors(n_neighbors=5)
            lshf.fit(X_train)
            clf = make_pipeline(LSHForestTransformer(lshf), SVC())
            param_grid = {'svc__C': [0.001, 0.01, 0.1], 'svc__gamma': [1, 10, 100]}
            best_params = GridSearchCV(clf, param_grid).fit(X_train, y_train).best_params_
            clf.set_params(**best_params)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
    except:
        print("")

print("Best Hyperparameters: ", best_params)
print("Accuracy: ", accuracy)