import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import warnings
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix,classification_report
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import warnings
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from collections import Counter
warnings.filterwarnings('ignore')

data=pd.read_csv("star_classification.csv") #CSV File is read
df=data.copy()

rows,columns = df.shape
unique=df.nunique()
unique["Total Rows"] = rows


df.drop('rerun_ID',axis=1, inplace=True) 
df.drop('spec_obj_ID', axis=1, inplace=True)

y = df['class']
x = df.drop('class', axis=1)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

# SMOTE ile dengelenmiş eğitim verilerini oluşturun
smote = SMOTE(random_state=42)
x_train_smote, y_train_smote = smote.fit_resample(x_train, y_train)

# Random Forest sınıflandırma modelini oluşturun ve eğitin
rf = RandomForestClassifier(max_depth=7, max_features=3, n_estimators=100)
rf.fit(x_train_smote, y_train_smote)

# Test verileri üzerinde modeli değerlendirin ve karışıklık matrisini oluşturun
v = confusion_matrix(y_test, rf.predict(x_test))

# Confusion matrix'i çizdirin
plot_confusion_matrix(v, class_names=["GALAXY", "QSO", "STAR"], cmap='YlOrRd')
plt.show()

# Classification report'u alın ve gereksiz kısımları çıkarın
report = classification_report(y_test, rf.predict(x_test), target_names=["GALAXY", "QSO", "STAR"], output_dict=True)
print("Precision :")
print("GALAXY:", report["GALAXY"]["precision"])
print("QSO:", report["QSO"]["precision"])
print("STAR:", report["STAR"]["precision"])
print("\nRecall:")
print("GALAXY:", report["GALAXY"]["recall"])
print("QSO:", report["QSO"]["recall"])
print("STAR:", report["STAR"]["recall"])
print("\nF1-Score:")
print("GALAXY:", report["GALAXY"]["f1-score"])
print("QSO:", report["QSO"]["f1-score"])
print("STAR:", report["STAR"]["f1-score"])
print("\nAccuracy :", accuracy_score(y_test, rf.predict(x_test)))


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import SMOTE

# SMOTE ile dengelenmiş eğitim verilerini oluşturun
smote = SMOTE(random_state=42)
x_train_smote, y_train_smote = smote.fit_resample(x_train, y_train)

# k-NN sınıflandırma modelini oluşturun ve eğitin
knn = KNeighborsClassifier(n_neighbors=5)  # Burada 5 komşu kullanıldı, kendi modelinize uygun bir değeri seçebilirsiniz
knn.fit(x_train_smote, y_train_smote)

# Test verileri üzerinde modeli değerlendirin ve karışıklık matrisini oluşturun
v = confusion_matrix(y_test, knn.predict(x_test))

# Confusion matrix'i çizdirin
plot_confusion_matrix(v, class_names=["GALAXY", "QSO", "STAR"], cmap='YlOrRd')
plt.show()

# Classification report'u alın ve gereksiz kısımları çıkarın
report = classification_report(y_test, knn.predict(x_test), target_names=["GALAXY", "QSO", "STAR"], output_dict=True)
print("Precision :")
print("GALAXY:", report["GALAXY"]["precision"])
print("QSO:", report["QSO"]["precision"])
print("STAR:", report["STAR"]["precision"])
print("\nRecall:")
print("GALAXY:", report["GALAXY"]["recall"])
print("QSO:", report["QSO"]["recall"])
print("STAR:", report["STAR"]["recall"])
print("\nF1-Score:")
print("GALAXY:", report["GALAXY"]["f1-score"])
print("QSO:", report["QSO"]["f1-score"])
print("STAR:", report["STAR"]["f1-score"])
print("\nAccuracy :", accuracy_score(y_test, knn.predict(x_test)))

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from imblearn.over_sampling import SMOTE

# SMOTE ile dengelenmiş eğitim verilerini oluşturun
smote = SMOTE(random_state=42)
x_train_smote, y_train_smote = smote.fit_resample(x_train, y_train)

# "Naive Bayes" sınıflandırma modelini oluşturun ve eğitin
nb = GaussianNB()
nb.fit(x_train_smote, y_train_smote)

# Test verileri üzerinde modeli değerlendirin ve karışıklık matrisini oluşturun
v = confusion_matrix(y_test, nb.predict(x_test))

# Confusion matrix'i çizdirin
plot_confusion_matrix(v, class_names=["GALAXY", "QSO", "STAR"], cmap='YlOrRd')
plt.show()

# Classification report'u alın ve gereksiz kısımları çıkarın
report = classification_report(y_test, nb.predict(x_test), target_names=["GALAXY", "QSO", "STAR"], output_dict=True)
print("Precision :")
print("GALAXY:", report["GALAXY"]["precision"])
print("QSO:", report["QSO"]["precision"])
print("STAR:", report["STAR"]["precision"])
print("\nRecall:")
print("GALAXY:", report["GALAXY"]["recall"])
print("QSO:", report["QSO"]["recall"])
print("STAR:", report["STAR"]["recall"])
print("\nF1-Score:")
print("GALAXY:", report["GALAXY"]["f1-score"])
print("QSO:", report["QSO"]["f1-score"])
print("STAR:", report["STAR"]["f1-score"])
print("\nAccuracy :", accuracy_score(y_test, nb.predict(x_test)))


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE

# SMOTE ile dengelenmiş eğitim verilerini oluşturun
smote = SMOTE(random_state=42)
x_train_smote, y_train_smote = smote.fit_resample(x_train, y_train)

# SVM sınıflandırma modelini oluşturun ve eğitin
svm = SVC(kernel='rbf')  # rbf çekirdek kullanıldı, modeliniz için uygun bir çekirdek seçebilirsiniz
svm.fit(x_train_smote, y_train_smote)

# Test verileri üzerinde modeli değerlendirin ve karışıklık matrisini oluşturun
v = confusion_matrix(y_test, svm.predict(x_test))

# Confusion matrix'i çizdirin
plot_confusion_matrix(v, class_names=["GALAXY", "QSO", "STAR"], cmap='YlOrRd')
plt.show()

# Classification report'u alın ve gereksiz kısımları çıkarın
report = classification_report(y_test, svm.predict(x_test), target_names=["GALAXY", "QSO", "STAR"], output_dict=True)
print("Precision :")
print("GALAXY:", report["GALAXY"]["precision"])
print("QSO:", report["QSO"]["precision"])
print("STAR:", report["STAR"]["precision"])
print("\nRecall:")
print("GALAXY:", report["GALAXY"]["recall"])
print("QSO:", report["QSO"]["recall"])
print("STAR:", report["STAR"]["recall"])
print("\nF1-Score:")
print("GALAXY:", report["GALAXY"]["f1-score"])
print("QSO:", report["QSO"]["f1-score"])
print("STAR:", report["STAR"]["f1-score"])
print("\nAccuracy :", accuracy_score(y_test, svm.predict(x_test)))


