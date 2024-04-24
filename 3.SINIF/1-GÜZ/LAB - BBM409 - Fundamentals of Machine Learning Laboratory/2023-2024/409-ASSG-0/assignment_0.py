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
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
warnings.filterwarnings('ignore')

data=pd.read_csv("star_classification.csv")
df=data.copy()

#print(df['class'].value_counts())

x=df.drop('class',axis=1).values
y=df['class'].values


print(df.shape)

x_train , x_test , y_train , y_test = train_test_split(x,y, test_size= 0.33 , random_state= 42)

print(1)
# SMOTE ile dengelenmiş eğitim verilerini oluşturun
smote = SMOTE(random_state=42)
x_train_smote, y_train_smote = smote.fit_resample(x_train, y_train)
print(2)
# SVM sınıflandırma modelini oluşturun ve eğitin
svm = SVC(kernel='rbf')  # rbf çekirdek kullanıldı, modeliniz için uygun bir çekirdek seçebilirsiniz
svm.fit(x_train_smote, y_train_smote)
print(3)
# Test verileri üzerinde modeli değerlendirin ve karışıklık matrisini oluşturun
v = confusion_matrix(y_test, svm.predict(x_test))
print(4)
# Confusion matrix'i çizdirin
plot_confusion_matrix(v, class_names=["GALAXY", "QSO", "STAR"], cmap='YlOrRd')
plt.show()
print(5)
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

print("6")