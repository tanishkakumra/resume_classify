import pandas as pd
import re
import nltk
import matplotlib.pyplot as plt
from ftfy import fix_text
from sklearn.feature_extraction.text import TfidfTransformer,CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model

df = pd.read_csv("data/raw/UpdatedResumeDataSet.csv")
df["Resume"] = df["Resume"].apply(fix_text)

df=df.drop_duplicates(keep='first').reset_index(drop=True)
df['Resume']=df['Resume'].str.lower()
df["Resume"] = df["Resume"].str.replace(
    r'https?://\S+|www\.\S+',
    '',
    regex=True
)


df["Resume"] = df["Resume"].str.replace(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}","",regex=True)

df["Resume"] = df["Resume"].str.replace(
    r"\+?\d[\d\s\-\(\)]{7,}\d",
    "",
    regex=True
)

df["Resume"] = df["Resume"].str.replace(
    r"<[^>]+>",
    "",
    regex=True
)

df["Resume"] = df["Resume"].str.replace(
    r"\s+",
    " ",
    regex=True
).str.strip()
df["Resume"] = df["Resume"].str.replace(
    r"[^\w\s]",
    " ",
    regex=True
)
import nltk


from nltk.corpus import stopwords
stop_words=set(stopwords.words('english'))
df["Resume"] = df["Resume"].apply(
    lambda text: " ".join(
        word for word in text.split()
        if word not in stop_words
    )
)


from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
df["Resume"] = df["Resume"].apply(
    lambda text: " ".join(
        lemmatizer.lemmatize(word)
        for word in text.split()
    )
)
X=df['Resume']
y=df["Category"]
cv=CountVectorizer(max_features=5000)
X_cv = cv.fit_transform(X)
lm=LabelEncoder()
y_lm=lm.fit_transform(y)
tfidf=TfidfTransformer()
X_tfidf=tfidf.fit_transform(X_cv)
X_train, X_test, y_train, y_test = train_test_split( X_tfidf, y_lm, test_size=0.2,random_state=42, stratify=y_lm)


num_classes=len(set(y_lm))
input_dim=X_tfidf.shape[1]
model=Sequential()
model.add(Dense(128, activation='relu',input_shape=(input_dim,),))
model.add(Dropout(0.5))
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(num_classes,activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

early_stopping=EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)

history=model.fit(X_train.toarray(),y_train,epochs=100,batch_size=8,validation_split=0.2,verbose=1,callbacks=[early_stopping])
# my X-train is a sprase matrix,so i need to convert it into an array before passing it to the model

 
# plt.plot(history.history['loss'], label='Training Loss')
# plt.plot(history.history['val_loss'], label='Validation Loss')

# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.legend()
# plt.show()
loss,accuracy=model.evaluate(X_test.toarray(),y_test)
print("loss:",loss)
print("accuracy:",accuracy)
model.save('model/resume_classifier_model.keras')