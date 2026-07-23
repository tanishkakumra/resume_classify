import pandas as pd
import re
import nltk
import matplotlib.pyplot as plt
import joblib
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from ftfy import fix_text
from sklearn.feature_extraction.text import TfidfTransformer,CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split,GridSearchCV
from scikeras.wrappers import KerasClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()
def clean_text(text):
    text = fix_text(text)

    text = text.lower()

    text = re.sub(r'https?://\S+|www\.\S+', '', text)

    text = re.sub(
        r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
        "",
        text
    )

    text = re.sub(
        r"\+?\d[\d\s\-\(\)]{7,}\d",
        "",
        text
    )

    text = re.sub(
        r"<[^>]+>",
        "",
        text
    )

    text = re.sub(
        r"[^\w\s]",
        " ",
        text
    )

    text = re.sub(
            r"\s+",
            " ",
            text
        ).strip()

    text = " ".join(
        word for word in text.split()
        if word not in stop_words
    )

    text = " ".join(
        lemmatizer.lemmatize(word)
        for word in text.split()
    )

    return text


df = pd.read_csv("data/raw/UpdatedResumeDataSet.csv")
df = df.drop_duplicates(keep="first").reset_index(drop=True)
df["Resume"] = df["Resume"].apply(clean_text)


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

 #Loss graph
# plt.plot(history.history['loss'], label='Training Loss')
# plt.plot(history.history['val_loss'], label='Validation Loss')

# plt.xlabel("Epoch")
# plt.ylabel("Loss")
#plt.title("Training and Validation Loss")
# plt.legend()
# plt.show()
loss,accuracy=model.evaluate(X_test.toarray(),y_test)
print("loss:",loss)
print("accuracy:",accuracy)
model.save('model/resume_classifier_model.keras')

joblib.dump(cv, "model/count_vectorizer.pkl")
joblib.dump(tfidf, "model/tfidf_transformer.pkl")
joblib.dump(lm, "model/label_encoder.pkl")


# def build_model(hidden_units=128, dropout_rate=0.5):
#     model = Sequential([
#         Dense(hidden_units, activation='relu', input_shape=(input_dim,)),
#         Dropout(dropout_rate),
#         Dense(64, activation='relu'),
#         Dropout(0.3),
#         Dense(num_classes, activation='softmax')
#     ])
#     model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
#     return model

# keras_clf = KerasClassifier(
#     model=build_model,
#     hidden_units=128,
#     dropout_rate=0.5,
#     epochs=50,
#     batch_size=8,
#     verbose=0,
#     validation_split=0.2
# )
# param_grid = {
#     'hidden_units': [64, 128],
#     'dropout_rate': [0.3, 0.5],
#     'batch_size': [8, 16]
# }

# grid = GridSearchCV(keras_clf, param_grid, cv=3, scoring='accuracy', n_jobs=1)
# grid.fit(X_train.toarray(), y_train)

# print("Best params:", grid.best_params_)
# print("Best CV score:", grid.best_score_)


model = load_model("model/resume_classifier_model.keras")
cv = joblib.load("model/count_vectorizer.pkl")
tfidf = joblib.load("model/tfidf_transformer.pkl")
lm = joblib.load("model/label_encoder.pkl")

predictions=model.predict(X_test.toarray())
print(classification_report(y_test,predictions.argmax(axis=1),target_names=lm.classes_))
print(confusion_matrix(y_test,predictions.argmax(axis=1)))

def predict_category(text):
    cleaned = clean_text(text)  
    vec = cv.transform([cleaned])
    vec_tfidf = tfidf.transform(vec)
    
    probs = model.predict(vec_tfidf.toarray())[0]
    pred_idx = probs.argmax()
    pred_label = lm.inverse_transform([pred_idx])[0]
    confidence = probs[pred_idx] * 100
    top3_idx = probs.argsort()[::-1][:3]
    top3 = []

    for idx in top3_idx:

       label = lm.inverse_transform([idx])[0]
       confidence = probs[idx] * 100
       top3.append((label, confidence))
    
    return pred_label, confidence,top3

resume = """..."""


label, confidence ,top3 = predict_category(resume)

print(f"Predicted category: {label}")
print(f"Confidence: {confidence:.2f}%")
print("Top 3 matches:")
for idx in top3:
    print(f"Category: {idx[0]}, Confidence: {idx[1]:.2f}%")