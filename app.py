import os.path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import timeit

st.set_option('deprecation.showPyplotGlobalUse', False)

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "attrition_data.xlsx")

@st.cache
def load_data():
    df = pd.read_excel(DATA)
    df.columns = [column.replace(" ", "_").lower() for column in df.columns]
    df = df.drop(columns=['employeecount','employeenumber','standardhours'])
    df['attrition'] = df['attrition'].map({'Yes':1, 'No':0})
    return df

df = load_data()

st.title("Attrition Modeling: Exploring Workforce Attrition")
st.header("**Intro**")
st.markdown("* This is a demonstration application developed by Andy Mean.")
st.markdown("* The data utilized is historical attrition data for a generic company that is publically available for use.")
st.markdown("* Please contact andy.d.mean@gmail.com if you have any questions.")

if st.checkbox('Click to show the sample of raw data'):
    st.write(df.head(10), index=False)

st.header("**Selecting Features**")

#feature selection - look for vars close to 1 or -1
plt.figure(figsize=(12,10))
cor = df.corr()
#mask redundancy
mask_ut=np.triu(np.ones(cor.shape)).astype(np.bool)
sns.heatmap(cor, annot=False, cmap=sns.diverging_palette(240, 10, n=20), mask=mask_ut, linewidth=1)
st.pyplot()

#or just corr against target
df_corr = df.corrwith(df['attrition']).round(decimals=2).abs().sort_values(ascending=False)
st.write(df_corr)
#how to determine which features to use is variable...going with .10+

st.header("**Applying Different Machine Learning Models**")
st.markdown('In order to test different models, you first need to select your desired features and prediction target. Features are the dataset attributes that you want to train the model on. Target is the label you are trying to predict. Finally you can select different algorithms from the radio list to compare accuracy and model diagnostics and determine the most promising one.')
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

sel_features = st.multiselect('Select Prediction Features:', ['totalworkingyears','joblevel','yearswithcurrmanager','monthlyincome','age','yearsincurrentrole','stockoptionlevel','yearsatcompany','jobinvolvement','jobsatisfaction','environmentsatisfaction'], default=['totalworkingyears','age','joblevel'])
sel_target = st.selectbox('Select Prediction Target:', ['attrition'])

features = df[sel_features].values
labels = df[sel_target].values

data_balance = st.radio('Do you want to balance data classes?', ('Yes, balance my data','No, use the raw data'))

if data_balance == 'No, use the raw data':
    X_train,X_test, y_train, y_test = train_test_split(features, labels, train_size=0.7, random_state=1)

if data_balance == 'Yes, balance my data':
    #balance dataset with undersampling
    from imblearn.under_sampling import RandomUnderSampler
    rus = RandomUnderSampler(random_state=0)
    X_rus, y_rus = rus.fit_resample(features, labels)
    #train model
    X_train,X_test, y_train, y_test = train_test_split(X_rus, y_rus, train_size=0.7, random_state=1)

    #scale values
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

alg =['Logisitic Regression', 'Random Forest', 'Decision Tree', 'Support Vector Machine', 'K-Nearest Neighbor', 'AdaBoost', 'Gradient Boosting', 'Naive Bayes', 'Linear Discriminant Analysis']
classifier = st.radio('Select an algorithm', alg)
with st.spinner('Model Running'):
    start_time = timeit.default_timer()
    if classifier =='Logisitic Regression':
        selected_class = LogisticRegression(random_state=0)
        selected_class.fit(X_train, y_train)
        accuracy = selected_class.score(X_test, y_test)
        accuracy_percentage = 100 * accuracy
        st.write('Logisitc Regression Model Accuracy: ', 
        accuracy_percentage.round(decimals=2),'%')
        pred_selected_class = selected_class.predict(X_test)
        cm_selected_class = confusion_matrix(y_test, pred_selected_class)
        st.write('Logistic Regression Confusion Matrix: ', cm_selected_class)
    
    elif classifier =='Random Forest':
        selected_class = RandomForestClassifier(oob_score=True,n_estimators=100, 
        random_state=0)
        selected_class.fit(X_train, y_train)
        accuracy = selected_class.score(X_test, y_test)
        accuracy_percentage = 100 * accuracy
        st.write('Random Forest Model Accuracy: ', 
        accuracy_percentage.round(decimals=2),'%')
        pred_selected_class = selected_class.predict(X_test)
        cm_selected_class = confusion_matrix(y_test, pred_selected_class)
        st.write('Random Forest Confusion Matrix: ', cm_selected_class)

    elif classifier =='Decision Tree':
        selected_class = DecisionTreeClassifier(random_state=0)
        selected_class.fit(X_train, y_train)
        accuracy = selected_class.score(X_test, y_test)
        accuracy_percentage = 100 * accuracy
        st.write('Decision Tree Model Accuracy: ', 
        accuracy_percentage.round(decimals=2),'%')
        pred_selected_class = selected_class.predict(X_test)
        cm_selected_class = confusion_matrix(y_test, pred_selected_class)
        st.write('Decision Tree Confusion Matrix: ', cm_selected_class)

    elif classifier =='Support Vector Machine':
        selected_class = SVC(random_state=0)
        selected_class.fit(X_train, y_train)
        accuracy = selected_class.score(X_test, y_test)
        accuracy_percentage = 100 * accuracy
        st.write('Support Vector Machine Model Accuracy: ', 
        accuracy_percentage.round(decimals=2),'%')
        pred_selected_class = selected_class.predict(X_test)
        cm_selected_class = confusion_matrix(y_test, pred_selected_class)
        st.write('Support Vector Machine Confusion Matrix: ', cm_selected_class)
    
    elif classifier =='K-Nearest Neighbor':
        selected_class = KNeighborsClassifier()
        selected_class.fit(X_train, y_train)
        accuracy = selected_class.score(X_test, y_test)
        accuracy_percentage = 100 * accuracy
        st.write('K-Nearest Neighbor Model Accuracy: ', 
        accuracy_percentage.round(decimals=2),'%')
        pred_selected_class = selected_class.predict(X_test)
        cm_selected_class = confusion_matrix(y_test, pred_selected_class)
        st.write('K-Nearest Neighbor Confusion Matrix: ', cm_selected_class)
        
    elif classifier =='AdaBoost':
        selected_class = AdaBoostClassifier()
        selected_class.fit(X_train, y_train)
        accuracy = selected_class.score(X_test, y_test)
        accuracy_percentage = 100 * accuracy
        st.write('AdaBoost Model Accuracy: ', 
        accuracy_percentage.round(decimals=2),'%')
        pred_selected_class = selected_class.predict(X_test)
        cm_selected_class = confusion_matrix(y_test, pred_selected_class)
        st.write('AdaBoost Confusion Matrix: ', cm_selected_class)

    elif classifier =='Gradient Boosting':
        selected_class = GradientBoostingClassifier(random_state=0)
        selected_class.fit(X_train, y_train)
        accuracy = selected_class.score(X_test, y_test)
        accuracy_percentage = 100 * accuracy
        st.write('Gradient Boosting Model Accuracy: ', 
        accuracy_percentage.round(decimals=2),'%')
        pred_selected_class = selected_class.predict(X_test)
        cm_selected_class = confusion_matrix(y_test, pred_selected_class)
        st.write('Gradient Boosting Confusion Matrix: ', cm_selected_class)
        
    elif classifier =='Naive Bayes':
        selected_class = GaussianNB()
        selected_class.fit(X_train, y_train)
        accuracy = selected_class.score(X_test, y_test)
        accuracy_percentage = 100 * accuracy
        st.write('Naive Bayes Model Accuracy: ', 
        accuracy_percentage.round(decimals=2),'%')
        pred_selected_class = selected_class.predict(X_test)
        cm_selected_class = confusion_matrix(y_test, pred_selected_class)
        st.write('Naive Bayes Confusion Matrix: ', cm_selected_class)
        
    elif classifier =='Linear Discriminant Analysis':
        selected_class = LinearDiscriminantAnalysis()
        selected_class.fit(X_train, y_train)
        accuracy = selected_class.score(X_test, y_test)
        accuracy_percentage = 100 * accuracy
        st.write('LDA Model Accuracy: ', 
        accuracy_percentage.round(decimals=2),'%')
        pred_selected_class = selected_class.predict(X_test)
        cm_selected_class = confusion_matrix(y_test, pred_selected_class)
        st.write('LDA Confusion Matrix: ', cm_selected_class)
        
    elapsed = timeit.default_timer() - start_time
st.header("**Model Results**")
'Model Computation Time: %f seconds' % elapsed
pred_df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': pred_selected_class.flatten()})
pred_df['Model_Outcome'] = pred_df.apply(lambda x: x['Actual'] == x['Predicted'], axis=1)
pred_df['Model_Outcome'] = pred_df['Model_Outcome'].replace((1,0), ('Correct','Incorrect'))

fig = px.histogram(pred_df, x='Model_Outcome', title='Prediction Accuracy Count', width=375, height=500, color='Model_Outcome', color_discrete_sequence = px.colors.qualitative.Prism).update_xaxes(categoryorder='total descending')
fig.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)'}, showlegend=False)
st.plotly_chart(fig)

st.markdown('Sample of Prediction Table (First 10 Rows)')
st.table(pred_df.head(10))

if st.checkbox('Show Model Diagnostics'):
            st.subheader('Confusion Matrix')
            'The confusion matrix indicates some stuff.'
            class_names=['No Attrition','Attrition']
            plot_confusion_matrix(selected_class, X_test, y_test, normalize='all', values_format='.0%', display_labels=class_names, cmap='Blues')
            st.pyplot()
            st.subheader('ROC Curve')
            'The reciever operating characteristic (ROC) indicates how capable the model is at distinguishing between classes (0s and 1s). A higher area under curve (AUC), the better the model is at distinguishing between employees that leave and those that do not. An excellent model will have an AUC close to 1, a model with an AUC close to 0.5 will have no predictive capacity.'
            plot_roc_curve(selected_class, X_test, y_test)
            plt.plot([0, 1], [0, 1],'r--')
            st.pyplot()
            st.subheader('Precision-Recall Curve')
            'The precision-recall curve indicates the positive predictive value of the model. Precision is a ratio of the number of true positives divided by the sum of true positives and true negatives. Recall is the ratio of the number of true positives divided by the sum of the true positives and the false negatives.'
            plot_precision_recall_curve(selected_class, X_test, y_test)
            st.pyplot()
            st.subheader('When to use ROC vs. Precision-Recall Curves')
            'Generally, ROC curves are best when there are roughly equal numbers of observations for each class. Precision-Recall curves are best when there is a large class imbalance.'
