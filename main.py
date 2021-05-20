from tkinter import *
from tkinter import ttk
#from tkinter.filedialog import askopenfile
from tkinter import filedialog
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, precision_score, plot_roc_curve, f1_score
from sklearn.metrics import plot_confusion_matrix, plot_precision_recall_curve, recall_score, precision_score
from sklearn.metrics import roc_curve, auc
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from statistics import mode
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import random as rd
import seaborn as sns
import numpy as np
import warnings
warnings.filterwarnings("ignore")



root=Tk()
root.title("Easy Churn Master")
win_wid=root.winfo_screenwidth()
win_heit=root.winfo_screenheight()
geo="%dx%d+0+0"%(win_wid,win_heit)
root.geometry(geo)
root.configure(bg="#21b7c4")



#----------------------------------main farme ui-------------------------
def mainfarme_ui():
    for child in root.winfo_children():
        child.destroy()

    main_frame=Frame(root,width=(win_wid//2),height=(win_heit//2))
    main_frame.grid(row=0,column=0,padx=300,pady=150,sticky = "nsew")
    main_frame.grid_propagate (False)


    head_lable=Label(main_frame,text="Easy Customer Churn Calculator",font=(10))
    head_lable.grid(row=1,column=1,columnspan=5,padx=200,pady=20)

    address_entry=Entry(main_frame,width=40,font=(5))
    address_entry.grid(row=3,column=1,pady=20,padx=5)

    def get_address():
        filename =filedialog.askopenfilename(initialdir = "/",title="select file",filetypes=(("excel files", "*.csv"),("csv files","*.csv"),("csv files","*.csv"),("excel files","*.dbf"),("excel files","*.dif"),("excel files","*.htm"),("excel files","*.html"),("excel files","*.mht"),("excel files","*.mhtml"),("excel files","*.ods"),("excels files","*.pdf"),("excel files","*.prn"),("excel files","*.slk"),("excel files","*.txt"),("excel files","*.txt"),("excel files","*.txt"),("excel files","*.txt"),("excel files","*.xla"),("excel files","*.xlam"),("excel files","*.xls"),("excel files","*.xls"),("excel files","*.xlsb"),("excel files","*.xlsm"),("excel files","*.xlsx"),("excel files","*.xlsx"),("excel files","*.xlt"),("excel files","*.xltm"),("excel files","*.xltx"),("excel files","*.xlw"),("excel files","*.xml"),("excel files","*.xml"),("excel files","*.xps"),("all files", "*.*")))
        if filename is not None:
            address_entry.delete(0,END)
            address_entry.insert(0,filename)

    browse_button=Button(main_frame,text="Browse",width=9,font=(5),command=get_address)
    browse_button.grid(row=3,column=3,pady=20,padx=5)

    def authentication():
        x=address_entry.get()

        if len(x)!=0:
            main_frame.destroy()
            usr=User_panel(x)

        else:
            print("no")

    report_button=Button(main_frame,text="GET REPORT",font=(7),command=authentication)
    report_button.grid(row=5,column=1,columnspan=5,pady=30,padx=5)


#------------------------------------------------------------------------
class User_panel:

    def __init__(self,x):

        self.tafrm=Frame(root)
        self.tafrm.grid(row=2,column=0,sticky=N+S+E+W,columnspan=3,padx=7,pady=1)
        l7=Label(self.tafrm,text="Machine Learning Report for the Accuracy of Software")
        l7.pack(padx=1)
        self.Tx = Text(self.tafrm, height=10, width=140)
        self.Tx.pack(pady=1,padx=1)
        try:

            self.file_data=pd.read_csv (x)
            self.df = pd.DataFrame(self.file_data, columns= ['customer_id','vintage','age','gender','dependents','occupation','city','customer_nw_category','branch_code','days_since_last_transaction','current_balance','previous_month_end_balance','average_monthly_balance_prevQ'	,'average_monthly_balance_prevQ2'	,'current_month_credit'	,'previous_month_credit'	,'current_month_debit'	,'previous_month_debit','current_month_balance','previous_month_balance'	,'churn'])
            self.df['gender'].fillna(value='Male',inplace=True)
            self.df['dependents'].fillna(0,inplace=True)
            self.df['occupation'].fillna(value = 'self_employed', inplace=True)
            self.df['days_since_last_transaction'].fillna(0,inplace=True)
            self.df['city'].fillna(0,inplace=True)
            self.df['vintage'] = self.df['vintage'].transform(func='sqrt')
            self.df['days_since_last_transaction'] = self.df['days_since_last_transaction'].transform(lambda x:x**0.5)

    #=====================================================================================================================
            #------------------molding----------------
            self.data = pd.get_dummies(self.df,dtype = 'int')
            x = self.data.drop('churn',axis=1)
            y = self.data['churn']
           # print(x.shape, y.shape)
            #-------------------------------------------
            #-----------------Logistic Regression-------
                #------------Splitting---------
            train_x, valid_x, train_y, valid_y= train_test_split(x, y, test_size = 0.20, stratify = y, random_state = 42)
                #------------------------------
                #--------------Scaling---------
            scaler = StandardScaler()
            train_x = scaler.fit_transform(train_x)
            valid_x = scaler.fit_transform(valid_x)
                #------------------------------
                #--------------Fitting---------
            lr = LogisticRegression()
            lr.fit(train_x, train_y)
                #------------------------------
                #--Predicting and Evaluating---
            pred_train = lr.predict_proba(train_x)
            pred_valid = lr.predict_proba(valid_x)
            q1="ROC score for predict_proba w.r.t train data: "+str(roc_auc_score(train_y, pred_train[:,1]))
            self.Tx.insert(END,q1)
            q1="\nROC score for predict_proba w.r.t validation data: "+str(roc_auc_score(valid_y, pred_valid[:,1]))
            self.Tx.insert(END,q1)
            valid_prediction = lr.predict(valid_x)
            self.Tx.insert(END,q1)
            q1="\nAccuracy score for predict: "+str(accuracy_score(valid_prediction, valid_y))
            self.Tx.insert(END,q1)
            # print(confusion_matrix(valid_prediction, valid_y))

            # ------------------------------
            # ---WITHOUT THE CITY AND BRANCH CODE VARIABLE-----
            without_city_branch = self.data.drop(['city', 'branch_code'], axis=1)
            # print(without_city_branch.columns)
            x_without_city_branch = without_city_branch.drop('churn', axis=1)
            y_without_city_branch = without_city_branch['churn']
            train_x_wc, valid_x_wc, train_y_wc, valid_y_wc = train_test_split(x_without_city_branch,
                                                                              y_without_city_branch, test_size=0.20,
                                                                              stratify=y_without_city_branch,
                                                                              random_state=42)
            train_x_wc = scaler.fit_transform(train_x_wc)
            valid_x_wc = scaler.fit_transform(valid_x_wc)
            lr_wc = LogisticRegression()
            lr_wc.fit(train_x_wc, train_y_wc)
            pred_train_wc = lr_wc.predict_proba(train_x_wc)
            pred_valid_wc = lr_wc.predict_proba(valid_x_wc)
            # probs = pred_valid_wc[:,1]
            valid_prediction_wc = lr_wc.predict(valid_x_wc)
            q1 = "\nROC score for predict_proba w.r.t train data without city: " + str(
                roc_auc_score(train_y_wc, pred_train_wc[:, 1]))
            self.Tx.insert(END, q1)
            q1 = "\nROC score for predict_proba w.r.t validation data without city: " + str(
                roc_auc_score(valid_y_wc, pred_valid_wc[:, 1]))
            self.Tx.insert(END, q1)
            q1 = "\nROC score for predict w.r.t validation data without city: " + str(
                roc_auc_score(valid_prediction_wc, valid_y_wc))

            q1 = "\nAccuracy score for predict without city " + str(accuracy_score(valid_prediction_wc, valid_y_wc))
            self.Tx.insert(END, q1)
            q1 = "\nRecall score for predict without city: " + str(recall_score(valid_prediction_wc, valid_y_wc))
            # print(confusion_matrix(valid_prediction_wc, valid_y_wc))
            self.Tx.insert(END, q1)
            # -------------------------------------------------
            # -----------Decision Tree Classification--------
            # --------Split and Scale---------------
            train_xd, valid_xd, train_yd, valid_yd = train_test_split(x, y, test_size=0.20, stratify=y, random_state=42)

            train_xd = scaler.fit_transform(train_xd)
            valid_xd = scaler.fit_transform(valid_xd)
            # print(train_xd.shape, valid_xd.shape)
            # --------------------------------------
            # ----------Training--------------------
            dt = DecisionTreeClassifier(criterion="gini", max_depth=3, splitter="random")
            dt.fit(train_xd, train_yd)
            # --------------------------------------
            # --------------Predicting--------------
            dt_pred = dt.predict(valid_xd)
            # --------------------------------------
            # --------------Evaluating--------------
            q1 = "\nDecision Trees Accuracy: " + str(accuracy_score(dt_pred, valid_yd))
            self.Tx.insert(END, q1)
            q1 = "\nF1 Score: " + str(f1_score(valid_yd, dt_pred, average='weighted'))
            self.Tx.insert(END, q1)
            # print(dt.score(train_xd,train_yd), dt.score(valid_xd,valid_yd))
            # --------------------------------------
            # ---WITHOUT THE CITY AND BRANCH CODE VARIABLE-----
            x_without_city_branch_d = without_city_branch.drop('churn', axis=1)
            y_without_city_branch_d = without_city_branch['churn']
            train_x_wc_d, valid_x_wc_d, train_y_wc_d, valid_y_wc_d = train_test_split(x_without_city_branch_d,
                                                                                      y_without_city_branch_d,
                                                                                      test_size=0.20,
                                                                                      stratify=y_without_city_branch,
                                                                                      random_state=42)
            train_x_wc_d = scaler.fit_transform(train_x_wc_d)
            valid_x_wc_d = scaler.fit_transform(valid_x_wc_d)
            valid_acc_score = []
            train_acc_score = []
            trainscore = []
            validscore = []
            for md in range(2, 10):
                dt_d = DecisionTreeClassifier(criterion="gini", max_depth=md, splitter="random")
                dt_d.fit(train_x_wc_d, train_y_wc_d)
                dt_pred_wc_d = dt_d.predict(valid_x_wc_d)
                valid_acc_score.append(accuracy_score(dt_pred_wc_d, valid_y_wc_d))
                trainscore.append(dt_d.score(train_x_wc_d, train_y_wc_d))
                validscore.append(dt_d.score(valid_x_wc_d, valid_y_wc_d))

            dt_d1 = DecisionTreeClassifier(criterion="gini", max_depth=2, splitter="random")
            dt_d1.fit(train_x_wc_d, train_y_wc_d)
            dt_pred_wc_d = dt_d1.predict(valid_x_wc_d)
            q1 = "\nAccuracy score for decisiontree-predict without city and branch code: " + str(
                accuracy_score(dt_pred_wc_d, valid_y_wc_d))
            self.Tx.insert(END, q1)
            q1 = "\nF1 Score without city and branch code: " + str(
                f1_score(dt_pred_wc_d, valid_y_wc_d, average='weighted'))
            self.Tx.insert(END, q1)

            # -------------------------------------------------

            # -------------------------------------------------
            # -------------------------------------------

            # =====================================================================================================================

        except:
            self.Tx.insert(END, "Invalid file...\n")

        def ge_list():
            for child in self.list_frame.winfo_children():
                child.destroy()

                # Using treeview widget
            treev = ttk.Treeview(self.list_frame, selectmode='browse', height=21)

            # Calling pack method w.r.to treeview
            treev.pack(side=LEFT)

            # Constructing vertical scrollbar
            # with treeview
            verscrlbar = ttk.Scrollbar(self.list_frame, orient="vertical", command=treev.yview)

            # Calling pack method w.r.to verical
            # scrollbar
            verscrlbar.pack(side='right', fill='x')

            # Configuring treeview
            treev.configure(xscrollcommand=verscrlbar.set)

            # Defining number of columns
            treev["columns"] = ("1", "2", "3", "4", "5", "6", "7", "8", "9", "11")

            # Defining heading
            treev['show'] = 'headings'

            # Assigning the width and anchor to  the
            # respective columns
            treev.column("1", width=100, anchor='c')
            treev.column("2", width=50, anchor='c')
            treev.column("3", width=75, anchor='se')
            treev.column("4", width=100, anchor='se')
            treev.column("5", width=75, anchor='c')
            treev.column("6", width=100, anchor='se')
            treev.column("7", width=100, anchor='se')
            treev.column("8", width=100, anchor='se')
            treev.column("9", width=100, anchor='se')
            treev.column("11", width=70, anchor='c')

            # Assigning the heading names to the
            # respective columns
            treev.heading("1", text="Customer id")
            treev.heading("2", text="Age")
            treev.heading("3", text="Dependent")
            treev.heading("4", text="Ocupation")
            treev.heading("5", text="City")
            treev.heading("6", text="Branch Code")
            treev.heading("7", text="Current Balance")
            treev.heading("8", text="Last Transaction")
            treev.heading("9", text="Churn rate")

            treev.heading("11", text="Churn")

            # Inserting the items and their features to the
            # columns built

            for j in range(len(self.df)):

                bal = (self.df.loc[j, 'current_month_balance'] + self.df.loc[j, 'previous_month_balance'] + self.df.loc[
                    j, 'average_monthly_balance_prevQ'] + self.df.loc[j, 'average_monthly_balance_prevQ2']) / 4
                cred = (int(self.df.loc[j, 'current_month_credit']) + int(self.df.loc[j, 'previous_month_credit'])) / 2
                debi = (int(self.df.loc[j, 'current_month_debit']) + int(self.df.loc[j, 'previous_month_debit'])) / 2
                av = bal - self.df.loc[j, 'current_balance']
                x = av / bal
                i = True
                l = 0
                if x >= 0.5:
                    l = 1
                if l != self.df.loc[j, 'churn']:
                    i = False

                treev.insert("", 'end', text="L1", values=(
                self.df.loc[j, 'customer_id'], self.df.loc[j, 'age'], self.df.loc[j, 'dependents'],
                self.df.loc[j, 'occupation'], self.df.loc[j, 'city'], self.df.loc[j, 'branch_code'],
                self.df.loc[j, 'current_balance'], self.df.loc[j, 'days_since_last_transaction'], x,
                self.df.loc[j, 'churn']))
            self.df['vintage'] = self.df['vintage'].transform(func='sqrt')
            self.df['days_since_last_transaction'] = self.df['days_since_last_transaction'].transform(
                lambda x: x ** 0.5)

            return 0

        self.bkfram = Frame(root)
        self.bkfram.grid(row=0, column=0, sticky=N + S + E + W, columnspan=3, padx=7, pady=1)
        self.bkfram.grid_propagate(False)

        def backk():

            mainfarme_ui()

        self.bk = Button(self.bkfram, text="BACK", command=backk)
        self.bk.pack(padx=5, pady=5, side=LEFT)

        self.button_frame = Frame(root)
        self.button_frame.grid(row=1, column=0, sticky=N + S + E + W, padx=7, pady=1)
        self.button_frame.grid_propagate(False)

        self.list_frame = Frame(root)
        self.list_frame.grid(row=1, column=1, sticky=N + S + E + W, padx=7, pady=1)
        self.list_frame.grid_propagate(False)

        self.acuracy_fram = Frame(root)
        self.acuracy_fram.grid(row=1, column=2, sticky=N + S + E + W, padx=7, pady=1)
        self.acuracy_fram.grid_propagate(False)

        try:
            ge_list()
        except:
            self.Tx.insert(END, "Invalid file...\n")

        l2 = Label(self.acuracy_fram, text="Receiver Operating")
        l2.pack(padx=1, pady=1)
        l2 = Label(self.acuracy_fram, text=" Characteristic Curve")
        l2.pack(padx=1, pady=1)

        def rocf():
            plt.close(1)
            fpr, tpr, threshold = roc_curve(valid_prediction_wc, valid_y_wc)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc, color='royalblue')
            plt.legend(loc='lower right')
            plt.plot([0, 1], [0, 1], 'r--', color='#8B0000')
            plt.xlim([0, 1])
            plt.ylim([0, 1])
            plt.rcParams['figure.figsize'] = (6, 5)
            plt.ylabel('True Positive Rate')
            plt.xlabel('False Positive Rate')
            plt.show()

        self.roccurv = Button(self.acuracy_fram, text="ROC CURV", width=20, command=rocf)
        self.roccurv.pack(padx=5, pady=1)

        l2 = Label(self.acuracy_fram, text="Decision Tree Classification")
        l2.pack(padx=5, pady=1)

        def acudp():
            plt.close(1)
            plt.plot(valid_acc_score, color='orangered')
            plt.ylabel('Accuracy score')
            plt.xlabel('Values for maximum depth')

            plt.show()

        self.acudpth = Button(self.acuracy_fram, text="Accuracy - Depth Tree", width=20, command=acudp)
        self.acudpth.pack(padx=5, pady=1)

        def perdp():
            plt.close(1)
            frame = pd.DataFrame({'max_depth': range(2, 10), 'train_acc': trainscore, 'valid_acc': validscore})
            plt.figure(figsize=(12, 6))
            plt.plot(frame['max_depth'], frame['train_acc'], marker='o', label='train_acc')
            plt.plot(frame['max_depth'], frame['valid_acc'], marker='o', label='valid_acc')
            plt.xlabel('Depth of tree')
            plt.ylabel('performance')
            plt.legend()
            plt.show()

        self.perdpth = Button(self.acuracy_fram, text="Performance - Depth Tree", width=20, command=perdp)
        self.perdpth.pack(padx=5, pady=1)

        l1 = Label(self.button_frame, text="Frequency Report", width=20)
        l1.pack(padx=1, pady=1)

        def chur():
            plt.close(1)
            self.df['churn'].value_counts().plot(kind='bar', color='r')
            plt.xlabel('Churn')
            plt.ylabel('Frequency')
            plt.show()

        self.genrate_bar = Button(self.button_frame, text="CHURN", width=20, command=chur)
        self.genrate_bar.pack(padx=1, pady=1)

        def frag():
            plt.close(1)
            self.df['age'].plot.hist(bins=20, color='c')
            plt.xlabel('age', fontsize=12)
            plt.xlabel('Age')
            plt.show()

        self.genrate_bar = Button(self.button_frame, text="AGE", width=20, command=frag)
        self.genrate_bar.pack(padx=1, pady=1)

        def vint():
            plt.close(1)
            plt.figure(figsize=(6, 3))
            self.df['vintage'].plot.hist(bins=30, color='orangered')
            plt.show()

        self.genrate_bar = Button(self.button_frame, text="VINTAGE", width=20, command=vint)
        self.genrate_bar.pack(padx=1, pady=1)

        def lasttr():
            plt.close(1)
            plt.figure(figsize=(6, 3))
            self.df['days_since_last_transaction'].plot.hist(bins=30, color='orangered')
            plt.show()

        self.genrate_bar = Button(self.button_frame, text="LAST TRANSACTION", width=20, command=lasttr)
        self.genrate_bar.pack(padx=1, pady=1)

        def occchu():
            plt.close(1)
            pd.crosstab(self.df['churn'], self.df['occupation']).plot.bar()
            plt.ylabel('Frequency')
            plt.xlabel('Churn')
            plt.show()

        self.genrate_bar = Button(self.button_frame, text="OCCUPATION - CHURN", width=20, command=occchu)
        self.genrate_bar.pack(padx=1, pady=1)

        def vinchu():
            plt.close(1)
            self.df.groupby('churn')['vintage'].mean().plot.bar(color='#b81c8b')
            plt.ylabel('Frequency')
            plt.xlabel('Churn')
            plt.show()

        self.genrate_bar = Button(self.button_frame, text="VINTAGE - CHURN", width=20, command=vinchu)
        self.genrate_bar.pack(padx=1, pady=1)

        def vino():
            plt.close(1)
            self.df.groupby('occupation')['vintage'].mean().plot.bar(color='hotpink')
            plt.xlabel('Occupation')
            plt.ylabel('Vintage')
            plt.show()

        self.genrate_bar = Button(self.button_frame, text="VINTAGE - OCCUPATION", width=20, command=vino)
        self.genrate_bar.pack(padx=1, pady=1)

        l1 = Label(self.button_frame, text="Bivariate Analysis Report", width=20)
        l1.pack(padx=1, pady=5)
        f = Frame(self.button_frame, width=20)
        f.pack(padx=1, pady=5)

        self.n = StringVar()
        self.c = ttk.Combobox(f, width=17, textvariable=self.n)
        self.c['values'] = ("company", "retired", "salaried", "self_employed", "student")
        self.c.grid(row=0, column=0)
        self.c.current(1)
        self.n1 = StringVar()
        age = []
        for i in range(1, 150):
            age.append(i)
        self.c1 = ttk.Combobox(f, width=3, textvariable=self.n1)
        self.c1['values'] = age
        self.c1.grid(row=0, column=1)
        self.c1.current(20)

        def foa():
            plt.close(1)
            a = str(self.c.get())
            b = int(self.c1.get())
            temp_data = self.df.loc[(self.df['occupation'] == a) & (self.df['age'] > 18) & (self.df['age'] < b)]
            temp_data['churn'].plot.hist(bins=50, color='r')
            t = "For " + a + " between 18-" + str(b) + "Age Group"
            plt.title(t)
            plt.xlabel('Churn Value')
            plt.show()

        self.genrate_bar = Button(self.button_frame, text="OCCUPATION - AGE", width=20, command=foa)
        self.genrate_bar.pack()

        def pmq():
            plt.close(1)
            fig, ax = plt.subplots()
            colors = {0: 'green', 1: 'red'}
            ax.scatter(self.df['vintage'], self.df['average_monthly_balance_prevQ'],
                       c=self.df['churn'].apply(lambda x: colors[x]))
            plt.title('plot between vintage, average_monthly_balance_prevQ and churn value')
            plt.xlabel('Vintage')
            plt.ylabel('Average Monthly Balance Previous Quarter')
            plt.legend()
            plt.show()

        self.genrate_bar = Button(self.button_frame, text="PRV MTH BLNC - CHURN", width=20, command=pmq)
        self.genrate_bar.pack(padx=1, pady=3)

        def ambpq():
            plt.close(1)
            fig, ax = plt.subplots()
            colors = {0: 'green', 1: 'red'}
            ax.scatter(self.df['vintage'], self.df['average_monthly_balance_prevQ'],
                       c=self.df['churn'].apply(lambda x: colors[x]))
            plt.title('plot between vintage, average_monthly_balance_prevQ and churn value')
            plt.xlabel('Vintage')
            plt.ylabel('Average Monthly Balance Previous Quarter')
            plt.show()

        self.genrate_bar = Button(self.button_frame, text="AV_MTH_PV_QT-VINTAGE", width=20, command=ambpq)
        self.genrate_bar.pack(padx=1, pady=1)

        def wort():
            plt.close(1)
            fig, ax = plt.subplots()
            colors = {'self_employed': 'red', 'salaried': 'blue', 'student': 'green', 'retired': 'yellow',
                      'company': 'black'}
            ax.scatter(self.df['customer_nw_category'], self.df['churn'],
                       c=self.df['occupation'].apply(lambda x: colors[x]))
            plt.title('plot between customer_nw_category, occupation and churn value')
            plt.xlabel('Customer Net Worth Category')
            plt.ylabel('Churn value')
            # plt.legend()
            plt.show()

        self.genrate_bar = Button(self.button_frame, text="OCCUPATION - WORTH", width=20, command=wort)
        self.genrate_bar.pack(padx=1, pady=1)

        def ageoccu():
            plt.close(1)
            fig, ax = plt.subplots()
            colors = {'self_employed': 'red', 'salaried': 'blue', 'student': 'green', 'retired': 'yellow',
                      'company': 'black'}
            ax.scatter(self.df['age'], self.df['churn'], c=self.df['occupation'].apply(lambda x: colors[x]))
            plt.title('Plot between age, occupation and churn value')
            plt.xlabel('Age')
            plt.ylabel('Churn Value')
            # plt.legend(['self_employed','salaried','student','retired','company'])
            plt.show()

        self.genrate_bar = Button(self.button_frame, text="OCCUPATION - AGE", width=20, command=ageoccu)
        self.genrate_bar.pack(padx=1, pady=1)

        def histo():
            plt.close(1)
            plt.rcParams['figure.figsize'] = (35, 20)
            sns.heatmap(self.df[['vintage', 'age', 'dependents', 'customer_nw_category', 'days_since_last_transaction',
                                 'current_balance', 'previous_month_end_balance', 'average_monthly_balance_prevQ',
                                 'average_monthly_balance_prevQ2', 'current_month_credit', 'previous_month_credit',
                                 'current_month_debit', 'previous_month_debit', 'current_month_balance',
                                 'previous_month_balance', 'churn', ]].corr(), annot=True)
            plt.title('Histogram of the Dataset', fontsize=25)
            plt.show()

        self.genrate_bar = Button(self.button_frame, text="DATASET HISTOGRAM", width=20, command=histo)
        self.genrate_bar.pack(padx=1, pady=1)


mainfarme_ui()
root.mainloop()