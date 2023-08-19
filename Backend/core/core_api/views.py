import pickle,pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status


class GetdiabetesPrediction(APIView):
    def post(self,request):
        try:
            print(request.POST)
            file = open(r'F:\All Projects\Python-AI-main Compalete\Python-AI-main\Python-AI-main\core\core_api\Diabetes_model.pkl', 'rb')
            get_trained_model = pickle.load(file)
            user_data = {
                'Pregnancies':int(request.POST.get("pregnancies",0)),
                'Glucose':int(request.POST.get("glucose",0)),
                'BloodPressure':int(request.POST.get("bloodPressure",0)),
                'SkinThickness':int(request.POST.get("skinThickness",0)),
                'Insulin':int(request.POST.get("insulin",0)),
                'BMI':int(request.POST.get("BMI",0)),
                'DiabetesPedigreeFunction':int(request.POST.get("diabetesPedigreeFunction",0)),
                'Age':int(request.POST.get("Age",0)),
            }
            df = pd.DataFrame(user_data,index=[0])
            get_result = get_trained_model.predict(df)[0]
            get_dict = {
                "0":"You Are Not Diabetic.",
                "1":"You Are Diabetic."
            }
            
            context = {
                "status":status.HTTP_200_OK,
                "success":True,
                "response":{
                    "result":get_dict.get(str(get_result),None),
                    # "accuracy":str(accuracy_score([get_result], get_trained_model.predict(df))*100)+'%'
                }
            }
            return Response(context,status=status.HTTP_200_OK)
        except Exception as exception:
            context = {
                "status":status.HTTP_400_BAD_REQUEST,
                "success":False,
                "response":str(exception)
            }
            return Response(context,status=status.HTTP_400_BAD_REQUEST)






# # df = pd.read_csv(![](diabetes.csv))
# df = pd.read_csv("diabetes.csv")
# # HEADINGS
# st.title('Diabetes Prediction Using Machine Learning')
# st.sidebar.header('Patient Data')
# st.subheader('Training Data')
# st.write(df.describe())


# # X AND Y DATA
# x = df.drop(['Outcome'], axis = 1)
# y = df.iloc[:, -1]
# x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 0)


# # FUNCTION
# def user_report():
#   Pregnancies = st.sidebar.slider('Pregnancies', 0,17, 3 )
#   Glucose = st.sidebar.slider('Glucose', 0,200, 120 )
#   BloodPressure = st.sidebar.slider('Blood Pressure', 0,122, 70 )
#   SkinThickness = st.sidebar.slider('Skin Thickness', 0,100, 20 )
#   Insulin = st.sidebar.slider('Insulin', 0,846, 79 )
#   BMI = st.sidebar.slider('BMI', 0,67, 20 )
#   DiabetesPedigreeFunction = st.sidebar.slider('Diabetes Pedigree Function', 0.0,2.4, 0.47 )
#   Age = st.sidebar.slider('Age', 21,88, 33 )

#   user_report_data = {
#       'Pregnancies':Pregnancies,
#       'Glucose':Glucose,
#       'BloodPressure':BloodPressure,
#       'SkinThickness':SkinThickness,
#       'Insulin':Insulin,
#       'BMI':BMI,
#       'DiabetesPedigreeFunction':DiabetesPedigreeFunction,
#       'Age':Age
#   }
#   report_data = pd.DataFrame(user_report_data, index=[0])
#   return report_data




# # PATIENT DATA
# user_data = user_report()
# st.subheader('Patient Data')
# st.write(user_data)




# # MODEL
# rf  = RandomForestClassifier()
# rf.fit(x_train, y_train)
# user_result = rf.predict(user_data)
# # VISUALISATIONS
# st.title('Visualised Patient Report')
# filename = 'Diabetes_model.pkl'
# pickle.dump(rf, open(filename, 'wb'))


# # COLOR FUNCTION
# if user_result[0]==0:
#   color = 'blue'
# else:
#   color = 'red'


# # Age vs Pregnancies
# st.header('Pregnancy count Graph (Others vs Yours)')
# fig_preg = plt.figure()
# ax1 = sns.scatterplot(x = 'Age', y = 'Pregnancies', data = df, hue = 'Outcome', palette = 'Greens')
# ax2 = sns.scatterplot(x = user_data['Age'], y = user_data['Pregnancies'], s = 150, color = color)
# plt.xticks(np.arange(10,100,5))
# plt.yticks(np.arange(0,20,2))
# plt.title('0 - Healthy & 1 - Unhealthy')
# st.pyplot(fig_preg)



# # Age vs Glucose
# st.header('Glucose Value Graph (Others vs Yours)')
# fig_glucose = plt.figure()
# ax3 = sns.scatterplot(x = 'Age', y = 'Glucose', data = df, hue = 'Outcome' , palette='magma')
# ax4 = sns.scatterplot(x = user_data['Age'], y = user_data['Glucose'], s = 150, color = color)
# plt.xticks(np.arange(10,100,5))
# plt.yticks(np.arange(0,220,10))
# plt.title('0 - Healthy & 1 - Unhealthy')
# st.pyplot(fig_glucose)



# # Age vs Bp
# st.header('Blood Pressure Value Graph (Others vs Yours)')
# fig_bp = plt.figure()
# ax5 = sns.scatterplot(x = 'Age', y = 'BloodPressure', data = df, hue = 'Outcome', palette='Reds')
# ax6 = sns.scatterplot(x = user_data['Age'], y = user_data['BloodPressure'], s = 150, color = color)
# plt.xticks(np.arange(10,100,5))
# plt.yticks(np.arange(0,130,10))
# plt.title('0 - Healthy & 1 - Unhealthy')
# st.pyplot(fig_bp)


# # Age vs St
# st.header('Skin Thickness Value Graph (Others vs Yours)')
# fig_st = plt.figure()
# ax7 = sns.scatterplot(x = 'Age', y = 'SkinThickness', data = df, hue = 'Outcome', palette='Blues')
# ax8 = sns.scatterplot(x = user_data['Age'], y = user_data['SkinThickness'], s = 150, color = color)
# plt.xticks(np.arange(10,100,5))
# plt.yticks(np.arange(0,110,10))
# plt.title('0 - Healthy & 1 - Unhealthy')
# st.pyplot(fig_st)


# # Age vs Insulin
# st.header('Insulin Value Graph (Others vs Yours)')
# fig_i = plt.figure()
# ax9 = sns.scatterplot(x = 'Age', y = 'Insulin', data = df, hue = 'Outcome', palette='rocket')
# ax10 = sns.scatterplot(x = user_data['Age'], y = user_data['Insulin'], s = 150, color = color)
# plt.xticks(np.arange(10,100,5))
# plt.yticks(np.arange(0,900,50))
# plt.title('0 - Healthy & 1 - Unhealthy')
# st.pyplot(fig_i)


# # Age vs BMI
# st.header('BMI Value Graph (Others vs Yours)')
# fig_bmi = plt.figure()
# ax11 = sns.scatterplot(x = 'Age', y = 'BMI', data = df, hue = 'Outcome', palette='rainbow')
# ax12 = sns.scatterplot(x = user_data['Age'], y = user_data['BMI'], s = 150, color = color)
# plt.xticks(np.arange(10,100,5))
# plt.yticks(np.arange(0,70,5))
# plt.title('0 - Healthy & 1 - Unhealthy')
# st.pyplot(fig_bmi)


# # Age vs Dpf
# st.header('DPF Value Graph (Others vs Yours)')
# fig_dpf = plt.figure()
# ax13 = sns.scatterplot(x = 'Age', y = 'DiabetesPedigreeFunction', data = df, hue = 'Outcome', palette='YlOrBr')
# ax14 = sns.scatterplot(x = user_data['Age'], y = user_data['DiabetesPedigreeFunction'], s = 150, color = color)
# plt.xticks(np.arange(10,100,5))
# plt.yticks(np.arange(0,3,0.2))
# plt.title('0 - Healthy & 1 - Unhealthy')
# st.pyplot(fig_dpf)



# # OUTPUT
# st.subheader('Your Report: ')
# output=''
# if user_result[0]==0:
#   output = 'You are not Diabetic'
# else:
#   output = 'You are Diabetic'
# st.title(output)
# st.subheader('Accuracy: ')
# st.write(str(accuracy_score(y_test, rf.predict(x_test))*100)+'%')
