from django.http import HttpResponse
from django.shortcuts import render
import joblib
from plagiarismchecker import feature_engineering



def home(request):
    return render(request,"home.html")

def result(request):
    # ans=feature_engineering.calculate_containment(feature_engineering.complete_df, 1, 'g0pB_taskd.txt')
    
    file_name=request.GET['x']

    ans = round(feature_engineering.calculate_containment(feature_engineering.complete_df,1,file_name)*100,2)

    return render(request,"result.html",{'ans':ans})
