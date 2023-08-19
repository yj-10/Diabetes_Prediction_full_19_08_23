from django.urls import path
from .views import (
    GetdiabetesPrediction
)
urlpatterns = [
    path("get_diabetes_data",GetdiabetesPrediction.as_view(),name="get-diabetes-data")
]