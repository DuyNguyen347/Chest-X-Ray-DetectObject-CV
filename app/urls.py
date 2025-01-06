from django.urls import path
from . import views
from django.conf import settings
from django.conf.urls.static import static

app_name = 'app'

urlpatterns = [
    path('', views.home, name='home'),
    path('services/', views.services, name='services'),
    path('about/', views.about, name='about'),
    path('contact/', views.contact, name='contact'),
    path('download-dicom/<str:filename>/', views.download_dicom, name='download_dicom'),
]

urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)