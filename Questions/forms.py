from django import forms

from .models import UserValues


class MakeDirForm(forms.Form):
    name = forms.CharField(max_length=255, required=True,
                           label="Directory Name")
    photo = forms.ImageField(required=True, label="Upload Photo")


class FileUploadForm(forms.Form):
    file = forms.FileField(required=True, label="Upload File")


class UserValueForm(forms.ModelForm):
    class Meta:
        model = UserValues
        fields = ['splitter', 'chunksize', 'overlap', 'temp']

    temp = forms.FloatField(
        min_value=0,
        max_value=1,
        widget=forms.NumberInput(attrs={
            'type': 'range',  # Set the input type to range
            'min': '0',      # Start value for the slider
            'max': '1',      # End value for the slider
            'step': '0.01',  # Increment step for the slider
            'value': '0'     # Default start value
        }),
        label="Temperature"
    )
