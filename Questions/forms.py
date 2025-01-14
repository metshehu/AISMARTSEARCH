from django import forms


class MakeDirForm(forms.Form):
    name = forms.CharField(max_length=255, required=True,
                           label="Directory Name")
    # Optional photo field
    photo = forms.ImageField(required=False, label="Upload Photo")


class FileUploadForm(forms.Form):
    file = forms.FileField(required=True, label="Upload File")
