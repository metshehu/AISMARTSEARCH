from django import forms

from .models import TestUPFILE


class FileUploadForm(forms.ModelForm):
    class Meta:
        model = TestUPFILE
        fields = [ 'file']
