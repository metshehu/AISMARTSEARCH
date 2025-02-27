# Generated by Django 5.1.3 on 2025-01-22 19:02

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('Questions', '0005_alter_uservalues_splitter'),
    ]

    operations = [
        migrations.AlterField(
            model_name='uservalues',
            name='splitter',
            field=models.CharField(choices=[('CharacterTextSplitter', 'CharacterTextSplitter'), ('RecursiveCharacterTextSplitter', 'RecursiveCharacterTextSplitter'), ('TokenTextSplitter', 'TokenTextSplitter'), ('MarkdownHeaderTextSplitter', 'MarkdownHeaderTextSplitter')], default='CharacterTextSplitter', help_text='Select the splitter', max_length=50),
        ),
    ]
