# Generated by Django 5.1.3 on 2024-11-21 14:40

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='TestUPFILE',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=600)),
                ('file', models.FileField(upload_to='uploadsT/')),
            ],
        ),
    ]
