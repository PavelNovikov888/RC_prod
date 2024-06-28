from flask_wtf import FlaskForm
from wtforms import StringField, IntegerField, SubmitField,HiddenField
from wtforms.validators import DataRequired

class EditForm(FlaskForm):
    cost = IntegerField('id')
    button_reccom = SubmitField('Получить рекомендации')
    button_item = SubmitField('Item')
    button_user = SubmitField('User')
    user = IntegerField('Введите userid')
    item = IntegerField('Введите itemid')
    
