from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileAllowed, FileRequired
from wtforms import StringField, PasswordField, BooleanField, SubmitField, IntegerField, SelectField, RadioField, FormField
from wtforms.validators import DataRequired, EqualTo, ValidationError, NumberRange, Optional
from .models import User

class LoginForm(FlaskForm):
    username = StringField('用户名', validators=[DataRequired()])
    password = PasswordField('密码', validators=[DataRequired()])
    remember_me = BooleanField('记住我')
    submit = SubmitField('登录')

class RegistrationForm(FlaskForm):
    username = StringField('用户名', validators=[DataRequired()])
    password = PasswordField('密码', validators=[DataRequired()])
    password2 = PasswordField(
        '重复密码', validators=[DataRequired(), EqualTo('password', message='两次输入的密码必须匹配。')])
    submit = SubmitField('注册')

    def validate_username(self, username):
        user = User.query.filter_by(username=username.data).first()
        if user is not None:
            raise ValidationError('该用户名已被使用，请输入不同的用户名。')

class SmokingHistoryForm(FlaskForm):
    class Meta:
        # 子表单不需要CSRF令牌
        csrf = False
    packs_per_day = IntegerField('每天吸烟包数', validators=[Optional(), NumberRange(min=0)])
    years_smoked = IntegerField('吸烟年限', validators=[Optional(), NumberRange(min=0)])

class ScreeningForm(FlaskForm):
    age = IntegerField('年龄', validators=[DataRequired(), NumberRange(min=0, max=150)])
    gender = SelectField('性别', choices=[('Male', '男'), ('Female', '女'), ('Other', '其他')], validators=[DataRequired()])
    smoking_status = RadioField('吸烟史', choices=[('never', '从未吸烟'), ('former', '曾经吸烟'), ('current', '目前吸烟')], validators=[DataRequired()])
    smoking_details = FormField(SmokingHistoryForm)
    family_history = RadioField('肺癌家族史', choices=[('yes', '有'), ('no', '无')], validators=[DataRequired()])
    occupational_exposure = RadioField('职业暴露史 (石棉/氡等)', choices=[('yes', '有'), ('no', '无')], validators=[DataRequired()])
    previous_cancer_history = RadioField('个人既往癌症史', choices=[('yes', '有'), ('no', '无')], validators=[DataRequired()])
    ct_scan = BooleanField('我有一个CT扫描文件需要上传。')
    submit = SubmitField('下一步：上传CT扫描')

class UploadForm(FlaskForm):
    mhd_file = FileField('MHD 元数据文件 (*.mhd)', validators=[
        FileRequired(message='请选择一个文件。'),
        FileAllowed(['mhd'], '请在此处上传 .mhd 文件。')
    ])
    raw_file = FileField('RAW 数据文件 (*.raw)', validators=[
        FileRequired(message='请选择一个文件。'),
        FileAllowed(['raw'], '请在此处上传 .raw 文件。')
    ])
    submit = SubmitField('上传并分析') 