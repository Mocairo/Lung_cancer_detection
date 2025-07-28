from flask import render_template, flash, redirect, url_for, session, current_app, request, make_response
from flask_login import login_required, current_user
from .routes import bp
from .forms import ScreeningForm, UploadForm
from .models import ScreeningSession
from . import db
from .ai_models import analyze_ct_scan
from .visualizer import generate_visualizations
from .comparison import perform_comparison
from .ai_summary import generate_ai_summary # Import the new function
import os
from werkzeug.utils import secure_filename
from pathlib import Path
import uuid

@bp.route('/')
@bp.route('/index')
@login_required
def index():
    return render_template('index.html', title='主页')

@bp.route('/start_screening')
@login_required
def start_screening():
    last_screening = ScreeningSession.query.filter_by(user_id=current_user.id).order_by(ScreeningSession.timestamp.desc()).first()

    if last_screening:
        session['questionnaire_data'] = {
            'age': last_screening.age,
            'gender': last_screening.gender,
            'smoking_status': last_screening.smoking_status,
            'packs_per_day': last_screening.packs_per_day,
            'years_smoked': last_screening.years_smoked,
            'pack_years': last_screening.pack_years,
            'family_history': last_screening.family_history,
            'occupational_exposure': last_screening.occupational_exposure,
            'previous_cancer_history': last_screening.previous_cancer_history,
        }
        flash('已加载您之前保存的信息。请上传新的CT扫描。', 'info')
        return redirect(url_for('main.upload_scan'))
    else:
        return redirect(url_for('main.screen'))

@bp.route('/screen', methods=['GET', 'POST'])
@login_required
def screen():
    form = ScreeningForm()
    
    if request.method == 'GET' and request.args.get('edit') == 'true':
        last_screening = ScreeningSession.query.filter_by(user_id=current_user.id).order_by(ScreeningSession.timestamp.desc()).first()
        if last_screening:
            form.age.data = last_screening.age
            form.gender.data = last_screening.gender
            form.smoking_status.data = last_screening.smoking_status
            form.smoking_details.packs_per_day.data = last_screening.packs_per_day
            form.smoking_details.years_smoked.data = last_screening.years_smoked
            form.family_history.data = last_screening.family_history
            form.occupational_exposure.data = last_screening.occupational_exposure
            form.previous_cancer_history.data = last_screening.previous_cancer_history
            flash('请检查并更新您的信息。', 'info')
        else:
            flash('未找到过往信息，请填写此问卷。', 'warning')
            
    if form.validate_on_submit():
        packs_per_day = 0
        years_smoked = 0
        pack_years = 0
        if form.smoking_status.data in ['former', 'current']:
            packs_per_day = form.smoking_details.packs_per_day.data or 0
            years_smoked = form.smoking_details.years_smoked.data or 0
            pack_years = packs_per_day * years_smoked
        
        session['questionnaire_data'] = {
            'age': form.age.data,
            'gender': form.gender.data,
            'smoking_status': form.smoking_status.data,
            'packs_per_day': packs_per_day,
            'years_smoked': years_smoked,
            'pack_years': pack_years,
            'family_history': form.family_history.data,
            'occupational_exposure': form.occupational_exposure.data,
            'previous_cancer_history': form.previous_cancer_history.data,
        }
        flash('问卷已成功提交！现在，请上传您的CT扫描文件。')
        return redirect(url_for('main.upload_scan'))
    return render_template('screen.html', title='临床问卷', form=form)

@bp.route('/upload_scan', methods=['GET', 'POST'])
@login_required
def upload_scan():
    if 'questionnaire_data' not in session:
        flash('请先填写问卷。')
        return redirect(url_for('main.screen'))

    form = UploadForm()
    if form.validate_on_submit():
        mhd_file = form.mhd_file.data
        raw_file = form.raw_file.data

        mhd_filename = secure_filename(mhd_file.filename)
        raw_filename = secure_filename(raw_file.filename)

        # 确保文件名是配对的 (例如 xxx.mhd 和 xxx.raw)
        if Path(mhd_filename).stem != Path(raw_filename).stem:
            flash('MHD 和 RAW 文件名必须匹配 (例如 "my_scan.mhd" 和 "my_scan.raw")。')
            return redirect(url_for('main.upload_scan'))

        upload_folder = os.path.join(current_app.root_path, '..', 'uploads')
        mhd_path = os.path.join(upload_folder, mhd_filename)
        raw_path = os.path.join(upload_folder, raw_filename)
        
        mhd_file.save(mhd_path)
        raw_file.save(raw_path)

        questionnaire_data = session.pop('questionnaire_data', None)

        new_screening = ScreeningSession(
            user_id=current_user.id,
            age=questionnaire_data['age'],
            gender=questionnaire_data['gender'],
            smoking_status=questionnaire_data['smoking_status'],
            packs_per_day=questionnaire_data['packs_per_day'],
            years_smoked=questionnaire_data['years_smoked'],
            pack_years=questionnaire_data['pack_years'],
            family_history=questionnaire_data['family_history'],
            occupational_exposure=questionnaire_data['occupational_exposure'],
            previous_cancer_history=questionnaire_data['previous_cancer_history'],
            original_filename=mhd_filename,
            saved_filename=mhd_filename, # 我们只需要保存mhd的路径
            status='Processing'
        )
        db.session.add(new_screening)
        db.session.commit()

        # --- AI Analysis Step ---
        analysis_result = analyze_ct_scan(mhd_path)
        
        # 将分析结果分为可存入DB的部分和仅用于可视化的部分
        db_analysis_result = {
            'status': analysis_result.get('status'),
            'nodules_found': analysis_result.get('nodules_found', 0),
            'nodules': analysis_result.get('nodules', []),
            'message': analysis_result.get('message')
        }

        # 更新数据库记录
        new_screening.analysis_result = db_analysis_result
        if analysis_result.get('status') == 'success':
            new_screening.status = 'Complete'

            # --- AI Summary Generation Step ---
            with current_app.app_context():
                ai_summary_text = generate_ai_summary(db_analysis_result)
            new_screening.ai_summary = ai_summary_text
            
            # --- 图像可视化步骤 ---
            base_filename = f"report_{new_screening.id}_{uuid.uuid4().hex[:8]}"
            
            # 直接将分析产出的数组传递给可视化函数，避免重复读取和处理
            viz_results = generate_visualizations(
                ct_array=analysis_result['resampled_ct_array'],
                mask_array=analysis_result['segmentation_mask'],
                raw_mask_array=analysis_result['raw_segmentation_mask'],
                itk_image=analysis_result['resampled_itk_image'],
                base_filename=base_filename
            )
            
            if viz_results.get('status') == 'success':
                filenames = viz_results['filenames']
                new_screening.image_2d_slice_top = filenames['2d_slice_top']
                new_screening.image_2d_slice_center = filenames['2d_slice_center']
                new_screening.image_2d_slice_bottom = filenames['2d_slice_bottom']
                new_screening.image_3d_default = filenames['3d_default']
                new_screening.image_3d_mip = filenames['3d_mip']
                new_screening.image_3d_iso = filenames['3d_iso']
            else:
                flash(f"无法生成可视化图像: {viz_results.get('message', '未知错误')}")

        else:
            new_screening.status = 'Failed'
        
        db.session.commit()

        return redirect(url_for('main.report', screening_id=new_screening.id))

    return render_template('upload_scan.html', title='上传CT扫描', form=form)

@bp.route('/report/<int:screening_id>')
@login_required
def report(screening_id):
    screening = ScreeningSession.query.get_or_404(screening_id)
    # Ensure the report belongs to the current user
    if screening.user_id != current_user.id:
        flash('您无权查看此报告。')
        return redirect(url_for('main.health_profile'))
    
    historical_screenings = ScreeningSession.query.filter(
        ScreeningSession.user_id == current_user.id,
        ScreeningSession.id != screening_id
    ).order_by(ScreeningSession.timestamp.desc()).all()
    
    return render_template('report.html', title=f'报告 - {screening.timestamp.strftime("%Y-%m-%d")}', screening=screening, historical_screenings=historical_screenings)

@bp.route('/report/<int:screening_id>/download')
@login_required
def download_report(screening_id):
    screening = ScreeningSession.query.get_or_404(screening_id)
    if screening.user_id != current_user.id:
        flash('您无权下载此报告。', 'danger')
        return redirect(url_for('main.health_profile'))

    if not screening.ai_summary:
        flash('该报告没有可供下载的AI智能解读文本。', 'info')
        return redirect(url_for('main.report', screening_id=screening_id))

    # Create a text response for downloading
    response = make_response(screening.ai_summary, 200)
    response.mimetype = "text/plain"
    response.headers.set(
        "Content-Disposition", "attachment", filename=f"AI_Report_Summary_{screening.id}.txt"
    )
    return response

@bp.route('/compare_reports')
@login_required
def compare_reports():
    report_ids_str = request.args.getlist('report_ids')

    if len(report_ids_str) != 2:
        flash('请选择正好两份报告进行比较。', 'warning')
        return redirect(url_for('main.health_profile'))

    try:
        report_ids = [int(id_str) for id_str in report_ids_str]
    except ValueError:
        flash('无效的报告选择。', 'danger')
        return redirect(url_for('main.health_profile'))

    report1 = ScreeningSession.query.get(report_ids[0])
    report2 = ScreeningSession.query.get(report_ids[1])

    # Ensure reports exist and belong to the current user
    if not all([report1, report2]) or report1.user_id != current_user.id or report2.user_id != current_user.id:
        flash('一份或多份选定的报告无法找到，或者您没有权限查看。', 'danger')
        return redirect(url_for('main.health_profile'))

    # Ensure reports are sorted by date
    if report1.timestamp > report2.timestamp:
        report1, report2 = report2, report1 # report1 is now the older one

    # 调用核心对比函数
    comparison_results = perform_comparison(report1, report2)

    return render_template('comparison_report.html', 
                           title='对比报告', 
                           report1=report1, 
                           report2=report2,
                           comparison=comparison_results)


@bp.route('/training_board')
@login_required
def training_board():
    return render_template('training_board.html', title='训练看板')


@bp.route('/health_profile')
@login_required
def health_profile():
    screenings = ScreeningSession.query.filter_by(user_id=current_user.id).order_by(ScreeningSession.timestamp.desc()).all()
    return render_template('health_profile.html', title='健康档案', screenings=screenings) 