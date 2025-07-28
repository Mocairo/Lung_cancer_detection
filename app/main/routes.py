from flask import Blueprint, render_template, redirect, url_for, flash, request
from flask_login import login_required, current_user
from .. import db
from ..models import User, Screening, ClinicalData
from .forms import ClinicalDataForm, FileUploadForm
from ..ai_models import process_image_and_run_models
from ..visualizer import generate_visualizations
from werkzeug.utils import secure_filename
import os
from flask import current_app as app
from flask_login import logout_user

main = Blueprint('main', __name__)

@main.route('/')
def index():
    return render_template('index.html')

@main.route('/about')
def about():
    return render_template('about.html')

@main.route('/contact')
def contact():
    return render_template('contact.html')

@main.route('/screening/new', methods=['GET', 'POST'])
@login_required
def new_screening():
    form = FileUploadForm()
    if form.validate_on_submit():
        mhd_file = form.mhd_file.data
        raw_file = form.raw_file.data

        if not mhd_file or not raw_file:
            flash('请选择一个MHD文件和RAW文件', 'warning')
            return render_template('screening_form.html', form=form)

        # 保存文件
        mhd_filename = secure_filename(mhd_file.filename)
        raw_filename = secure_filename(raw_file.filename)
        mhd_path = os.path.join(app.config['UPLOAD_FOLDER'], mhd_filename)
        raw_path = os.path.join(app.config['UPLOAD_FOLDER'], raw_filename)

        try:
            mhd_file.save(mhd_path)
            raw_file.save(raw_path)
        except Exception as e:
            flash(f"保存文件失败: {e}", 'danger')
            return render_template('screening_form.html', form=form)

        # --- AI模型处理 ---
        try:
            analysis_results = process_image_and_run_models(mhd_path, raw_path)
            if analysis_results['status'] != 'success':
                flash(f"AI模型处理失败: {analysis_results['message']}", 'danger')
                return render_template('screening_form.html', form=form)
        except Exception as e:
            flash(f"调用AI模型时发生严重错误: {e}", 'danger')
            return render_template('screening_form.html', form=form)

        # --- 结果可视化 ---
        base_filename = mhd_file.filename.rsplit('.', 2)[0]
        
        vis_results = generate_visualizations(
            analysis_results['resampled_ct_array'],
            analysis_results['segmentation_mask'],
            analysis_results['raw_segmentation_mask'],
            analysis_results['resampled_itk_image'],
            base_filename
        )
        if vis_results['status'] != 'success':
            flash(f"生成可视化图像失败: {vis_results['message']}", 'warning')

        # --- 保存到数据库 ---
        screening = Screening(
            analysis_result=analysis_results,
            original_filename=mhd_file.filename,
            saved_filename=base_filename,
            status=analysis_results.get('status', 'Failed'),
            image_2d_slice_top=vis_results.get('filenames', {}).get('2d_slice_top'),
            image_2d_slice_center=vis_results.get('filenames', {}).get('2d_slice_center'),
            image_2d_slice_bottom=vis_results.get('filenames', {}).get('2d_slice_bottom'),
            image_3d_default=vis_results.get('filenames', {}).get('3d_default'),
            image_3d_mip=vis_results.get('filenames', {}).get('3d_mip'),
            image_3d_iso=vis_results.get('filenames', {}).get('3d_iso'),
            user_id=current_user.id
        )
        db.session.add(screening)
        db.session.commit()

        flash('筛查完成！正在生成报告...', 'success')
        return redirect(url_for('main.report', screening_id=screening.id))
        
    return render_template('screening_form.html', form=form)

@main.route('/screening/<int:screening_id>/report')
@login_required
def report(screening_id):
    screening = Screening.query.get_or_404(screening_id)
    if screening.user_id != current_user.id:
        flash('您没有权限查看此报告', 'danger')
        return redirect(url_for('main.index'))

    # 假设您有一个 report.html 模板来显示报告内容
    return render_template('report.html', screening=screening)

@main.route('/screening/list')
@login_required
def screening_list():
    screenings = Screening.query.filter_by(user_id=current_user.id).order_by(Screening.date_created.desc()).all()
    return render_template('screening_list.html', screenings=screenings)

@main.route('/screening/<int:screening_id>/delete', methods=['POST'])
@login_required
def delete_screening(screening_id):
    screening = Screening.query.get_or_404(screening_id)
    if screening.user_id != current_user.id:
        flash('您没有权限删除此筛查', 'danger')
        return redirect(url_for('main.screening_list'))

    try:
        # 删除相关文件
        if screening.image_3d_iso:
            os.remove(os.path.join(app.config['UPLOAD_FOLDER'], screening.image_3d_iso))

        db.session.delete(screening)
        db.session.commit()
        flash('筛查已删除', 'success')
    except Exception as e:
        db.session.rollback()
        flash(f'删除筛查失败: {e}', 'danger')
    return redirect(url_for('main.screening_list'))

@main.route('/profile')
@login_required
def profile():
    return render_template('profile.html', title='Profile')

@main.route('/logout')
def logout():
    logout_user()
    return redirect(url_for('main.index'))

@main.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

@main.errorhandler(500)
def internal_server_error(e):
    return render_template('500.html'), 500 