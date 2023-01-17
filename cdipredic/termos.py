import functools

from flask import (
    Blueprint, flash, g, redirect, render_template, request, session, url_for
)
from werkzeug.security import check_password_hash, generate_password_hash

bp = Blueprint('termos', __name__, url_prefix='/')

@bp.route('/termos', methods=('GET', 'POST'))
def termos():

    return render_template('termos/termos.html')

@bp.route('/api', methods=(['GET']))
def login():
    return {"data": "dados"}