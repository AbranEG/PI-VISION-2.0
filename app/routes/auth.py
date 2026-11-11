from flask import Blueprint, render_template, request, redirect, url_for, session, flash
import sqlite3
from werkzeug.security import check_password_hash

auth_bp = Blueprint('auth', __name__)

from db import DB_PATH  # base unificada: database/pivision.db

# Página de Login
@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT id, username, password_hash, role FROM users WHERE username = ?", (username,))
        user = cursor.fetchone()
        conn.close()

        if user and check_password_hash(user[2], password):
            session['user_id'] = user[0]
            session['username'] = user[1]
            session['role'] = user[3]
            flash("Inicio de sesión exitoso", "success")
            return redirect(url_for('dashboard.index'))  # Ajusta según tu ruta principal
        else:
            flash("Credenciales inválidas", "danger")

    return render_template('login.html')


# Logout
@auth_bp.route('/logout')
def logout():
    session.clear()
    flash("Sesión cerrada correctamente", "info")
    return redirect(url_for('auth.login'))


# Decorador para rutas protegidas
from functools import wraps

def login_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if 'user_id' not in session:
            flash("Debes iniciar sesión", "warning")
            return redirect(url_for('auth.login'))
        return f(*args, **kwargs)
    return wrapper

def admin_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if session.get('role') != 'admin':
            flash("Acceso restringido a administradores", "danger")
            return redirect(url_for('dashboard.index'))
        return f(*args, **kwargs)
    return wrapper
