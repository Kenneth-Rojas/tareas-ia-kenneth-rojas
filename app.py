from flask import Flask, request, jsonify, render_template
from models import db, Task
from datetime import datetime

app = Flask(__name__)

# Configuración de la base de datos
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///tasks.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)

# Crear base de datos si no existe
with app.app_context():
    db.create_all()

# Ruta principal para mostrar el frontend
@app.route('/')
def index():
    return render_template('index.html')

# -----------------------
# Rutas API REST
# -----------------------

# Obtener todas las tareas
@app.route('/api/tasks', methods=['GET'])
def get_tasks():
    tasks = Task.query.all()
    return jsonify([task.to_dict() for task in tasks])

# Crear una nueva tarea
@app.route('/api/tasks', methods=['POST'])
def create_task():
    data = request.json
    due_date = datetime.strptime(data['due_date'], '%Y-%m-%d').date() if data.get('due_date') else None
    task = Task(
        title=data['title'],
        description=data.get('description', ''),
        status=data.get('status', 'Por hacer'),
        due_date=due_date
    )
    db.session.add(task)
    db.session.commit()
    return jsonify(task.to_dict()), 201

# Actualizar una tarea
@app.route('/api/tasks/<int:task_id>', methods=['PUT'])
def update_task(task_id):
    task = Task.query.get_or_404(task_id)
    data = request.json
    task.title = data.get('title', task.title)
    task.description = data.get('description', task.description)
    task.status = data.get('status', task.status)
    if data.get('due_date'):
        task.due_date = datetime.strptime(data['due_date'], '%Y-%m-%d').date()
    db.session.commit()
    return jsonify(task.to_dict())

# Eliminar una tarea
@app.route('/api/tasks/<int:task_id>', methods=['DELETE'])
def delete_task(task_id):
    task = Task.query.get_or_404(task_id)
    db.session.delete(task)
    db.session.commit()
    return jsonify({'message': 'Tarea eliminada'})

# Obtener una sola tarea por ID
@app.route('/api/tasks/<int:task_id>', methods=['GET'])
def get_task(task_id):
    task = Task.query.get_or_404(task_id)
    return jsonify(task.to_dict())

if __name__ == '__main__':
    app.run(debug=True)
