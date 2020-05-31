from flask import Flask, render_template, request

app = Flask(__name__)
app.config['SECRET_KEY'] = 'inspectext'

####################################################################################################################

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/start')
def about():
    return render_template('start.html')

@app.route('/start/whatsapp')
def whatsapp():
    return render_template('whatsapp.html')

@app.route('/start/telegram')
def telegram():
    return render_template('telegram.html')

@app.route('/aims')
def aims():
    return render_template('aims.html')

@app.route('/features')
def features():
    return render_template('features.html')

###################################################################################################################

if __name__ == '__main__':
    app.run()
