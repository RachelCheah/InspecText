from flask import Flask, render_template, request

app = Flask(__name__)
app.config['SECRET_KEY'] = 'inspectext'

####################################################################################################################

@app.route('/')
def home():
    return render_template('index.html')

# this is for the page to be redirected to home page when logo is pressed - TO FIND A MORE EFFICIENT WAY!
@app.route('/index.html')
def home():
    return render_template('index.html')
##########################################################################################################

@app.route('/start.html')
def about():
    return render_template('start.html')

@app.route('/whatsapp.html')
def whatsapp():
    return render_template('whatsapp.html')

@app.route('/telegram.html')
def telegram():
    return render_template('telegram.html')

@app.route('/aims.html')
def aims():
    return render_template('aims.html')

@app.route('/features.html')
def features():
    return render_template('features.html')

###################################################################################################################

if __name__ == '__main__':
    app.run()