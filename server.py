from flask import Flask, render_template, request

app = Flask(__name__)
app.config['SECRET_KEY'] = 'inspectext'

####################################################################################################################

@app.route('/')
@app.route('/home')
def gobackhome():
    return render_template('index.html')
##########################################################################################################

@app.route('/start/')
def about():
    return render_template('start.html')

@app.route('/start/whatsapp/', methods=['POST', 'GET'])
def whatsapp():
    if request.method == "POST":
        file = request.files["fileWhatsapp"].read() #file is now the text file
        if file:
            return redirect('/output')
    return render_template('whatsapp.html')

@app.route('/start/telegram/', methods=['POST', 'GET'])
def telegram():
    if request.method == "POST":
        file = request.files["fileTelegram"].read() #file is now the text file
        if file:
            return redirect('/output')
    return render_template('telegram.html')

@app.route('/aims/')
def aims():
    return render_template('aims.html')

@app.route('/features/')
def features():
    return render_template('features.html')

@app.route('/customize/')
def customize():
    return render_template('customize.html')

@app.route('/output/')
def output():
    return render_template('output.html')
    
###################################################################################################################

if __name__ == '__main__':
    app.run(debug=True)
