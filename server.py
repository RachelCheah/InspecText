from flask import Flask, flash, render_template, request, redirect, make_response
import pdfkit

app = Flask(__name__)
app.config['SECRET_KEY'] = 'inspectext'
file = '' 

def process(textFile):
    short = '' 
    for i in str(textFile):
        short += i
        if i == ']' or i == '\\':
            break
    return short

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
    global file
    if request.method == "POST":
        file = request.files["fileWhatsapp"].read() #file is now the text file
        if file:
            return redirect('/output')
        else:
            flash("No file selected for uploading")
            redirect('/start/whatsapp/')
    return render_template('whatsapp.html')

@app.route('/start/telegram/', methods=['POST', 'GET'])
def telegram():
    global file
    if request.method == "POST":
        file = request.files["fileTelegram"].read() #file is now the text file
        if file:
            return redirect('/output')
        else:
            flash("No file selected for uploading")
            redirect('/start/telegram/')
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
    pdf = pdfkit.from_string(process(file), False) #False keeps the pdf in memory
    response = make_response(pdf)
    response.headers['Content-Type'] = 'application/pdf'
    response.headers['Content-Disposition'] = 'inline; filename=inspecText.pdf'
    return response    
    
###################################################################################################################

if __name__ == '__main__':
    app.run(debug=True)
