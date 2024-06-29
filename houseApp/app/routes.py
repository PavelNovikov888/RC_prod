from app import app
from app.forms import EditForm
from flask import render_template, redirect, request
from app.ml_model import user_service
from app.ml_model import item_service
from app.ml_model import item_service_unknown
import pickle
# import os

# PATH = '/home/user/Рабочий стол/Prod_f/houseApp/app/files/'
PATH = './app/files/'

@app.route('/', methods=['GET'])
def button():
    return render_template('button.html')


@app.route('/submit', methods=['POST'])
def submit():
    redirect_value = request.form['redirect']
    if redirect_value == "users":
        return redirect("/users")
    elif redirect_value == "items":
        return redirect("/items")
    else:
        return "Неправильное значение кнопки."

    
@app.route('/users', methods=['GET','POST'])
def get_hot1():
    if request.method == 'GET':
        form = EditForm()
        return render_template('user_history.html', form=form, cost = "")
    if request.method == 'POST':
        args = request.form
        user = args['user']

        form = EditForm()
        form.cost.data = user
        
        g, g1, g2, g3, g4 = user_service(int(user))
        return render_template('user_history.html', form=form, cost = str(g), cost1 = str(g1), cost2 = str(g2), cost3 = str(g3), cost4 = str(g4))
    
    
@app.route('/items', methods=['GET','POST'])
def items_recom():
    if request.method == 'GET':
        form = EditForm()
        return render_template('item_history.html', form=form, cost = "")
    if request.method == 'POST':
        args = request.form
        itemid = args['item']

        form = EditForm()
        form.cost.data = itemid
        
        with open (f'{PATH}inv_item_mappings', 'rb') as fp:
            inv_item_mappings = pickle.load(fp)
    
        if int(itemid) in list(set(inv_item_mappings.keys())):
            g, g1, g2, g3, g4 = item_service(int(itemid))
            return render_template('item_history.html', form=form, cost = str(g), cost1 = str(g1), cost2 = str(g2), cost3 = str(g3), cost4 = str(g4))
        else:
            f1,f2,f3,f4,f5,f6,f7 = item_service_unknown(int(itemid))
            return render_template('item_history_unknown.html', form=form, \
                                   cost1 = str(f1), cost2 = str(f2), cost3 = str(f3), cost4 = str(f4), cost5 = f5,\
                                    cost6 = str(f6), cost7 = f7) 

# if __name__ == '__main__':
#     app.run(debug=True)