#/src/controllers/TodoController.py
from flask import request, g, Blueprint, json, Response
from ..models.Connect4Model import connect4Predict
import requests
import threading

connect4_api = Blueprint('connect4_api', __name__)


@connect4_api.route('/', methods=['GET'])
def get_all():
  """
  Get All Connect4
  """
  all_args = request.args
  # Static method in Connect4 model
  # connect4 = Connect4Model.get_all_connect4(all_args)
  connect4 = "THIS IS ALL THE CONNECT4"
  return custom_response(connect4, 200)

@connect4_api.route('/predict', methods=['POST'])
def do_predict():
  """
  Get All Connect4
  """
  all_args = request.args
  req_data = request.get_json()
  
  if req_data.get('state') and req_data.get('available_actions'):
    prediction = connect4Predict(req_data.get('state'), req_data.get('available_actions'))
    print("====================================")
    print(prediction)
    print("====================================")
    connect4 = {'column': prediction}
    return custom_response(connect4, 200)
  return custom_response({'error': 'Missing the state or available actions'}, 400)

@connect4_api.route('/train', methods=['GET'])
def do_train():
  """
  Get All Connect4
  """
  all_args = request.args
  # Static method in Connect4 model
  # connect4 = Connect4Model.get_all_connect4(all_args)
  connect4 = "I SHOULD REALLY BE TRAINING RIGHT NOW (LAZY)"
  return custom_response(connect4, 200)
  

def custom_response(res, status_code):
  """
  Custom Response Function
  """
  return Response(
    mimetype="application/json",
    response=json.dumps(res),
    status=status_code
  )
