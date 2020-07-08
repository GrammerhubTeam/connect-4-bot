#/src/controllers/TodoController.py
from flask import Flask, make_response, current_app
from datetime import timedelta
from functools import update_wrapper
from flask import request, g, Blueprint, json, Response
from ..models.Connect4Model import connect4Predict
import requests
import threading

connect4_api = Blueprint('connect4_api', __name__)


def crossdomain(origin=None, methods=None, headers=None,
              max_age=21600, attach_to_all=True,
              automatic_options=True):
  if methods is not None:
      methods = ', '.join(sorted(x.upper() for x in methods))
  if headers is not None and not isinstance(headers, str):
      headers = ', '.join(x.upper() for x in headers)
  if not isinstance(origin, str):
      origin = ', '.join(origin)
  if isinstance(max_age, timedelta):
      max_age = max_age.total_seconds()

  def get_methods():
      if methods is not None:
          return methods

      options_resp = current_app.make_default_options_response()
      return options_resp.headers['allow']

  def decorator(f):
      def wrapped_function(*args, **kwargs):
          if automatic_options and request.method == 'OPTIONS':
              resp = current_app.make_default_options_response()
          else:
              resp = make_response(f(*args, **kwargs))
          if not attach_to_all and request.method != 'OPTIONS':
              return resp

          h = resp.headers

          h['Access-Control-Allow-Origin'] = origin
          h['Access-Control-Allow-Methods'] = get_methods()
          h['Access-Control-Max-Age'] = str(max_age)
          if headers is not None:
              h['Access-Control-Allow-Headers'] = headers
          return resp

      f.provide_automatic_options = False
      return update_wrapper(wrapped_function, f)
  return decorator


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
@crossdomain(origin="*")
def do_predict():
  """
  Get All Connect4
  """
  all_args = request.args
  req_data = request.get_json()
  
  if req_data.get('state') and req_data.get('available_actions'):
    prediction = connect4Predict(req_data.get('state'), req_data.get('available_actions'), req_data.get('steps_done'))
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
