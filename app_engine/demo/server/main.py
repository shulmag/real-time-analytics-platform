'''
'''
import os

from flask import Flask, Blueprint
from flask_restful import Api
from flask_cors import CORS
from firebase_admin import credentials
import firebase_admin

from resources import YieldCurvePlot, GetPriceAndYTW, DailySchoonover, YieldCurveTable, BatchPricing, NewProspect, GetSimilarBonds, Compliance


# defined in `deploy-ficc-servers.sh` with `--set-env-vars` flag
ROUTE_TO_DIFFERENT_SERVER = os.getenv('ROUTE_TO_DIFFERENT_SERVER')
assert ROUTE_TO_DIFFERENT_SERVER in ('True', 'False')
ROUTE_TO_DIFFERENT_SERVER = True if ROUTE_TO_DIFFERENT_SERVER == 'True' else False
print('ROUTE_TO_DIFFERENT_SERVER:', ROUTE_TO_DIFFERENT_SERVER, type(ROUTE_TO_DIFFERENT_SERVER))

# firebase Initialization; check first if `firebase_admin` is already loaded to limit memory usage required for initialization
if not firebase_admin._apps:
    cred = credentials.Certificate('fbAdminConfig.json')
    firebase_admin.initialize_app(cred)

app = Flask(__name__)
CORS(app)
api = Api(app)

blueprint = Blueprint('api', __name__, url_prefix='/api')
api.init_app(blueprint)
app.register_blueprint(blueprint)

# Register routes
api.add_resource(YieldCurvePlot, '/yield')
api.add_resource(GetPriceAndYTW, '/price')
api.add_resource(DailySchoonover, '/dailySchoonover')
api.add_resource(YieldCurveTable, '/realtimeyieldcurve')
api.add_resource(BatchPricing, '/batchpricing', resource_class_args=(ROUTE_TO_DIFFERENT_SERVER,))
api.add_resource(NewProspect, '/createcustomer')
api.add_resource(GetSimilarBonds, '/getsimilarbonds')
api.add_resource(Compliance, '/compliance', resource_class_args=(ROUTE_TO_DIFFERENT_SERVER,))


if __name__ == '__main__':    # this `if` statement is never entered in production
    app.run(port=5000)    # (port=8001)
