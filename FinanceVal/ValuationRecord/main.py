import json
from ValuationRecord import SaveValuationRecord
from flask import Flask, request
app = Flask(__name__)


@app.route("/saveValuationData", methods=['POST'])
def start():
    if request.method == 'POST':
        all_data = request.form
        if not all_data:
            return "缺少参数"
        else:
            if 'enterprise_name' not in all_data:
                return "缺少参数enterprise_name"
            elif 'mode' not in all_data:
                return "缺少参数mode"
            elif 'industry' not in all_data:
                return "缺少参数industry"
            elif 'peer' not in all_data:
                return "缺少参数peer"
            elif 'bal' not in all_data:
                return "缺少参数bal"
            elif 'flow' not in all_data:
                return "缺少参数flow"
            elif 'hypo' not in all_data:
                return "缺少参数hypo"
            elif 'bm' not in all_data:
                return "缺少参数bm"
            elif 'type' not in all_data:
                return "缺少参数type"
            elif 'basic' not in all_data:
                return "缺少参数basic"
            return json.dumps(SaveValuationRecord.start(all_data))


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80, debug=True)
