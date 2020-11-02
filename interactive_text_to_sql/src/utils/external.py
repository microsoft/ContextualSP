import json
import requests


def complex_rephrase(query, span, target_span):
    url = 'http://172.16.13.25:9009'
    headers = {'apikey': '9212c3f48ad1766bcf63535710686d07'}
    params = {'query': query,
              'span': span,
              'target': target_span}
    r = requests.get(url, params=params, headers=headers)
    return json.loads(r.text['modified_query'])
