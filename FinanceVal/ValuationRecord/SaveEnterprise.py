import requests

from log.log import logger
from CONFIG.globalENV import global_var


def save(name):
    try:
        url = global_var.url
        check_url = url + 'pythonCheckEnterpriseName?enterpriseName='+name
        check_r = requests.get(url=check_url, timeout=5).json()
        if check_r is None:
            logger('【%s】的添加公司有误' % name)
        else:

            if check_r["code"] in 'S01':
                enterprise_id = check_r["results"]
                return enterprise_id
            else:
                logger('【%s】的添加公司有误' % name)
    except Exception as e:
        print('save', e)

if __name__ == '__main__':
    e_id = save(name = '上海中建东孚物业管理有限公司上海分公司')
    print(e_id)
