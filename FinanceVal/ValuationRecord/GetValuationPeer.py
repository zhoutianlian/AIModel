import pymysql

from CONFIG.globalENV import global_var


def get_peer(peer_s):
    peer_ids = {}
    db = pymysql.connect(global_var.name, global_var.user, global_var.password, 'rdt_fintech')
    cursor = db.cursor()
    try:
        for peer in peer_s:
            values = peer_s[peer]
            peer_ids[peer] = []
            for value in values:
                sql = "select id from t_company_industry_code where stock_code='" + str(value) + "'"
                cursor.execute(sql)
                find = cursor.fetchall()
                if find:
                    ind_code = int(find[0][0])
                    peer_ids[peer].append(ind_code)
    except Exception as e:
        print('peer', e)
    db.close()
    return peer_ids
