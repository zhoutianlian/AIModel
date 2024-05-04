from pymongo import MongoClient

from CONFIG.globalENV import global_var


def get_table_id(table_name):
    try:
        conn = MongoClient('mongodb://hub:hubhub@' + global_var.name + ':%s/' % global_var.port)
        db = conn.rdt_fintech
        table_sequence = db.sequence
        query = {"collName": table_name}
        table_result = table_sequence.find_one(query)
        if table_result:
            num = table_result['seqId'] + 1
            update_query = {"$set": {'seqId': num}}
            table_sequence.update_one(query, update_query)
        else:
            num = 1
            table_sequence.insert({'collName': table_name, 'seqId': 1})
        return num
    except Exception as e:
        print(e)
