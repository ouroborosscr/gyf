from pymongo import MongoClient, DESCENDING
from utils import config

def _col(name):
    c = MongoClient(config.DATABASE["mongo"]["uri"])
    return c[config.DATABASE["mongo"]["db_name"]][name]

def find_ssl_by_uid(collection, uid):
    return _col(collection).find_one({"uid": uid})

def find_ssl_by_5tuple(collection, sip, sport, dip, dport):
    return _col(collection).find_one({
        "id.orig_h": sip, "id.orig_p": sport,
        "id.resp_h": dip, "id.resp_p": dport,
    })

def find_resumed_session_origin(collection, session_id, before_uid=None):
    """根据 session_id 反查首次完整握手的那条 ssl 记录"""
    q = {"session_id": session_id, "resumed": {"$ne": True}}
    return _col(collection).find_one(q, sort=[("ts", DESCENDING)])