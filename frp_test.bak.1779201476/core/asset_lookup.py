"""
asset_lookup.py — 资产查询模块

设计目标：
    1. 给定目标 IP（通常是 anchor 流的 frps 端），返回资产指纹信息
    2. 抽象 lookup 接口，方便未来切换到真正的知识图谱
    3. 当前用 SIM 数据库（硬编码 6 个实验室 IP:port 的资产信息）
    4. 未来知识图谱建好后，只需实现 query_asset_from_knowledge_graph 函数
       并放到 rag.py 里，本模块会自动优先用它

资产信息典型用途：
    - 在 final_verdict 时与子 skill 的检测结果交叉印证
    - 比如 frp_wss skill 命中"SNI 可疑 + Go JA3" 时，资产信息显示
      "目标 IP:443 是 nginx + 自签证书" → 三个独立证据形成强闭环
    - 对于 frp_quic，资产显示 UDP 端口 ALPN 包含 'frp' 是决定性
    - 对于 frp_tcp（无 hard fingerprint），资产显示"目标端口未指纹化
      的服务" 是关键辅助证据
"""

import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════
# 模拟资产数据库（SIM）
# 待 gyf 项目知识图谱（rag.py + Neo4j gyf_history db）建好后，
# 通过 lookup_asset() 函数自动切换到真实数据，本 SIM_DB 仅作回退。
# ════════════════════════════════════════════════════════════════════════

ASSET_SIM_DB = {
    "192.168.100.10": {
        "ip": "192.168.100.10",
        "asset_type": "server",
        "hostname_cache": ["c2.attacker-lab.test"],
        "open_services": [
            {
                "port": 7000, "proto": "tcp",
                "service": "unknown",
                "banner": "",
                "tls": True,                                # frp v0.50+ 默认 TLS
                "tls_cert": {
                    "subject_cn": "c2.attacker-lab.test",
                    "issuer_cn": "attacker-lab.test CA",
                    "self_signed": True,
                    "san": ["c2.attacker-lab.test"]
                },
                "fingerprint_tags": ["non_standard_port", "self_signed_cert", "unknown_service"]
            },
            {
                "port": 7001, "proto": "tcp",
                "service": "unknown_tls",
                "banner": "",
                "tls": True,
                "tls_cert": {
                    "subject_cn": "c2.attacker-lab.test",
                    "issuer_cn": "attacker-lab.test CA",
                    "self_signed": True,
                    "san": ["c2.attacker-lab.test"]
                },
                "fingerprint_tags": ["non_standard_port", "self_signed_cert", "forced_tls"]
            },
            {
                "port": 7002, "proto": "udp",
                "service": "kcp_or_custom_udp",
                "banner": "",
                "fingerprint_tags": ["long_udp_session", "non_dns_non_quic_udp"]
            },
            {
                "port": 7003, "proto": "udp",
                "service": "quic",
                "alpn": ["frp"],                            # ★★★ 决定性
                "fingerprint_tags": ["quic_non_h3", "alpn_anomaly_frp"]
            },
            {
                "port": 7004, "proto": "tcp",
                "service": "http",
                "server_header": "(none)",                  # frp ws 服务端不带 Server 头
                "tls": False,
                "fingerprint_tags": ["http_no_server_header", "frp_websocket_suspect"]
            },
            {
                "port": 443, "proto": "tcp",
                "service": "http",
                "server_header": "nginx/1.20.1",            # ★★★ WSS 的强信号
                "tls": True,
                "tls_cert": {
                    "subject_cn": "c2.attacker-lab.test",
                    "issuer_cn": "attacker-lab.test CA",
                    "self_signed": True,
                    "san": ["c2.attacker-lab.test"]
                },
                "fingerprint_tags": ["nginx", "self_signed_cert", "reverse_proxy_candidate", "wss_terminator"]
            }
        ],
        "passive_dns": ["c2.attacker-lab.test"],
        "geo_country": "internal/lab",
        "as_owner": "private network",
        "first_seen": "2026-05-01",
        "last_seen": "2026-05-17",
        "risk_score": 9.5,
        "tags": [
            "suspicious_self_signed_cert",
            "multiple_non_standard_ports_open",
            "frp_indicator_strong"
        ],
        "source": "SIM (待替换为 rag.py 知识图谱真实数据)"
    },
}


# ════════════════════════════════════════════════════════════════════════
# 接口函数
# ════════════════════════════════════════════════════════════════════════

def lookup_asset(ip: str, port: Optional[int] = None) -> Dict[str, Any]:
    """
    查询 IP（或 IP:Port）的资产信息。
    
    优先尝试 rag.py 提供的知识图谱接口；失败则回退到 SIM 数据。
    
    Args:
        ip: 目标 IP（通常是 anchor 流的 frps 端）
        port: 可选端口，提供则在结果里挂上对应服务详情到 service_on_query_port
    
    Returns:
        {
            "ip": str,
            "asset_type": "server" | "client" | "unknown",
            "open_services": [...],                       # 该 IP 开放的所有服务
            "service_on_query_port": {...},               # 仅当 port 不为 None
            "tags": [...],                                # 高层标签
            "risk_score": float,
            "source": "knowledge_graph" | "SIM" | "not_found"
        }
    """
    # 1. 尝试真实知识图谱（未来 rag.py 会提供）
    asset = _try_knowledge_graph(ip, port)
    if asset is not None:
        return asset
    
    # 2. 回退到 SIM 数据库
    sim = ASSET_SIM_DB.get(ip)
    if sim:
        result = {k: v for k, v in sim.items() if k != "open_services"}
        result["open_services"] = list(sim.get("open_services", []))
        if port is not None:
            for svc in result["open_services"]:
                if svc.get("port") == port:
                    result["service_on_query_port"] = dict(svc)
                    break
        result["source"] = "SIM (sandbox/lab data)"
        return result
    
    # 3. 完全没记录
    return {
        "ip": ip,
        "asset_type": "unknown",
        "open_services": [],
        "tags": [],
        "risk_score": 0.0,
        "source": "not_found"
    }


def _try_knowledge_graph(ip: str, port: Optional[int]) -> Optional[Dict[str, Any]]:
    """
    尝试调用 rag.py 的知识图谱查询。
    
    本函数预留切换点：未来 rag.py 实现 query_asset_from_knowledge_graph(ip, port)
    后，自动启用真实数据。
    """
    try:
        from rag import query_asset_from_knowledge_graph  # type: ignore
    except ImportError:
        return None  # rag.py 还没实现这个函数，回退到 SIM
    
    try:
        result = query_asset_from_knowledge_graph(ip, port=port)
        if result:
            result.setdefault("source", "knowledge_graph (Neo4j via rag.py)")
            return result
    except Exception as e:
        logger.warning(f"知识图谱查询异常（回退 SIM）: {e}")
    
    return None


def summarize_asset_for_llm(asset: Dict[str, Any], port: Optional[int] = None) -> str:
    """
    把资产信息浓缩成一段适合喂给 LLM 的多行文本。
    高亮关键风险信号，方便 LLM 在 final_verdict 时与 skill 证据交叉印证。
    """
    if not asset or asset.get("asset_type") == "unknown":
        return f"- 资产查询: {asset.get('ip')} 无记录"
    
    lines = []
    ip = asset.get("ip")
    src = asset.get("source", "?")
    tags = asset.get("tags", [])
    risk = asset.get("risk_score")
    
    lines.append(f"- 资产记录 ({ip}, source={src}): "
                 f"risk_score={risk}, tags={tags[:5]}")
    
    # 查询端口的服务详情
    svc = asset.get("service_on_query_port")
    if svc:
        svc_name = svc.get("service", "?")
        server = svc.get("server_header", "")
        tls = svc.get("tls", False)
        cert = svc.get("tls_cert", {}) or {}
        self_signed = cert.get("self_signed", False)
        cert_cn = cert.get("subject_cn", "")
        alpn = svc.get("alpn", [])
        svc_tags = svc.get("fingerprint_tags", [])
        
        port_line = f"  目标端口 {svc.get('port')}/{svc.get('proto')}: service={svc_name}"
        if server:
            port_line += f", Server={server}"
        if tls:
            port_line += f", TLS=true (cert_cn='{cert_cn}', self_signed={self_signed})"
        if alpn:
            port_line += f", ALPN={alpn}"
        port_line += f", 端口标签={svc_tags}"
        lines.append(port_line)
    
    # 主机名缓存
    hostnames = asset.get("hostname_cache", [])
    if hostnames:
        lines.append(f"  历史主机名: {hostnames[:3]}")
    
    return "\n".join(lines)


def is_using_simulated_data() -> bool:
    """供调试/告警用：返回当前是否在用 SIM 数据"""
    try:
        from rag import query_asset_from_knowledge_graph  # type: ignore
        return False
    except ImportError:
        return True
