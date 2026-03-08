import subprocess
import os
import json
import logging
import shutil
import uuid

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def _repair_pcap_via_tmp(filepath):
    """
    通过将文件移动到 /tmp 来绕过 AppArmor/权限问题。
    """
    # 生成唯一的临时文件名，防止冲突
    unique_id = str(uuid.uuid4())[:8]
    filename = os.path.basename(filepath)
    
    # 1. 定义临时路径
    tmp_source = f"/tmp/{unique_id}_src_{filename}"
    tmp_fixed = f"/tmp/{unique_id}_fixed_{filename}"
    
    try:
        # A. 将原文件拷贝到 /tmp (绕过读取权限限制)
        shutil.copy2(filepath, tmp_source)
        
        # B. 把临时文件的权限全开 (chmod 777)
        os.chmod(tmp_source, 0o777)
        
        # C. 运行 tcpdump
        # 注意：不再需要 -Z root，因为在 /tmp 下通常 tcpdump 用户也能读
        # 如果还是不行，我们在外部用 sudo python 运行脚本即可
        cmd = ["tcpdump", "-r", tmp_source, "-w", tmp_fixed]
        
        # 允许报错退出 (check=False)
        subprocess.run(cmd, check=False, capture_output=True)
        
        # D. 检查修复结果
        if os.path.exists(tmp_fixed) and os.path.getsize(tmp_fixed) > 24:
            # 记录大小变化
            old_size = os.path.getsize(filepath)
            new_size = os.path.getsize(tmp_fixed)
            
            # E. 覆盖回原位置
            shutil.move(tmp_fixed, filepath)
            
            logging.info(f"修复成功: {filename} (Size: {old_size} -> {new_size})")
            return True
        else:
            logging.warning(f"修复无效: {filename} (生成文件为空或过小)")
            return False
            
    except Exception as e:
        logging.error(f"处理异常 {filename}: {e}")
        return False
        
    finally:
        # F. 清理垃圾
        if os.path.exists(tmp_source):
            os.remove(tmp_source)
        if os.path.exists(tmp_fixed):
            # 如果上面 move 成功了，这里就不存在了；如果失败了，这里清理掉
            try:
                os.remove(tmp_fixed)
            except OSError:
                pass

def main():
    pcap_dir = "./pcap"
    json_file = "failed_files.json"
    
    if not os.path.exists(json_file):
        logging.error("未找到 failed_files.json")
        return

    with open(json_file, "r") as f:
        failed_files = json.load(f)

    if not failed_files:
        logging.info("failed_files.json 为空。")
        return

    logging.info(f"Step 2 (AppArmor Bypass): 开始修复 {len(failed_files)} 个文件...")
    logging.info("提示: 建议使用 'sudo python step2_repair.py' 运行以获得最佳权限支持")
    
    repaired_files = []
    
    for idx, f in enumerate(failed_files):
        full_path = os.path.join(pcap_dir, f)
        
        if not os.path.exists(full_path):
            logging.error(f"文件丢失: {f}")
            continue

        if _repair_pcap_via_tmp(full_path):
            repaired_files.append(f)

    # 写入成功列表
    with open("repaired_files.json", "w") as f:
        json.dump(repaired_files, f)
        
    logging.info(f"Step 2 完成. 提交: {len(failed_files)}, 成功: {len(repaired_files)}")

if __name__ == "__main__":
    main()