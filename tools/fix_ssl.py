"""修复 SSL 证书问题"""
import ssl
import certifi

# 安装证书
import os
import subprocess

print("修复 SSL 证书...")

# 对于 macOS，运行证书安装命令
cert_script = "/Applications/Python 3.13/Install Certificates.command"
if os.path.exists(cert_script):
    print(f"运行: {cert_script}")
    subprocess.run([cert_script], shell=True)
    print("✅ 证书已安装")
else:
    print(f"证书脚本不存在: {cert_script}")
    print("尝试手动设置...")
    
# 显示当前证书位置
print(f"\nCertifi 证书位置: {certifi.where()}")

# 设置环境变量
print("\n请运行以下命令:")
print(f"export SSL_CERT_FILE={certifi.where()}")
print(f"export REQUESTS_CA_BUNDLE={certifi.where()}")
