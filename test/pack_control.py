import subprocess
import sys
import json
import os
import argparse


def run_command(command):
    """执行系统命令并返回结果"""
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"执行命令失败: {e}")
        return None


def export_conda_environment(output_file="environment.yml"):
    """导出当前conda环境到YAML文件"""
    print(f"正在导出当前conda环境到 {output_file}...")
    try:
        # 使用conda env export命令导出环境
        result = subprocess.run(['conda', 'env', 'export'], capture_output=True, text=True, check=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(result.stdout)

        print(f"环境已成功导出到 {output_file}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"导出环境失败: {e}")
        return False
    except Exception as e:
        print(f"保存文件时出错: {e}")
        return False


def export_conda_list(output_file="conda_list.txt"):
    """导出当前conda包列表到文本文件"""
    print(f"正在导出conda包列表到 {output_file}...")
    try:
        result = subprocess.run(['conda', 'list'], capture_output=True, text=True, check=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(result.stdout)

        print(f"包列表已成功导出到 {output_file}")
        return True
    except Exception as e:
        print(f"导出包列表失败: {e}")
        return False


def import_conda_environment(env_file="environment.yml"):
    """从YAML文件导入conda环境"""
    if not os.path.exists(env_file):
        print(f"环境文件 {env_file} 不存在")
        return False

    print(f"正在从 {env_file} 导入conda环境...")
    try:
        # 使用conda env create命令创建环境
        subprocess.run(['conda', 'env', 'create', '-f', env_file], check=True)
        print("环境导入成功")
        return True
    except subprocess.CalledProcessError as e:
        print(f"导入环境失败: {e}")
        return False


def create_requirements_txt():
    """创建pip格式的requirements.txt文件"""
    print("正在创建requirements.txt文件...")
    try:
        # 获取pip包列表
        result = subprocess.run([sys.executable, '-m', 'pip', 'list', '--format=json'],
                                capture_output=True, text=True, check=True)

        packages = json.loads(result.stdout)

        with open('requirements.txt', 'w', encoding='utf-8') as f:
            for package in packages:
                f.write(f"{package['name']}=={package['version']}\n")

        print("requirements.txt文件创建成功")
        return True
    except Exception as e:
        print(f"创建requirements.txt失败: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Conda环境管理工具')
    parser.add_argument('--export', action='store_true', help='导出当前环境')
    parser.add_argument('--import-env', action='store_true', help='导入环境')
    parser.add_argument('--env-file', default='environment.yml', help='环境文件路径')
    parser.add_argument('--list-export', action='store_true', help='导出包列表')
    parser.add_argument('--pip-requirements', action='store_true', help='创建pip requirements.txt')

    args = parser.parse_args()

    if args.export:
        export_conda_environment(args.env_file)
    elif args.import_env:
        import_conda_environment(args.env_file)
    elif args.list_export:
        export_conda_list()
    elif args.pip_requirements:
        create_requirements_txt()
    else:
        # 默认导出所有格式
        export_conda_environment()
        export_conda_list()
        create_requirements_txt()


if __name__ == "__main__":
    main()