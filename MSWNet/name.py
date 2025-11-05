import os
import re

# 自然排序函数
def natural_key(text):
    return [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', text)]

folder_path = r"C:\Users\PC\Desktop\LUT-Fuse-main\CPSTN(5）完全损失\result\MSRS"
save_ext = ".png"
valid_exts = ['.jpg', '.jpeg', '.png', '.bmp', '.tif']

# 读取并自然排序
files = [f for f in os.listdir(folder_path) if os.path.splitext(f)[-1].lower() in valid_exts]
files.sort(key=natural_key)

# 重命名
start_index = 1
for idx, filename in enumerate(files):
    old_path = os.path.join(folder_path, filename)
    new_name = f"{idx+1:05d}{save_ext}"
    new_path = os.path.join(folder_path, new_name)
    os.rename(old_path, new_path)
    print(f"重命名: {filename} → {new_name}")

print("✅ 重命名完成，总数：", len(files))
# import os
# import re
#
# # 自然排序函数
# def natural_key(text):
#     return [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', text)]
#
# folder_path = r"C:\Users\PC\Desktop\com\data\road\TNO\vi"
# save_ext = ".png"
# valid_exts = ['.jpg', '.jpeg', '.png', '.bmp', '.tif']
#
# # 读取并自然排序
# files = [f for f in os.listdir(folder_path) if os.path.splitext(f)[-1].lower() in valid_exts]
# files.sort(key=natural_key)
#
# # 重命名，从00222开始
# start_index = 222
# for idx, filename in enumerate(files):
#     old_path = os.path.join(folder_path, filename)
#     new_name = f"{start_index + idx:05d}{save_ext}"
#     new_path = os.path.join(folder_path, new_name)
#     os.rename(old_path, new_path)
#     print(f"重命名: {filename} → {new_name}")
