import os
import json
import shutil

# 1. 创建符合实验代码要求的全部标准目录（新增 val2014 文件夹）
os.makedirs('RSTPReid_new/annotations', exist_ok=True)
os.makedirs('RSTPReid_new/train2014', exist_ok=True)
os.makedirs('RSTPReid_new/val2014', exist_ok=True)
os.makedirs('RSTPReid_new/test2014', exist_ok=True)

source_json = 'data_captions.json'
source_img_dir = 'imgs'  # 对应你现有的图片文件夹

if not os.path.exists(source_json):
    print(f"❌ 错误：未在当前目录下找到 {source_json}，请确认原始标注文件名！")
    exit()
if not os.path.exists(source_img_dir):
    print(f"❌ 错误：未在当前目录下找到 {source_img_dir} 文件夹！")
    exit()

with open(source_json, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 初始化 3 个独立的 COCO 格式容器
train_coco = {"images": [], "annotations": []}
val_coco = {"images": [], "annotations": []}
test_coco = {"images": [], "annotations": []}

ann_id = 0
print("正在为你严格切分【训练/验证/测试】三路数据集并自动分流图片，请稍候...")

for idx, item in enumerate(data):
    split = item.get('split', 'train')

    img_id = item.get('id') if item.get('id') is not None else item.get('img_id', idx)
    image_path = item.get('img_path') or item.get('image_path') or item.get('image_name') or item.get(
        'img_name') or item.get('file_name')
    captions = item.get('captions', [])

    if image_path is None:
        print(f"\n❌ 错误：在数据第 {idx} 个条目中找不到任何代表图片文件名的键！")
        print(f"该条目实际包含的键名为: {list(item.keys())}")
        exit()

    image_info = {"id": img_id, "file_name": image_path}

    # 🎯 严格判定并分流到对应的文件夹和 JSON 容器中
    if split == 'train':
        target_coco = train_coco
        target_img_folder = 'RSTPReid_new/train2014'
    elif split == 'val':
        target_coco = val_coco
        target_img_folder = 'RSTPReid_new/val2014'
    elif split == 'test':
        target_coco = test_coco
        target_img_folder = 'RSTPReid_new/test2014'
    else:
        target_coco = train_coco
        target_img_folder = 'RSTPReid_new/train2014'

    target_coco["images"].append(image_info)

    for cap in captions:
        target_coco["annotations"].append({
            "id": ann_id,
            "image_id": img_id,
            "caption": cap
        })
        ann_id += 1

    # 自动将图片分流复制到特定的 train2014 / val2014 / test2014 文件夹下
    src_img_path = os.path.join(source_img_dir, image_path)
    if os.path.exists(src_img_path):
        shutil.copy(src_img_path, os.path.join(target_img_folder, image_path))
    else:
        found = False
        for root, dirs, files in os.walk(source_img_dir):
            if image_path in files:
                shutil.copy(os.path.join(root, image_path), os.path.join(target_img_folder, image_path))
                found = True
                break

# 写入标准 COCO 格式的 JSON（保持 ensure_ascii=True 防止 GBK 报错）
with open('RSTPReid_new/annotations/captions_train2014.json', 'w', encoding='utf-8') as f:
    json.dump(train_coco, f, ensure_ascii=True, indent=4)
with open('RSTPReid_new/annotations/captions_val2014.json', 'w', encoding='utf-8') as f:
    json.dump(val_coco, f, ensure_ascii=True, indent=4)
with open('RSTPReid_new/annotations/captions_test2014.json', 'w', encoding='utf-8') as f:
    json.dump(test_coco, f, ensure_ascii=True, indent=4)

print("🎉 恭喜！完美的【训练+验证+测试】三路数据集重构全部完成！")